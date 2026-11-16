//! Batched (prefill) versions of the `qwen35moe` glue kernels in `qwen_ops.rs`: a whole chunk of `m` tokens at positions `p0 ..` goes through
//! the norms, the causal conv and delta rule, the gated attention and the router in a few launches per layer. The matrix products are the
//! tiled GEMMs of `qgemm.rs`. Written to give the same per-token result as the decode kernels (same formulas; sums in a different order).

use crate::qwen_ops::{sigmoid_pub, silu_pub};
use crate::store::Gpu;
use crate::lane_sum;
use cubecl::prelude::*;

type Handle = cubecl::server::Handle;

/// `out[t, r] = w[r] . x[t]` for f32 matrices (router, beta/alpha) over `m` token rows; a plane an output.
#[cube(launch)]
fn qp_f32_rows_kernel(w: &[f32], x: &[f32], out: &mut [f32], #[comptime] cols: usize, #[comptime] rows: usize) {
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let mut o = CUBE_POS as usize * ppc + plane_id;
    while o < out.len() {
        let t = o / rows;
        let r = o % rows;
        let mut part = 0.0f32;
        let mut c = lane;
        while c < cols {
            part += w[r * cols + c] * x[t * cols + c];
            c += 32usize;
        }
        let tot = plane_sum(part);
        if lane == 0usize {
            out[o] = tot;
        }
        o += step;
    }
}

/// Router over `m` tokens: logits row stride `stride` (= experts + 1); routes `[t * k + rank] = (expert, t)`, weights `[t * k + rank]`. A cube a token.
#[cube(launch)]
fn qp_route_rows_kernel(logits: &[f32], routes: &mut [u32], weights: &mut [f32], #[comptime] ne: usize, #[comptime] k: usize, #[comptime] stride: usize) {
    let tok = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let lb = tok * stride;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut p = Shared::<[f32]>::new_slice(512usize);
    let mut mx = logits[lb] * 0.0f32 - 3.0e38f32;
    let mut i = t;
    while i < ne {
        if logits[lb + i] > mx {
            mx = logits[lb + i];
        }
        i += 256usize;
    }
    sm[t] = mx;
    sync_cube();
    let mut st = 128usize;
    while st > 0usize {
        if t < st {
            let o2 = sm[t + st];
            if o2 > sm[t] {
                sm[t] = o2;
            }
        }
        sync_cube();
        st = st / 2usize;
    }
    let m = sm[0usize];
    sync_cube();
    let mut ss = 0.0f32;
    let mut i = t;
    while i < ne {
        let e = Exp::exp(logits[lb + i] - m);
        p[i] = e;
        ss += e;
        i += 256usize;
    }
    let total = lane_sum(&mut sm, ss, false);
    let mut i = t;
    while i < ne {
        p[i] = p[i] / total;
        i += 256usize;
    }
    sync_cube();
    if t < ne {
        let pt = p[t];
        let mut rank = 0usize;
        let mut j = 0usize;
        while j < ne {
            let pj = p[j];
            if pj > pt || (pj == pt && j < t) {
                rank += 1usize;
            }
            j += 1usize;
        }
        if rank < k {
            routes[2usize * (tok * k + rank)] = t as u32;
            routes[2usize * (tok * k + rank) + 1usize] = tok as u32;
            weights[tok * k + rank] = pt;
        }
    }
    sync_cube();
    if t == 0usize {
        let mut sel = 0.0f32;
        let mut r = 0usize;
        while r < k {
            sel += weights[tok * k + r];
            r += 1usize;
        }
        let mut r = 0usize;
        while r < k {
            weights[tok * k + r] = weights[tok * k + r] / sel;
            r += 1usize;
        }
    }
}

/// Causal width-4 conv + SiLU over the chunk: token `t`, channel `c` reads tokens `t-3 .. t` (before the chunk: the stored columns, `3 + tau`).
/// `pp[0]` = `p0`; the state read is the buffer of `p0`'s parity. `out` `[m, ch]`.
#[cube(launch)]
fn qp_conv_rows_kernel(qkv: &[f32], cw: &[f32], st2: &[f32], out: &mut [f32], pp: &[u32], #[comptime] ch: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let t = i / ch;
        let c = i % ch;
        let rd = (pp[0] % 2u32) as usize * 3usize * ch;
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0..4usize {
            // token index relative to the chunk start: t - 3 + j
            let mut v = 0.0f32;
            if t + j >= 3usize {
                v = qkv[(t + j - 3usize) * ch + c];
            } else {
                v = st2[rd + (t + j) * ch + c];
            }
            sum += cw[c * 4usize + j] * v;
        }
        out[i] = sum / (1.0f32 + Exp::exp(0.0f32 - sum));
    }
}

/// After the chunk: the last three tokens' `qkv` columns become the state, in the buffer of parity `(p0 + m) % 2`. A thread a channel; it reads
/// its old columns before writing, so the two buffers may coincide. `pp` = [p0, m].
#[cube(launch)]
fn qp_conv_state_kernel(qkv: &[f32], st2: &mut [f32], pp: &[u32], #[comptime] ch: usize) {
    let c = ABSOLUTE_POS as usize;
    if c < ch {
        let p0 = pp[0] as usize;
        let m = pp[1] as usize;
        let rd = (p0 % 2usize) * 3usize * ch;
        let wr = ((p0 + m) % 2usize) * 3usize * ch;
        let o0 = st2[rd + c];
        let o1 = st2[rd + ch + c];
        let o2 = st2[rd + 2usize * ch + c];
        #[unroll]
        for j in 0..3usize {
            // new column j = token m - 3 + j of the chunk, or old column m + j when that is before the chunk
            let mut v = 0.0f32;
            if m + j >= 3usize {
                v = qkv[(m + j - 3usize) * ch + c];
            } else {
                let oc = m + j;
                if oc == 0usize {
                    v = o0;
                } else if oc == 1usize {
                    v = o1;
                } else {
                    v = o2;
                }
            }
            st2[wr + j * ch + c] = v;
        }
    }
}

/// L2-normalise the q and k head vectors of every token in place (`conv` `[m, ch]`; heads `0 .. 2 nk` of `hk`): a cube a (token, head).
#[cube(launch)]
fn qp_l2_rows_kernel(conv: &mut [f32], eps: &[f32], #[comptime] hk: usize, #[comptime] ch: usize) {
    let t = CUBE_POS_Y as usize;
    let hd = CUBE_POS_X as usize;
    let u = UNIT_POS as usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let v = conv[t * ch + hd * hk + u];
    let tot = lane_sum(&mut sm, v * v, false);
    let r = 1.0f32 / Sqrt::sqrt(tot + eps[0]);
    conv[t * ch + hd * hk + u] = v * r;
}

/// `beta[t, h] = sigmoid(ba[t, h])`, `g[t, h] = softplus(ba[t, nv + h] + dt[h]) * a[h]`.
#[cube(launch)]
fn qp_scalars_rows_kernel(ba: &[f32], dt: &[f32], a: &[f32], beta: &mut [f32], g: &mut [f32], #[comptime] nv: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < beta.len() {
        let t = i / nv;
        let h = i % nv;
        beta[i] = sigmoid_pub(ba[t * 2usize * nv + h]);
        let x = ba[t * 2usize * nv + nv + h] + dt[h];
        let mut sp = x;
        if x <= 20.0f32 {
            sp = Log1p::log1p(Exp::exp(x));
        }
        g[i] = sp * a[h];
    }
}

/// The delta rule over the chunk, one cube (4 planes) a v head looping over the tokens with the head's state matrix in global memory
/// (a plane a row, lanes along the row); then the gated RMS norm of each token's head output. `out` `[m, nv * hk]`. `hk` = 128.
#[cube(launch)]
fn qp_dn_rows_kernel(
    s: &mut [f32], conv: &[f32], beta: &[f32], g: &[f32], z: &[f32], nw: &[f32], out: &mut [f32], eps: &[f32], mtok: &[u32],
    #[comptime] nk: usize, #[comptime] nv: usize, #[comptime] hk: usize, #[comptime] ch: usize,
) {
    let hh = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let w = (UNIT_POS / PLANE_DIM) as usize;
    let nwp = (CUBE_DIM_X / PLANE_DIM) as usize;
    let kh = hh % nk;
    let m = mtok[0] as usize;
    let mut os = Shared::<[f32]>::new_slice(128usize);
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let qscale = 1.0f32 / Sqrt::sqrt(hk as f32);
    let mut tok = 0usize;
    while tok < m {
        let cb = tok * ch;
        let bt = beta[tok * nv + hh];
        let decay = Exp::exp(g[tok * nv + hh]);
        let mut kr = Array::<f32>::new(4usize);
        let mut qr = Array::<f32>::new(4usize);
        #[unroll]
        for mm in 0..4usize {
            qr[mm] = conv[cb + kh * hk + lane + 32usize * mm];
            kr[mm] = conv[cb + nk * hk + kh * hk + lane + 32usize * mm];
        }
        let mut i = w;
        while i < hk {
            let base = hh * hk * hk + i * hk;
            let mut vr = Array::<f32>::new(4usize);
            let mut ps = 0.0f32;
            #[unroll]
            for mm in 0..4usize {
                let v = s[base + lane + 32usize * mm] * decay;
                vr[mm] = v;
                ps += v * kr[mm];
            }
            let skv = plane_sum(ps);
            let d = (conv[cb + 2usize * nk * hk + hh * hk + i] - skv) * bt;
            let mut po = 0.0f32;
            #[unroll]
            for mm in 0..4usize {
                let v = vr[mm] + kr[mm] * d;
                s[base + lane + 32usize * mm] = v;
                po += v * qr[mm];
            }
            let o = plane_sum(po);
            if lane == 0usize {
                os[i] = o * qscale;
            }
            i += nwp;
        }
        sync_cube();
        let ov = os[t];
        let so = lane_sum(&mut sm, ov * ov, false);
        let r = 1.0f32 / Sqrt::sqrt(so / (hk as f32) + eps[0]);
        out[tok * nv * hk + hh * hk + t] = ov * r * nw[t] * silu_pub(z[tok * nv * hk + hh * hk + t]);
        sync_cube();
        tok += 1usize;
    }
}

/// The delta rule over the chunk with a plane a state row: the row lives in registers (4 values a lane) for the whole chunk, each token costs two
/// plane sums; rows are independent (`d_i = (v_i - S_i . k) beta` needs only its own row), so a cube of 4 planes takes 4 rows of one head and the grid is
/// `nv * hk / 4` cubes. Writes the raw (un-normed) head output `raw[t, h * hk + i]`. `hk` = 128, conv width of the buffer `ch`.
#[cube(launch)]
fn qp_dn_rec_kernel(s: &mut [f32], conv: &[f32], beta: &[f32], g: &[f32], raw: &mut [f32], mtok: &[u32], #[comptime] nk: usize, #[comptime] nv: usize, #[comptime] hk: usize, #[comptime] ch: usize) {
    let cube = CUBE_POS as usize;
    let per_head = hk / 4usize;
    let hh = cube / per_head;
    let w = (UNIT_POS / PLANE_DIM) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let i = (cube % per_head) * 4usize + w;
    let kh = hh % nk;
    let m = mtok[0] as usize;
    let base = hh * hk * hk + i * hk;
    let mut sr = Array::<f32>::new(4usize);
    #[unroll]
    for mm in 0..4usize {
        sr[mm] = s[base + lane + 32usize * mm];
    }
    let qscale = 1.0f32 / Sqrt::sqrt(hk as f32);
    let mut tok = 0usize;
    while tok < m {
        let cb = tok * ch;
        let decay = Exp::exp(g[tok * nv + hh]);
        let bt = beta[tok * nv + hh];
        let mut ps = 0.0f32;
        let mut kr = Array::<f32>::new(4usize);
        let mut qr = Array::<f32>::new(4usize);
        #[unroll]
        for mm in 0..4usize {
            kr[mm] = conv[cb + nk * hk + kh * hk + lane + 32usize * mm];
            qr[mm] = conv[cb + kh * hk + lane + 32usize * mm];
            sr[mm] = sr[mm] * decay;
            ps += sr[mm] * kr[mm];
        }
        let skv = plane_sum(ps);
        let d = (conv[cb + 2usize * nk * hk + hh * hk + i] - skv) * bt;
        let mut po = 0.0f32;
        #[unroll]
        for mm in 0..4usize {
            sr[mm] = sr[mm] + kr[mm] * d;
            po += sr[mm] * qr[mm];
        }
        let o = plane_sum(po);
        if lane == 0usize {
            raw[tok * nv * hk + hh * hk + i] = o * qscale;
        }
        tok += 1usize;
    }
    #[unroll]
    for mm in 0..4usize {
        s[base + lane + 32usize * mm] = sr[mm];
    }
}

/// Gated RMS norm of each token's head output: `out = rms(raw) * w * silu(z)`; a cube (128 units) a (token, head).
#[cube(launch)]
fn qp_gnorm_rows_kernel(raw: &[f32], z: &[f32], nw: &[f32], out: &mut [f32], eps: &[f32], #[comptime] nv: usize, #[comptime] hk: usize) {
    let hh = CUBE_POS_X as usize;
    let tok = CUBE_POS_Y as usize;
    let u = UNIT_POS as usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let idx = tok * nv * hk + hh * hk + u;
    let ov = raw[idx];
    let so = lane_sum(&mut sm, ov * ov, false);
    let r = 1.0f32 / Sqrt::sqrt(so / (hk as f32) + eps[0]);
    out[idx] = ov * r * nw[u] * silu_pub(z[idx]);
}

/// Per-head norm + partial rotary for a chunk and the KV-cache append: a cube a (token, head); heads `0..nh` are query heads (`qf` `[m, nh*2*hd]`),
/// `nh..nh+nkv` key heads (`kraw`, `vraw` `[m, nkv*hd]`). Position of token `t` is `pp[0] + t`.
#[cube(launch)]
fn qp_attn_prep_rows_kernel(
    qf: &[f32], kraw: &[f32], vraw: &[f32], qn: &[f32], kn: &[f32],
    qo: &mut [f32], gate: &mut [f32], kc: &mut [f32], vc: &mut [f32], pp: &[u32], pf: &[f32],
    #[comptime] nh: usize, #[comptime] nkv: usize, #[comptime] hd: usize, #[comptime] half: usize,
) {
    let c = CUBE_POS_X as usize;
    let tok = CUBE_POS_Y as usize;
    let d = UNIT_POS as usize;
    let pos = pp[0] as usize + tok;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut sv = Shared::<[f32]>::new_slice(256usize);
    let mut src = 0.0f32;
    let mut wn = 0.0f32;
    if c < nh {
        src = qf[tok * nh * 2usize * hd + c * 2usize * hd + d];
        wn = qn[d];
        gate[tok * nh * hd + c * hd + d] = qf[tok * nh * 2usize * hd + c * 2usize * hd + hd + d];
    } else {
        let j = c - nh;
        src = kraw[tok * nkv * hd + j * hd + d];
        wn = kn[d];
        vc[(pos * nkv + j) * hd + d] = vraw[tok * nkv * hd + j * hd + d];
    }
    let tot = lane_sum(&mut sm, src * src, false);
    let r = 1.0f32 / Sqrt::sqrt(tot / (hd as f32) + pf[0]);
    sv[d] = src * r * wn;
    sync_cube();
    let mut val = sv[d];
    if d < 2usize * half {
        let i = d % half;
        let mut theta = pos as f32;
        let mut tt = 0usize;
        while tt < i {
            theta = theta * pf[1];
            tt += 1usize;
        }
        let cs = Cos::cos(theta);
        let sn = Sin::sin(theta);
        let x0 = sv[i];
        let x1 = sv[i + half];
        if d < half {
            val = x0 * cs - x1 * sn;
        } else {
            val = x0 * sn + x1 * cs;
        }
    }
    if c < nh {
        qo[tok * nh * hd + c * hd + d] = val;
    } else {
        kc[(pos * nkv + (c - nh)) * hd + d] = val;
    }
}

/// Causal attention for a chunk: a cube (256 units, 8 planes) a (query head, token); keys `0 ..= p0 + t`; scores live in shared memory (so at most
/// 4096 keys); then `* sigmoid(gate)`. `out` `[m, nh * hd]`. `hd` = 256.
#[cube(launch)]
fn qp_attn_rows_kernel(
    q: &[f32], kc: &[f32], vc: &[f32], gate: &[f32], out: &mut [f32], pp: &[u32],
    #[comptime] nh: usize, #[comptime] nkv: usize, #[comptime] hd: usize,
) {
    let h = CUBE_POS_X as usize;
    let tok = CUBE_POS_Y as usize;
    let u = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let w = (UNIT_POS / PLANE_DIM) as usize;
    let nw = (CUBE_DIM_X / PLANE_DIM) as usize;
    let n = pp[0] as usize + tok + 1usize;
    let kvh = h / (nh / nkv);
    let scale = 1.0f32 / Sqrt::sqrt(hd as f32);
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut sc = Shared::<[f32]>::new_slice(4096usize);
    let qb = tok * nh * hd + h * hd;
    let mut qr = Array::<f32>::new(8usize);
    #[unroll]
    for mm in 0..8usize {
        qr[mm] = q[qb + lane + 32usize * mm];
    }
    let mut t = w;
    while t < n {
        let kb = (t * nkv + kvh) * hd;
        let mut part = 0.0f32;
        #[unroll]
        for mm in 0..8usize {
            part += qr[mm] * kc[kb + lane + 32usize * mm];
        }
        let sv = plane_sum(part) * scale;
        if lane == 0usize {
            sc[t] = sv;
        }
        t += nw;
    }
    sync_cube();
    let mut mx = q[0usize] * 0.0f32 - 3.0e38f32;
    let mut t = u;
    while t < n {
        if sc[t] > mx {
            mx = sc[t];
        }
        t += 256usize;
    }
    sm[u] = mx;
    sync_cube();
    let mut st = 128usize;
    while st > 0usize {
        if u < st {
            let o2 = sm[u + st];
            if o2 > sm[u] {
                sm[u] = o2;
            }
        }
        sync_cube();
        st = st / 2usize;
    }
    let m = sm[0usize];
    sync_cube();
    let mut sum = 0.0f32;
    let mut t = u;
    while t < n {
        let e = Exp::exp(sc[t] - m);
        sc[t] = e;
        sum += e;
        t += 256usize;
    }
    let total = lane_sum(&mut sm, sum, false);
    let mut t = u;
    while t < n {
        sc[t] = sc[t] / total;
        t += 256usize;
    }
    sync_cube();
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    let mut t = 0usize;
    while t + 3usize < n {
        o0 += sc[t] * vc[(t * nkv + kvh) * hd + u];
        o1 += sc[t + 1usize] * vc[((t + 1usize) * nkv + kvh) * hd + u];
        o2 += sc[t + 2usize] * vc[((t + 2usize) * nkv + kvh) * hd + u];
        o3 += sc[t + 3usize] * vc[((t + 3usize) * nkv + kvh) * hd + u];
        t += 4usize;
    }
    while t < n {
        o0 += sc[t] * vc[(t * nkv + kvh) * hd + u];
        t += 1usize;
    }
    out[tok * nh * hd + h * hd + u] = (o0 + o1 + o2 + o3) * sigmoid_pub(gate[tok * nh * hd + h * hd + u]);
}

/// `out[t, i] = x[t, i] + sum_k w[t, k] down[t * kk + k, i] + sigmoid(sg[t * stride + sg_off]) * shared[t, i]`.
#[cube(launch)]
fn qp_combine_rows_kernel(x: &[f32], down: &[f32], weights: &[f32], shared: &[f32], sgate: &[f32], out: &mut [f32], #[comptime] n: usize, #[comptime] kk: usize, #[comptime] stride: usize, #[comptime] sg_off: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let t = i / n;
        let c = i % n;
        let mut acc = 0.0f32;
        let mut r = 0usize;
        while r < kk {
            acc += weights[t * kk + r] * down[(t * kk + r) * n + c];
            r += 1usize;
        }
        acc += sigmoid_pub(sgate[t * stride + sg_off]) * shared[i];
        out[i] = x[i] + acc;
    }
}

/// RMS norm of rows (no permuted copy): `out[row] = x[row] * rsqrt(mean(x^2) + eps) * w`. A cube a row.
#[cube(launch)]
fn qp_rms_rows_kernel(x: &[f32], w: &[f32], out: &mut [f32], eps: &[f32], #[comptime] n: usize) {
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let row = CUBE_POS as usize;
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < n {
        let v = x[row * n + i];
        ss += v * v;
        i += 256usize;
    }
    let tot = lane_sum(&mut sm, ss, false);
    let r = 1.0f32 / Sqrt::sqrt(tot / (n as f32) + eps[0]);
    let mut i = UNIT_POS as usize;
    while i < n {
        out[row * n + i] = x[row * n + i] * r * w[i];
        i += 256usize;
    }
}

impl Gpu {
    pub fn qp_rms_rows(&self, x: &Handle, w: &Handle, rows: usize, n: usize, eps: f32) -> Handle {
        let out = self.scratch(rows * n * 4);
        let e = self.const_f32(&[eps]);
        unsafe {
            qp_rms_rows_kernel::launch(&self.client, CubeCount::Static(rows as u32, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(x.clone(), rows * n), BufferArg::from_raw_parts(w.clone(), n), BufferArg::from_raw_parts(out.clone(), rows * n), BufferArg::from_raw_parts(e, 1), n);
        }
        out
    }

    /// f32 matrix `w` `[rows, cols]` times `m` token rows: `[m, rows]`.
    pub fn qp_f32_rows(&self, w: &Handle, x: &Handle, m: usize, rows: usize, cols: usize) -> Handle {
        let out = self.scratch(m * rows * 4);
        unsafe {
            qp_f32_rows_kernel::launch(&self.client, crate::cube_count(m * rows, self.per_cube()), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(w.clone(), rows * cols), BufferArg::from_raw_parts(x.clone(), m * cols), BufferArg::from_raw_parts(out.clone(), m * rows), cols, rows);
        }
        out
    }

    /// Returns (`routes` `[m * k]` pairs, `weights` `[m * k]`).
    pub fn qp_route_rows(&self, logits: &Handle, m: usize, ne: usize, k: usize) -> (Handle, Handle) {
        let routes = self.scratch(2 * m * k * 4);
        let weights = self.scratch(m * k * 4);
        unsafe {
            qp_route_rows_kernel::launch(&self.client, CubeCount::Static(m as u32, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(logits.clone(), m * (ne + 1)), BufferArg::from_raw_parts(routes.clone(), 2 * m * k), BufferArg::from_raw_parts(weights.clone(), m * k), ne, k, ne + 1);
        }
        (routes, weights)
    }

    pub fn qp_combine_rows(&self, x: &Handle, down: &Handle, weights: &Handle, shared: &Handle, sgate: &Handle, m: usize, n: usize, k: usize, ne: usize) -> Handle {
        let out = self.scratch(m * n * 4);
        unsafe {
            qp_combine_rows_kernel::launch(&self.client, crate::cube_count(m * n, self.cube_dim), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(x.clone(), m * n), BufferArg::from_raw_parts(down.clone(), m * k * n), BufferArg::from_raw_parts(weights.clone(), m * k), BufferArg::from_raw_parts(shared.clone(), m * n), BufferArg::from_raw_parts(sgate.clone(), m * (ne + 1)), BufferArg::from_raw_parts(out.clone(), m * n), n, k, ne + 1, ne);
        }
        out
    }

    /// Delta-net core for a chunk of `m` tokens at positions `p0 ..` (`pp` = u32 buffer `[p0, m]`). `qkv` `[m, ch]`, `z` `[m, d_inner]`, `ba` `[m, 2 nv]`;
    /// `conv_state` (`2 * 3 * ch`, double buffered by position parity) and `s` are updated in place. Returns `[m, d_inner]` gated-normed.
    #[allow(clippy::too_many_arguments)]
    pub fn qp_delta_net(&self, qkv: &Handle, z: &Handle, ba: &Handle, conv_w: &Handle, dt: &Handle, a: &Handle, norm_w: &Handle, conv_state: &Handle, s: &Handle, pp: &Handle, m: usize, (ch, nk, nv, hk): (usize, usize, usize, usize), eps: f32) -> Handle {
        let conv = self.scratch(m * ch * 4);
        let beta = self.scratch(m * nv * 4);
        let g = self.scratch(m * nv * 4);
        let out = self.scratch(m * nv * hk * 4);
        let raw = self.scratch(m * nv * hk * 4);
        let e = self.const_f32(&[eps]);
        let mh = self.const_u32(&[m as u32]);
        unsafe {
            qp_conv_rows_kernel::launch(&self.client, crate::cube_count(m * ch, self.cube_dim), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(qkv.clone(), m * ch), BufferArg::from_raw_parts(conv_w.clone(), ch * 4), BufferArg::from_raw_parts(conv_state.clone(), 2 * 3 * ch), BufferArg::from_raw_parts(conv.clone(), m * ch), BufferArg::from_raw_parts(pp.clone(), 2), ch);
            qp_conv_state_kernel::launch(&self.client, crate::cube_count(ch, self.cube_dim), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(qkv.clone(), m * ch), BufferArg::from_raw_parts(conv_state.clone(), 2 * 3 * ch), BufferArg::from_raw_parts(pp.clone(), 2), ch);
            qp_l2_rows_kernel::launch(&self.client, CubeCount::Static((2 * nk) as u32, m as u32, 1), CubeDim::new_1d(128), BufferArg::from_raw_parts(conv.clone(), m * ch), BufferArg::from_raw_parts(e.clone(), 1), hk, ch);
            qp_scalars_rows_kernel::launch(&self.client, crate::cube_count(m * nv, self.cube_dim), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(ba.clone(), m * 2 * nv), BufferArg::from_raw_parts(dt.clone(), nv), BufferArg::from_raw_parts(a.clone(), nv), BufferArg::from_raw_parts(beta.clone(), m * nv), BufferArg::from_raw_parts(g.clone(), m * nv), nv);
            qp_dn_rec_kernel::launch(
                &self.client, CubeCount::Static((nv * hk / 4) as u32, 1, 1), CubeDim::new_1d(128),
                BufferArg::from_raw_parts(s.clone(), nv * hk * hk), BufferArg::from_raw_parts(conv.clone(), m * ch), BufferArg::from_raw_parts(beta.clone(), m * nv), BufferArg::from_raw_parts(g.clone(), m * nv),
                BufferArg::from_raw_parts(raw.clone(), m * nv * hk), BufferArg::from_raw_parts(mh, 1), nk, nv, hk, ch,
            );
            qp_gnorm_rows_kernel::launch(&self.client, CubeCount::Static(nv as u32, m as u32, 1), CubeDim::new_1d(128), BufferArg::from_raw_parts(raw, m * nv * hk), BufferArg::from_raw_parts(z.clone(), m * nv * hk), BufferArg::from_raw_parts(norm_w.clone(), hk), BufferArg::from_raw_parts(out.clone(), m * nv * hk), BufferArg::from_raw_parts(e, 1), nv, hk);
        }
        out
    }

    /// Gated attention for a chunk; `pp` = `[p0]` (u32). `kc`/`vc` are the persistent `[cap][nkv][hd]` caches. Returns `[m, nh * hd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn qp_attention(&self, qf: &Handle, kraw: &Handle, vraw: &Handle, qn: &Handle, kn: &Handle, kc: &Handle, vc: &Handle, pp: &Handle, m: usize, (nh, nkv, hd, half, cap): (usize, usize, usize, usize, usize), eps: f32, theta_scale: f32) -> Handle {
        let qo = self.scratch(m * nh * hd * 4);
        let gate = self.scratch(m * nh * hd * 4);
        let out = self.scratch(m * nh * hd * 4);
        let pf = self.const_f32(&[eps, theta_scale]);
        unsafe {
            qp_attn_prep_rows_kernel::launch(
                &self.client, CubeCount::Static((nh + nkv) as u32, m as u32, 1), CubeDim::new_1d(hd as u32),
                BufferArg::from_raw_parts(qf.clone(), m * nh * 2 * hd), BufferArg::from_raw_parts(kraw.clone(), m * nkv * hd), BufferArg::from_raw_parts(vraw.clone(), m * nkv * hd),
                BufferArg::from_raw_parts(qn.clone(), hd), BufferArg::from_raw_parts(kn.clone(), hd),
                BufferArg::from_raw_parts(qo.clone(), m * nh * hd), BufferArg::from_raw_parts(gate.clone(), m * nh * hd),
                BufferArg::from_raw_parts(kc.clone(), cap * nkv * hd), BufferArg::from_raw_parts(vc.clone(), cap * nkv * hd),
                BufferArg::from_raw_parts(pp.clone(), 1), BufferArg::from_raw_parts(pf, 2), nh, nkv, hd, half,
            );
            qp_attn_rows_kernel::launch(
                &self.client, CubeCount::Static(nh as u32, m as u32, 1), CubeDim::new_1d(256),
                BufferArg::from_raw_parts(qo, m * nh * hd), BufferArg::from_raw_parts(kc.clone(), cap * nkv * hd), BufferArg::from_raw_parts(vc.clone(), cap * nkv * hd),
                BufferArg::from_raw_parts(gate, m * nh * hd), BufferArg::from_raw_parts(out.clone(), m * nh * hd), BufferArg::from_raw_parts(pp.clone(), 1), nh, nkv, hd,
            );
        }
        out
    }
}
