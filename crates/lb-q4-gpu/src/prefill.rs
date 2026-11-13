//! The glue kernels of a device-resident **prefill** pass: the decode kernels of `decode.rs`
//! batched over `m` prompt rows, so a whole prompt chunk runs from the embedding upload to the
//! last row's logits with one read at the end (plus one per MoE layer only if a caller wants the
//! routes back early; the prefill path does not).
//!
//! Before this, a multi-token pass ran every operator around the matmuls on the host through
//! Burn: attention built a full `[heads, m, total]` score matrix and a host-side bool mask (at an
//! 8k prompt that is ~8.6 GB of scores), and every quantized matmul round-tripped its activation
//! through host memory. Measured on a P100 (X6, `docs/WORK_TO_WEIGHTS.md`): 63-69 prompt tok/s
//! against llama.cpp's 600-1000 on the same card, and out of memory at a 2k prompt.
//!
//! Kernels, each the `m`-row form of its decode twin (same arithmetic per row, same summation
//! order inside a row, so a row here equals the decode kernel's result for that token):
//!
//! * [`rms_norm_rows_kernel`]: `x <- x + y` (optional), `h = rms(x) * w`, one cube per row.
//! * [`qk_norm_rope_rows_kernel`]: per-head norm + NEOX RoPE for `m` rows at positions
//!   `pos0 + t`, K and V appended to the device cache. Cube `(head, t)`.
//! * [`attn_prefill_kernel`]: causal attention, query row `t` against cached positions
//!   `0 ..= pos0 + t` (so earlier chunks and this chunk's earlier rows). Cube `(head, t)`, the
//!   decode kernel's online softmax per cube. O(m * total) work with no K/V reuse across
//!   queries: correct first, a tiled (flash) version is the next step once it shows in the profile.
//! * [`conv_prefill_kernel`]: the LFM2 short conv over `m` rows with the rolling state, one unit
//!   per channel walking the rows in order; leaves the state exactly where `m` decode steps would.
//! * [`router_topk_rows_kernel`]: the sigmoid router + top-k for `m` rows; row `t` writes slots
//!   `t * k ..` as `[expert, t]` pairs, which is what the routed expert kernels read (they take the
//!   token index from the pair), so a whole chunk's experts run in the stacked layer's two
//!   dispatches.
//! * [`weighted_sum_rows_kernel`]: `y[t] = sum_s w[t k + s] out[t k + s]`, ascending expert id.

use crate::store::Gpu;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Units per cube for the row kernels (norm, router). A power of two.
pub const ROW_CUBE: usize = 256;
/// Units per attention cube (one query row of one head); a power of two.
pub const ATTN_LANES: usize = 64;
/// Most experts the router kernel handles.
pub const MAX_EXPERTS: usize = 256;

#[cube]
fn cube_sum(sm: &mut Shared<[f32]>, v: f32) -> f32 {
    sm[UNIT_POS as usize] = v;
    sync_cube();
    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if UNIT_POS < s {
            sm[UNIT_POS as usize] += sm[(UNIT_POS + s) as usize];
        }
        sync_cube();
        s = s / 2u32;
    }
    let r = sm[0usize];
    sync_cube();
    r
}

/// Row `CUBE_POS` of `x` (`[m, n]`): `x <- x + y` when `add`, `h = x * rsqrt(mean(x^2) + eps) * w`.
/// `p` = `[n]`.
#[cube(launch)]
pub fn rms_norm_rows_kernel(x: &mut [f32], y: &[f32], w: &[f32], h: &mut [f32], p: &[u32], eps: &[f32], #[comptime] add: bool) {
    let n = p[0] as usize;
    let row = CUBE_POS as usize;
    let o = row * n;
    let mut sm = Shared::<[f32]>::new_slice(ROW_CUBE);
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < n {
        let mut v = x[o + i];
        if add {
            v += y[o + i];
            x[o + i] = v;
        }
        ss += v * v;
        i += CUBE_DIM_X as usize;
    }
    let total = cube_sum(&mut sm, ss);
    let r = 1.0f32 / Sqrt::sqrt(total / n as f32 + eps[0]);
    let mut i = UNIT_POS as usize;
    while i < n {
        h[o + i] = x[o + i] * r * w[i];
        i += CUBE_DIM_X as usize;
    }
}

/// Per-head RMS norm + NEOX RoPE for `m` rows, and the KV append. Cube `(head, t)`: `head < nh`
/// is a query head (written to `qo[t, head]`), `nh + j` is KV head `j` (rotated K and plain V
/// written to row `pos0 + t` of the caches `[kvh, cap, hd]`). `p` = `[nh, kvh, pos0, cap]`.
#[cube(launch)]
pub fn qk_norm_rope_rows_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qn: &[f32],
    kn: &[f32],
    rope: &[f32],
    kc: &mut [f32],
    vc: &mut [f32],
    qo: &mut [f32],
    p: &[u32],
    eps: &[f32],
    #[comptime] hd: usize,
) {
    let nh = p[0] as usize;
    let kvh = p[1] as usize;
    let pos0 = p[2] as usize;
    let cap = p[3] as usize;
    let head = CUBE_POS_X as usize;
    let t = CUBE_POS_Y as usize;
    let pos = pos0 + t;
    let j = UNIT_POS as usize;
    let is_q = head < nh;
    let mut val = 0.0f32;
    let mut nw = 0.0f32;
    if j < hd {
        if is_q {
            val = q[(t * nh + head) * hd + j];
            nw = qn[j];
        } else {
            val = k[(t * kvh + head - nh) * hd + j];
            nw = kn[j];
        }
    }
    let mut sm = Shared::<[f32]>::new_slice(hd);
    let total = cube_sum(&mut sm, val * val);
    let r = 1.0f32 / Sqrt::sqrt(total / hd as f32 + eps[0]);
    let nv = val * r * nw;
    let mut buf = Shared::<[f32]>::new_slice(hd);
    if j < hd {
        buf[j] = nv;
    }
    sync_cube();
    if j < hd {
        let half = hd / 2usize;
        let mut rot = 0.0f32;
        if j < half {
            rot = -buf[j + half];
        } else {
            rot = buf[j - half];
        }
        let c = rope[pos * 2usize * hd + j];
        let s = rope[pos * 2usize * hd + hd + j];
        let out = nv * c + rot * s;
        if is_q {
            qo[(t * nh + head) * hd + j] = out;
        } else {
            let kh = head - nh;
            kc[(kh * cap + pos) * hd + j] = out;
            vc[(kh * cap + pos) * hd + j] = v[(t * kvh + kh) * hd + j];
        }
    }
}

/// Causal attention for `m` query rows. Cube `(head, t)`: query `qo[t, head]` against cached
/// positions `0 .. pos0 + t + 1` of KV head `head / group`; the decode kernel's online softmax
/// (unit `u` takes positions `u, u + lanes, ...`, combined through shared memory).
/// Writes `out[t, head]`. `p` = `[group, pos0, cap, nh]`, `scale` = `[hd^-0.5]`.
#[cube(launch)]
pub fn attn_prefill_kernel(q: &[f32], kc: &[f32], vc: &[f32], out: &mut [f32], p: &[u32], scale: &[f32], #[comptime] hd: usize, #[comptime] lanes: usize) {
    let group = p[0] as usize;
    let pos0 = p[1] as usize;
    let cap = p[2] as usize;
    let nh = p[3] as usize;
    let head = CUBE_POS_X as usize;
    let t = CUBE_POS_Y as usize;
    let total = pos0 + t + 1usize;
    let kh = head / group;
    let u = UNIT_POS as usize;
    let sc = scale[0];
    let qb = (t * nh + head) * hd;

    let mut qs = Shared::<[f32]>::new_slice(hd);
    let mut j = u;
    while j < hd {
        qs[j] = q[qb + j];
        j += lanes;
    }
    sync_cube();

    let mut acc = Array::<f32>::new(hd);
    let mut d = 0usize;
    while d < hd {
        acc[d] = 0.0f32;
        d += 1usize;
    }
    let mut mx = f32::new(-3.0e38f32);
    let mut s = 0.0f32;
    let mut pos = u;
    while pos < total {
        let base = (kh * cap + pos) * hd;
        let mut dot = 0.0f32;
        let mut d = 0usize;
        while d < hd {
            dot += qs[d] * kc[base + d];
            d += 1usize;
        }
        let score = dot * sc;
        if score > mx {
            let c = Exp::exp(mx - score);
            s *= c;
            let mut d = 0usize;
            while d < hd {
                acc[d] *= c;
                d += 1usize;
            }
            mx = score;
        }
        let e = Exp::exp(score - mx);
        s += e;
        let mut d = 0usize;
        while d < hd {
            acc[d] += e * vc[base + d];
            d += 1usize;
        }
        pos += lanes;
    }

    let mut ms = Shared::<[f32]>::new_slice(lanes);
    let mut ss = Shared::<[f32]>::new_slice(lanes);
    let mut accs = Shared::<[f32]>::new_slice(lanes * hd);
    ms[u] = mx;
    ss[u] = s;
    let mut d = 0usize;
    while d < hd {
        accs[u * hd + d] = acc[d];
        d += 1usize;
    }
    sync_cube();
    let mut big = f32::new(-3.0e38f32);
    let mut i = 0usize;
    while i < lanes {
        if ms[i] > big {
            big = ms[i];
        }
        i += 1usize;
    }
    let mut denom = 0.0f32;
    let mut i = 0usize;
    while i < lanes {
        denom += ss[i] * Exp::exp(ms[i] - big);
        i += 1usize;
    }
    let mut j = u;
    while j < hd {
        let mut num = 0.0f32;
        let mut i = 0usize;
        while i < lanes {
            num += accs[i * hd + j] * Exp::exp(ms[i] - big);
            i += 1usize;
        }
        out[qb + j] = num / denom;
        j += lanes;
    }
}

/// Query rows per tiled-attention cube (each with all `group` query heads of one KV head).
pub const ATTN_BQ: usize = 32;
/// Key positions staged per tile.
pub const ATTN_BK: usize = 32;

/// Causal attention for `m` query rows, tiled (flash style). Cube `(kv head kh, query tile)`:
/// its `group * ATTN_BQ` units are the query heads `kh * group ..` of rows `tile * ATTN_BQ ..`,
/// unit `u` = head `u / ATTN_BQ`, row `u % ATTN_BQ`. K and V of `ATTN_BK` positions at a time go to
/// shared memory once and serve every query of the cube (the naive kernel re-reads them per query);
/// per tile a unit computes its scores, rescales its running sum once, then accumulates.
/// Same inputs and outputs as [`attn_prefill_kernel`]; the summation order differs, so results agree
/// to rounding. `p` = `[group, pos0, cap, nh, m]`.
#[cube(launch)]
pub fn attn_prefill_tiled_kernel(q: &[f32], kc: &[f32], vc: &[f32], out: &mut [f32], p: &[u32], scale: &[f32], #[comptime] hd: usize) {
    let group = p[0] as usize;
    let pos0 = p[1] as usize;
    let cap = p[2] as usize;
    let nh = p[3] as usize;
    let m = p[4] as usize;
    let kh = CUBE_POS_X as usize;
    let tile = CUBE_POS_Y as usize;
    let u = UNIT_POS as usize;
    let nunits = CUBE_DIM_X as usize;
    let g = u / ATTN_BQ;
    let t = tile * ATTN_BQ + u % ATTN_BQ;
    let head = kh * group + g;
    let valid = t < m;
    let qpos = pos0 + t;
    let sc = scale[0];

    let mut qv = Array::<f32>::new(hd);
    let mut acc = Array::<f32>::new(hd);
    #[unroll]
    for d in 0..hd {
        let mut v = 0.0f32;
        if valid {
            v = q[(t * nh + head) * hd + d];
        }
        qv[d] = v;
        acc[d] = 0.0f32;
    }
    let mut mx = f32::new(-3.0e38f32);
    let mut s = 0.0f32;

    let mut ks = Shared::<[f32]>::new_slice(ATTN_BK * hd);
    let mut vs = Shared::<[f32]>::new_slice(ATTN_BK * hd);
    let mut scs = Array::<f32>::new(ATTN_BK);
    // positions this tile's last valid query can see: 0 ..= pos0 + min(tile_end, m) - 1
    let mut tend = tile * ATTN_BQ + ATTN_BQ;
    if tend > m {
        tend = m;
    }
    let kend = pos0 + tend;
    let mut k0 = 0usize;
    while k0 < kend {
        let mut i = u;
        while i < ATTN_BK * hd {
            let j = i / hd;
            let pos = k0 + j;
            let mut kv = 0.0f32;
            let mut vv = 0.0f32;
            if pos < kend {
                let o = (kh * cap + pos) * hd + i % hd;
                kv = kc[o];
                vv = vc[o];
            }
            ks[i] = kv;
            vs[i] = vv;
            i += nunits;
        }
        sync_cube();
        if valid {
            let mut tmax = f32::new(-3.0e38f32);
            #[unroll]
            for j in 0..ATTN_BK {
                let mut dot = 0.0f32;
                #[unroll]
                for d in 0..hd {
                    dot += qv[d] * ks[j * hd + d];
                }
                let mut sv = f32::new(-3.0e38f32);
                if k0 + j <= qpos {
                    sv = dot * sc;
                }
                scs[j] = sv;
                if sv > tmax {
                    tmax = sv;
                }
            }
            if tmax > mx {
                let c = Exp::exp(mx - tmax);
                s *= c;
                #[unroll]
                for d in 0..hd {
                    acc[d] *= c;
                }
                mx = tmax;
            }
            #[unroll]
            for j in 0..ATTN_BK {
                if k0 + j <= qpos {
                    let e = Exp::exp(scs[j] - mx);
                    s += e;
                    #[unroll]
                    for d in 0..hd {
                        acc[d] += e * vs[j * hd + d];
                    }
                }
            }
        }
        sync_cube();
        k0 += ATTN_BK;
    }
    if valid {
        #[unroll]
        for d in 0..hd {
            out[(t * nh + head) * hd + d] = acc[d] / s;
        }
    }
}

/// The short conv over `m` rows. `bcx` `[m, 3e]` (`[b | c | x]` per row), `conv` taps `[l, e]`,
/// `state` `[l - 1, e]` (the last `l - 1` values of `b * x` before this chunk; advanced in place
/// to the last `l - 1` of this chunk), `y` `[m, e]`. One unit per channel, rows in order, the
/// decode kernel's arithmetic per row: `y[t] = c[t] * (sum_{k<l-1} conv[k] z[t+k] + conv[l-1] bx[t])`
/// with `z` = state then this chunk's `b * x`. `dims` = `[e, l, m]`; `l <= 8`.
#[cube(launch)]
pub fn conv_prefill_kernel(bcx: &[f32], conv: &[f32], state: &mut [f32], y: &mut [f32], dims: &[u32]) {
    let e = dims[0] as usize;
    let l = dims[1] as usize;
    let m = dims[2] as usize;
    let mut i = ABSOLUTE_POS as usize;
    let stride = (CUBE_COUNT as usize) * (CUBE_DIM_X as usize);
    while i < e {
        // the rolling window lives in registers: win[k] = the value k + 1 steps... oldest first
        let mut win = Array::<f32>::new(8usize);
        let mut k = 0usize;
        while k + 1usize < l {
            win[k] = state[k * e + i];
            k += 1usize;
        }
        let mut t = 0usize;
        while t < m {
            let row = t * 3usize * e;
            let bx = bcx[row + i] * bcx[row + 2usize * e + i];
            let c = bcx[row + e + i];
            let mut acc = 0.0f32;
            let mut k = 0usize;
            while k + 1usize < l {
                acc += conv[k * e + i] * win[k];
                k += 1usize;
            }
            acc += conv[(l - 1usize) * e + i] * bx;
            y[t * e + i] = c * acc;
            let mut k = 0usize;
            while k + 2usize < l {
                win[k] = win[k + 1usize];
                k += 1usize;
            }
            win[l - 2usize] = bx;
            t += 1usize;
        }
        let mut k = 0usize;
        while k + 1usize < l {
            state[k * e + i] = win[k];
            k += 1usize;
        }
        i += stride;
    }
}

/// The LFM2-MoE router for `m` rows: row `t` (cube `t`) ranks its experts exactly as
/// `decode::router_topk_kernel` does and writes slots `t * k + i` as `[expert, t]` with weight.
/// `p` = `[n_expert, k, cols]`. With `seq` each logit is one unit's sequential dot (the host's
/// order); otherwise one plane per expert at a time with `plane_sum`.
#[cube(launch)]
pub fn router_topk_rows_kernel(h: &[f32], router: &[f32], bias: &[f32], routes: &mut [u32], weights: &mut [f32], p: &[u32], #[comptime] seq: bool) {
    let n_exp = p[0] as usize;
    let k = p[1] as usize;
    let cols = p[2] as usize;
    let t = CUBE_POS as usize;
    let hb = t * cols;
    let base = t * k;
    let u = UNIT_POS as usize;
    let mut probs = Shared::<[f32]>::new_slice(MAX_EXPERTS);
    let mut score = Shared::<[f32]>::new_slice(MAX_EXPERTS);
    let mut sel = Shared::<[u32]>::new_slice(MAX_EXPERTS);
    if seq {
        let mut e = u;
        while e < n_exp {
            let mut acc = 0.0f32;
            let mut c = 0usize;
            while c < cols {
                acc += router[e * cols + c] * h[hb + c];
                c += 1usize;
            }
            let pr = 1.0f32 / (1.0f32 + Exp::exp(-acc));
            probs[e] = pr;
            score[e] = pr + bias[e];
            e += CUBE_DIM_X as usize;
        }
    } else {
        let width = PLANE_DIM as usize;
        let planes = CUBE_DIM_X as usize / width;
        let lane = UNIT_POS_PLANE as usize;
        let mut e = (UNIT_POS / PLANE_DIM) as usize;
        while e < n_exp {
            let mut acc = 0.0f32;
            let mut c = lane;
            while c < cols {
                acc += router[e * cols + c] * h[hb + c];
                c += width;
            }
            let logit = plane_sum(acc);
            if lane == 0usize {
                let pr = 1.0f32 / (1.0f32 + Exp::exp(-logit));
                probs[e] = pr;
                score[e] = pr + bias[e];
            }
            e += planes;
        }
    }
    sync_cube();
    let mut e = u;
    while e < n_exp {
        let se = score[e];
        let mut rank = 0usize;
        let mut j = 0usize;
        while j < n_exp {
            let sj = score[j];
            if sj > se || (sj == se && j < e) {
                rank += 1usize;
            }
            j += 1usize;
        }
        if rank < k {
            sel[rank] = e as u32;
        }
        e += CUBE_DIM_X as usize;
    }
    sync_cube();
    if u == 0usize {
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < k {
            sum += probs[sel[i] as usize];
            i += 1usize;
        }
        if sum < 6.103515625e-5f32 {
            sum = 6.103515625e-5f32;
        }
        let mut i = 0usize;
        while i < k {
            let ex = sel[i];
            routes[2usize * (base + i)] = ex;
            routes[2usize * (base + i) + 1usize] = t as u32;
            weights[base + i] = probs[ex as usize] / sum;
            i += 1usize;
        }
    }
}

/// `y[t, i] = sum_s weights[t k + s] * outs[(t k + s) n + i]`, slots taken in ascending expert id
/// (the host's order). `p` = `[k, n, m]`.
#[cube(launch)]
pub fn weighted_sum_rows_kernel(outs: &[f32], weights: &[f32], routes: &[u32], y: &mut [f32], p: &[u32]) {
    let k = p[0] as usize;
    let n = p[1] as usize;
    let m = p[2] as usize;
    let mut idx = ABSOLUTE_POS as usize;
    let stride = CUBE_COUNT as usize * CUBE_DIM_X as usize;
    while idx < m * n {
        let t = idx / n;
        let i = idx % n;
        let base = t * k;
        let mut acc = 0.0f32;
        let mut last = 0u32;
        let mut first = true;
        let mut step = 0usize;
        while step < k {
            let mut best = 0usize;
            let mut best_e = 4294967295u32;
            let mut s = 0usize;
            while s < k {
                let e = routes[2usize * (base + s)];
                if (first || e > last) && e < best_e {
                    best_e = e;
                    best = s;
                }
                s += 1usize;
            }
            acc += weights[base + best] * outs[(base + best) * n + i];
            last = best_e;
            first = false;
            step += 1usize;
        }
        y[idx] = acc;
        idx += stride;
    }
}

/// `x[i] += y[i]` over `n` values (the residual add at a layer's end, so the layer's scratch can be
/// recycled with only persistent buffers live).
#[cube(launch)]
pub fn add_rows_kernel(x: &mut [f32], y: &[f32], p: &[u32]) {
    let n = p[0] as usize;
    let mut i = ABSOLUTE_POS as usize;
    let stride = CUBE_COUNT as usize * CUBE_DIM_X as usize;
    while i < n {
        x[i] += y[i];
        i += stride;
    }
}

fn grid(n: usize, per: usize) -> CubeCount {
    crate::cube_count(n, per)
}

/// Launchers. Every per-call parameter buffer is a fresh upload (not `const_u32`, which caches
/// by value forever: a prefill's `pos0` changes every chunk).
impl Gpu {
    /// Rows of `x` (`[m, n]`): optional residual add, then the RMS norm into a new `[m, n]`.
    pub fn rms_norm_rows(&self, x: &Handle, y: Option<&Handle>, w: &Handle, m: usize, n: usize, eps: f32) -> Handle {
        let h = self.scratch(m * n * 4);
        let yh = y.cloned().unwrap_or_else(|| x.clone());
        unsafe {
            rms_norm_rows_kernel::launch(
                &self.client,
                CubeCount::Static(m as u32, 1, 1),
                CubeDim::new_1d(ROW_CUBE as u32),
                BufferArg::from_raw_parts(x.clone(), m * n),
                BufferArg::from_raw_parts(yh, m * n),
                BufferArg::from_raw_parts(w.clone(), n),
                BufferArg::from_raw_parts(h.clone(), m * n),
                BufferArg::from_raw_parts(self.const_u32(&[n as u32]), 1),
                BufferArg::from_raw_parts(self.const_f32(&[eps]), 1),
                y.is_some(),
            );
        }
        h
    }

    /// `x += y` over `n` values, in place.
    pub fn add_rows(&self, x: &Handle, y: &Handle, n: usize) {
        unsafe {
            add_rows_kernel::launch(
                &self.client,
                grid(n, 256),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x.clone(), n),
                BufferArg::from_raw_parts(y.clone(), n),
                BufferArg::from_raw_parts(self.upload_u32(vec![n as u32]), 1),
            );
        }
    }

    /// Norm + RoPE of `m` rows of `q` / `k` at positions `pos0 ..`, K/V appended to the caches.
    /// Returns rotated `q` `[m, nh * hd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_rows(
        &self,
        q: &Handle,
        k: &Handle,
        v: &Handle,
        q_norm: &Handle,
        k_norm: &Handle,
        rope: &Handle,
        rope_rows: usize,
        kc: &Handle,
        vc: &Handle,
        (m, pos0, cap): (usize, usize, usize),
        (nh, kvh, hd): (usize, usize, usize),
        eps: f32,
    ) -> Handle {
        assert!(hd.is_power_of_two() && (2..=1024).contains(&hd), "head_dim {hd}");
        assert!(pos0 + m <= cap && pos0 + m <= rope_rows, "rows {pos0}+{m} past the cache ({cap}) or rope table ({rope_rows})");
        let qo = self.scratch(m * nh * hd * 4);
        let p = self.upload_u32(vec![nh as u32, kvh as u32, pos0 as u32, cap as u32]);
        unsafe {
            qk_norm_rope_rows_kernel::launch(
                &self.client,
                CubeCount::Static((nh + kvh) as u32, m as u32, 1),
                CubeDim::new_1d(hd as u32),
                BufferArg::from_raw_parts(q.clone(), m * nh * hd),
                BufferArg::from_raw_parts(k.clone(), m * kvh * hd),
                BufferArg::from_raw_parts(v.clone(), m * kvh * hd),
                BufferArg::from_raw_parts(q_norm.clone(), hd),
                BufferArg::from_raw_parts(k_norm.clone(), hd),
                BufferArg::from_raw_parts(rope.clone(), rope_rows * 2 * hd),
                BufferArg::from_raw_parts(kc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(vc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(qo.clone(), m * nh * hd),
                BufferArg::from_raw_parts(p, 4),
                BufferArg::from_raw_parts(self.const_f32(&[eps]), 1),
                hd,
            );
        }
        qo
    }

    /// Causal attention for `m` rows at `pos0 ..` over the caches; returns `[m, nh * hd]`. The tiled
    /// kernel unless `LB_ATTN_NAIVE=1` (or a shape it does not cover: `group * ATTN_BQ` > 1024 units).
    pub fn attn_prefill(&self, q: &Handle, kc: &Handle, vc: &Handle, (m, pos0, cap): (usize, usize, usize), (nh, kvh, hd): (usize, usize, usize)) -> Handle {
        let naive = std::env::var("LB_ATTN_NAIVE").is_ok_and(|v| !v.is_empty() && v != "0");
        if !naive && kvh > 0 && nh % kvh == 0 && (nh / kvh) * ATTN_BQ <= 1024 {
            return self.attn_prefill_tiled(q, kc, vc, (m, pos0, cap), (nh, kvh, hd));
        }
        self.attn_prefill_naive(q, kc, vc, (m, pos0, cap), (nh, kvh, hd))
    }

    /// [`attn_prefill_tiled_kernel`]: one cube per (KV head, `ATTN_BQ` query rows).
    pub fn attn_prefill_tiled(&self, q: &Handle, kc: &Handle, vc: &Handle, (m, pos0, cap): (usize, usize, usize), (nh, kvh, hd): (usize, usize, usize)) -> Handle {
        assert!(kvh > 0 && nh % kvh == 0, "{nh} query heads over {kvh} KV heads");
        assert!(pos0 + m <= cap, "attention rows {pos0}+{m} past a {cap} cache");
        let group = nh / kvh;
        let out = self.scratch(m * nh * hd * 4);
        let p = self.upload_u32(vec![group as u32, pos0 as u32, cap as u32, nh as u32, m as u32]);
        unsafe {
            attn_prefill_tiled_kernel::launch(
                &self.client,
                CubeCount::Static(kvh as u32, m.div_ceil(ATTN_BQ) as u32, 1),
                CubeDim::new_1d((group * ATTN_BQ) as u32),
                BufferArg::from_raw_parts(q.clone(), m * nh * hd),
                BufferArg::from_raw_parts(kc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(vc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(out.clone(), m * nh * hd),
                BufferArg::from_raw_parts(p, 5),
                BufferArg::from_raw_parts(self.const_f32(&[(hd as f32).powf(-0.5)]), 1),
                hd,
            );
        }
        out
    }

    /// [`attn_prefill_kernel`]: one cube per (query head, row); the reference for the tiled kernel.
    pub fn attn_prefill_naive(&self, q: &Handle, kc: &Handle, vc: &Handle, (m, pos0, cap): (usize, usize, usize), (nh, kvh, hd): (usize, usize, usize)) -> Handle {
        assert!(kvh > 0 && nh % kvh == 0, "{nh} query heads over {kvh} KV heads");
        assert!(pos0 + m <= cap, "attention rows {pos0}+{m} past a {cap} cache");
        let out = self.scratch(m * nh * hd * 4);
        let p = self.upload_u32(vec![(nh / kvh) as u32, pos0 as u32, cap as u32, nh as u32]);
        unsafe {
            attn_prefill_kernel::launch(
                &self.client,
                CubeCount::Static(nh as u32, m as u32, 1),
                CubeDim::new_1d(ATTN_LANES as u32),
                BufferArg::from_raw_parts(q.clone(), m * nh * hd),
                BufferArg::from_raw_parts(kc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(vc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(out.clone(), m * nh * hd),
                BufferArg::from_raw_parts(p, 4),
                BufferArg::from_raw_parts(self.const_f32(&[(hd as f32).powf(-0.5)]), 1),
                hd,
                ATTN_LANES,
            );
        }
        out
    }

    /// The short conv over `m` rows of `bcx` (`[m, 3e]`), advancing `state` (`[l - 1, e]`).
    /// Returns `[m, e]`.
    pub fn conv_prefill(&self, bcx: &Handle, taps: &Handle, state: &Handle, m: usize, e: usize, l: usize) -> Handle {
        assert!((2..=8).contains(&l), "conv length {l}: the kernel's window holds up to 8");
        let y = self.scratch(m * e * 4);
        unsafe {
            conv_prefill_kernel::launch(
                &self.client,
                grid(e, 256),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(bcx.clone(), m * 3 * e),
                BufferArg::from_raw_parts(taps.clone(), l * e),
                BufferArg::from_raw_parts(state.clone(), (l - 1) * e),
                BufferArg::from_raw_parts(y.clone(), m * e),
                BufferArg::from_raw_parts(self.upload_u32(vec![e as u32, l as u32, m as u32]), 3),
            );
        }
        y
    }

    /// Route `m` rows of `h` (`[m, cols]`) into `routes` (`[m k, 2]`) and `weights` (`[m k]`).
    #[allow(clippy::too_many_arguments)]
    pub fn router_topk_rows(&self, h: &Handle, router: &Handle, bias: &Handle, routes: &Handle, weights: &Handle, (m, n_expert, k, cols): (usize, usize, usize, usize), seq: bool) {
        assert!(n_expert <= MAX_EXPERTS && k <= n_expert, "{n_expert} experts, top {k}");
        assert!(seq || self.use_plane, "the plane router needs plane ops");
        unsafe {
            router_topk_rows_kernel::launch(
                &self.client,
                CubeCount::Static(m as u32, 1, 1),
                CubeDim::new_1d(ROW_CUBE as u32),
                BufferArg::from_raw_parts(h.clone(), m * cols),
                BufferArg::from_raw_parts(router.clone(), n_expert * cols),
                BufferArg::from_raw_parts(bias.clone(), n_expert),
                BufferArg::from_raw_parts(routes.clone(), 2 * m * k),
                BufferArg::from_raw_parts(weights.clone(), m * k),
                BufferArg::from_raw_parts(self.const_u32(&[n_expert as u32, k as u32, cols as u32]), 3),
                seq,
            );
        }
    }

    /// `y[t] = sum over t's k slots of weight * expert output` (`outs` `[m k, n]`); returns `[m, n]`.
    pub fn weighted_sum_rows(&self, outs: &Handle, weights: &Handle, routes: &Handle, (m, k, n): (usize, usize, usize)) -> Handle {
        let y = self.scratch(m * n * 4);
        unsafe {
            weighted_sum_rows_kernel::launch(
                &self.client,
                grid(m * n, 256),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(outs.clone(), m * k * n),
                BufferArg::from_raw_parts(weights.clone(), m * k),
                BufferArg::from_raw_parts(routes.clone(), 2 * m * k),
                BufferArg::from_raw_parts(y.clone(), m * n),
                BufferArg::from_raw_parts(self.upload_u32(vec![k as u32, n as u32, m as u32]), 3),
            );
        }
        y
    }
}
