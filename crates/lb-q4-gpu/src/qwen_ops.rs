//! Device kernels for the `qwen35moe` family that are not matrix products: RMS norms, the gated delta-net pieces (causal conv,
//! per-head L2 norm, the recurrent state update, gated norm), partial-rotary attention with a per-head output gate, the router
//! (softmax, top-k, renormalised), and the MoE combine. Each mirrors the CPU decoder (`lb-qwen35`) operation for operation.
//! All buffers are f32 unless named `routes`/`pos`; sizes are comptime.

use crate::kquant::GpuQ5KNative;
use crate::store::Gpu;
use crate::{cube_count, lane_sum};
use cubecl::prelude::*;

type Handle = cubecl::server::Handle;

#[cube]
pub(crate) fn silu_pub(x: f32) -> f32 {
    x / (1.0f32 + Exp::exp(-x))
}

#[cube]
pub(crate) fn sigmoid_pub(x: f32) -> f32 {
    1.0f32 / (1.0f32 + Exp::exp(-x))
}

#[cube]
fn silu_f(x: f32) -> f32 {
    x / (1.0f32 + Exp::exp(-x))
}

pub(crate) fn _marker() {}

/// Index of value `i` in the 16-value-permuted order of `perm16_kernel` (`y[16a + 4j + i'] = x[16a + 4i' + j]`).
#[cube]
fn perm16_idx(i: usize) -> usize {
    let rem = i % 16usize;
    (i / 16usize) * 16usize + (rem % 4usize) * 4usize + rem / 4usize
}

#[cube]
pub(crate) fn sigmoid_f(x: f32) -> f32 {
    1.0f32 / (1.0f32 + Exp::exp(-x))
}

/// `out[row] = x[row] * rsqrt(mean(x[row]^2) + eps) * w`, one cube (256 units) per row of `n`.
#[cube(launch)]
fn qw_rms_kernel(x: &[f32], w: &[f32], out: &mut [f32], outp: &mut [f32], eps: &[f32], #[comptime] n: usize) {
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let row = CUBE_POS as usize;
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < n {
        let v = x[row * n + i];
        ss += v * v;
        i += CUBE_DIM_X as usize;
    }
    let tot = lane_sum(&mut sm, ss, false);
    let r = 1.0f32 / Sqrt::sqrt(tot / (n as f32) + eps[0]);
    let mut i = UNIT_POS as usize;
    while i < n {
        let v = x[row * n + i] * r * w[i];
        out[row * n + i] = v;
        outp[row * n + perm16_idx(i)] = v;
        i += CUBE_DIM_X as usize;
    }
}

/// `out[o] = sum_c w[o, c] x[c]` for f32 matrices (router, beta/alpha, shared gate); a plane (or cube) per row.
#[cube(launch)]
fn qw_f32_matvec_kernel(w: &[f32], x: &[f32], out: &mut [f32], #[comptime] cols: usize, #[comptime] use_plane: bool) {
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut width = CUBE_DIM_X as usize;
    let mut plane_id = 0usize;
    let mut lane = UNIT_POS as usize;
    if use_plane {
        width = PLANE_DIM as usize;
        plane_id = (UNIT_POS / PLANE_DIM) as usize;
        lane = UNIT_POS_PLANE as usize;
    }
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut o = CUBE_POS as usize * ppc + plane_id;
    while o < out.len() {
        let mut part = 0.0f32;
        let mut c = lane;
        while c < cols {
            part += w[o * cols + c] * x[c];
            c += width;
        }
        let tot = lane_sum(&mut sm, part, use_plane);
        if lane == 0usize {
            out[o] = tot;
        }
        o += step;
    }
}

/// `sum = x + y`, `h = rms(sum) * w` (one row of `n`, one cube of 256 units).
#[cube(launch)]
fn qw_add_rms_kernel(x: &[f32], y: &[f32], w: &[f32], sum: &mut [f32], h: &mut [f32], hp: &mut [f32], eps: &[f32], #[comptime] n: usize) {
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < n {
        let v = x[i] + y[i];
        sum[i] = v;
        ss += v * v;
        i += CUBE_DIM_X as usize;
    }
    let tot = lane_sum(&mut sm, ss, false);
    let r = 1.0f32 / Sqrt::sqrt(tot / (n as f32) + eps[0]);
    let mut i = UNIT_POS as usize;
    while i < n {
        let v = sum[i] * r * w[i];
        h[i] = v;
        hp[perm16_idx(i)] = v;
        i += CUBE_DIM_X as usize;
    }
}

#[cube(launch)]
fn qw_add_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        out[i] = a[i] + b[i];
    }
}

#[cube(launch)]
fn qw_copy_kernel(src: &[f32], dst: &mut [f32]) {
    let i = ABSOLUTE_POS as usize;
    if i < dst.len() {
        dst[i] = src[i];
    }
}

#[cube(launch)]
fn qw_silu_mul_kernel(g: &[f32], u: &[f32], out: &mut [f32]) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        out[i] = silu_f(g[i]) * u[i];
    }
}

/// Causal depthwise conv of width `kd` over [`kd-1` stored columns | new column], SiLU; the state keeps the last `kd-1` columns.
#[cube(launch)]
fn qw_dn_conv_kernel(qkv: &[f32], cw: &[f32], state: &mut [f32], out: &mut [f32], #[comptime] ch: usize, #[comptime] kd: usize) {
    let c = ABSOLUTE_POS as usize;
    if c < ch {
        let mut sum = 0.0f32;
        let mut t = 0usize;
        while t < kd - 1usize {
            sum += state[t * ch + c] * cw[c * kd + t];
            t += 1usize;
        }
        sum += qkv[c] * cw[c * kd + kd - 1usize];
        out[c] = silu_f(sum);
        let mut t = 0usize;
        while t + 2usize < kd {
            state[t * ch + c] = state[(t + 1usize) * ch + c];
            t += 1usize;
        }
        state[(kd - 2usize) * ch + c] = qkv[c];
    }
}

/// `x / sqrt(sum(x^2) + eps)` per head of `hk`, in place; one cube (128 units) per head.
#[cube(launch)]
fn qw_l2_kernel(x: &mut [f32], eps: &[f32], #[comptime] hk: usize) {
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let hh = CUBE_POS as usize;
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < hk {
        let v = x[hh * hk + i];
        ss += v * v;
        i += CUBE_DIM_X as usize;
    }
    let tot = lane_sum(&mut sm, ss, false);
    let r = 1.0f32 / Sqrt::sqrt(tot + eps[0]);
    let mut i = UNIT_POS as usize;
    while i < hk {
        x[hh * hk + i] = x[hh * hk + i] * r;
        i += CUBE_DIM_X as usize;
    }
}

/// `beta = sigmoid(braw)`, `g = softplus(araw + dt_bias) * a` per v head.
#[cube(launch)]
fn qw_dn_scalars_kernel(ba: &[f32], dt: &[f32], a: &[f32], beta: &mut [f32], g: &mut [f32], #[comptime] nv: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < beta.len() {
        beta[i] = sigmoid_f(ba[i]);
        let x = ba[nv + i] + dt[i];
        let mut sp = x;
        if x <= 20.0f32 {
            sp = Log1p::log1p(Exp::exp(x));
        }
        g[i] = sp * a[i];
    }
}

/// The delta rule, one cube (4 planes of 32) per v head, one plane per state row, lanes along the row so every state access is
/// coalesced: `S *= exp(g); d = (v - S k) beta; S += d k^T; o = S q / sqrt(hk)`. `hk` = 128 (four columns a lane). `conv` is the
/// post-conv, L2-normalised row `[q (nk*hk) | k (nk*hk) | v (nv*hv)]`; v head `h` uses q/k head `h % nk`. Needs plane ops (CUDA).
#[cube(launch)]
fn qw_dn_recur_kernel(s: &mut [f32], conv: &[f32], beta: &[f32], g: &[f32], out: &mut [f32], #[comptime] nk: usize, #[comptime] hv: usize, #[comptime] hk: usize) {
    let hh = CUBE_POS as usize;
    let kh = hh % nk;
    let lane = UNIT_POS_PLANE as usize;
    let w = (UNIT_POS / PLANE_DIM) as usize;
    let nw = (CUBE_DIM_X / PLANE_DIM) as usize;
    let decay = Exp::exp(g[hh]);
    let bt = beta[hh];
    let mut kr = Array::<f32>::new(4usize);
    let mut qr = Array::<f32>::new(4usize);
    #[unroll]
    for m in 0..4usize {
        kr[m] = conv[nk * hk + kh * hk + lane + 32usize * m];
        qr[m] = conv[kh * hk + lane + 32usize * m];
    }
    let qscale = 1.0f32 / Sqrt::sqrt(hk as f32);
    let mut i = w;
    while i < hv {
        let base = hh * hv * hk + i * hk;
        let mut vr = Array::<f32>::new(4usize);
        let mut ps = 0.0f32;
        #[unroll]
        for m in 0..4usize {
            let v = s[base + lane + 32usize * m] * decay;
            vr[m] = v;
            ps += v * kr[m];
        }
        let sk = plane_sum(ps);
        let d = (conv[2usize * nk * hk + hh * hv + i] - sk) * bt;
        let mut po = 0.0f32;
        #[unroll]
        for m in 0..4usize {
            let v = vr[m] + kr[m] * d;
            s[base + lane + 32usize * m] = v;
            po += v * qr[m];
        }
        let o = plane_sum(po);
        if lane == 0usize {
            out[hh * hv + i] = o * qscale;
        }
        i += nw;
    }
}

/// The whole delta-net core of one token in one launch (CUDA, plane ops): a cube (4 planes) per v head `hh` does the causal conv + SiLU
/// of its q/k head's channels (`hh % nk`) and its own v channels, the L2 norms, beta and g, the delta rule (a plane a state row, lanes along the
/// row), and the gated RMS norm; it writes the result natural and in the 16-value-permuted order. The conv state is double buffered
/// (`st2` = two copies, the token's position parity picks the one to read; the cubes sharing a q/k head write identical values), so
/// no cube reads a column another has already shifted. `nk` = 16 q/k heads, `hk` = `hv` = 128, conv width 4.
#[cube(launch)]
fn qw_dn_fused_kernel(
    qkv: &[f32], z: &[f32], ba: &[f32], cw: &[f32], dt: &[f32], a: &[f32], nw: &[f32],
    st2: &mut [f32], s: &mut [f32], out: &mut [f32], outp: &mut [f32], pp: &[u32], eps: &[f32],
    #[comptime] nk: usize, #[comptime] nv: usize, #[comptime] hk: usize, #[comptime] ch: usize,
) {
    let hh = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let w = (UNIT_POS / PLANE_DIM) as usize;
    let nwp = (CUBE_DIM_X / PLANE_DIM) as usize;
    let kh = hh % nk;
    let par = (pp[0] % 2u32) as usize;
    let rd = par * 3usize * ch;
    let wr = (1usize - par) * 3usize * ch;
    let mut qs = Shared::<[f32]>::new_slice(128usize);
    let mut ks = Shared::<[f32]>::new_slice(128usize);
    let mut vs = Shared::<[f32]>::new_slice(128usize);
    let mut os = Shared::<[f32]>::new_slice(128usize);
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    // conv + SiLU of the three channels this thread owns (q, k of head kh; v of head hh), and the state shift
    let cq = kh * hk + t;
    let ck = nk * hk + kh * hk + t;
    let cv = 2usize * nk * hk + hh * hk + t;
    let mut sum = cw[cq * 4usize + 3usize] * qkv[cq];
    sum += st2[rd + cq] * cw[cq * 4usize];
    sum += st2[rd + ch + cq] * cw[cq * 4usize + 1usize];
    sum += st2[rd + 2usize * ch + cq] * cw[cq * 4usize + 2usize];
    qs[t] = silu_f(sum);
    st2[wr + cq] = st2[rd + ch + cq];
    st2[wr + ch + cq] = st2[rd + 2usize * ch + cq];
    st2[wr + 2usize * ch + cq] = qkv[cq];
    let mut sumk = cw[ck * 4usize + 3usize] * qkv[ck];
    sumk += st2[rd + ck] * cw[ck * 4usize];
    sumk += st2[rd + ch + ck] * cw[ck * 4usize + 1usize];
    sumk += st2[rd + 2usize * ch + ck] * cw[ck * 4usize + 2usize];
    ks[t] = silu_f(sumk);
    st2[wr + ck] = st2[rd + ch + ck];
    st2[wr + ch + ck] = st2[rd + 2usize * ch + ck];
    st2[wr + 2usize * ch + ck] = qkv[ck];
    let mut sumv = cw[cv * 4usize + 3usize] * qkv[cv];
    sumv += st2[rd + cv] * cw[cv * 4usize];
    sumv += st2[rd + ch + cv] * cw[cv * 4usize + 1usize];
    sumv += st2[rd + 2usize * ch + cv] * cw[cv * 4usize + 2usize];
    vs[t] = silu_f(sumv);
    st2[wr + cv] = st2[rd + ch + cv];
    st2[wr + ch + cv] = st2[rd + 2usize * ch + cv];
    st2[wr + 2usize * ch + cv] = qkv[cv];
    sync_cube();
    // L2 norms of q and k
    let qv = qs[t];
    let kv = ks[t];
    let sq = lane_sum(&mut sm, qv * qv, false);
    let sk = lane_sum(&mut sm, kv * kv, false);
    let rq = 1.0f32 / Sqrt::sqrt(sq + eps[0]);
    let rk = 1.0f32 / Sqrt::sqrt(sk + eps[0]);
    qs[t] = qv * rq;
    ks[t] = kv * rk;
    sync_cube();
    let beta = sigmoid_f(ba[hh]);
    let xg = ba[nv + hh] + dt[hh];
    let mut sp = xg;
    if xg <= 20.0f32 {
        sp = Log1p::log1p(Exp::exp(xg));
    }
    let decay = Exp::exp(sp * a[hh]);
    let mut kr = Array::<f32>::new(4usize);
    let mut qr = Array::<f32>::new(4usize);
    #[unroll]
    for m in 0..4usize {
        kr[m] = ks[lane + 32usize * m];
        qr[m] = qs[lane + 32usize * m];
    }
    let qscale = 1.0f32 / Sqrt::sqrt(hk as f32);
    let mut i = w;
    while i < hk {
        let base = hh * hk * hk + i * hk;
        let mut vr = Array::<f32>::new(4usize);
        let mut ps = 0.0f32;
        #[unroll]
        for m in 0..4usize {
            let v = s[base + lane + 32usize * m] * decay;
            vr[m] = v;
            ps += v * kr[m];
        }
        let skv = plane_sum(ps);
        let d = (vs[i] - skv) * beta;
        let mut po = 0.0f32;
        #[unroll]
        for m in 0..4usize {
            let v = vr[m] + kr[m] * d;
            s[base + lane + 32usize * m] = v;
            po += v * qr[m];
        }
        let o = plane_sum(po);
        if lane == 0usize {
            os[i] = o * qscale;
        }
        i += nwp;
    }
    sync_cube();
    // gated RMS norm of the head's output
    let ov = os[t];
    let so = lane_sum(&mut sm, ov * ov, false);
    let r = 1.0f32 / Sqrt::sqrt(so / (hk as f32) + eps[0]);
    let val = ov * r * nw[t] * silu_f(z[hh * hk + t]);
    out[hh * hk + t] = val;
    outp[hh * hk + perm16_idx(t)] = val;
}

/// `out = rms(o) * w * silu(z)` per head of `hv`; one cube per head.
#[cube(launch)]
fn qw_dn_gnorm_kernel(o: &[f32], z: &[f32], w: &[f32], out: &mut [f32], outp: &mut [f32], eps: &[f32], #[comptime] hv: usize) {
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let hh = CUBE_POS as usize;
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < hv {
        let v = o[hh * hv + i];
        ss += v * v;
        i += CUBE_DIM_X as usize;
    }
    let tot = lane_sum(&mut sm, ss, false);
    let r = 1.0f32 / Sqrt::sqrt(tot / (hv as f32) + eps[0]);
    let mut i = UNIT_POS as usize;
    while i < hv {
        let v = o[hh * hv + i] * r * w[i] * silu_f(z[hh * hv + i]);
        out[hh * hv + i] = v;
        outp[hh * hv + perm16_idx(i)] = v;
        i += CUBE_DIM_X as usize;
    }
}

/// Per-head RMS norm + partial NEOX rotary (`2*half` of `hd` values, pairs `(i, i + half)`, `theta_i = pos * ts^i` accumulated in f32)
/// for `nh` query heads (cubes `0..nh`, `qf` = `[head][q | gate]`) and `nkv` key heads (cubes `nh..`), and the KV-cache append
/// (`[pos][kv head][hd]`). `hd` units a cube. `pp` = [pos]; `pf` = [eps, ts].
#[cube(launch)]
fn qw_attn_prep_kernel(
    qf: &[f32], kraw: &[f32], vraw: &[f32], qn: &[f32], kn: &[f32],
    qo: &mut [f32], gate: &mut [f32], kc: &mut [f32], vc: &mut [f32], pp: &[u32], pf: &[f32],
    #[comptime] nh: usize, #[comptime] nkv: usize, #[comptime] hd: usize, #[comptime] half: usize,
) {
    let c = CUBE_POS as usize;
    let d = UNIT_POS as usize;
    let pos = pp[0] as usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut sv = Shared::<[f32]>::new_slice(256usize);
    let mut src = 0.0f32;
    let mut wn = 0.0f32;
    if c < nh {
        src = qf[c * 2usize * hd + d];
        wn = qn[d];
        gate[c * hd + d] = qf[c * 2usize * hd + hd + d];
    } else {
        let j = c - nh;
        src = kraw[j * hd + d];
        wn = kn[d];
        vc[(pos * nkv + j) * hd + d] = vraw[j * hd + d];
    }
    let tot = lane_sum(&mut sm, src * src, false);
    let r = 1.0f32 / Sqrt::sqrt(tot / (hd as f32) + pf[0]);
    sv[d] = src * r * wn;
    sync_cube();
    let mut val = sv[d];
    if d < 2usize * half {
        let i = d % half;
        let mut theta = pos as f32;
        let mut t = 0usize;
        while t < i {
            theta = theta * pf[1];
            t += 1usize;
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
        qo[c * hd + d] = val;
    } else {
        kc[(pos * nkv + (c - nh)) * hd + d] = val;
    }
}

/// Softmax attention for one token over `pos + 1` cached positions, then `* sigmoid(gate)`; one cube (256 units) per query head.
/// `scores` is scratch `[nh][cap]`.
#[cube(launch)]
fn qw_attn_decode_kernel(
    q: &[f32], kc: &[f32], vc: &[f32], gate: &[f32], out: &mut [f32], scores: &mut [f32], pp: &[u32],
    #[comptime] nh: usize, #[comptime] nkv: usize, #[comptime] hd: usize, #[comptime] cap: usize,
) {
    let h = CUBE_POS as usize;
    let u = UNIT_POS as usize;
    let n = pp[0] as usize + 1usize;
    let kvh = h / (nh / nkv);
    let scale = 1.0f32 / Sqrt::sqrt(hd as f32);
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut mx = q[0usize] * 0.0f32 - 3.0e38f32;
    let mut t = u;
    while t < n {
        let mut s = 0.0f32;
        let mut k = 0usize;
        while k < hd {
            s += q[h * hd + k] * kc[(t * nkv + kvh) * hd + k];
            k += 1usize;
        }
        s = s * scale;
        scores[h * cap + t] = s;
        if s > mx {
            mx = s;
        }
        t += CUBE_DIM_X as usize;
    }
    // max reduce
    sm[u] = mx;
    sync_cube();
    let mut st = CUBE_DIM_X / 2u32;
    while st > 0u32 {
        if UNIT_POS < st {
            let o2 = sm[u + st as usize];
            if o2 > sm[u] {
                sm[u] = o2;
            }
        }
        sync_cube();
        st = st / 2u32;
    }
    let m = sm[0usize];
    sync_cube();
    let mut sum = 0.0f32;
    let mut t = u;
    while t < n {
        let e = Exp::exp(scores[h * cap + t] - m);
        scores[h * cap + t] = e;
        sum += e;
        t += CUBE_DIM_X as usize;
    }
    let total = lane_sum(&mut sm, sum, false);
    let mut d = u;
    while d < hd {
        let mut o = 0.0f32;
        let mut t = 0usize;
        while t < n {
            o += scores[h * cap + t] / total * vc[(t * nkv + kvh) * hd + d];
            t += 1usize;
        }
        out[h * hd + d] = o * sigmoid_f(gate[h * hd + d]);
        d += CUBE_DIM_X as usize;
    }
}

/// Plane-wide attention for one token (CUDA): the scores are computed a position per plane with the lanes along the head dimension
/// (coalesced reads of K, one `plane_sum` a position), `p = e / sum` is stored once, and the output loop is a plain FMA over positions.
/// One cube of 256 units (8 planes of 32) per query head; `hd` = 256. Then `* sigmoid(gate)`.
#[cube(launch)]
fn qw_attn_decode_wide_kernel(
    q: &[f32], kc: &[f32], vc: &[f32], gate: &[f32], out: &mut [f32], scores: &mut [f32], pp: &[u32],
    #[comptime] nh: usize, #[comptime] nkv: usize, #[comptime] hd: usize, #[comptime] cap: usize,
) {
    let h = CUBE_POS as usize;
    let u = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let w = (UNIT_POS / PLANE_DIM) as usize;
    let nw = (CUBE_DIM_X / PLANE_DIM) as usize;
    let n = pp[0] as usize + 1usize;
    let kvh = h / (nh / nkv);
    let scale = 1.0f32 / Sqrt::sqrt(hd as f32);
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut qr = Array::<f32>::new(8usize);
    #[unroll]
    for m in 0..8usize {
        qr[m] = q[h * hd + lane + 32usize * m];
    }
    let mut t = w;
    while t < n {
        let kb = (t * nkv + kvh) * hd;
        let mut part = 0.0f32;
        #[unroll]
        for m in 0..8usize {
            part += qr[m] * kc[kb + lane + 32usize * m];
        }
        let s = plane_sum(part) * scale;
        if lane == 0usize {
            scores[h * cap + t] = s;
        }
        t += nw;
    }
    sync_cube();
    // max over positions
    let mut mx = q[0usize] * 0.0f32 - 3.0e38f32;
    let mut t = u;
    while t < n {
        let sv = scores[h * cap + t];
        if sv > mx {
            mx = sv;
        }
        t += CUBE_DIM_X as usize;
    }
    sm[u] = mx;
    sync_cube();
    let mut st = CUBE_DIM_X / 2u32;
    while st > 0u32 {
        if UNIT_POS < st {
            let o2 = sm[u + st as usize];
            if o2 > sm[u] {
                sm[u] = o2;
            }
        }
        sync_cube();
        st = st / 2u32;
    }
    let m = sm[0usize];
    sync_cube();
    let mut sum = 0.0f32;
    let mut t = u;
    while t < n {
        let e = Exp::exp(scores[h * cap + t] - m);
        scores[h * cap + t] = e;
        sum += e;
        t += CUBE_DIM_X as usize;
    }
    let total = lane_sum(&mut sm, sum, false);
    let mut t = u;
    while t < n {
        scores[h * cap + t] = scores[h * cap + t] / total;
        t += CUBE_DIM_X as usize;
    }
    sync_cube();
    // output: unit d owns dimension d (hd == cube size); positions in groups of four for independent loads
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    let mut t = 0usize;
    while t + 3usize < n {
        o0 += scores[h * cap + t] * vc[(t * nkv + kvh) * hd + u];
        o1 += scores[h * cap + t + 1usize] * vc[((t + 1usize) * nkv + kvh) * hd + u];
        o2 += scores[h * cap + t + 2usize] * vc[((t + 2usize) * nkv + kvh) * hd + u];
        o3 += scores[h * cap + t + 3usize] * vc[((t + 3usize) * nkv + kvh) * hd + u];
        t += 4usize;
    }
    while t < n {
        o0 += scores[h * cap + t] * vc[(t * nkv + kvh) * hd + u];
        t += 1usize;
    }
    out[h * hd + u] = (o0 + o1 + o2 + o3) * sigmoid_f(gate[h * hd + u]);
}

// ---- `LB_QW_ATTN_SPLIT`: the wide kernel's arithmetic in three launches that fill the card (P100 tuning, 2026-11-21).
// The wide kernel runs one cube per query head (16 cubes: 16 of a P100's 56 SMs), every head re-reads its KV head's whole K and V, and the
// output loop walks all positions serially; at 4k context it costs ~0.8 ms a layer. Every value below is computed with exactly the
// operations, operands and order of the wide kernel (per-position scores with the same lane split and `plane_sum`; the max/exp/total code
// unchanged; the four output chains o0..o3, split by position mod 4 with the tail folded into o0, as separate threads, combined as
// ((o0 + o1) + o2) + o3), so the output is bit-identical; only the work is spread over many more cubes.

/// Scores of one token for all query heads: a plane takes one position of one KV head and serves the `nh / nkv` query heads that share it
/// (K read once). `hd` = 256 (8 values a lane). Grid: any number of 256-unit cubes, a multiple of `nkv` planes.
#[cube(launch)]
fn qw_attn_scores_split_kernel(q: &[f32], kc: &[f32], scores: &mut [f32], pp: &[u32], #[comptime] nh: usize, #[comptime] nkv: usize, #[comptime] hd: usize, #[comptime] cap: usize) {
    let g = nh / nkv;
    let lane = UNIT_POS_PLANE as usize;
    let ppc = (CUBE_DIM_X / PLANE_DIM) as usize;
    let planes = CUBE_COUNT as usize * ppc;
    let per_kv = planes / nkv;
    let p = CUBE_POS as usize * ppc + (UNIT_POS / PLANE_DIM) as usize;
    let kvh = p / per_kv;
    let n = pp[0] as usize + 1usize;
    let scale = 1.0f32 / Sqrt::sqrt(hd as f32);
    if kvh < nkv {
        let mut qr = Array::<f32>::new(64usize);
        #[unroll]
        for j in 0..g {
            #[unroll]
            for m in 0..8usize {
                qr[j * 8usize + m] = q[(kvh * g + j) * hd + lane + 32usize * m];
            }
        }
        let mut t = p % per_kv;
        while t < n {
            let kb = (t * nkv + kvh) * hd;
            let mut kr = Array::<f32>::new(8usize);
            #[unroll]
            for m in 0..8usize {
                kr[m] = kc[kb + lane + 32usize * m];
            }
            #[unroll]
            for j in 0..g {
                let mut part = 0.0f32;
                #[unroll]
                for m in 0..8usize {
                    part += qr[j * 8usize + m] * kr[m];
                }
                let s = plane_sum(part) * scale;
                if lane == 0usize {
                    scores[(kvh * g + j) * cap + t] = s;
                }
            }
            t += per_kv;
        }
    }
}

/// Softmax of the scores in place (`p = exp(s - max) / sum`), one cube of 256 units per query head: the wide kernel's code verbatim.
#[cube(launch)]
fn qw_attn_softmax_split_kernel(scores: &mut [f32], pp: &[u32], #[comptime] cap: usize) {
    let h = CUBE_POS as usize;
    let u = UNIT_POS as usize;
    let n = pp[0] as usize + 1usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut mx = scores[0usize] * 0.0f32 - 3.0e38f32;
    let mut t = u;
    while t < n {
        let sv = scores[h * cap + t];
        if sv > mx {
            mx = sv;
        }
        t += CUBE_DIM_X as usize;
    }
    sm[u] = mx;
    sync_cube();
    let mut st = CUBE_DIM_X / 2u32;
    while st > 0u32 {
        if UNIT_POS < st {
            let o2 = sm[u + st as usize];
            if o2 > sm[u] {
                sm[u] = o2;
            }
        }
        sync_cube();
        st = st / 2u32;
    }
    let m = sm[0usize];
    sync_cube();
    let mut sum = 0.0f32;
    let mut t = u;
    while t < n {
        let e = Exp::exp(scores[h * cap + t] - m);
        scores[h * cap + t] = e;
        sum += e;
        t += CUBE_DIM_X as usize;
    }
    let total = lane_sum(&mut sm, sum, false);
    let mut t = u;
    while t < n {
        scores[h * cap + t] = scores[h * cap + t] / total;
        t += CUBE_DIM_X as usize;
    }
}

/// Output of one token: cube `(dim chunk of 64, query head)`, 256 units = 4 chains x 64 dimensions; chain `r` sums positions `t = r mod 4`
/// below `4 * floor(n / 4)` (chain 0 also takes the tail), as the wide kernel's o0..o3. Loads of four chain steps are issued before their
/// FMAs (the sums keep their order). Heads sharing a KV head are adjacent cubes, so their V reads meet in the cache.
#[cube(launch)]
fn qw_attn_out_split_kernel(p: &[f32], vc: &[f32], gate: &[f32], out: &mut [f32], pp: &[u32], #[comptime] nh: usize, #[comptime] nkv: usize, #[comptime] hd: usize, #[comptime] cap: usize) {
    let h = CUBE_POS as usize % nh;
    let chunk = CUBE_POS as usize / nh;
    let u = UNIT_POS as usize;
    let r = u / 64usize;
    let d = chunk * 64usize + u % 64usize;
    let n = pp[0] as usize + 1usize;
    let n4 = (n / 4usize) * 4usize;
    let kvh = h / (nh / nkv);
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let pb = h * cap;
    let mut o = 0.0f32;
    let mut t = r;
    while t + 12usize < n4 {
        let p0 = p[pb + t];
        let v0 = vc[(t * nkv + kvh) * hd + d];
        let p1 = p[pb + t + 4usize];
        let v1 = vc[((t + 4usize) * nkv + kvh) * hd + d];
        let p2 = p[pb + t + 8usize];
        let v2 = vc[((t + 8usize) * nkv + kvh) * hd + d];
        let p3 = p[pb + t + 12usize];
        let v3 = vc[((t + 12usize) * nkv + kvh) * hd + d];
        o += p0 * v0;
        o += p1 * v1;
        o += p2 * v2;
        o += p3 * v3;
        t += 16usize;
    }
    while t < n4 {
        o += p[pb + t] * vc[(t * nkv + kvh) * hd + d];
        t += 4usize;
    }
    if r == 0usize {
        let mut t = n4;
        while t < n {
            o += p[pb + t] * vc[(t * nkv + kvh) * hd + d];
            t += 1usize;
        }
    }
    sm[u] = o;
    sync_cube();
    if r == 0usize {
        let dl = u % 64usize;
        out[h * hd + d] = (o + sm[64usize + dl] + sm[128usize + dl] + sm[192usize + dl]) * sigmoid_f(gate[h * hd + d]);
    }
}

/// Index of the largest of `n` values (ties: the lowest index), one cube of 256 units; the index is written as the bits of a u32 in `out[0]`.
#[cube(launch)]
fn qw_argmax_kernel(x: &[f32], out: &mut [f32], #[comptime] n: usize) {
    let t = UNIT_POS as usize;
    let mut sv = Shared::<[f32]>::new_slice(256usize);
    let mut si = Shared::<[u32]>::new_slice(256usize);
    let mut bv = x[0usize] * 0.0f32 - 3.0e38f32;
    let mut bi = 0u32;
    let mut i = t;
    while i < n {
        let v = x[i];
        if v > bv {
            bv = v;
            bi = i as u32;
        }
        i += 256usize;
    }
    sv[t] = bv;
    si[t] = bi;
    sync_cube();
    let mut st = 128usize;
    while st > 0usize {
        if t < st {
            let ov = sv[t + st];
            let oi = si[t + st];
            if ov > sv[t] || (ov == sv[t] && oi < si[t]) {
                sv[t] = ov;
                si[t] = oi;
            }
        }
        sync_cube();
        st = st / 2usize;
    }
    if t == 0usize {
        out[0usize] = f32::reinterpret(si[0usize]);
    }
}

/// [`qw_argmax_kernel`] over the `m` rows of an `[m, n]` matrix, one cube a row; row `r`'s index is written in `out[r]`.
#[cube(launch)]
fn qw_argmax_rows_kernel(x: &[f32], out: &mut [f32], #[comptime] n: usize) {
    let t = UNIT_POS as usize;
    let base = CUBE_POS_X as usize * n;
    let mut sv = Shared::<[f32]>::new_slice(256usize);
    let mut si = Shared::<[u32]>::new_slice(256usize);
    let mut bv = x[base] * 0.0f32 - 3.0e38f32;
    let mut bi = 0u32;
    let mut i = t;
    while i < n {
        let v = x[base + i];
        if v > bv {
            bv = v;
            bi = i as u32;
        }
        i += 256usize;
    }
    sv[t] = bv;
    si[t] = bi;
    sync_cube();
    let mut st = 128usize;
    while st > 0usize {
        if t < st {
            let ov = sv[t + st];
            let oi = si[t + st];
            if ov > sv[t] || (ov == sv[t] && oi < si[t]) {
                sv[t] = ov;
                si[t] = oi;
            }
        }
        sync_cube();
        st = st / 2usize;
    }
    if t == 0usize {
        out[CUBE_POS_X as usize] = f32::reinterpret(si[0usize]);
    }
}

/// Softmax over `ne` (<= 512) router logits, top `k` by probability (ties: lowest index), weights renormalised; writes `[expert, 0]`
/// pairs. One cube of 256 units (`ne` <= 256): max and sum reductions, then every expert counts how many outrank it.
#[cube(launch)]
fn qw_route_kernel(logits: &[f32], routes: &mut [u32], weights: &mut [f32], #[comptime] ne: usize, #[comptime] k: usize) {
    let t = UNIT_POS as usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut p = Shared::<[f32]>::new_slice(512usize);
    let mut mx = logits[0usize] * 0.0f32 - 3.0e38f32;
    let mut i = t;
    while i < ne {
        if logits[i] > mx {
            mx = logits[i];
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
        let e = Exp::exp(logits[i] - m);
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
    // rank of every expert by (probability desc, index asc), one pass; the top k write themselves out in rank order
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
            routes[2usize * rank] = t as u32;
            routes[2usize * rank + 1usize] = 0u32;
            weights[rank] = pt;
        }
    }
    sync_cube();
    if t == 0usize {
        let mut sel_sum = 0.0f32;
        let mut r = 0usize;
        while r < k {
            sel_sum += weights[r];
            r += 1usize;
        }
        let mut r = 0usize;
        while r < k {
            weights[r] = weights[r] / sel_sum;
            r += 1usize;
        }
    }
}

/// `out = x + (sum_r w_r down_r) + sigmoid(sgate) * shared`, in the CPU decoder's order.
#[cube(launch)]
fn qw_moe_combine_kernel(x: &[f32], down: &[f32], weights: &[f32], shared: &[f32], sgate: &[f32], out: &mut [f32], #[comptime] n: usize, #[comptime] k: usize, #[comptime] sg_off: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let mut acc = 0.0f32;
        let mut r = 0usize;
        while r < k {
            acc += weights[r] * down[r * n + i];
            r += 1usize;
        }
        acc += sigmoid_f(sgate[sg_off]) * shared[i];
        out[i] = x[i] + acc;
    }
}

/// Launchers. Buffer arguments count f32 elements (u32 for `routes`/`pos`).
impl Gpu {
    fn qw_cubes(&self, n: usize) -> CubeCount {
        cube_count(n, self.cube_dim)
    }

    /// Returns (`rms(x) * w`, the same values in the 16-value-permuted order the native Q5_K/Q6_K kernels read; `n` a multiple of 16).
    pub fn qw_rms(&self, x: &Handle, w: &Handle, rows: usize, n: usize, eps: f32) -> (Handle, Handle) {
        let out = self.scratch(rows * n * 4);
        let outp = self.scratch(rows * n * 4);
        let e = self.const_f32(&[eps]);
        unsafe {
            qw_rms_kernel::launch(&self.client, CubeCount::Static(rows as u32, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(x.clone(), rows * n), BufferArg::from_raw_parts(w.clone(), n), BufferArg::from_raw_parts(out.clone(), rows * n), BufferArg::from_raw_parts(outp.clone(), rows * n), BufferArg::from_raw_parts(e, 1), n);
        }
        (out, outp)
    }

    /// `w` is `rows x cols` f32 on the device.
    pub fn qw_f32_matvec(&self, w: &Handle, x: &Handle, rows: usize, cols: usize) -> Handle {
        let out = self.scratch(rows * 4);
        unsafe {
            qw_f32_matvec_kernel::launch(&self.client, cube_count(rows, self.per_cube()), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(w.clone(), rows * cols), BufferArg::from_raw_parts(x.clone(), cols), BufferArg::from_raw_parts(out.clone(), rows), cols, self.use_plane);
        }
        out
    }

    /// Returns (`x + y`, `rms(x + y) * w`, the latter permuted).
    pub fn qw_add_rms(&self, x: &Handle, y: &Handle, w: &Handle, n: usize, eps: f32) -> (Handle, Handle, Handle) {
        let sum = self.scratch(n * 4);
        let h = self.scratch(n * 4);
        let hp = self.scratch(n * 4);
        let e = self.const_f32(&[eps]);
        unsafe {
            qw_add_rms_kernel::launch(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(x.clone(), n), BufferArg::from_raw_parts(y.clone(), n), BufferArg::from_raw_parts(w.clone(), n), BufferArg::from_raw_parts(sum.clone(), n), BufferArg::from_raw_parts(h.clone(), n), BufferArg::from_raw_parts(hp.clone(), n), BufferArg::from_raw_parts(e, 1), n);
        }
        (sum, h, hp)
    }

    pub fn qw_add(&self, a: &Handle, b: &Handle, n: usize) -> Handle {
        let out = self.scratch(n * 4);
        unsafe {
            qw_add_kernel::launch(&self.client, self.qw_cubes(n), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(a.clone(), n), BufferArg::from_raw_parts(b.clone(), n), BufferArg::from_raw_parts(out.clone(), n));
        }
        out
    }

    /// Device-to-device copy of the first `n` f32 of `src` into `dst` (both persistent or scratch; `dst` may be larger).
    pub fn qw_copy(&self, src: &Handle, dst: &Handle, n: usize) {
        unsafe {
            qw_copy_kernel::launch(&self.client, self.qw_cubes(n), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(src.clone(), n), BufferArg::from_raw_parts(dst.clone(), n));
        }
    }

    pub fn qw_silu_mul(&self, g: &Handle, u: &Handle, n: usize) -> Handle {
        let out = self.scratch(n * 4);
        unsafe {
            qw_silu_mul_kernel::launch(&self.client, self.qw_cubes(n), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(g.clone(), n), BufferArg::from_raw_parts(u.clone(), n), BufferArg::from_raw_parts(out.clone(), n));
        }
        out
    }

    /// Delta-net core for one token, one launch. `qkv` `[ch]`, `z` `[d_inner]`, `ba` `[2 nv]` (beta logits then alpha logits); `conv_state` is
    /// `2 * (kd-1) * ch` (double buffered, see the kernel) and `s` `[nv * hv * hk]`, both persistent (`upload_f32`), updated in place;
    /// `pos_h` holds the token position (its parity picks the conv-state buffer). Returns the gated-normed `[d_inner]` (natural, permuted).
    #[allow(clippy::too_many_arguments)]
    pub fn qw_delta_net(&self, qkv: &Handle, z: &Handle, ba: &Handle, conv_w: &Handle, dt: &Handle, a: &Handle, norm_w: &Handle, conv_state: &Handle, s: &Handle, pos_h: &Handle, (ch, kd, nk, nv, hk, hv): (usize, usize, usize, usize, usize, usize), eps: f32) -> (Handle, Handle) {
        assert!(kd == 4 && hk == 128 && hv == 128, "fused delta-net kernel is written for conv width 4 and 128-wide heads");
        let out = self.scratch(nv * hv * 4);
        let outp = self.scratch(nv * hv * 4);
        let e = self.const_f32(&[eps]);
        unsafe {
            qw_dn_fused_kernel::launch(
                &self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(128),
                BufferArg::from_raw_parts(qkv.clone(), ch), BufferArg::from_raw_parts(z.clone(), nv * hv), BufferArg::from_raw_parts(ba.clone(), 2 * nv),
                BufferArg::from_raw_parts(conv_w.clone(), ch * kd), BufferArg::from_raw_parts(dt.clone(), nv), BufferArg::from_raw_parts(a.clone(), nv),
                BufferArg::from_raw_parts(norm_w.clone(), hv), BufferArg::from_raw_parts(conv_state.clone(), 2 * (kd - 1) * ch), BufferArg::from_raw_parts(s.clone(), nv * hv * hk),
                BufferArg::from_raw_parts(out.clone(), nv * hv), BufferArg::from_raw_parts(outp.clone(), nv * hv), BufferArg::from_raw_parts(pos_h.clone(), 1), BufferArg::from_raw_parts(e, 1),
                nk, nv, hk, ch,
            );
        }
        (out, outp)
    }

    /// Gated attention core for one token at `pos` (`pos_h` = a u32 buffer holding it, rewritten by the caller each token with
    /// [`qw_write_u32`](Self::qw_write_u32)). `kc`/`vc` are persistent `[cap][nkv][hd]`. Returns `[nh * hd]` already multiplied
    /// by `sigmoid(gate)`.
    #[allow(clippy::too_many_arguments)]
    pub fn qw_attention(&self, qf: &Handle, kraw: &Handle, vraw: &Handle, qn: &Handle, kn: &Handle, kc: &Handle, vc: &Handle, pos_h: &Handle, (nh, nkv, hd, half, cap): (usize, usize, usize, usize, usize), eps: f32, theta_scale: f32) -> Handle {
        let qo = self.scratch(nh * hd * 4);
        let gate = self.scratch(nh * hd * 4);
        let scores = self.scratch(nh * cap * 4);
        let out = self.scratch(nh * hd * 4);
        let pf = self.const_f32(&[eps, theta_scale]);
        unsafe {
            qw_attn_prep_kernel::launch(
                &self.client, CubeCount::Static((nh + nkv) as u32, 1, 1), CubeDim::new_1d(hd as u32),
                BufferArg::from_raw_parts(qf.clone(), nh * 2 * hd), BufferArg::from_raw_parts(kraw.clone(), nkv * hd), BufferArg::from_raw_parts(vraw.clone(), nkv * hd),
                BufferArg::from_raw_parts(qn.clone(), hd), BufferArg::from_raw_parts(kn.clone(), hd),
                BufferArg::from_raw_parts(qo.clone(), nh * hd), BufferArg::from_raw_parts(gate.clone(), nh * hd),
                BufferArg::from_raw_parts(kc.clone(), cap * nkv * hd), BufferArg::from_raw_parts(vc.clone(), cap * nkv * hd),
                BufferArg::from_raw_parts(pos_h.clone(), 1), BufferArg::from_raw_parts(pf, 2), nh, nkv, hd, half,
            );
            let wide = self.use_plane && self.plane_size.1 == 32 && hd == 256 && std::env::var("LB_QW_ATTN_OLD").is_err();
            if wide && self.kq.attn_split && nh % nkv == 0 && nh / nkv <= 8 {
                let cubes = self.kq.attn_cubes.div_ceil(nkv).max(1) * nkv;
                qw_attn_scores_split_kernel::launch(
                    &self.client, CubeCount::Static(cubes as u32, 1, 1), CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(qo, nh * hd), BufferArg::from_raw_parts(kc.clone(), cap * nkv * hd), BufferArg::from_raw_parts(scores.clone(), nh * cap),
                    BufferArg::from_raw_parts(pos_h.clone(), 1), nh, nkv, hd, cap,
                );
                qw_attn_softmax_split_kernel::launch(
                    &self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(scores.clone(), nh * cap), BufferArg::from_raw_parts(pos_h.clone(), 1), cap,
                );
                qw_attn_out_split_kernel::launch(
                    &self.client, CubeCount::Static((nh * hd / 64) as u32, 1, 1), CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(scores, nh * cap), BufferArg::from_raw_parts(vc.clone(), cap * nkv * hd), BufferArg::from_raw_parts(gate, nh * hd),
                    BufferArg::from_raw_parts(out.clone(), nh * hd), BufferArg::from_raw_parts(pos_h.clone(), 1), nh, nkv, hd, cap,
                );
                return out;
            }
            if wide {
                qw_attn_decode_wide_kernel::launch(
                    &self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(qo, nh * hd), BufferArg::from_raw_parts(kc.clone(), cap * nkv * hd), BufferArg::from_raw_parts(vc.clone(), cap * nkv * hd),
                    BufferArg::from_raw_parts(gate, nh * hd), BufferArg::from_raw_parts(out.clone(), nh * hd), BufferArg::from_raw_parts(scores, nh * cap),
                    BufferArg::from_raw_parts(pos_h.clone(), 1), nh, nkv, hd, cap,
                );
                return out;
            }
            qw_attn_decode_kernel::launch(
                &self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(256),
                BufferArg::from_raw_parts(qo, nh * hd), BufferArg::from_raw_parts(kc.clone(), cap * nkv * hd), BufferArg::from_raw_parts(vc.clone(), cap * nkv * hd),
                BufferArg::from_raw_parts(gate, nh * hd), BufferArg::from_raw_parts(out.clone(), nh * hd), BufferArg::from_raw_parts(scores, nh * cap),
                BufferArg::from_raw_parts(pos_h.clone(), 1), nh, nkv, hd, cap,
            );
        }
        out
    }

    /// Overwrite a persistent u32 buffer (a position) in place.
    pub fn qw_write_u32(&self, h: &Handle, v: &[u32]) {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.client.write(h, cubecl::bytes::Bytes::from_bytes_vec(b));
    }

    /// Router: `logits` `[ne]` -> (`routes` `[2k]` u32, `weights` `[k]`).
    pub fn qw_route(&self, logits: &Handle, ne: usize, k: usize) -> (Handle, Handle) {
        let routes = self.scratch(2 * k * 4);
        let weights = self.scratch(k * 4);
        unsafe {
            qw_route_kernel::launch(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(logits.clone(), ne), BufferArg::from_raw_parts(routes.clone(), 2 * k), BufferArg::from_raw_parts(weights.clone(), k), ne, k);
        }
        (routes, weights)
    }

    /// `sgate` is a buffer whose element `sg_off` is the shared-expert gate logit (the router matvec's extra row).
    pub fn qw_moe_combine(&self, x: &Handle, down: &Handle, weights: &Handle, shared: &Handle, sgate: &Handle, sg_len: usize, sg_off: usize, n: usize, k: usize) -> Handle {
        let out = self.scratch(n * 4);
        unsafe {
            qw_moe_combine_kernel::launch(&self.client, self.qw_cubes(n), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(x.clone(), n), BufferArg::from_raw_parts(down.clone(), k * n), BufferArg::from_raw_parts(weights.clone(), k), BufferArg::from_raw_parts(shared.clone(), n), BufferArg::from_raw_parts(sgate.clone(), sg_len), BufferArg::from_raw_parts(out.clone(), n), n, k, sg_off);
        }
        out
    }

    /// Argmax index of `n` f32 on the device, as a 1-element f32 buffer holding the u32 bits.
    pub fn qw_argmax(&self, x: &Handle, n: usize) -> Handle {
        let out = self.scratch(4);
        unsafe {
            qw_argmax_kernel::launch(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(x.clone(), n), BufferArg::from_raw_parts(out.clone(), 1), n);
        }
        out
    }

    /// Argmax of each of the `m` rows of `[m, n]`: `m` u32 bit patterns in an f32 buffer.
    pub fn qw_argmax_rows(&self, x: &Handle, m: usize, n: usize) -> Handle {
        let out = self.scratch(m * 4);
        unsafe {
            qw_argmax_rows_kernel::launch(&self.client, CubeCount::Static(m as u32, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(x.clone(), m * n), BufferArg::from_raw_parts(out.clone(), m), n);
        }
        out
    }

    pub fn qw_permute_x(&self, x: &Handle, n: usize) -> Handle {
        GpuQ5KNative::permute_x(self, x, n)
    }
}
