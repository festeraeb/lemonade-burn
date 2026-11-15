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
fn silu_f(x: f32) -> f32 {
    x / (1.0f32 + Exp::exp(-x))
}

#[cube]
fn sigmoid_f(x: f32) -> f32 {
    1.0f32 / (1.0f32 + Exp::exp(-x))
}

/// `out[row] = x[row] * rsqrt(mean(x[row]^2) + eps) * w`, one cube (256 units) per row of `n`.
#[cube(launch)]
fn qw_rms_kernel(x: &[f32], w: &[f32], out: &mut [f32], eps: &[f32], #[comptime] n: usize) {
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
        out[row * n + i] = x[row * n + i] * r * w[i];
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

#[cube(launch)]
fn qw_add_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        out[i] = a[i] + b[i];
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
fn qw_dn_scalars_kernel(braw: &[f32], araw: &[f32], dt: &[f32], a: &[f32], beta: &mut [f32], g: &mut [f32]) {
    let i = ABSOLUTE_POS as usize;
    if i < beta.len() {
        beta[i] = sigmoid_f(braw[i]);
        let x = araw[i] + dt[i];
        let mut sp = x;
        if x <= 20.0f32 {
            sp = Log1p::log1p(Exp::exp(x));
        }
        g[i] = sp * a[i];
    }
}

/// The delta rule, one cube per v head, one unit per state row: `S *= exp(g); d = (v - S k) beta; S += d k^T; o = S q / sqrt(hk)`.
/// `conv` is the post-conv, L2-normalised row `[q (nk*hk) | k (nk*hk) | v (nv*hv)]`; v head `h` uses q/k head `h % nk`.
#[cube(launch)]
fn qw_dn_recur_kernel(s: &mut [f32], conv: &[f32], beta: &[f32], g: &[f32], out: &mut [f32], #[comptime] nk: usize, #[comptime] hv: usize, #[comptime] hk: usize) {
    let hh = CUBE_POS as usize;
    let i = UNIT_POS as usize;
    let kh = hh % nk;
    let mut ks = Shared::<[f32]>::new_slice(256usize);
    let mut qs = Shared::<[f32]>::new_slice(256usize);
    let mut j = UNIT_POS as usize;
    while j < hk {
        ks[j] = conv[nk * hk + kh * hk + j];
        qs[j] = conv[kh * hk + j];
        j += CUBE_DIM_X as usize;
    }
    sync_cube();
    if i < hv {
        let decay = Exp::exp(g[hh]);
        let base = hh * hv * hk + i * hk;
        let mut sk = 0.0f32;
        let mut j = 0usize;
        while j < hk {
            let v = s[base + j] * decay;
            s[base + j] = v;
            sk += v * ks[j];
            j += 1usize;
        }
        let d = (conv[2usize * nk * hk + hh * hv + i] - sk) * beta[hh];
        let mut o = 0.0f32;
        let mut j = 0usize;
        while j < hk {
            let v = s[base + j] + ks[j] * d;
            s[base + j] = v;
            o += v * qs[j];
            j += 1usize;
        }
        out[hh * hv + i] = o / Sqrt::sqrt(hk as f32);
    }
}

/// `out = rms(o) * w * silu(z)` per head of `hv`; one cube per head.
#[cube(launch)]
fn qw_dn_gnorm_kernel(o: &[f32], z: &[f32], w: &[f32], out: &mut [f32], eps: &[f32], #[comptime] hv: usize) {
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
        out[hh * hv + i] = o[hh * hv + i] * r * w[i] * silu_f(z[hh * hv + i]);
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

/// Softmax over `ne` router logits, top `k` by probability (ties: lowest index), weights renormalised; writes `[expert, 0]` pairs.
/// One unit.
#[cube(launch)]
fn qw_route_kernel(logits: &[f32], routes: &mut [u32], weights: &mut [f32], #[comptime] ne: usize, #[comptime] k: usize) {
    let mut p = Shared::<[f32]>::new_slice(512usize);
    if UNIT_POS == 0u32 {
        let mut m = logits[0usize];
        let mut i = 1usize;
        while i < ne {
            if logits[i] > m {
                m = logits[i];
            }
            i += 1usize;
        }
        let mut s = 0.0f32;
        let mut i = 0usize;
        while i < ne {
            let e = Exp::exp(logits[i] - m);
            p[i] = e;
            s += e;
            i += 1usize;
        }
        let mut i = 0usize;
        while i < ne {
            p[i] = p[i] / s;
            i += 1usize;
        }
        let mut sel_sum = 0.0f32;
        let mut r = 0usize;
        while r < k {
            let mut best = 0usize;
            let mut bv = logits[0usize] * 0.0f32 - 1.0f32;
            let mut i = 0usize;
            while i < ne {
                if p[i] > bv {
                    bv = p[i];
                    best = i;
                }
                i += 1usize;
            }
            routes[2usize * r] = best as u32;
            routes[2usize * r + 1usize] = 0u32;
            weights[r] = bv;
            sel_sum += bv;
            p[best] = logits[0usize] * 0.0f32 - 1.0f32;
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
fn qw_moe_combine_kernel(x: &[f32], down: &[f32], weights: &[f32], shared: &[f32], sgate: &[f32], out: &mut [f32], #[comptime] n: usize, #[comptime] k: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let mut acc = 0.0f32;
        let mut r = 0usize;
        while r < k {
            acc += weights[r] * down[r * n + i];
            r += 1usize;
        }
        acc += sigmoid_f(sgate[0usize]) * shared[i];
        out[i] = x[i] + acc;
    }
}

/// Launchers. Buffer arguments count f32 elements (u32 for `routes`/`pos`).
impl Gpu {
    fn qw_cubes(&self, n: usize) -> CubeCount {
        cube_count(n, self.cube_dim)
    }

    pub fn qw_rms(&self, x: &Handle, w: &Handle, rows: usize, n: usize, eps: f32) -> Handle {
        let out = self.scratch(rows * n * 4);
        let e = self.const_f32(&[eps]);
        unsafe {
            qw_rms_kernel::launch(&self.client, CubeCount::Static(rows as u32, 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(x.clone(), rows * n), BufferArg::from_raw_parts(w.clone(), n), BufferArg::from_raw_parts(out.clone(), rows * n), BufferArg::from_raw_parts(e, 1), n);
        }
        out
    }

    /// `w` is `rows x cols` f32 on the device.
    pub fn qw_f32_matvec(&self, w: &Handle, x: &Handle, rows: usize, cols: usize) -> Handle {
        let out = self.scratch(rows * 4);
        unsafe {
            qw_f32_matvec_kernel::launch(&self.client, cube_count(rows, self.per_cube()), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(w.clone(), rows * cols), BufferArg::from_raw_parts(x.clone(), cols), BufferArg::from_raw_parts(out.clone(), rows), cols, self.use_plane);
        }
        out
    }

    pub fn qw_add(&self, a: &Handle, b: &Handle, n: usize) -> Handle {
        let out = self.scratch(n * 4);
        unsafe {
            qw_add_kernel::launch(&self.client, self.qw_cubes(n), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(a.clone(), n), BufferArg::from_raw_parts(b.clone(), n), BufferArg::from_raw_parts(out.clone(), n));
        }
        out
    }

    pub fn qw_silu_mul(&self, g: &Handle, u: &Handle, n: usize) -> Handle {
        let out = self.scratch(n * 4);
        unsafe {
            qw_silu_mul_kernel::launch(&self.client, self.qw_cubes(n), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(g.clone(), n), BufferArg::from_raw_parts(u.clone(), n), BufferArg::from_raw_parts(out.clone(), n));
        }
        out
    }

    /// Delta-net core for one token. `qkv` `[ch]`, `z` `[d_inner]`, `braw`/`araw` `[nv]`; `conv_state` `[(kd-1) * ch]` and `s`
    /// `[nv * hv * hk]` are persistent (create with `upload_f32`), updated in place. Returns the gated-normed `[d_inner]`.
    #[allow(clippy::too_many_arguments)]
    pub fn qw_delta_net(&self, qkv: &Handle, z: &Handle, braw: &Handle, araw: &Handle, conv_w: &Handle, dt: &Handle, a: &Handle, norm_w: &Handle, conv_state: &Handle, s: &Handle, (ch, kd, nk, nv, hk, hv): (usize, usize, usize, usize, usize, usize), eps: f32) -> Handle {
        let conv = self.scratch(ch * 4);
        let beta = self.scratch(nv * 4);
        let g = self.scratch(nv * 4);
        let o = self.scratch(nv * hv * 4);
        let e = self.const_f32(&[eps]);
        unsafe {
            qw_dn_conv_kernel::launch(&self.client, self.qw_cubes(ch), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(qkv.clone(), ch), BufferArg::from_raw_parts(conv_w.clone(), ch * kd), BufferArg::from_raw_parts(conv_state.clone(), (kd - 1) * ch), BufferArg::from_raw_parts(conv.clone(), ch), ch, kd);
            qw_l2_kernel::launch(&self.client, CubeCount::Static((2 * nk) as u32, 1, 1), CubeDim::new_1d(128), BufferArg::from_raw_parts(conv.clone(), ch), BufferArg::from_raw_parts(e.clone(), 1), hk);
            qw_dn_scalars_kernel::launch(&self.client, self.qw_cubes(nv), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(braw.clone(), nv), BufferArg::from_raw_parts(araw.clone(), nv), BufferArg::from_raw_parts(dt.clone(), nv), BufferArg::from_raw_parts(a.clone(), nv), BufferArg::from_raw_parts(beta.clone(), nv), BufferArg::from_raw_parts(g.clone(), nv));
            qw_dn_recur_kernel::launch(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(128), BufferArg::from_raw_parts(s.clone(), nv * hv * hk), BufferArg::from_raw_parts(conv.clone(), ch), BufferArg::from_raw_parts(beta, nv), BufferArg::from_raw_parts(g, nv), BufferArg::from_raw_parts(o.clone(), nv * hv), nk, hv, hk);
        }
        let out = self.scratch(nv * hv * 4);
        unsafe {
            qw_dn_gnorm_kernel::launch(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(128), BufferArg::from_raw_parts(o, nv * hv), BufferArg::from_raw_parts(z.clone(), nv * hv), BufferArg::from_raw_parts(norm_w.clone(), hv), BufferArg::from_raw_parts(out.clone(), nv * hv), BufferArg::from_raw_parts(e, 1), hv);
        }
        out
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
            qw_route_kernel::launch(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1), BufferArg::from_raw_parts(logits.clone(), ne), BufferArg::from_raw_parts(routes.clone(), 2 * k), BufferArg::from_raw_parts(weights.clone(), k), ne, k);
        }
        (routes, weights)
    }

    pub fn qw_moe_combine(&self, x: &Handle, down: &Handle, weights: &Handle, shared: &Handle, sgate: &Handle, n: usize, k: usize) -> Handle {
        let out = self.scratch(n * 4);
        unsafe {
            qw_moe_combine_kernel::launch(&self.client, self.qw_cubes(n), CubeDim::new_1d(self.cube_dim as u32), BufferArg::from_raw_parts(x.clone(), n), BufferArg::from_raw_parts(down.clone(), k * n), BufferArg::from_raw_parts(weights.clone(), k), BufferArg::from_raw_parts(shared.clone(), n), BufferArg::from_raw_parts(sgate.clone(), 1), BufferArg::from_raw_parts(out.clone(), n), n, k);
        }
        out
    }

    pub fn qw_permute_x(&self, x: &Handle, n: usize) -> Handle {
        GpuQ5KNative::permute_x(self, x, n)
    }
}
