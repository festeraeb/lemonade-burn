//! The glue kernels of a device-resident decode token (Phase D1 of `docs/DEVICE_DECODE_PLAN.md`).
//!
//! A decode token used to leave the card about 145 times: every quantized matvec read its result
//! back so Burn could do the norms, RoPE, attention, the router and the expert sum on the host.
//! These kernels are those host steps, so that a token can run from the embedding upload to the
//! logits with nothing read back in between:
//!
//! * [`add_rms_norm_kernel`]: `x <- x + y`, `h = rms(x) * w` (the residual add fused into the next
//!   pre-norm; the first norm of a token has no `y`).
//! * [`qk_norm_rope_kernel`]: per-head RMS norm of `q` and `k` over `head_dim`, NEOX RoPE at the
//!   token's position, and `k`, `v` written into the device KV cache.
//! * [`attn_decode_kernel`]: one query row against `0..total` cached positions, online softmax,
//!   grouped-query heads.
//! * [`router_topk_kernel`]: the LFM2-MoE sigmoid router with selection bias, top-k and the
//!   normalisation floor, bit for bit the rule of `lb_lfm2::moe::route_sigmoid`.
//! * [`weighted_sum_kernel`]: `y = sum_k w_k * out_k`, accumulated in ascending expert id, which
//!   is the order the host path adds expert outputs in.
//!
//! Every kernel here is checked against a host reference in `tests/decode_kernels_match_host.rs`.
//! Summation order differs from the Burn ops they replace (a reduce tree against a sequential
//! sum), so the results agree to rounding, not bit for bit; the router's selection rule itself is
//! exact, including ties.

use crate::store::Gpu;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Units per cube for the single-cube kernels (`add_rms_norm`, `router_topk`). A power of two.
pub const NORM_CUBE: usize = 256;
/// Most experts the router kernel handles (its shared arrays are this long).
pub const MAX_EXPERTS: usize = 256;
/// Units per cube (one cube per query head) for [`attn_decode_kernel`]; each unit owns the
/// positions `u, u + ATTN_LANES, ...` of the cache. Must be a power of two.
pub const ATTN_LANES: usize = 64;
/// A running max that is below any real score but finite, so `exp(m - M)` is never `inf - inf`.
// The running max starts at -3e38: below any real score but finite, so `exp(m - M)` is never
// `inf - inf` (written as a literal in the kernel: a Rust const does not expand).

/// Sum `v` over the whole cube through shared memory (`sm` at least `CUBE_DIM_X` long; the cube
/// size a power of two). Valid in every unit; ends on a barrier so `sm` can be reused at once.
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

/// `x <- x + y` (when `add`), then `h = x * (1 / sqrt(mean(x^2) + eps)) * w`, over one row of
/// `x.len()` values. One cube. The order of the multiplications is the host's
/// (`x * rsqrt * w`, see `Lfm2::rms_norm`).
#[cube(launch)]
pub fn add_rms_norm_kernel(x: &mut [f32], y: &[f32], w: &[f32], h: &mut [f32], eps: &[f32], #[comptime] add: bool) {
    let n = x.len();
    let mut sm = Shared::<[f32]>::new_slice(NORM_CUBE);
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < n {
        let mut v = x[i];
        if add {
            v += y[i];
            x[i] = v;
        }
        ss += v * v;
        i += CUBE_DIM_X as usize;
    }
    let total = cube_sum(&mut sm, ss);
    let r = 1.0f32 / Sqrt::sqrt(total / n as f32 + eps[0]);
    let mut i = UNIT_POS as usize;
    while i < n {
        h[i] = x[i] * r * w[i];
        i += CUBE_DIM_X as usize;
    }
}

/// Per-head RMS norm + NEOX RoPE for one token, and the KV-cache append.
///
/// Cube `c < nh` is query head `c`: norm with `qn`, rotate, write `qo[c * hd ..]`. Cube
/// `nh + j` is KV head `j`: norm `k` with `kn`, rotate, write row `pos` of head `j` in `kc`
/// (`[kvh, cap, hd]`), and copy `v` into the same row of `vc`. One unit per value (`hd` units).
///
/// `rope` is the cos/sin table, row `p` = `[cos(hd) | sin(hd)]` for position `p`, built on the
/// host in f64 exactly as `Lfm2::rope_tables` builds it (so the rotation is the host's to the
/// last bit, see the module doc of `lb_lfm2::device_decode`). `p` = `[nh, kvh]` (fixed per
/// model), `tp` = `[pos, total, cap]` (the token's, written once a token: see [`TokenParams`]).
#[cube(launch)]
pub fn qk_norm_rope_kernel(
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
    tp: &[u32],
    eps: &[f32],
    #[comptime] hd: usize,
) {
    let nh = p[0] as usize;
    let pos = tp[0] as usize;
    let cap = tp[2] as usize;
    let head = CUBE_POS;
    let j = UNIT_POS as usize;
    let is_q = head < nh;
    let mut val = 0.0f32;
    let mut nw = 0.0f32;
    if j < hd {
        if is_q {
            val = q[head * hd + j];
            nw = qn[j];
        } else {
            val = k[(head - nh) * hd + j];
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
            qo[head * hd + j] = out;
        } else {
            let kh = head - nh;
            kc[(kh * cap + pos) * hd + j] = out;
            vc[(kh * cap + pos) * hd + j] = v[kh * hd + j];
        }
    }
}

/// One decode query against the cache: `out[h] = softmax(q_h . K^T * scale) V` over positions
/// `0..total`, KV head `h / group`. One cube per query head, `ATTN_LANES` units; unit `u` keeps
/// an online softmax (running max, sum, `acc[hd]`) over positions `u, u + lanes, ...`, then the
/// units combine through shared memory. `p` = `[group]`, `tp` = `[pos, total, cap]`,
/// `scale` = `[hd^-0.5]`.
#[cube(launch)]
pub fn attn_decode_kernel(q: &[f32], kc: &[f32], vc: &[f32], out: &mut [f32], p: &[u32], tp: &[u32], scale: &[f32], #[comptime] hd: usize, #[comptime] lanes: usize) {
    let group = p[0] as usize;
    let total = tp[1] as usize;
    let cap = tp[2] as usize;
    let head = CUBE_POS;
    let kh = head / group;
    let u = UNIT_POS as usize;
    let sc = scale[0];

    let mut qs = Shared::<[f32]>::new_slice(hd);
    let mut j = u;
    while j < hd {
        qs[j] = q[head * hd + j];
        j += lanes;
    }
    sync_cube();

    let mut acc = Array::<f32>::new(hd);
    let mut d = 0usize;
    while d < hd {
        acc[d] = 0.0f32;
        d += 1usize;
    }
    let mut m = f32::new(-3.0e38f32);
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
        if score > m {
            let c = Exp::exp(m - score);
            s *= c;
            let mut d = 0usize;
            while d < hd {
                acc[d] *= c;
                d += 1usize;
            }
            m = score;
        }
        let e = Exp::exp(score - m);
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
    ms[u] = m;
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
        out[head * hd + j] = num / denom;
        j += lanes;
    }
}

/// The LFM2-MoE router for one token, written into slot `base` of a per-token routes buffer.
///
/// `logit_e = h . router_e`, `p_e = 1 / (1 + exp(-logit_e))`; experts are ranked by `p_e + bias_e`
/// descending with the lower id first on a tie (a stable sort, as `route_sigmoid` does); the
/// top `k` are selected; their weights are `p_e / max(sum, 2^-14)` with the sum taken in rank
/// order. Writes `routes[2 (base + i)] = [expert, 0]` and `weights[base + i]` for rank `i < k`.
///
/// `p` = `[n_expert, k, base]`. One cube of `NORM_CUBE` units. With `seq` each logit is one unit's
/// sequential dot (the host's summation order, for parity checks); otherwise one plane per expert
/// at a time, lanes over the columns, `plane_sum` (needs plane ops).
#[cube(launch)]
pub fn router_topk_kernel(h: &[f32], router: &[f32], bias: &[f32], routes: &mut [u32], weights: &mut [f32], p: &[u32], #[comptime] seq: bool) {
    let n_exp = p[0] as usize;
    let k = p[1] as usize;
    let base = p[2] as usize;
    let cols = h.len();
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
                acc += router[e * cols + c] * h[c];
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
                acc += router[e * cols + c] * h[c];
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
            routes[2usize * (base + i) + 1usize] = 0u32;
            weights[base + i] = probs[ex as usize] / sum;
            i += 1usize;
        }
    }
}

/// `y[i] = sum_s weights[base + s] * outs[s * n + i]` over the `k` slots of one token, the slots
/// taken in ascending expert id (`routes[2 (base + s)]`): the host path sums expert outputs in that
/// order, so this keeps the same rounding. `p` = `[k, n, base]`.
#[cube(launch)]
pub fn weighted_sum_kernel(outs: &[f32], weights: &[f32], routes: &[u32], y: &mut [f32], p: &[u32]) {
    let k = p[0] as usize;
    let n = p[1] as usize;
    let base = p[2] as usize;
    let mut i = ABSOLUTE_POS as usize;
    let stride = CUBE_COUNT as usize * CUBE_DIM_X as usize;
    while i < n {
        let mut acc = 0.0f32;
        // walk the slots in ascending expert id: k is 4, so a selection pass per step is free
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
            acc += weights[base + best] * outs[best * n + i];
            last = best_e;
            first = false;
            step += 1usize;
        }
        y[i] = acc;
        i += stride;
    }
}

/// Launchers for the kernels above: small fixed shapes, so these only pick the geometry and
/// count elements. Every buffer argument counts **elements** of the kernel's slice type.
impl Gpu {
    /// A device buffer of f32 constants, shared by every caller asking for the same values.
    pub fn const_f32(&self, v: &[f32]) -> Handle {
        self.const_u32(&v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
    }

    /// `x <- x + y` (if `y` is given), `h = rms(x) * w`; returns `h` (scratch, `n` values).
    pub fn add_rms_norm(&self, x: &Handle, y: Option<&Handle>, w: &Handle, n: usize, eps: f32) -> Handle {
        let h = self.scratch(n * 4);
        let eps_h = self.const_f32(&[eps]);
        // without `y` the kernel never reads it; bind `x` there so the argument is a real buffer
        let yh = y.cloned().unwrap_or_else(|| x.clone());
        unsafe {
            add_rms_norm_kernel::launch(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(NORM_CUBE as u32),
                BufferArg::from_raw_parts(x.clone(), n),
                BufferArg::from_raw_parts(yh, n),
                BufferArg::from_raw_parts(w.clone(), n),
                BufferArg::from_raw_parts(h.clone(), n),
                BufferArg::from_raw_parts(eps_h, 1),
                y.is_some(),
            );
        }
        h
    }

    /// Per-head norm + RoPE of `q` (`nh` heads) and `k` (`kvh` heads) at `pos`; `k`, `v` go into
    /// row `pos` of the caches (`[kvh, cap, hd]`). Returns the rotated `q` (scratch, `nh * hd`).
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
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
        tp: &TokenParams,
        (nh, kvh, hd): (usize, usize, usize),
        eps: f32,
    ) -> Handle {
        assert!(hd.is_power_of_two() && (2..=1024).contains(&hd), "head_dim {hd}: the per-head reduce needs a power of two");
        let (pos, cap) = (tp.pos, tp.cap);
        assert!(pos < cap && pos < rope_rows, "position {pos} past the cache ({cap}) or the rope table ({rope_rows})");
        let qo = self.scratch(nh * hd * 4);
        let p = self.const_u32(&[nh as u32, kvh as u32]);
        unsafe {
            qk_norm_rope_kernel::launch(
                &self.client,
                CubeCount::Static((nh + kvh) as u32, 1, 1),
                CubeDim::new_1d(hd as u32),
                BufferArg::from_raw_parts(q.clone(), nh * hd),
                BufferArg::from_raw_parts(k.clone(), kvh * hd),
                BufferArg::from_raw_parts(v.clone(), kvh * hd),
                BufferArg::from_raw_parts(q_norm.clone(), hd),
                BufferArg::from_raw_parts(k_norm.clone(), hd),
                BufferArg::from_raw_parts(rope.clone(), rope_rows * 2 * hd),
                BufferArg::from_raw_parts(kc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(vc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(qo.clone(), nh * hd),
                BufferArg::from_raw_parts(p, 2),
                BufferArg::from_raw_parts(tp.handle.clone(), 3),
                BufferArg::from_raw_parts(self.const_f32(&[eps]), 1),
                hd,
            );
        }
        qo
    }

    /// Decode attention over positions `0..tp.total` of the caches; returns `[nh * hd]` (scratch).
    pub fn attn_decode(&self, q: &Handle, kc: &Handle, vc: &Handle, tp: &TokenParams, (nh, kvh, hd): (usize, usize, usize)) -> Handle {
        assert!(kvh > 0 && nh % kvh == 0, "{nh} query heads over {kvh} KV heads");
        let (total, cap) = (tp.total, tp.cap);
        assert!(total >= 1 && total <= cap, "attention over {total} positions of a {cap} cache");
        let out = self.scratch(nh * hd * 4);
        let p = self.const_u32(&[(nh / kvh) as u32]);
        let scale = self.const_f32(&[(hd as f32).powf(-0.5)]);
        unsafe {
            attn_decode_kernel::launch(
                &self.client,
                CubeCount::Static(nh as u32, 1, 1),
                CubeDim::new_1d(ATTN_LANES as u32),
                BufferArg::from_raw_parts(q.clone(), nh * hd),
                BufferArg::from_raw_parts(kc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(vc.clone(), kvh * cap * hd),
                BufferArg::from_raw_parts(out.clone(), nh * hd),
                BufferArg::from_raw_parts(p, 1),
                BufferArg::from_raw_parts(tp.handle.clone(), 3),
                BufferArg::from_raw_parts(scale, 1),
                hd,
                ATTN_LANES,
            );
        }
        out
    }

    /// Route one token (`h`, `n_embd` values) into slot `base` of `routes` (`[.., 2k]` u32) and
    /// `weights` (`[.., k]` f32). `router` is `[n_expert, n_embd]` f32, `bias` `[n_expert]`.
    /// `seq` forces the sequential-dot variant (also taken when the device has no plane ops).
    #[allow(clippy::too_many_arguments)]
    pub fn router_topk(&self, h: &Handle, router: &Handle, bias: &Handle, routes: &Handle, weights: &Handle, (n_expert, k, n_embd, base, n_slots): (usize, usize, usize, usize, usize), seq: bool) {
        assert!(n_expert <= MAX_EXPERTS && k >= 1 && k <= n_expert, "router: {n_expert} experts, top-{k}");
        assert!(base + k <= n_slots, "router slot {base}+{k} past {n_slots}");
        let p = self.const_u32(&[n_expert as u32, k as u32, base as u32]);
        unsafe {
            router_topk_kernel::launch(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(NORM_CUBE as u32),
                BufferArg::from_raw_parts(h.clone(), n_embd),
                BufferArg::from_raw_parts(router.clone(), n_expert * n_embd),
                BufferArg::from_raw_parts(bias.clone(), n_expert),
                BufferArg::from_raw_parts(routes.clone(), 2 * n_slots),
                BufferArg::from_raw_parts(weights.clone(), n_slots),
                BufferArg::from_raw_parts(p, 3),
                seq || !self.use_plane,
            );
        }
    }

    /// `y = sum_s w_s * outs[s]` over the `k` slots at `base` (ascending expert id); `outs` is
    /// `[k, n]`. Returns `y` (scratch, `n` values).
    pub fn weighted_sum(&self, outs: &Handle, weights: &Handle, routes: &Handle, (k, n, base, n_slots): (usize, usize, usize, usize)) -> Handle {
        assert!(base + k <= n_slots);
        let y = self.scratch(n * 4);
        let p = self.const_u32(&[k as u32, n as u32, base as u32]);
        unsafe {
            weighted_sum_kernel::launch(
                &self.client,
                crate::cube_count(n, NORM_CUBE),
                CubeDim::new_1d(NORM_CUBE as u32),
                BufferArg::from_raw_parts(outs.clone(), k * n),
                BufferArg::from_raw_parts(weights.clone(), n_slots),
                BufferArg::from_raw_parts(routes.clone(), 2 * n_slots),
                BufferArg::from_raw_parts(y.clone(), n),
                BufferArg::from_raw_parts(p, 3),
            );
        }
        y
    }
}

/// A token's position values on the device, `[pos, total, cap]`: one small buffer written once a
/// token instead of a constant buffer per distinct position (those are cached forever, and a
/// long context would leave thousands of them behind).
pub struct TokenParams {
    pub handle: Handle,
    pub pos: usize,
    pub total: usize,
    pub cap: usize,
}

impl TokenParams {
    /// A fresh buffer for these values.
    pub fn new(gpu: &Gpu, pos: usize, total: usize, cap: usize) -> Self {
        let handle = gpu.upload_u32(vec![pos as u32, total as u32, cap as u32]);
        Self { handle, pos, total, cap }
    }

    /// Overwrite in place for the next token. Stream-ordered: kernels already queued read the
    /// old values only if they have not run yet, so call this after the previous token's sync.
    pub fn set(&mut self, gpu: &Gpu, pos: usize, total: usize, cap: usize) {
        gpu.client.write(&self.handle, cubecl::bytes::Bytes::from_elems(vec![pos as u32, total as u32, cap as u32]));
        (self.pos, self.total, self.cap) = (pos, total, cap);
    }
}

/// The NEOX RoPE table the decode kernels read: row `p` (for `p < rows`) is `cos` then `sin`
/// over `hd` values, computed in f64 and rounded once, exactly as `Lfm2::rope_tables` does on
/// the host, so the device rotation matches the host's to the bit.
pub fn rope_table(theta: f64, hd: usize, rows: usize) -> Vec<f32> {
    let mut t = vec![0f32; rows * 2 * hd];
    for p in 0..rows {
        let row = &mut t[p * 2 * hd..(p + 1) * 2 * hd];
        for j in 0..hd / 2 {
            let inv = 1.0 / theta.powf(2.0 * j as f64 / hd as f64);
            let (s, c) = (p as f64 * inv).sin_cos();
            for jj in [j, j + hd / 2] {
                row[jj] = c as f32;
                row[hd + jj] = s as f32;
            }
        }
    }
    t
}
