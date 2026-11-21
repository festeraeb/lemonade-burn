//! The scalar pieces of the deepseek41 forward, each written to give ggml's CPU result (same formula, same f32/f64
//! choice, same order where it is cheap to keep). Citations are into llama.cpp `dadb89f180`
//! (`/mnt/fast/scrape-v2/llama-v8`): `OPS` = ggml/src/ggml-cpu/ops.cpp, `D41` = src/models/deepseek41.cpp,
//! `D4` = src/models/deepseek4.cpp, `GR` = src/llama-graph.cpp.


/// `1 / sqrt(mean(x^2) + eps)` as ggml's `rms_norm` (OPS:4115-4122): squares in f32, summed in f64, mean and scale in f32.
pub fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0f64;
    for &v in x {
        sum += (v * v) as f64;
    }
    let mean = (sum / x.len() as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

/// RMS norm of each row of `n = out.len() / rows` values; `w` (if any) multiplies after the scale, as
/// `build_norm` does with a separate `ggml_mul`.
pub fn rms_norm(x: &[f32], w: Option<&[f32]>, n: usize, eps: f32, out: &mut [f32]) {
    for (xr, or) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let s = rms_scale(xr, eps);
        match w {
            Some(w) => {
                for i in 0..n {
                    or[i] = (xr[i] * s) * w[i];
                }
            }
            None => {
                for i in 0..n {
                    or[i] = xr[i] * s;
                }
            }
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ggml_silu_f32 (vec.h:1096).
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `op_softplus` (unary-ops.cpp:80-82).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Gating function 4 (SQRT_SOFTPLUS, GR:2316-2318).
pub fn sqrt_softplus(x: f32) -> f32 {
    softplus(x).sqrt()
}

/// `ggml_swiglu_clamp` (OPS:3585-3589): the gate is clamped from above only, before the SiLU; up to [-limit, limit].
pub fn swiglu_clamp(gate: f32, up: f32, limit: f32) -> f32 {
    let g = gate.min(limit);
    let u = up.clamp(-limit, limit);
    g / (1.0 + (-g).exp()) * u
}

pub fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

// Activation rounding to the weight's ggml vec_dot_type lives in lb-gguf (shared with lb-mimo2).
pub use lb_gguf::act::{act_round, bf16r, q8_0_roundtrip, q8_k_roundtrip};

// ---------------------------------------------------------------------------------------------------------------
// RoPE (NORM type, adjacent pairs) with ggml's yarn, OPS:6076-6112, 6186-6203, 6232-6340; corr dims ggml.c:4373-4388.

/// `ggml_rope_yarn_corr_dims` with truncate = true (the default, floor/ceil).
pub fn yarn_corr_dims(n_dims: usize, n_ctx_orig: u32, base: f32, beta_fast: f32, beta_slow: f32) -> [f32; 2] {
    let corr_dim = |n_rot: f32| n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
    let start = corr_dim(beta_fast).floor();
    let end = corr_dim(beta_slow).ceil();
    // MAX/MIN macros of ggml: (a > b ? a : b)
    let lo = if 0.0 > start { 0.0 } else { start };
    let hi = if (n_dims as f32 - 1.0) < end { n_dims as f32 - 1.0 } else { end };
    [lo, hi]
}

/// `rope_yarn_ramp` (OPS:6076-6079); `i0` is the even channel index inside the rotated block (pair = i0/2).
pub fn yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = ((i0 / 2) as f32 - low) / 0.001f32.max(high - low);
    1.0 - 1f32.min(0f32.max(y))
}

/// Parameters of one `ggml_rope_ext` call (per layer kind), rotating `n_dims` channels starting at `offset`.
#[derive(Clone, Debug)]
pub struct Rope {
    pub n_dims: usize,
    pub offset: usize,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub corr: [f32; 2],
    pub theta_scale: f32,
}

impl Rope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(n_dims: usize, offset: usize, n_ctx_orig: u32, base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) -> Self {
        Rope {
            n_dims,
            offset,
            freq_scale,
            ext_factor,
            attn_factor,
            corr: yarn_corr_dims(n_dims, n_ctx_orig, base, beta_fast, beta_slow),
            theta_scale: base.powf(-2.0 / n_dims as f32),
        }
    }

    /// (cos, sin) per pair at position `pos`; `sin_sign` -1 for `rope_back` (OPS:6296). `ggml_rope_cache_init`.
    pub fn cache(&self, pos: i64, sin_sign: f32) -> Vec<(f32, f32)> {
        let mut theta = pos as f32;
        let mut out = Vec::with_capacity(self.n_dims / 2);
        for i0 in (0..self.n_dims).step_by(2) {
            let theta_extrap = theta;
            let theta_interp = self.freq_scale * theta_extrap;
            let mut th = theta_interp;
            let mut mscale = self.attn_factor;
            if self.ext_factor != 0.0 {
                let ramp_mix = yarn_ramp(self.corr[0], self.corr[1], i0) * self.ext_factor;
                th = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / self.freq_scale).ln();
            }
            out.push((th.cos() * mscale, th.sin() * mscale * sin_sign));
            theta *= self.theta_scale;
        }
        out
    }

    /// Rotate adjacent pairs of `x[offset .. offset + n_dims]` with a cache from [`Rope::cache`] (`rotate_pairs`, NORM).
    pub fn apply(&self, x: &mut [f32], cache: &[(f32, f32)]) {
        let r = &mut x[self.offset..self.offset + self.n_dims];
        for (i, &(c, s)) in cache.iter().enumerate() {
            let (x0, x1) = (r[2 * i], r[2 * i + 1]);
            r[2 * i] = x0 * c - x1 * s;
            r[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connections: the fused CPU ops (OPS:11216-11508), same formulas as D4:261-448.

/// `ggml_dsv4_hc_comb` for one token (OPS:11290-11330): `m` = the 24 mixes, comb logits `m[8+idx]*scale + base[8+idx]`
/// with idx = dst + 4*src; softmax over dst per src row, `+ eps`; one column norm, then `iters-1` x (row, column),
/// each divided by (eps + sum). Result index = dst + 4*src.
pub fn hc_comb(m: &[f32], scale_comb: f32, base: &[f32], eps: f32, iters: usize) -> [f32; 16] {
    let mut c = [0f32; 16];
    for i in 0..16 {
        c[i] = m[8 + i] * scale_comb + base[8 + i];
    }
    sinkhorn(&mut c, eps, iters);
    c
}

/// Softmax rows + Sinkhorn on comb logits laid out `dst + 4*src` (row = src, column = dst).
pub fn sinkhorn(c: &mut [f32; 16], eps: f32, iters: usize) {
    const HC: usize = 4;
    for src in 0..HC {
        let mut max = f32::NEG_INFINITY;
        for dst in 0..HC {
            max = max.max(c[dst + HC * src]);
        }
        let mut sum = 0f32;
        for dst in 0..HC {
            let v = (c[dst + HC * src] - max).exp();
            c[dst + HC * src] = v;
            sum += v;
        }
        let inv = 1.0 / sum;
        for dst in 0..HC {
            c[dst + HC * src] = c[dst + HC * src] * inv + eps;
        }
    }
    let norm_cols = |c: &mut [f32; 16]| {
        for dst in 0..HC {
            let mut sum = eps;
            for src in 0..HC {
                sum += c[dst + HC * src];
            }
            let inv = 1.0 / sum;
            for src in 0..HC {
                c[dst + HC * src] *= inv;
            }
        }
    };
    let norm_rows = |c: &mut [f32; 16]| {
        for src in 0..HC {
            let mut sum = eps;
            for dst in 0..HC {
                sum += c[dst + HC * src];
            }
            let inv = 1.0 / sum;
            for dst in 0..HC {
                c[dst + HC * src] *= inv;
            }
        }
    };
    norm_cols(c);
    for _ in 1..iters {
        norm_rows(c);
        norm_cols(c);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Engram hash (D41:240-278).

pub const ENGRAM_DEAD: i64 = -1;

/// The four (normalised) token ids of one position's n-gram, shift 0 = the token itself. `prev(s)` = the token at
/// distance `s` (only asked for `s <= pos`). Out-of-range ids and map entries equal to DEAD block this and every
/// longer shift, which then take `pad = token_map[pad_id]` (D41:257-267).
pub fn engram_toks(token: i64, pos: i64, n_gram: usize, token_map: &[i64], pad: i64, prev: impl Fn(usize) -> i64) -> Vec<u64> {
    let cid = |t: i64| -> i64 {
        if t < 0 || t >= token_map.len() as i64 {
            ENGRAM_DEAD
        } else {
            token_map[t as usize]
        }
    };
    let mut toks = vec![0u64; n_gram];
    let mut c = cid(token);
    let mut blocked = c == ENGRAM_DEAD;
    toks[0] = if blocked { pad as u64 } else { c as u64 };
    for s in 1..n_gram {
        c = if pos - (s as i64) < 0 { ENGRAM_DEAD } else { cid(prev(s)) };
        blocked = blocked || c == ENGRAM_DEAD;
        toks[s] = if blocked { pad as u64 } else { c as u64 };
    }
    toks
}

/// Table rows of one position for one Engram module: `(n_gram-1) * n_head` ids, column `(o-1)*n_head + h`
/// (D41:269-276). `mult` = the module's `n_gram` multipliers, `primes`/`offs` its `(n_gram-1)*n_head` columns.
/// uint64 wrapping arithmetic as in C++.
pub fn engram_rows(toks: &[u64], mult: &[u64], primes: &[u64], offs: &[u64], n_head: usize) -> Vec<i32> {
    let n_gram = toks.len();
    let mut out = vec![0i32; (n_gram - 1) * n_head];
    let mut rolling = toks[0].wrapping_mul(mult[0]);
    for o in 1..n_gram {
        rolling ^= toks[o].wrapping_mul(mult[o]);
        for h in 0..n_head {
            let col = (o - 1) * n_head + h;
            out[col] = (rolling % primes[col]).wrapping_add(offs[col]) as i32;
        }
    }
    out
}

/// Grouped output projection (D41:757-763, wo_a loaded as `[heads/groups * hd, o_lora_rank, groups]`): group `g`
/// reads attention output channels `[g * hpg * hd, (g+1) * hpg * hd)` (heads `g*hpg ..` in order, each head's `hd`
/// contiguous) and uses `attn_output_a` file rows `[g * o_lora, (g+1) * o_lora)`; its result lands at
/// `[g * o_lora, (g+1) * o_lora)` of the concatenated input of `attn_output_b`.
/// Returns (input offset, input length, wo_a first row, rows).
pub fn out_group(g: usize, heads_per_group: usize, hd: usize, o_lora: usize) -> (usize, usize, usize, usize) {
    (g * heads_per_group * hd, heads_per_group * hd, g * o_lora, o_lora)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn sinkhorn_rows_and_cols_sum_to_one() {
        let mut s = 7u64;
        for _ in 0..20 {
            let mut c = [0f32; 16];
            for v in c.iter_mut() {
                *v = lcg(&mut s) * 1.5;
            }
            sinkhorn(&mut c, 1e-6, 20);
            for k in 0..4 {
                // the column norm runs last, so columns are exact up to eps; rows converge geometrically
                let col: f32 = (0..4).map(|src| c[k + 4 * src]).sum();
                let row: f32 = (0..4).map(|dst| c[dst + 4 * k]).sum();
                assert!((col - 1.0).abs() < 1e-5, "col {k} sums to {col}");
                assert!((row - 1.0).abs() < 1e-3, "row {k} sums to {row}");
                assert!(c.iter().all(|&v| v > 0.0));
            }
        }
        // a known matrix: equal logits give the uniform doubly stochastic matrix
        let mut c = [0.3f32; 16];
        sinkhorn(&mut c, 1e-6, 20);
        assert!(c.iter().all(|&v| (v - 0.25).abs() < 1e-5));
    }

    #[test]
    fn hc_comb_uses_dst_plus_4src_layout() {
        // logits strongly favour dst == src: the result is near the identity in the dst + 4*src layout
        let mut m = vec![0f32; 24];
        for src in 0..4 {
            m[8 + src + 4 * src] = 10.0;
        }
        let c = hc_comb(&m, 1.0, &[0f32; 24], 1e-6, 20);
        for src in 0..4 {
            for dst in 0..4 {
                let v = c[dst + 4 * src];
                if src == dst {
                    assert!(v > 0.99);
                } else {
                    assert!(v < 0.01);
                }
            }
        }
    }

    #[test]
    fn yarn_corr_dims_and_ramp_for_deepseek41() {
        let c = yarn_corr_dims(64, 65536, 160000.0, 32.0, 1.0);
        assert_eq!(c, [15.0, 25.0]);
        assert_eq!(yarn_ramp(15.0, 25.0, 0), 1.0);
        assert_eq!(yarn_ramp(15.0, 25.0, 30), 1.0); // pair 15
        assert!((yarn_ramp(15.0, 25.0, 32) - 0.9).abs() < 1e-6); // pair 16
        assert!((yarn_ramp(15.0, 25.0, 40) - 0.5).abs() < 1e-6); // pair 20
        assert!((yarn_ramp(15.0, 25.0, 48) - 0.1).abs() < 1e-6); // pair 24
        assert_eq!(yarn_ramp(15.0, 25.0, 50), 0.0); // pair 25
        // net mscale of a yarn layer: attn_factor_l = 1/(1 + 0.1 ln 16) cancels ggml's (1 + 0.1 ln(1/freq_scale))
        let af = 1.0 / (1.0 + 0.1 * (1.0f32 / 0.0625).ln());
        let r = Rope::new(64, 448, 65536, 160000.0, 0.0625, 1.0, af, 32.0, 1.0);
        let c0 = r.cache(0, 1.0);
        assert!((c0[0].0 - 1.0).abs() < 1e-6 && c0[0].1 == 0.0);
    }

    #[test]
    fn rope_norm_rotates_adjacent_pairs_after_offset() {
        let r = Rope::new(4, 2, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        let mut x = vec![9.0, 8.0, 1.0, 0.0, 0.0, 1.0];
        let c = r.cache(1, 1.0);
        r.apply(&mut x, &c);
        assert_eq!(&x[..2], &[9.0, 8.0], "channels below the offset are untouched");
        // pair 0 = (x[2], x[3]) rotates by theta_0 = 1 rad
        assert!((x[2] - 1f32.cos()).abs() < 1e-6 && (x[3] - 1f32.sin()).abs() < 1e-6);
        // pair 1 = (x[4], x[5]) rotates by 1 * 10000^(-2/4) = 0.01 rad
        let t = 0.01f32;
        assert!((x[4] + t.sin()).abs() < 1e-6 && (x[5] - t.cos()).abs() < 1e-6);
        // position 0 is the identity
        let mut y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        r.apply(&mut y, &r.cache(0, 1.0));
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rope_back_inverts_rope() {
        let af = 1.0 / (1.0 + 0.1 * 16f32.ln());
        for r in [Rope::new(64, 448, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0), Rope::new(64, 448, 65536, 160000.0, 0.0625, 1.0, af, 32.0, 1.0)] {
            let mut s = 3u64;
            let x: Vec<f32> = (0..512).map(|_| lcg(&mut s)).collect();
            for pos in [0i64, 1, 5, 77, 127] {
                let mut y = x.clone();
                r.apply(&mut y, &r.cache(pos, 1.0));
                if pos > 0 {
                    assert!(y[448..].iter().zip(&x[448..]).any(|(a, b)| (a - b).abs() > 1e-3));
                }
                r.apply(&mut y, &r.cache(pos, -1.0));
                for (a, b) in y.iter().zip(&x) {
                    assert!((a - b).abs() < 2e-6, "pos {pos}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn swiglu_clamp_limits() {
        assert_eq!(swiglu_clamp(20.0, 20.0, 10.0), silu(10.0) * 10.0);
        assert_eq!(swiglu_clamp(-3.0, -15.0, 10.0), silu(-3.0) * -10.0);
        assert_eq!(swiglu_clamp(1.0, 2.0, 10.0), silu(1.0) * 2.0);
        // the gate has no lower clamp
        assert_eq!(swiglu_clamp(-50.0, 1.0, 10.0), silu(-50.0));
    }

    #[test]
    fn sqrt_softplus_branches() {
        assert_eq!(sqrt_softplus(25.0), 5.0);
        assert_eq!(sqrt_softplus(21.0), 21f32.sqrt());
        assert!((sqrt_softplus(0.0) - 2f32.ln().sqrt()).abs() < 1e-7);
        let tiny = sqrt_softplus(-30.0);
        assert!((tiny - (-30f32).exp().sqrt()).abs() / tiny < 1e-5);
    }

    #[test]
    fn engram_hash_hand_example() {
        let toks = [5u64, 7, 2, 2];
        let rows = engram_rows(&toks, &[3, 5, 7, 11], &[13, 17, 19, 23, 29, 31], &[0, 100, 200, 300, 400, 500], 2);
        // rolling: 15 -> 15^35 = 44 -> 44^14 = 34 -> 34^22 = 52
        assert_eq!(rows, vec![5, 110, 215, 311, 423, 521]);
    }

    #[test]
    fn engram_hash_wraps_in_u64() {
        let m0 = 1u64 << 63;
        let toks = [3u64, 0, 0, 0];
        let rows = engram_rows(&toks, &[m0, 1, 1, 1], &[1_000_003; 3], &[7; 3], 1);
        let want = ((3u128 * m0 as u128) % (1u128 << 64)) as u64 % 1_000_003 + 7;
        assert_eq!(rows, vec![want as i32; 3]);
        // a product past 2^64 (multiplier from the file, id near the vocab size)
        let (t, m) = (129_279u64, 82_619_226_485_591u64 * 1000);
        let rows = engram_rows(&[t, 0, 0, 0], &[m, 0, 0, 0], &[16_000_057; 3], &[0; 3], 1);
        let want = ((t as u128 * m as u128) % (1u128 << 64)) as u64 % 16_000_057;
        assert_eq!(rows[0], want as i32);
    }

    #[test]
    fn engram_toks_pad_and_block() {
        let map: Vec<i64> = vec![0, 1, 2, 3, 3, -1, 6];
        // pos 1: one real predecessor, longer shifts pad
        let hist = [4i64, 6];
        let t = engram_toks(6, 1, 4, &map, 2, |s| hist[1 - s]);
        assert_eq!(t, vec![6, 3, 2, 2]);
        // a dead predecessor (map -1) blocks it and every longer shift even when history exists
        let hist = [1i64, 5, 3, 6];
        let t = engram_toks(6, 3, 4, &map, 2, |s| hist[3 - s]);
        assert_eq!(t, vec![6, 3, 2, 2]);
        // an out-of-range token blocks everything
        let t = engram_toks(99, 3, 4, &map, 2, |s| hist[3 - s]);
        assert_eq!(t, vec![2, 2, 2, 2]);
    }

    #[test]
    fn grouped_output_mapping() {
        // 64 heads x 512 in 8 groups -> attn_output_a [4096 in, 1024 out, 8]
        assert_eq!(out_group(0, 8, 512, 1024), (0, 4096, 0, 1024));
        assert_eq!(out_group(3, 8, 512, 1024), (3 * 4096, 4096, 3 * 1024, 1024));
        // head h, channel c sits in group h/8 at offset (h%8)*512 + c of that group's input
        for h in [0usize, 7, 8, 63] {
            let (off, len, _, _) = out_group(h / 8, 8, 512, 1024);
            let flat = h * 512 + 100;
            assert!(flat >= off && flat < off + len);
            assert_eq!(flat - off, (h % 8) * 512 + 100);
        }
    }

    #[test]
    fn activation_rounding() {
        assert_eq!(bf16r(1.0 + 1.0 / 256.0), 1.0); // tie -> even
        assert_eq!(bf16r(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0); // tie -> even (up)
        assert_eq!(bf16r(-2.5), -2.5);
        let mut s = 11u64;
        let x: Vec<f32> = (0..256).map(|_| lcg(&mut s)).collect();
        let mut a = x.clone();
        q8_k_roundtrip(&mut a);
        let mut b = x.clone();
        q8_0_roundtrip(&mut b);
        for i in 0..256 {
            assert!((a[i] - x[i]).abs() <= 1.0 / 127.0 * 0.5 + 1e-6);
            assert!((b[i] - x[i]).abs() <= 1.0 / 127.0 * 0.51 + 1e-3);
        }
        // the largest magnitude survives exactly up to the scale's rounding
        let mut z = vec![0f32; 256];
        z[3] = -2.0;
        z[4] = 1.0;
        q8_k_roundtrip(&mut z);
        assert!((z[3] + 2.0).abs() < 1e-6);
        assert!((z[4] - 2.0 * 64.0 / 127.0).abs() < 1e-6, "{}", z[4]); // nearest_int(63.5) = 64 (tie to even)
    }
}
