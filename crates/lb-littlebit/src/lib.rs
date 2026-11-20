//! Sub-1-bit factorised linear layers in the LittleBit / LittleBit-2 representation.
//!
//! Written from the papers' equations (Lee et al., "LittleBit", NeurIPS 2025, arXiv 2506.13771, Eq. 4-5
//! and 9-10; Lee & Kim, "LittleBit-2", ICML 2026, arXiv 2603.00042, which changes the initialisation only).
//! No code from the authors' repository is used.
//!
//! A weight `W [out, in]` is replaced by one or two *paths* (primary, residual). Each path is
//!
//! ```text
//! W_p = diag(s_out) · U · diag(s_lat) · Vᵀ · diag(s_in)     U ∈ {±1}^[out, r], V ∈ {±1}^[in, r]
//! ```
//!
//! and `W ≈ Σ_p W_p`. The forward never builds `W`:
//!
//! ```text
//! y = Σ_p  s_out ⊙ ( U · ( s_lat ⊙ ( Vᵀ · (s_in ⊙ x) ) ) )
//! ```
//!
//! Storage: signs packed 1 bit each, little-endian u32 words, bit `b` of word `w` = column `32 w + b`,
//! bit set = −1 (rows padded with +1 = 0 bits). `U` is stored as rows of `out` over `r`, `V` as rows of
//! `r` over `in` (so both matvecs read along rows). Scales are f16 on disk, f32 in memory.

pub mod convert;
pub mod sidecar;

/// Words per packed row of `cols` signs.
pub const fn words(cols: usize) -> usize {
    cols.div_ceil(32)
}

/// One path: two sign matrices and three scale vectors.
#[derive(Clone, Debug)]
pub struct Path {
    pub out: usize,
    pub inp: usize,
    pub r: usize,
    /// `[out][words(r)]`: row j holds the signs of U[j, 0..r].
    pub u: Vec<u32>,
    /// `[r][words(inp)]`: row i holds the signs of V[0..in, i] (the i-th column of V, i.e. row i of Vᵀ).
    pub v: Vec<u32>,
    pub s_out: Vec<f32>,
    pub s_lat: Vec<f32>,
    pub s_in: Vec<f32>,
}

/// A factorised linear layer `[out, in]`: one path, or primary + residual.
#[derive(Clone, Debug)]
pub struct Linear {
    pub out: usize,
    pub inp: usize,
    pub paths: Vec<Path>,
}

/// Pack a row of ±1 values given as `f32` signs (>= 0 → +1) into words.
pub fn pack_row(signs: impl Iterator<Item = bool>, cols: usize, dst: &mut [u32]) {
    debug_assert_eq!(dst.len(), words(cols));
    dst.fill(0);
    for (c, neg) in signs.enumerate().take(cols) {
        if neg {
            dst[c / 32] |= 1 << (c % 32);
        }
    }
}

#[inline]
fn sign_at(row: &[u32], c: usize) -> f32 {
    if row[c / 32] >> (c % 32) & 1 == 1 {
        -1.0
    } else {
        1.0
    }
}

/// `Σ_c sign[c] · x[c]` for one packed row: `total − 2 · Σ_{bit set} x[c]`.
#[inline]
fn signed_dot(row: &[u32], x: &[f32], total: f32) -> f32 {
    let mut neg = 0f32;
    for (w, &word) in row.iter().enumerate() {
        let mut bits = word;
        let base = w * 32;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            neg += x[base + b];
            bits &= bits - 1;
        }
    }
    total - 2.0 * neg
}

impl Path {
    pub fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.u.len() == self.out * words(self.r), "U: {} words, expected {}", self.u.len(), self.out * words(self.r));
        anyhow::ensure!(self.v.len() == self.r * words(self.inp), "V: {} words, expected {}", self.v.len(), self.r * words(self.inp));
        anyhow::ensure!(self.s_out.len() == self.out && self.s_lat.len() == self.r && self.s_in.len() == self.inp, "scale lengths");
        // padding bits must be 0 (+1) so a kernel may run whole words
        let pad_ok = |rows: &[u32], cols: usize| {
            let wpr = words(cols);
            let tail = cols % 32;
            tail == 0 || rows.chunks(wpr).all(|r| r[wpr - 1] >> tail == 0)
        };
        anyhow::ensure!(pad_ok(&self.u, self.r) && pad_ok(&self.v, self.inp), "padding bits set");
        Ok(())
    }

    /// `y[t] += s_out ⊙ U (s_lat ⊙ Vᵀ (s_in ⊙ x[t]))` for `m` rows of `x [m, in]`, `y [m, out]`.
    pub fn forward_add(&self, x: &[f32], m: usize, y: &mut [f32]) {
        use rayon::prelude::*;
        let (wi, wr) = (words(self.inp), words(self.r));
        let mut xs = vec![0f32; m * self.inp];
        let mut xs_tot = vec![0f32; m];
        for t in 0..m {
            let xt = &x[t * self.inp..(t + 1) * self.inp];
            let d = &mut xs[t * self.inp..(t + 1) * self.inp];
            for c in 0..self.inp {
                d[c] = xt[c] * self.s_in[c];
            }
            xs_tot[t] = d.iter().sum();
        }
        // z [m, r]
        let mut z = vec![0f32; m * self.r];
        z.par_chunks_mut(self.r).enumerate().for_each(|(t, zt)| {
            let xt = &xs[t * self.inp..(t + 1) * self.inp];
            for i in 0..self.r {
                zt[i] = signed_dot(&self.v[i * wi..(i + 1) * wi], xt, xs_tot[t]) * self.s_lat[i];
            }
        });
        let z_tot: Vec<f32> = z.chunks(self.r).map(|zt| zt.iter().sum()).collect();
        y.par_chunks_mut(self.out).enumerate().for_each(|(t, yt)| {
            let zt = &z[t * self.r..(t + 1) * self.r];
            for j in 0..self.out {
                yt[j] += self.s_out[j] * signed_dot(&self.u[j * wr..(j + 1) * wr], zt, z_tot[t]);
            }
        });
    }

    /// The dense `[out, in]` matrix this path stands for (tests and error reports only).
    pub fn dense(&self) -> Vec<f32> {
        let (wi, wr) = (words(self.inp), words(self.r));
        let mut w = vec![0f32; self.out * self.inp];
        for j in 0..self.out {
            let urow = &self.u[j * wr..(j + 1) * wr];
            for i in 0..self.r {
                let a = self.s_out[j] * sign_at(urow, i) * self.s_lat[i];
                let vrow = &self.v[i * wi..(i + 1) * wi];
                for k in 0..self.inp {
                    w[j * self.inp + k] += a * sign_at(vrow, k) * self.s_in[k];
                }
            }
        }
        w
    }

    /// Bits on disk: two sign matrices + three f16 vectors.
    pub fn bits(&self) -> usize {
        self.r * (self.out + self.inp) + 16 * (self.out + self.r + self.inp)
    }
}

impl Linear {
    /// `y [m, out] = x [m, in] · Wᵀ` through every path.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.inp, "littlebit forward: input width");
        let mut y = vec![0f32; m * self.out];
        for p in &self.paths {
            p.forward_add(x, m, &mut y);
        }
        y
    }

    pub fn dense(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.out * self.inp];
        for p in &self.paths {
            for (a, b) in w.iter_mut().zip(p.dense()) {
                *a += b;
            }
        }
        w
    }

    pub fn bits_per_weight(&self) -> f64 {
        self.paths.iter().map(Path::bits).sum::<usize>() as f64 / (self.out * self.inp) as f64
    }
}

/// The rank for a bits-per-weight budget (paper's accounting: per path `r(out+in)` sign bits and
/// `16(out+r+in)` scale bits), rounded down to a multiple of 8, at least 8, at most `min(out, in)`.
pub fn rank_for_bpw(out: usize, inp: usize, bpw: f64, paths: usize) -> usize {
    let (o, i, p) = (out as f64, inp as f64, paths as f64);
    let r = (bpw * o * i - p * 16.0 * (o + i)) / (p * (o + i + 16.0));
    let r = ((r.max(0.0) as usize) / 8 * 8).max(8);
    r.min(out.min(inp))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub(crate) fn random_path(out: usize, inp: usize, r: usize, seed: &mut u64) -> Path {
        let mut u = vec![0u32; out * words(r)];
        for j in 0..out {
            let s: Vec<bool> = (0..r).map(|_| lcg(seed) < 0.0).collect();
            pack_row(s.into_iter(), r, &mut u[j * words(r)..(j + 1) * words(r)]);
        }
        let mut v = vec![0u32; r * words(inp)];
        for i in 0..r {
            let s: Vec<bool> = (0..inp).map(|_| lcg(seed) < 0.0).collect();
            pack_row(s.into_iter(), inp, &mut v[i * words(inp)..(i + 1) * words(inp)]);
        }
        let pos = |n: usize, seed: &mut u64| (0..n).map(|_| lcg(seed).abs() + 0.05).collect::<Vec<f32>>();
        Path { out, inp, r, u, v, s_out: pos(out, seed), s_lat: pos(r, seed), s_in: pos(inp, seed) }
    }

    #[test]
    fn factored_forward_matches_the_dense_matrix() {
        let mut seed = 7;
        // odd sizes so the padding bits and partial words are exercised
        let lin = Linear { out: 45, inp: 70, paths: vec![random_path(45, 70, 37, &mut seed), random_path(45, 70, 37, &mut seed)] };
        for p in &lin.paths {
            p.check().unwrap();
        }
        let m = 3;
        let x: Vec<f32> = (0..m * 70).map(|_| lcg(&mut seed)).collect();
        let y = lin.forward(&x, m);
        let w = lin.dense();
        for t in 0..m {
            for j in 0..45 {
                let r: f32 = (0..70).map(|k| w[j * 70 + k] * x[t * 70 + k]).sum();
                let d = (r - y[t * 45 + j]).abs();
                assert!(d <= 1e-3 * (1.0 + r.abs()), "t {t} j {j}: dense {r} factored {}", y[t * 45 + j]);
            }
        }
    }

    #[test]
    fn rank_matches_the_paper_examples() {
        // LittleBit Appendix D: 4096x4096 at 0.55 bpw -> 546, 4096x11008 at 0.1 -> 133 (before the multiple-of-8 floor)
        assert_eq!(rank_for_bpw(4096, 4096, 0.55, 2), 544);
        assert_eq!(rank_for_bpw(4096, 11008, 0.1, 2), 128);
    }
}
