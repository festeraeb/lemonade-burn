//! Init-only conversion of a dense `W [out, in]` into a factorised layer, from the papers' equations:
//!
//! 1. truncated SVD `W ≈ U' V'ᵀ` with `U' = P_r √Σ_r`, `V' = Q_r √Σ_r` (LittleBit Eq. 6);
//! 2. optional Joint-ITQ (LittleBit-2): an orthogonal `R [r, r]` minimising `‖sign(ZR) − ZR‖` for
//!    `Z = [U'; V']`, found by alternating `B = sign(ZR)` and the Procrustes step `R = A Cᵀ` from the SVD
//!    `Zᵀ B = A S Cᵀ`; then `U' ← U'R`, `V' ← V'R` (the product is unchanged);
//! 3. signs `U = sign(U')`, `V = sign(V')`; magnitudes by rank-1 fits `|U'| ≈ s_out ℓ_uᵀ`,
//!    `|V'| ≈ s_in ℓ_vᵀ`, `s_lat = ℓ_u ⊙ ℓ_v` (Eq. 7-8);
//! 4. the residual path: the same on `W − Ŵ_primary` (Eq. 9-10).
//!
//! This is the starting point the papers then train (QAT with distillation); without that training
//! the quality is far below their tables. It exists to exercise the runtime on real weights.

use crate::{pack_row, words, Linear, Path};
use nalgebra::DMatrix;

#[derive(Clone, Debug)]
pub struct Opts {
    pub r: usize,
    pub residual: bool,
    /// Joint-ITQ iterations (0 = LittleBit v1 initialisation).
    pub itq_iters: usize,
    pub seed: u64,
}

/// Rank-1 fit of a non-negative matrix `a [n, r] ≈ σ x yᵀ`: returns `(x √σ, y √σ)`, both >= 0.
fn rank_one(a: &DMatrix<f32>) -> (Vec<f32>, Vec<f32>) {
    let mut y = nalgebra::DVector::<f32>::from_element(a.ncols(), 1.0 / (a.ncols() as f32).sqrt());
    let mut x = a * &y;
    let mut sigma = 0f32;
    for _ in 0..100 {
        x = a * &y;
        let nx = x.norm();
        if nx == 0.0 {
            break;
        }
        x /= nx;
        y = a.transpose() * &x;
        let s = y.norm();
        if s == 0.0 {
            break;
        }
        y /= s;
        let done = (s - sigma).abs() <= 1e-6 * s;
        sigma = s;
        if done {
            break;
        }
    }
    let rs = sigma.max(0.0).sqrt();
    (x.iter().map(|v| v.abs() * rs).collect(), y.iter().map(|v| v.abs() * rs).collect())
}

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        let (a, b) = (self.uniform(), self.uniform());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

/// Joint-ITQ rotation for `z [n, r]`.
fn itq(z: &DMatrix<f32>, iters: usize, seed: u64) -> DMatrix<f32> {
    let r = z.ncols();
    let mut rng = Rng(seed ^ 0x9e37_79b9_7f4a_7c15);
    let g = DMatrix::<f32>::from_fn(r, r, |_, _| rng.gauss());
    let mut rot = g.qr().q();
    for _ in 0..iters {
        let zr = z * &rot;
        let b = zr.map(|v| if v < 0.0 { -1.0 } else { 1.0 });
        let m = z.transpose() * b; // Zᵀ B = A S Cᵀ  ->  R = A Cᵀ
        let svd = m.svd(true, true);
        rot = svd.u.unwrap() * svd.v_t.unwrap();
    }
    rot
}

/// Share of the factor entries whose sign is at risk: |value| below 10 % of the column's RMS (a small
/// change flips them). Joint-ITQ rotates the factors toward the ±1 corners, which should lower it.
pub fn flip_risk(z: &DMatrix<f32>) -> f32 {
    let mut risky = 0usize;
    for c in 0..z.ncols() {
        let col = z.column(c);
        let rms = (col.norm_squared() / col.len() as f32).sqrt();
        risky += col.iter().filter(|v| v.abs() < 0.1 * rms).count();
    }
    risky as f32 / z.len().max(1) as f32
}

/// The ITQ objective per entry, columns scaled to unit RMS: mean of (|z| - 1)^2, i.e. how far the
/// entries sit from the nearest ±1 corner.
pub fn corner_distance(z: &DMatrix<f32>) -> f32 {
    let mut acc = 0f64;
    for c in 0..z.ncols() {
        let col = z.column(c);
        let rms = (col.norm_squared() / col.len() as f32).sqrt().max(f32::MIN_POSITIVE);
        acc += col.iter().map(|v| ((v.abs() / rms) as f64 - 1.0).powi(2)).sum::<f64>();
    }
    (acc / z.len().max(1) as f64) as f32
}

/// Per-path numbers the converter reports.
#[derive(Clone, Copy, Debug, Default)]
pub struct PathStats {
    /// flip_risk of [U'; V'] before and after the rotation (equal without ITQ).
    pub risk_before: f32,
    pub risk_after: f32,
    pub corner_before: f32,
    pub corner_after: f32,
}

/// Top-`r` singular triplets of `w`: `(P_r, σ_r, Q_r)` with `w ≈ P diag(σ) Qᵀ`, largest first.
/// For `r` well below the smaller side, a randomized range finder (Gaussian sketch of `r + 16`
/// columns, two power iterations with re-orthonormalisation) and an SVD of the small projected
/// matrix; otherwise the full SVD.
fn top_svd(w: &DMatrix<f32>, r: usize, seed: u64) -> (DMatrix<f32>, Vec<f32>, DMatrix<f32>) {
    let (m, n) = (w.nrows(), w.ncols());
    let k = (r + 16).min(m.min(n));
    if 2 * k >= m.min(n) {
        let svd = w.clone().svd(true, true);
        let mut idx: Vec<usize> = (0..svd.singular_values.len()).collect();
        idx.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
        let (pu, qt) = (svd.u.as_ref().unwrap(), svd.v_t.as_ref().unwrap());
        let p = DMatrix::from_fn(m, r, |i, c| pu[(i, idx[c])]);
        let q = DMatrix::from_fn(n, r, |i, c| qt[(idx[c], i)]);
        return (p, idx.iter().take(r).map(|&c| svd.singular_values[c]).collect(), q);
    }
    let mut rng = Rng(seed ^ 0x5eed_5eed);
    let omega = DMatrix::<f32>::from_fn(n, k, |_, _| rng.gauss());
    let mut y = w * omega;
    for _ in 0..2 {
        let qy = y.qr().q();
        let z = (w.transpose() * qy).qr().q();
        y = w * z;
    }
    let qm = y.qr().q(); // m x k
    let b = qm.transpose() * w; // k x n
    let svd = b.svd(true, true);
    let mut idx: Vec<usize> = (0..svd.singular_values.len()).collect();
    idx.sort_by(|&a, &c| svd.singular_values[c].total_cmp(&svd.singular_values[a]));
    let ub = svd.u.as_ref().unwrap(); // k x k
    let vt = svd.v_t.as_ref().unwrap(); // k x n
    let ub_r = DMatrix::from_fn(k, r, |i, c| ub[(i, idx[c])]);
    let p = &qm * ub_r;
    let q = DMatrix::from_fn(n, r, |i, c| vt[(idx[c], i)]);
    (p, idx.iter().take(r).map(|&c| svd.singular_values[c]).collect(), q)
}

/// One path from `w` at rank `r`; also returns the dense approximation it stands for.
fn path(w: &DMatrix<f32>, o: &Opts, seed: u64) -> (Path, DMatrix<f32>, PathStats) {
    let (out, inp) = (w.nrows(), w.ncols());
    let r = o.r.min(out.min(inp));
    let (pu, sv, qv) = top_svd(w, r, seed);
    let mut up = DMatrix::<f32>::zeros(out, r);
    let mut vp = DMatrix::<f32>::zeros(inp, r);
    for c in 0..r {
        let s = sv[c].max(0.0).sqrt();
        up.set_column(c, &(pu.column(c) * s));
        vp.set_column(c, &(qv.column(c) * s));
    }
    let stack = |a: &DMatrix<f32>, b: &DMatrix<f32>| {
        let mut z = DMatrix::<f32>::zeros(out + inp, r);
        z.rows_mut(0, out).copy_from(a);
        z.rows_mut(out, inp).copy_from(b);
        z
    };
    let z0 = stack(&up, &vp);
    let mut st = PathStats { risk_before: flip_risk(&z0), corner_before: corner_distance(&z0), ..Default::default() };
    if o.itq_iters > 0 {
        let rot = itq(&z0, o.itq_iters, seed);
        up = &up * &rot;
        vp = &vp * &rot;
    }
    let z1 = stack(&up, &vp);
    st.risk_after = flip_risk(&z1);
    st.corner_after = corner_distance(&z1);
    let (s_out, lu) = rank_one(&up.abs());
    let (s_in, lv) = rank_one(&vp.abs());
    let s_lat: Vec<f32> = lu.iter().zip(&lv).map(|(a, b)| a * b).collect();

    let (wr, wi) = (words(r), words(inp));
    let mut u = vec![0u32; out * wr];
    for j in 0..out {
        pack_row((0..r).map(|i| up[(j, i)] < 0.0), r, &mut u[j * wr..(j + 1) * wr]);
    }
    let mut v = vec![0u32; r * wi];
    for i in 0..r {
        pack_row((0..inp).map(|k| vp[(k, i)] < 0.0), inp, &mut v[i * wi..(i + 1) * wi]);
    }
    // dense approximation: (diag(s_out) U diag(s_lat)) · (diag(s_in) V)ᵀ, rounded through f16 like the file
    let h = |x: f32| half::f16::from_f32(x).to_f32();
    let a = DMatrix::<f32>::from_fn(out, r, |j, i| h(s_out[j]) * if up[(j, i)] < 0.0 { -1.0 } else { 1.0 } * h(s_lat[i]));
    let b = DMatrix::<f32>::from_fn(inp, r, |k, i| h(s_in[k]) * if vp[(k, i)] < 0.0 { -1.0 } else { 1.0 });
    let approx = a * b.transpose();
    (Path { out, inp, r, u, v, s_out, s_lat, s_in }, approx, st)
}

/// Convert `w` (row-major `[out, in]`). Returns the layer, `‖W − Ŵ‖_F / ‖W‖_F` and per-path stats.
pub fn convert(w: &[f32], out: usize, inp: usize, o: &Opts) -> (Linear, f32, Vec<PathStats>) {
    let wm = DMatrix::<f32>::from_row_slice(out, inp, w);
    let (p0, a0, s0) = path(&wm, o, o.seed);
    let mut paths = vec![p0];
    let mut stats = vec![s0];
    let mut rest = &wm - a0;
    if o.residual {
        let (p1, a1, s1) = path(&rest, o, o.seed.wrapping_add(1));
        rest -= a1;
        paths.push(p1);
        stats.push(s1);
    }
    let rel = rest.norm() / wm.norm().max(f32::MIN_POSITIVE);
    (Linear { out, inp, paths }, rel, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_svd_finds_the_top_subspace() {
        // a 300 x 200 matrix with a clear spectrum: the randomized path (r = 20) must match the full SVD's top values
        let mut rng = Rng(5);
        let a = DMatrix::<f32>::from_fn(300, 30, |_, _| rng.gauss());
        let b = DMatrix::<f32>::from_fn(30, 200, |_, _| rng.gauss());
        let w = a * b + DMatrix::<f32>::from_fn(300, 200, |_, _| 0.01 * rng.gauss());
        let (_, s_fast, _) = top_svd(&w, 20, 1);
        let mut full: Vec<f32> = w.clone().svd(false, false).singular_values.iter().copied().collect();
        full.sort_by(|a, b| b.total_cmp(a));
        for i in 0..20 {
            assert!((s_fast[i] - full[i]).abs() <= 2e-3 * full[i], "sigma {i}: {} vs {}", s_fast[i], full[i]);
        }
    }

    #[test]
    fn the_layer_reproduces_its_own_dense_approximation_and_more_rank_helps() {
        let (out, inp) = (48, 40);
        let mut rng = Rng(11);
        // a matrix with decaying spectrum, like trained weights
        let a = DMatrix::<f32>::from_fn(out, 8, |_, _| rng.gauss());
        let b = DMatrix::<f32>::from_fn(8, inp, |_, _| rng.gauss());
        let noise = DMatrix::<f32>::from_fn(out, inp, |_, _| 0.05 * rng.gauss());
        let w = a * b + noise;
        let wv: Vec<f32> = w.transpose().iter().copied().collect(); // row-major
        let mut last = f32::INFINITY;
        for (r, itq_iters) in [(8, 0), (16, 0), (16, 20), (32, 20)] {
            let (lin, rel, st) = convert(&wv, out, inp, &Opts { r, residual: true, itq_iters, seed: 1 });
            if itq_iters > 0 {
                // LittleBit-2's point: the rotation moves the entries toward the ±1 corners
                assert!(st[0].corner_after < st[0].corner_before, "r {r}: ITQ did not move toward the corners: {:?}", st[0]);
                assert!(st[0].risk_after <= st[0].risk_before + 1e-6, "r {r}: ITQ raised the flip risk: {:?}", st[0]);
            }
            // the factored layer is what the file holds: its dense form gives the same error
            let d = lin.dense();
            let err: f32 = d.iter().zip(&wv).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() / w.norm();
            assert!((err - rel).abs() < 2e-3, "r {r}: file error {err} vs reported {rel}");
            assert!(rel < 1.0, "r {r}: error {rel} is no better than zero");
            if itq_iters == 0 {
                assert!(rel <= last + 1e-4, "r {r}: more rank made it worse ({rel} > {last})");
                last = rel;
            }
        }
    }
}
