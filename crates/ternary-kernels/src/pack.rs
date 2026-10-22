//! Absmean ternarization and 2-bit packing.

use crate::{padded_k, BLOCK, BLOCK_BYTES};

/// Result of ternarizing one weight tensor the way
/// `transformers.integrations.bitnet.WeightQuant` does:
///
/// ```text
/// scale = 1 / mean(|W|).clamp(min=1e-5)          (f32)
/// T     = round_half_even(W * scale).clamp(-1, 1)
/// W'    = T / scale                               (so dequant = 1 / scale, f32)
/// ```
#[derive(Debug, Clone)]
pub struct Ternarized {
    pub values: Vec<i8>,
    /// `scale` as computed by WeightQuant (f32).
    pub scale: f32,
    /// Magnitude of a +-1 weight after dequantization: `1.0f32 / scale`.
    pub dequant: f32,
}

/// Mean of |w| computed the way torch's f32 `mean()` behaves to within an ulp: we sum in
/// f64 (more accurate than torch's cascade sum) and round to f32 once. The converter
/// checks the resulting ternary values against a torch-produced hash, so any ulp-level
/// disagreement that flips a value is detected rather than assumed away.
pub fn absmean(w: &[f32]) -> f32 {
    let s: f64 = w.iter().map(|v| v.abs() as f64).sum();
    (s / w.len() as f64) as f32
}

pub fn ternarize_with_mean(w: &[f32], mean: f32) -> Ternarized {
    let scale = 1.0f32 / mean.max(1e-5);
    let values = w
        .iter()
        .map(|&v| (v * scale).round_ties_even().clamp(-1.0, 1.0) as i8)
        .collect();
    Ternarized { values, scale, dequant: 1.0f32 / scale }
}

pub fn ternarize_absmean(w: &[f32]) -> Ternarized {
    ternarize_with_mean(w, absmean(w))
}

/// Pack a row-major `[n, k]` ternary matrix (values in {-1,0,1}) into the LB2 layout.
/// Columns are zero-padded (code 1) up to a multiple of 128.
pub fn pack_ternary(t: &[i8], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(t.len(), n * k);
    let kp = padded_k(k);
    let row_bytes = kp / 4;
    // code 1 in all four 2-bit lanes = 0b01010101: padding weights are 0.
    let mut out = vec![0x55u8; n * row_bytes];
    for r in 0..n {
        let row = &t[r * k..(r + 1) * k];
        let dst = &mut out[r * row_bytes..(r + 1) * row_bytes];
        for (c, &v) in row.iter().enumerate() {
            debug_assert!((-1..=1).contains(&v));
            let code = (v + 1) as u8;
            let b = c / BLOCK;
            let within = c % BLOCK;
            let i = within / 32;
            let j = within % 32;
            let byte = &mut dst[b * BLOCK_BYTES + j];
            *byte = (*byte & !(3 << (2 * i))) | (code << (2 * i));
        }
    }
    out
}

/// Inverse of [`pack_ternary`] for one row (returns the padded length `kp`).
pub fn unpack_row(packed_row: &[u8], kp: usize) -> Vec<i8> {
    let mut out = vec![0i8; kp];
    for (c, o) in out.iter_mut().enumerate() {
        let b = c / BLOCK;
        let within = c % BLOCK;
        let byte = packed_row[b * BLOCK_BYTES + within % 32];
        *o = ((byte >> (2 * (within / 32))) & 3) as i8 - 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let n = 3;
        let k = 300; // not a multiple of 128: exercises padding
        let t: Vec<i8> = (0..n * k).map(|i| ((i * 7 + i / 5) % 3) as i8 - 1).collect();
        let p = pack_ternary(&t, n, k);
        let kp = padded_k(k);
        assert_eq!(p.len(), n * kp / 4);
        for r in 0..n {
            let u = unpack_row(&p[r * kp / 4..(r + 1) * kp / 4], kp);
            assert_eq!(&u[..k], &t[r * k..(r + 1) * k]);
            assert!(u[k..].iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn ternarize_ties_even() {
        // mean |w| = 1.0 -> scale 1; 0.5 rounds to 0 (ties-to-even), 1.5 -> 2 -> clamp 1
        let w = [0.5f32, -0.5, 1.5, -1.5, 0.0, 1.0, -1.0, 2.0];
        let t = ternarize_with_mean(&w, 1.0);
        assert_eq!(t.values, vec![0, 0, 1, -1, 0, 1, -1, 1]);
        assert_eq!(t.dequant, 1.0);
    }
}
