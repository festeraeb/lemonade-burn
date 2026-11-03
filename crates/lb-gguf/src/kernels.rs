//! CPU reference kernels: `out[t, r] = sum_c W[r, c] * x[t, c]` with `W` kept quantized.
//!
//! These are the scalar references the CubeCL kernels are checked against (see
//! `crates/lb-q4-gpu`). Rows are split across rayon threads; each row is dequantized
//! block by block in f32 and dotted with f32 activations, so the result is the
//! "true" dequantized matmul (no activation quantization).

use crate::types::GgmlType;
use half::f16;
use rayon::prelude::*;

/// A quantized row-major matrix view: `rows` rows of `cols` values, each row
/// `ty.row_bytes(cols)` bytes, contiguous.
#[derive(Clone, Copy, Debug)]
pub struct QuantMat<'a> {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
}

impl<'a> QuantMat<'a> {
    pub fn new(ty: GgmlType, rows: usize, cols: usize, bytes: &'a [u8]) -> Self {
        let rb = ty.row_bytes(cols).expect("cols must be a whole number of blocks");
        assert_eq!(bytes.len(), rows * rb, "matrix byte length");
        Self { ty, rows, cols, bytes }
    }

    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.cols).unwrap()
    }

    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.bytes[r * rb..(r + 1) * rb]
    }

    /// Dequantize row `r` into `dst` (`cols` values).
    pub fn dequant_row(&self, r: usize, dst: &mut [f32]) {
        self.ty.dequant_row(self.row(r), dst)
    }

    /// Dot of row `r` with `x` (`cols` values).
    pub fn dot(&self, r: usize, x: &[f32]) -> f32 {
        let row = self.row(r);
        match self.ty {
            GgmlType::Q4_0 => dot_q4_0(row, x),
            GgmlType::Q8_0 => dot_q8_0(row, x),
            _ => {
                let mut buf = vec![0f32; self.cols];
                self.ty.dequant_row(row, &mut buf);
                buf.iter().zip(x).map(|(a, b)| a * b).sum()
            }
        }
    }
}

/// Q4_0 row dot product without a dequantized scratch row.
pub fn dot_q4_0(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() / 18 * 32, x.len());
    let mut acc = 0f32;
    for (b, xs) in row.chunks_exact(18).zip(x.chunks_exact(32)) {
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let mut s = 0f32;
        for j in 0..16 {
            let byte = b[2 + j];
            s += ((byte & 0x0f) as i32 - 8) as f32 * xs[j];
            s += ((byte >> 4) as i32 - 8) as f32 * xs[j + 16];
        }
        acc += d * s;
    }
    acc
}

/// Q8_0 row dot product.
pub fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() / 34 * 32, x.len());
    let mut acc = 0f32;
    for (b, xs) in row.chunks_exact(34).zip(x.chunks_exact(32)) {
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let mut s = 0f32;
        for j in 0..32 {
            s += (b[2 + j] as i8) as f32 * xs[j];
        }
        acc += d * s;
    }
    acc
}

/// `out[t * rows + r] = sum_c W[r, c] * x[t * cols + c]` for `m` activation rows.
pub fn matvec_f32(w: QuantMat<'_>, x: &[f32], m: usize, out: &mut [f32]) {
    assert_eq!(x.len(), m * w.cols);
    assert_eq!(out.len(), m * w.rows);
    let rows = w.rows;
    let cols = w.cols;
    // Work unit: a chunk of rows; the per-row dequantized buffer is reused per token.
    let chunk = 16usize;
    let results: Vec<(usize, Vec<f32>)> = (0..rows.div_ceil(chunk))
        .into_par_iter()
        .map(|ci| {
            let r0 = ci * chunk;
            let r1 = (r0 + chunk).min(rows);
            let mut vals = vec![0f32; (r1 - r0) * m];
            let mut buf = vec![0f32; cols];
            for r in r0..r1 {
                match w.ty {
                    GgmlType::Q4_0 | GgmlType::Q8_0 if m == 1 => {
                        vals[r - r0] = w.dot(r, x);
                    }
                    _ => {
                        w.dequant_row(r, &mut buf);
                        for t in 0..m {
                            let xr = &x[t * cols..(t + 1) * cols];
                            vals[t * (r1 - r0) + (r - r0)] = buf.iter().zip(xr).map(|(a, b)| a * b).sum();
                        }
                    }
                }
            }
            (r0, vals)
        })
        .collect();
    for (r0, vals) in results {
        let n = vals.len() / m;
        for t in 0..m {
            out[t * rows + r0..t * rows + r0 + n].copy_from_slice(&vals[t * n..(t + 1) * n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{quantize_q4_0, quantize_q8_0};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn q4_0_matvec_matches_dequantized_f32() {
        let (rows, cols, m) = (37, 96, 3);
        let mut s = 5u64;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..m * cols).map(|_| lcg(&mut s)).collect();
        let bytes: Vec<u8> = w.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect();
        let qm = QuantMat::new(GgmlType::Q4_0, rows, cols, &bytes);
        let mut out = vec![0f32; m * rows];
        matvec_f32(qm, &x, m, &mut out);
        let mut buf = vec![0f32; cols];
        for r in 0..rows {
            qm.dequant_row(r, &mut buf);
            for t in 0..m {
                let want: f32 = buf.iter().zip(&x[t * cols..(t + 1) * cols]).map(|(a, b)| a * b).sum();
                assert!((out[t * rows + r] - want).abs() < 1e-4, "t{t} r{r}: {} vs {want}", out[t * rows + r]);
            }
        }
    }

    #[test]
    fn q8_0_single_token_uses_fused_dot() {
        let (rows, cols) = (5, 64);
        let mut s = 9u64;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..cols).map(|_| lcg(&mut s)).collect();
        let bytes: Vec<u8> = w.chunks_exact(cols).flat_map(|r| quantize_q8_0(r)).collect();
        let qm = QuantMat::new(GgmlType::Q8_0, rows, cols, &bytes);
        let mut out = vec![0f32; rows];
        matvec_f32(qm, &x, 1, &mut out);
        let mut buf = vec![0f32; cols];
        for r in 0..rows {
            qm.dequant_row(r, &mut buf);
            let want: f32 = buf.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((out[r] - want).abs() < 1e-4);
        }
    }
}
