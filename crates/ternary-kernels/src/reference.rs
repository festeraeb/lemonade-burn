//! Slow, obviously-correct reference implementations used by golden tests.

use crate::pack::unpack_row;

/// Exact int32 dot products of int8 activations with ternary weights, decoding each
/// weight individually (no code-offset trick, no SIMD).
pub fn gemv_i32_reference(packed: &[u8], n: usize, kp: usize, q: &[i8]) -> Vec<i32> {
    let rb = kp / 4;
    (0..n)
        .map(|r| {
            let w = unpack_row(&packed[r * rb..(r + 1) * rb], kp);
            w.iter().zip(q).map(|(&a, &b)| a as i32 * b as i32).sum()
        })
        .collect()
}

/// f64 reference of the dequantized linear: `y = (q / a_scale) . (t * dequant)`.
pub fn linear_f64_reference(t: &[i8], n: usize, k: usize, dequant: f32, q: &[i8], a_scale: f32) -> Vec<f64> {
    (0..n)
        .map(|r| {
            let mut s = 0f64;
            for c in 0..k {
                s += (q[c] as f64 / a_scale as f64) * (t[r * k + c] as f64 * dequant as f64);
            }
            s
        })
        .collect()
}
