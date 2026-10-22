//! Per-token int8 absmax activation quantization (BitNet b1.58 "A8").
//!
//! Mirrors `transformers.integrations.bitnet.ActQuant` in f32:
//! ```text
//! scale = 127 / max(|x|).clamp(min=1e-5)
//! q     = round_half_even(x * scale).clamp(-128, 127)
//! x'    = q / scale
//! ```
//! Because `|x * scale| <= 127` (up to f32 rounding of the product, which cannot push it
//! past 127.5), `q` is in `[-127, 127]`; -128 never occurs.

/// Quantize one row. Returns `(scale, sum(q))`. `q.len()` may exceed `x.len()`
/// (padding); the tail is zeroed.
pub fn quantize_row(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    debug_assert!(q.len() >= x.len());
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = 127.0f32 / amax.max(1e-5);
    let mut sum = 0i32;
    for (o, &v) in q.iter_mut().zip(x) {
        let r = (v * scale).round_ties_even().clamp(-128.0, 127.0) as i8;
        sum += r as i32;
        *o = r;
    }
    for o in q[x.len()..].iter_mut() {
        *o = 0;
    }
    (scale, sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic() {
        let x = [1.0f32, -2.0, 0.5, 0.0];
        let mut q = [0i8; 6];
        let (s, sum) = quantize_row(&x, &mut q);
        assert_eq!(s, 63.5);
        // 63.5 -> 64 (ties-to-even), -127, 31.75 -> 32, 0
        assert_eq!(q, [64, -127, 32, 0, 0, 0]);
        assert_eq!(sum, 64 - 127 + 32);
    }
}
