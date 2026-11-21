//! What ggml's CPU matmul does to an f32 activation before the dot with a quantised weight: it converts the activation
//! row to the weight type's `vec_dot_type` (ggml-cpu.c `type_traits_cpu`): Q8_0 for Q4_0/Q8_0/IQ4_NL/MXFP4, Q8_K for the
//! k- and i-quants, BF16 for BF16, F16 for F16, nothing for F32. These round trips let a CPU reference reproduce
//! llama.cpp's numbers instead of the exact f32 product (the difference is large where an activation block holds an
//! outlier: the other values of its 32/256-block lose most of their bits). Semantics from ggml-quants.c and
//! arch/x86/quants.c (MIT); code is this project's own.

use crate::types::GgmlType;

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// fp32 -> bf16 -> fp32 as `ggml_compute_fp32_to_bf16` (round to nearest even; NaN kept quiet).
pub fn bf16r(x: f32) -> f32 {
    let u = x.to_bits();
    let h = if (u & 0x7fff_ffff) > 0x7f80_0000 { (u >> 16) | 64 } else { (u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16 };
    f32::from_bits(h << 16)
}

/// `nearest_int` of ggml-quants.c (round half to even through the 1.5 * 2^23 trick).
fn nearest_int(v: f32) -> i32 {
    let val = v + 12582912.0;
    (val.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// One activation row through Q8_0 and back: x86 `quantize_row_q8_0`: d = amax/127 stored as f16,
/// q = round-nearest-even(x * 127/amax). The value ggml's integer dot effectively uses is `f16(d) * q`.
pub fn q8_0_roundtrip(x: &mut [f32]) {
    for b in x.chunks_exact_mut(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = f16r(amax / 127.0);
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        for v in b.iter_mut() {
            let q = (*v * id).round_ties_even().clamp(-128.0, 127.0);
            *v = d * q;
        }
    }
}

/// One activation row through Q8_K and back (`quantize_row_q8_K_ref`, also the x86 `quantize_row_q8_K`): signed max of
/// the largest magnitude, iscale = -127/max, q = min(127, nearest_int(iscale*x)), d = 1/iscale.
pub fn q8_k_roundtrip(x: &mut [f32]) {
    for b in x.chunks_exact_mut(256) {
        let (mut max, mut amax) = (0f32, 0f32);
        for &v in b.iter() {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            b.iter_mut().for_each(|v| *v = 0.0);
            continue;
        }
        let iscale = -127.0 / max;
        let d = 1.0 / iscale;
        for v in b.iter_mut() {
            let q = nearest_int(iscale * *v).min(127);
            *v = d * q as f32;
        }
    }
}

/// Round `x` (rows of `cols`) as ggml's CPU matmul rounds the activation for a weight of type `ty`.
pub fn act_round(ty: GgmlType, x: &mut [f32], cols: usize) {
    match ty {
        GgmlType::F32 | GgmlType::Other(_) => {}
        GgmlType::BF16 => x.iter_mut().for_each(|v| *v = bf16r(*v)),
        GgmlType::F16 => x.iter_mut().for_each(|v| *v = f16r(*v)),
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::IQ4NL | GgmlType::MXFP4 => {
            assert_eq!(cols % 32, 0);
            x.chunks_exact_mut(cols).for_each(q8_0_roundtrip)
        }
        _ => {
            assert_eq!(cols % 256, 0, "Q8_K activation needs rows of 256");
            x.chunks_exact_mut(cols).for_each(q8_k_roundtrip)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q8_roundtrips_bound_error_by_half_step() {
        let mut a: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let orig = a.clone();
        let amax = orig.iter().fold(0f32, |m, v| m.max(v.abs()));
        q8_k_roundtrip(&mut a);
        for (x, y) in a.iter().zip(&orig) {
            assert!((x - y).abs() <= amax / 127.0 * 0.5 + 1e-6);
        }
        let mut b = orig.clone();
        q8_0_roundtrip(&mut b);
        assert!(b.iter().zip(&orig).all(|(x, y)| (x - y).abs() <= 0.06));
        assert_eq!(bf16r(1.0), 1.0);
    }
}
