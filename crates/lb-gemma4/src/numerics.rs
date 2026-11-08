//! llama.cpp-compatible numerics for the quantized matmuls and the KV cache.
//!
//! llama.cpp never multiplies a quantized weight by an f32 activation: it first quantizes the
//! activation row to the weight type's `vec_dot_type` (CPU: `Q8_0` for Q4_0/Q8_0, `Q8_K` for
//! the K-quants, f16 for F16 weights; CUDA: `q8_1`, the same values as `Q8_0`) and takes an
//! integer dot. Measured on gemma-4-26B Q4_0, layer 0 `Qcur`: llama.cpp's value -4.1107 is
//! reproduced to every printed digit by an f64 matmul over Q8_0-quantized activations, while the
//! exact f32 product is -3.9429 (`docs/GEMMA4.md`). Over 30 layers of 128-expert routing those
//! few-percent differences decide near-tied distributions, so parity against llama.cpp needs
//! the same rounding. It also stores K and V as f16 (the default cache type on every backend).
//!
//! [`Numerics::LlamaCpp`] (the default) fake-quantizes each activation row the way llama.cpp
//! quantizes it (quantize, then dequantize in f32) and then runs lb-gguf's dequantized-weight
//! kernel, so the products equal llama.cpp's integer dot up to f32 summation order.
//! [`Numerics::Exact`] (`LB_GEMMA4_NUMERICS=exact`) keeps f32 activations and an f32 cache.

use half::f16;
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Numerics {
    LlamaCpp = 0,
    Exact = 1,
}

static MODE: AtomicU8 = AtomicU8::new(u8::MAX);

pub fn mode() -> Numerics {
    match MODE.load(Ordering::Relaxed) {
        0 => Numerics::LlamaCpp,
        1 => Numerics::Exact,
        _ => {
            let m = match std::env::var("LB_GEMMA4_NUMERICS").as_deref() {
                Ok("exact") => Numerics::Exact,
                _ => Numerics::LlamaCpp,
            };
            MODE.store(m as u8, Ordering::Relaxed);
            m
        }
    }
}

pub fn set_mode(m: Numerics) {
    MODE.store(m as u8, Ordering::Relaxed);
}

/// Q8_0 quantize + dequantize, blocks of 32 (`quantize_row_q8_0`, x86 path:
/// `id = 127 / amax`, round half to even, `d = amax / 127` stored as f16).
pub fn fake_q8_0(x: &mut [f32]) {
    for b in x.chunks_mut(32) {
        let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = f16::from_f32(amax / 127.0).to_f32();
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        for v in b.iter_mut() {
            *v = (*v * id).round_ties_even() * d;
        }
    }
}

/// Q8_K quantize + dequantize, blocks of 256 (`quantize_row_q8_K_ref`: `iscale = -127 / max`
/// with `max` the signed value of largest magnitude, `q = min(127, round(iscale x))`,
/// `d = 1 / iscale` in f32).
pub fn fake_q8_k(x: &mut [f32]) {
    for b in x.chunks_mut(256) {
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in b.iter() {
            if v.abs() > amax {
                amax = v.abs();
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
            // nearest_int: round half away from zero is what the magic-number trick gives for
            // these magnitudes up to ties; ties are measure-zero here
            let q = (iscale * *v).round().min(127.0);
            *v = q * d;
        }
    }
}

pub fn round_f16(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = f16::from_f32(*v).to_f32());
}

/// Quantize `x` (rows of `cols`) the way llama.cpp does before a dot with a `ty` weight.
pub fn quantize_activations(ty: GgmlType, x: &mut [f32], cols: usize) {
    match ty {
        GgmlType::Q4_0 | GgmlType::Q8_0 => x.chunks_mut(cols).for_each(fake_q8_0),
        GgmlType::Q6K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q2K | GgmlType::Q3K => {
            x.chunks_mut(cols).for_each(fake_q8_k)
        }
        GgmlType::F16 => round_f16(x),
        _ => {}
    }
}

/// `out = x @ W^T` with llama.cpp's activation rounding in [`Numerics::LlamaCpp`] mode.
pub fn matmul(w: QuantMat<'_>, x: &[f32], m: usize, out: &mut [f32]) {
    if mode() == Numerics::Exact {
        return matvec_f32(w, x, m, out);
    }
    let mut xq = x.to_vec();
    quantize_activations(w.ty, &mut xq, w.cols);
    matvec_f32(w, &xq, m, out)
}

/// Round a K/V row as the f16 cache stores it (no-op in exact mode).
pub fn kv_store(x: &mut [f32]) {
    if mode() == Numerics::LlamaCpp {
        round_f16(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_keeps_the_block_max_and_snaps_the_rest_to_its_grid() {
        let mut x: Vec<f32> = (0..32).map(|i| i as f32 / 31.0 * 2.0 - 1.0).collect();
        x[5] = 2.54;
        let orig = x.clone();
        fake_q8_0(&mut x);
        let d = f16::from_f32(2.54 / 127.0).to_f32();
        assert!((x[5] - 127.0 * d).abs() < 1e-6);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() <= d * 0.5 + 1e-3);
            assert!(((a / d) - (a / d).round()).abs() < 1e-3);
        }
    }

    #[test]
    fn q8_k_maps_the_signed_max_to_minus_127() {
        let mut x = vec![0.01f32; 256];
        x[3] = -4.0;
        fake_q8_k(&mut x);
        assert!((x[3] + 4.0).abs() < 1e-6);
        let d = 4.0 / 127.0;
        assert!((x[0] - 0.0).abs() < d);
    }

    #[test]
    fn exact_mode_is_a_plain_matvec() {
        set_mode(Numerics::Exact);
        let bytes = lb_gguf::types::quantize_q4_0(
            &(0..64)
                .map(|i| (i as f32 - 32.0) / 10.0)
                .collect::<Vec<_>>(),
        );
        let w = QuantMat::new(GgmlType::Q4_0, 1, 64, &bytes);
        let x: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let (mut a, mut b) = (vec![0f32; 1], vec![0f32; 1]);
        matmul(w, &x, 1, &mut a);
        matvec_f32(w, &x, 1, &mut b);
        assert_eq!(a, b);
        set_mode(Numerics::LlamaCpp);
    }
}
