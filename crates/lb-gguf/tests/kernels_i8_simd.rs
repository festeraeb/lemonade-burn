//! Every SIMD level this CPU has must agree with the scalar int8 kernel.
use lb_gguf::kernels_i8::{dot_q4_0_i8_at, dot_q4_0_i8_scalar, quantize_act_q8, simd_level, ActQ8, SimdLevel};
use lb_gguf::types::quantize_q4_0;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn levels() -> Vec<SimdLevel> {
    match simd_level() {
        SimdLevel::Avx512 => vec![SimdLevel::Avx2, SimdLevel::Avx512],
        SimdLevel::Avx2 => vec![SimdLevel::Avx2],
        SimdLevel::Scalar => vec![],
    }
}

fn check(row: &[u8], a: &ActQ8, what: &str) {
    let want = dot_q4_0_i8_scalar(row, a);
    // Bound on float reordering: 1e-5 of the sum of |block terms|.
    let mut mag = 0f32;
    for (b, blk) in row.chunks_exact(18).enumerate() {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let mut s = 0i32;
        for j in 0..16 {
            s += ((blk[2 + j] & 15) as i32 - 8) * a.q[b * 32 + j] as i32;
            s += ((blk[2 + j] >> 4) as i32 - 8) * a.q[b * 32 + j + 16] as i32;
        }
        mag += (d * a.scale[b] * s as f32).abs();
    }
    for lv in levels() {
        let got = dot_q4_0_i8_at(lv, row, a);
        assert!((got - want).abs() <= 1e-5 * mag + 1e-12, "{what} {lv:?}: {got} vs scalar {want} (mag {mag})");
    }
}

#[test]
fn random_rows_of_many_widths() {
    let mut s = 21u64;
    for &cols in &[32usize, 64, 96, 160, 224, 256, 1024, 2048, 2080, 8192] {
        for _ in 0..8 {
            let w: Vec<f32> = (0..cols).map(|_| lcg(&mut s) * 2.0).collect();
            let x: Vec<f32> = (0..cols).map(|_| lcg(&mut s) * 5.0).collect();
            check(&quantize_q4_0(&w), &quantize_act_q8(&x), &format!("cols {cols}"));
        }
    }
}

#[test]
fn extreme_nibbles_and_activations() {
    // All nibbles 0 (value -8) or 15 (value +7) against activations pinned at +-127:
    // the largest block sums there are (32 * 8 * 127 = 32512).
    for &cols in &[32usize, 96, 2048] {
        for &nib in &[0x00u8, 0xff, 0x0f, 0xf0] {
            for &qv in &[127i8, -127, 1, 0] {
                let mut row = Vec::new();
                for b in 0..cols / 32 {
                    row.extend_from_slice(&half::f16::from_f32(0.01 + b as f32 * 1e-4).to_le_bytes());
                    row.extend_from_slice(&[nib; 16]);
                }
                let a = ActQ8 { q: vec![qv; cols], scale: vec![0.03; cols / 32] };
                check(&row, &a, &format!("cols {cols} nib {nib:#x} q {qv}"));
            }
        }
    }
}

#[test]
fn alternating_signs_do_not_saturate() {
    let cols = 2048;
    let mut row = Vec::new();
    for b in 0..cols / 32 {
        row.extend_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
        for j in 0..16 {
            row.push(if (b + j) % 2 == 0 { 0xf0 } else { 0x0f });
        }
    }
    let q: Vec<i8> = (0..cols).map(|c| if c % 3 == 0 { 127 } else { -127 }).collect();
    check(&row, &ActQ8 { q, scale: vec![0.5; cols / 32] }, "alternating");
}
