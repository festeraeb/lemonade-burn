//! Times the int8 CPU expert kernel on one expert-sized Q4_0 matrix, single thread.
//! Prints ms per matvec for the f32 reference dot, the scalar int8 dot and each SIMD level,
//! and `speedup_<level>` = scalar int8 time / that level's time.
//! With `--gate X`, exits 1 unless every SIMD level this CPU has is at least X times the scalar int8 kernel.
use lb_gguf::kernels::dot_q4_0;
use lb_gguf::kernels_i8::{dot_q4_0_i8_at, quantize_act_q8, simd_level, SimdLevel};
use lb_gguf::types::quantize_q4_0;
use std::time::Instant;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn best_ms(mut f: impl FnMut() -> f32, reps: usize) -> (f64, f32) {
    let (mut best, mut sink) = (f64::MAX, 0f32);
    for _ in 0..reps {
        let t = Instant::now();
        sink += f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, sink)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gate: Option<f64> = args.iter().position(|a| a == "--gate").map(|i| args[i + 1].parse().expect("--gate X"));
    // An LFM2.5-24B expert is about 5.3 MB of Q4_0: 4608 rows of 2048 columns is 5.3 MB.
    let (rows, cols) = (4608usize, 2048usize);
    let mut s = 1u64;
    let bytes: Vec<u8> = (0..rows).flat_map(|_| quantize_q4_0(&(0..cols).map(|_| lcg(&mut s)).collect::<Vec<_>>())).collect();
    let x: Vec<f32> = (0..cols).map(|_| lcg(&mut s)).collect();
    let a = quantize_act_q8(&x);
    let rb = cols / 32 * 18;
    let row = |r: usize| &bytes[r * rb..(r + 1) * rb];
    let reps = 15;
    let (f32_ms, _) = best_ms(|| (0..rows).map(|r| dot_q4_0(row(r), &x)).sum(), reps);
    let (sc_ms, sc_sum) = best_ms(|| (0..rows).map(|r| dot_q4_0_i8_at(SimdLevel::Scalar, row(r), &a)).sum(), reps);
    println!("matrix {rows}x{cols} Q4_0 ({:.2} MB), single thread, best of {reps}", bytes.len() as f64 / 1e6);
    println!("f32_reference_ms {f32_ms:.3}");
    println!("i8_scalar_ms {sc_ms:.3}");
    let levels: &[SimdLevel] = match simd_level() {
        SimdLevel::Avx512 => &[SimdLevel::Avx2, SimdLevel::Avx512],
        SimdLevel::Avx2 => &[SimdLevel::Avx2],
        SimdLevel::Scalar => &[],
    };
    let mut ok = true;
    for &lv in levels {
        let (ms, sum) = best_ms(|| (0..rows).map(|r| dot_q4_0_i8_at(lv, row(r), &a)).sum(), reps);
        let sp = sc_ms / ms;
        println!("i8_{lv:?}_ms {ms:.3}  speedup_{lv:?} {sp:.2}  checksum_rel_diff {:.2e}", ((sum - sc_sum) / sc_sum).abs());
        if let Some(g) = gate {
            if sp < g {
                println!("GATE FAIL: {lv:?} is {sp:.2}x the scalar int8 kernel, need {g}x");
                ok = false;
            }
        }
    }
    if levels.is_empty() {
        println!("no SIMD level on this CPU");
    }
    if !ok {
        std::process::exit(1);
    }
    if gate.is_some() {
        println!("GATE PASS");
    }
}
