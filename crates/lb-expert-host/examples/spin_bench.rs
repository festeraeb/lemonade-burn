//! Times the CPU-miss batch (`spin::swiglu_batch`) on synthetic LFM2-sized experts (5.3 MB each),
//! rotating through 96 experts so every call reads cold weights, with a gap between calls like a
//! layer's device time. `spin_bench [threads] [gap_us]`; prints ms per call for 1, 2, 4 experts.
use lb_expert_host::spin::{swiglu_batch, SpinPool};
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use std::time::{Duration, Instant};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let threads: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
    let gap: u64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(400);
    let (n_ff, n_embd) = (1536usize, 2048usize);
    let mut s = 1u64;
    let mut mk = |rows: usize, cols: usize| -> Vec<u8> { (0..rows).flat_map(|_| quantize_q4_0(&(0..cols).map(|_| lcg(&mut s)).collect::<Vec<_>>())).collect() };
    let base: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..4).map(|_| (mk(n_ff, n_embd), mk(n_ff, n_embd), mk(n_embd, n_ff))).collect();
    // 96 experts: 24 copies of each distinct set, as distinct heap blocks so they do not share cache lines
    let sets: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..96).map(|i| base[i % 4].clone()).collect();
    let x: Vec<f32> = (0..n_embd).map(|_| lcg(&mut 7u64.clone()) ).collect();
    let pool = SpinPool::new(threads, Duration::from_millis(20));
    println!("threads {threads}, gap {gap} us, expert {:.2} MB", (sets[0].0.len() * 3) as f64 / 1e6);
    for int8 in [false, true, false] {
        for k in [1usize, 2, 4] {
            let mut tot = Duration::ZERO;
            let calls = 300;
            for c in 0..calls {
                let mats: Vec<_> = (0..k)
                    .map(|j| {
                        let e = &sets[(c * k + j) % 96];
                        (QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &e.0), QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &e.1), QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &e.2))
                    })
                    .collect();
                std::thread::sleep(Duration::from_micros(gap));
                let t = Instant::now();
                let o = swiglu_batch(&pool, &mats, &x, int8);
                tot += t.elapsed();
                std::hint::black_box(o);
            }
            let ms = tot.as_secs_f64() * 1e3 / calls as f64;
            println!("{} experts {k}: {ms:.3} ms per call, {:.3} ms per expert, {:.1} GB/s", if int8 { "int8 " } else { "exact" }, ms / k as f64, k as f64 * 5.31e6 / (ms * 1e-3) / 1e9);
        }
    }
}
