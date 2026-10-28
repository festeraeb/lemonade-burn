//! Simple timed microbenchmark (no criterion): `cargo bench -p lb-ternary --bench kbench`.
//! Env: `KB_QUANT=1` (quantizer probe), `KB_STREAM=1` (DRAM read ceiling), `KB_THREADS` (comma list, default "1,<all>"), `KB_ITERS` (default auto ~0.3 s).
//! Reports per-call time, weight bandwidth (packed bytes / time) and int8 GOPS.

use lb_ternary::pack::ternarize_absmean;
use lb_ternary::{quantize_acts, ternary_gemm, Kernel, PackedTernary, TernaryView};
use std::time::Instant;

fn main() {
    let all = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads: Vec<usize> = std::env::var("KB_THREADS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, all]);
    // BitNet 2B4T shapes: (rows=out, cols=in)
    let shapes = [("qo 2560x2560", 2560, 2560), ("kv 640x2560", 640, 2560), ("gate/up 6912x2560", 6912, 2560), ("down 2560x6912", 2560, 6912)];
    if std::env::var("KB_QUANT").is_ok() {
        quant_probe();
        return;
    }
    if std::env::var("KB_MODEL").is_ok() {
        model_decode(&threads);
        return;
    }
    if std::env::var("KB_STREAM").is_ok() {
        stream(&threads);
        return;
    }
    println!("cpu kernels available: {:?}", Kernel::available().iter().map(|k| k.name()).collect::<Vec<_>>());
    for &nt in &threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        pool.install(|| {
            for &(label, rows, cols) in &shapes {
                let mut s = 0x9e3779b97f4a7c15u64;
                let w: Vec<f32> = (0..rows * cols)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .collect();
                let (t, sc) = ternarize_absmean(&w);
                let p = PackedTernary::from_ternary(rows, cols, &t, sc);
                for m in [1usize, 16, 64] {
                    let x: Vec<f32> = (0..m * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
                    let qa = quantize_acts(&x, m, cols);
                    let mut out = vec![0f32; m * rows];
                    for k in Kernel::available() {
                        ternary_gemm(p.view(), &qa, k, &mut out); // warm
                        let t0 = Instant::now();
                        let mut iters = 0;
                        while t0.elapsed().as_secs_f64() < 0.3 {
                            ternary_gemm(p.view(), &qa, k, &mut out);
                            iters += 1;
                        }
                        let dt = t0.elapsed().as_secs_f64() / iters as f64;
                        println!(
                            "threads {nt:>2} | {label:<18} m={m:<2} | {:<10} | {:>9.1} us/call | {:>6.2} GB/s weights | {:>7.1} GOPS",
                            k.name(),
                            dt * 1e6,
                            p.data.len() as f64 / dt / 1e9,
                            2.0 * (rows * cols * m) as f64 / dt / 1e9
                        );
                    }
                }
            }
        });
    }
}

/// Decode-like streaming: one 520 MB packed matrix (same bytes as all BitNet 2B4T
/// ternary layers) read once per GEMV, so it cannot sit in cache. Also reports a plain
/// parallel u64 sum over the same buffer as the DRAM read ceiling.
fn stream(threads: &[usize]) {
    use rayon::prelude::*;
    let cols = 2560;
    let rows = 520_000_000 / (cols / 4);
    let t: Vec<i8> = (0..rows * cols).map(|i| ((i * 2654435761usize) >> 7) as i8 % 2).collect();
    let p = PackedTernary::from_ternary(rows, cols, &t, 1.0);
    drop(t);
    let x: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let qa = quantize_acts(&x, 1, cols);
    let mut out = vec![0f32; rows];
    let words: &[u64] = unsafe { std::slice::from_raw_parts(p.data.as_ptr() as *const u64, p.data.len() / 8) };
    if let Some(np) = lb_ternary::numa::NodePools::global() {
        // two-socket split: rows [0, s) placed on node 0, [s, rows) on node 1
        let s = np.split_rows(rows);
        let rb = p.view().row_bytes();
        let d0 = np.place(0, &p.data[..s * rb]);
        let d1 = np.place(1, &p.data[s * rb..]);
        let (v0, v1) = (TernaryView::new(s, cols, &d0, 1.0), TernaryView::new(rows - s, cols, &d1, 1.0));
        let (o0, o1) = out.split_at_mut(s);
        for k in Kernel::available().into_iter().filter(|k| *k != Kernel::Scalar) {
            for rep in 0..4 {
                let t0 = Instant::now();
                for _ in 0..3 {
                    np.join(|| ternary_gemm(v0, &qa, k, o0), || ternary_gemm(v1, &qa, k, o1));
                }
                let dt = t0.elapsed().as_secs_f64() / 3.0;
                println!("numa share {:.2} rep {rep} | 520MB gemv {:<10} | {:>7.2} GB/s  {:>6.1} ms/token-equivalent", np.share, k.name(), p.data.len() as f64 / dt / 1e9, dt * 1e3);
            }
        }
        return;
    }
    for &nt in threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        pool.install(|| {
            let t0 = Instant::now();
            let mut s = 0u64;
            for _ in 0..3 {
                s = s.wrapping_add(words.par_chunks(1 << 16).map(|c| c.iter().fold(0u64, |a, &b| a ^ b)).reduce(|| 0, |a, b| a ^ b));
            }
            let dt = t0.elapsed().as_secs_f64() / 3.0;
            println!("threads {nt:>2} | read-only xor sweep      | {:>7.2} GB/s ({s:x})", p.data.len() as f64 / dt / 1e9);
            for k in Kernel::available().into_iter().filter(|k| *k != Kernel::Scalar) {
                ternary_gemm(p.view(), &qa, k, &mut out);
                let t0 = Instant::now();
                for _ in 0..3 {
                    ternary_gemm(p.view(), &qa, k, &mut out);
                }
                let dt = t0.elapsed().as_secs_f64() / 3.0;
                println!("threads {nt:>2} | 520MB gemv {:<10}    | {:>7.2} GB/s  {:>6.1} ms/token-equivalent", k.name(), p.data.len() as f64 / dt / 1e9, dt * 1e3);
            }
        });
    }
}

/// `KB_QUANT=1`: single-thread activation quantizer speed, scalar vs dispatched.
fn quant_probe() {
    let k = 2560;
    let x: Vec<f32> = (0..k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let mut q = vec![0i8; k];
    let n = 2000;
    for (name, f) in [
        ("scalar", lb_ternary::quant::quantize_row_scalar as fn(&[f32], &mut [i8]) -> (f32, i32)),
        ("dispatched", lb_ternary::quant::quantize_row),
    ] {
        let t0 = Instant::now();
        let mut s = 0i32;
        for _ in 0..n {
            s = s.wrapping_add(f(std::hint::black_box(&x), &mut q).1);
        }
        println!("quantize_row {name:<10} k={k}: {:.2} us/row ({s})", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
    }
}

/// `KB_MODEL=1`: decode-like sequence over 30 layers of BitNet 2B4T-shaped matrices
/// (distinct, cold weights, 520 MB), grouped like the model (q/k/v fused, o, gate/up
/// fused, down = 120 parallel regions per token), with the activation quantization in
/// between. Reports ms per token-equivalent and effective weight bandwidth.
fn model_decode(threads: &[usize]) {
    use lb_ternary::ternary_gemm_multi;
    let shapes: [&[(usize, usize)]; 4] = [&[(2560, 2560), (640, 2560), (640, 2560)], &[(2560, 2560)], &[(6912, 2560), (6912, 2560)], &[(2560, 6912)]];
    let mut layers: Vec<Vec<Vec<PackedTernary>>> = Vec::new();
    let mut bytes = 0usize;
    for l in 0..30 {
        let mut groups = Vec::new();
        for g in shapes.iter() {
            let mats: Vec<PackedTernary> = g
                .iter()
                .map(|&(rows, cols)| {
                    let t: Vec<i8> = (0..rows * cols).map(|i| (((i + l * 7) * 2654435761usize) >> 13) as i8 % 2).collect();
                    let p = PackedTernary::from_ternary(rows, cols, &t, 1.0);
                    bytes += p.data.len();
                    p
                })
                .collect();
            groups.push(mats);
        }
        layers.push(groups);
    }
    let x6912: Vec<f32> = (0..6912).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let k = Kernel::detect();
    for &nt in threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        pool.install(|| {
            for rep in 0..5 {
                let t0 = Instant::now();
                for l in &layers {
                    for g in l {
                        let cols = g[0].cols;
                        let qa = quantize_acts(&x6912[..cols], 1, cols);
                        let views: Vec<_> = g.iter().map(|p| p.view()).collect();
                        let mut bufs: Vec<Vec<f32>> = g.iter().map(|p| vec![0f32; p.rows]).collect();
                        let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
                        ternary_gemm_multi(&views, &qa, k, &mut outs);
                    }
                }
                let dt = t0.elapsed().as_secs_f64();
                println!("threads {nt:>2} rep {rep} | model-like decode, 120 regions | {:>6.2} ms/token | {:>6.2} GB/s", dt * 1e3, bytes as f64 / dt / 1e9);
            }
        });
    }
}
