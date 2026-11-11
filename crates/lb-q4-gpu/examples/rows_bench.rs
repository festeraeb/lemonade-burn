//! D2 A/B: the one-row Q4_0 kernels against R rows per plane (4, 8) and x in shared memory, in
//! one process on one upload, arms interleaved round-robin so drift hits every arm alike.
//!
//! Headline is `distinct24_m1` as `lb-gpu-bench` measures it: 24 distinct resident experts
//! (5.3 MB each, no cache reuse between them), m = 1, one upload, 24 launches, one readback;
//! per-expert ms and the weight stream in GB/s. Also each dense shape of the 24B on its own
//! (launch + readback, m = 1), as GB/s of its weights, and the same shape as a chain of
//! `LB_BENCH_CHAIN` (default 16) launches with one readback (`*_chain_ms` per launch): a single
//! launch + readback on wgpu is mostly round-trip latency, the chain is mostly the kernel's
//! own streaming rate, which is what the roofline comparison wants. A warm-up of ~2 s of
//! launches runs first so the card is at its clocks. Prints one JSON line per (arm, metric)
//! with the median over `LB_BENCH_REPS` (default 30) after a warm-up, per round.
//!
//! `LB_BENCH_ROUNDS` (default 3) rounds; card: `LB_WGPU_DEVICE` or `LB_CUDA_DEVICE`.
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert, GpuQ4Mat, GpuQ8Mat, Q4Math};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn q4(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<(), String> {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (reps, rounds, chain) = (env("LB_BENCH_REPS", 30), env("LB_BENCH_ROUNDS", 3), env("LB_BENCH_CHAIN", 16));
    let gpu = Gpu::open()?;
    let load = || std::fs::read_to_string("/proc/loadavg").unwrap_or_default().split_whitespace().next().unwrap_or("").to_string();
    println!("{{\"cc\":\"{:?}\",\"device\":{:?},\"cube_dim\":{},\"use_plane\":{},\"plane\":{:?},\"xshared_max_cols\":{}}}", gpu.compute_capability, gpu.name, gpu.cube_dim, gpu.use_plane, gpu.plane_size, gpu.xshared_max_cols);
    let (n_embd, n_ff) = (2048usize, 1536usize);
    let mut rng = Rng(3);
    let mut es = Vec::new();
    for _ in 0..24 {
        let (g, u, d) = (q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_embd, n_ff));
        es.push(GpuExpert::upload(&gpu, QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &g), QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &u), QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &d))?);
    }
    let expert_bytes = es[0].bytes_on_device() as f64;
    // `big_65536x2048` (75 MB, X7a): one launch long enough (~120 us at HBM2 rate) that the
    // per-launch host cost (~28 us through CubeCL on c4) no longer hides the kernel's own rate
    let mut shapes = vec![("in_proj_6144x2048", 6144usize, 2048usize), ("out_proj_2048x2048", 2048, 2048), ("kv_512x2048", 512, 2048), ("ffn_down_2048x11776", 2048, 11776), ("ffn_up_11776x2048", 11776, 2048)];
    if std::env::var_os("LB_BENCH_MATHS").is_some() {
        shapes.push(("big_65536x2048", 65536, 2048));
    }
    let dense: Vec<(&str, GpuQ4Mat)> = shapes
        .iter()
        .map(|&(n, r, c)| (n, GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, r, c, &q4(&mut rng, r, c))).expect("upload")))
        .collect();
    let x: Vec<f32> = (0..11776).map(|_| rng.unit()).collect();
    // X7a: `LB_BENCH_MATHS=f32,f32m,f16,dp4a` replaces the D2 arms with rows4 (and rows4_xs) per math
    let maths: Vec<Q4Math> = std::env::var("LB_BENCH_MATHS").ok().map(|v| v.split(',').filter_map(Q4Math::parse).collect()).unwrap_or_default();
    let mut arms: Vec<(String, usize, bool, Q4Math)> = Vec::new();
    if maths.is_empty() {
        for (a, r, xs) in [("rows1", 1, false), ("rows4", 4, false), ("rows8", 8, false), ("rows4_xs", 4, true), ("rows8_xs", 8, true)] {
            arms.push((a.to_string(), r, xs, Q4Math::F32));
        }
    } else {
        for &m in &maths {
            arms.push((format!("rows4_{}", m.name()), 4, false, m));
            arms.push((format!("rows4_xs_{}", m.name()), 4, true, m));
        }
    }
    let time = |f: &mut dyn FnMut() -> Result<(), String>| -> Result<f64, String> {
        f()?;
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            f()?;
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(median(ts))
    };
    // warm-up: the card idles at its lowest clocks; ~2 s of work before anything is timed
    {
        let xh = gpu.upload_f32(&x[..n_embd]);
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < 2.0 {
            let hs: Vec<_> = es.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
            gpu.read_all(hs)?;
        }
    }
    // X7b follow-up, `LB_BENCH_Q8=1`: the same weights as Q4_0 (rows 4, f32) and as Q8_0 (the
    // transcode target that keeps quality), as chains of `chain` launches with one readback
    if std::env::var_os("LB_BENCH_Q8").is_some() {
        gpu.set_q4_rows(4);
        gpu.set_q4_xshared(false);
        gpu.set_q4_math(Q4Math::F32);
        for &(name, r, c) in &[("expert_1536x2048", 1536usize, 2048usize), ("expert_2048x1536", 2048, 1536), ("big_65536x2048", 65536, 2048)] {
            let wf: Vec<f32> = (0..r * c).map(|_| rng.unit()).collect();
            let q4b: Vec<u8> = wf.chunks_exact(c).flat_map(|row| quantize_q4_0(row)).collect();
            let q8b: Vec<u8> = wf.chunks_exact(c).flat_map(|row| lb_gguf::types::quantize_q8_0(row)).collect();
            let m4 = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, r, c, &q4b))?;
            let m8 = GpuQ8Mat::upload(&gpu, QuantMat::new(GgmlType::Q8_0, r, c, &q8b))?;
            let xh = gpu.upload_f32(&x[..c]);
            for round in 0..rounds {
                let t4 = time(&mut || { let hs: Vec<_> = (0..chain).map(|_| (m4.launch(&gpu, &xh, 1), r)).collect(); gpu.read_all(hs)?; Ok(()) })? / chain as f64;
                let t8 = time(&mut || { let hs: Vec<_> = (0..chain).map(|_| (m8.launch(&gpu, &xh, 1), r)).collect(); gpu.read_all(hs)?; Ok(()) })? / chain as f64;
                println!("{{\"round\":{round},\"arm\":\"q4_0_f32\",\"metric\":\"{name}_chain_ms\",\"value\":{t4:.4},\"gb_s\":{:.1}}}", m4.bytes_on_device() as f64 / 1e9 / (t4 / 1e3));
                println!("{{\"round\":{round},\"arm\":\"q8_0_f32\",\"metric\":\"{name}_chain_ms\",\"value\":{t8:.4},\"gb_s\":{:.1}}}", m8.bytes_on_device() as f64 / 1e9 / (t8 / 1e3));
            }
        }
        return Ok(());
    }
    for round in 0..rounds {
        for (arm, r, xs, math) in &arms {
            let (arm, r, xs) = (arm.as_str(), *r, *xs);
            gpu.set_q4_rows(r);
            gpu.set_q4_xshared(xs);
            gpu.set_q4_math(*math);
            let xh = gpu.upload_f32(&x[..n_embd]);
            let ms = time(&mut || {
                let hs: Vec<_> = es.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
                gpu.read_all(hs)?;
                Ok(())
            })? / 24.0;
            println!("{{\"round\":{round},\"arm\":\"{arm}\",\"metric\":\"distinct24_m1_ms_per_expert\",\"value\":{ms:.4},\"gb_s\":{:.1},\"load\":\"{}\"}}", expert_bytes / 1e9 / (ms / 1e3), load());
            for (name, m) in &dense {
                let xh = gpu.upload_f32(&x[..m.cols]);
                let ms = time(&mut || {
                    let o = m.launch(&gpu, &xh, 1);
                    gpu.read_all(vec![(o, m.rows)])?;
                    Ok(())
                })?;
                println!("{{\"round\":{round},\"arm\":\"{arm}\",\"metric\":\"{name}_ms\",\"value\":{ms:.4},\"gb_s\":{:.1}}}", m.bytes_on_device() as f64 / 1e9 / (ms / 1e3));
                let ms = time(&mut || {
                    let hs: Vec<_> = (0..chain).map(|_| (m.launch(&gpu, &xh, 1), m.rows)).collect();
                    gpu.read_all(hs)?;
                    Ok(())
                })? / chain as f64;
                println!("{{\"round\":{round},\"arm\":\"{arm}\",\"metric\":\"{name}_chain_ms\",\"value\":{ms:.4},\"gb_s\":{:.1}}}", m.bytes_on_device() as f64 / 1e9 / (ms / 1e3));
            }
        }
    }
    Ok(())
}
