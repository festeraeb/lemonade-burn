//! Device vs CPU reference for every layer of a LittleBit sidecar: max |diff| relative to the output
//! scale, and the time per call for m = 1 (decode) and m = 64 (a prompt chunk).
//! `LB_CUDA_DEVICE=0 cargo run --release --features cuda --example littlebit_parity -- m.lb.safetensors`
use lb_q4_gpu::{littlebit::GpuLittleBit, Gpu};
use std::time::Instant;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: littlebit_parity <sidecar.safetensors>")?;
    let layers = lb_littlebit::sidecar::read(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let gpu = Gpu::open()?;
    println!("device: {} (plane ops {}, cube {})", gpu.name, gpu.use_plane, gpu.cube_dim);
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let (mut worst, mut t_gpu1, mut t_cpu1, mut t_gpu64, mut bytes) = (0f32, 0f64, 0f64, 0f64, 0usize);
    for (name, l) in &layers {
        let g = GpuLittleBit::upload(&gpu, l)?;
        bytes += g.bytes_on_device();
        for &m in &[1usize, 64] {
            let x: Vec<f32> = (0..m * l.inp).map(|_| rnd()).collect();
            let t = Instant::now();
            let want = l.forward(&x, m);
            let tc = t.elapsed().as_secs_f64();
            let _ = g.matvec(&gpu, &x, m)?; // warm (kernel compile on first use)
            let reps = if m == 1 { 20 } else { 5 };
            let t = Instant::now();
            let mut got = Vec::new();
            for _ in 0..reps {
                got = g.matvec(&gpu, &x, m)?;
            }
            let tg = t.elapsed().as_secs_f64() / reps as f64;
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            let diff = want.iter().zip(&got).fold(0f32, |a, (w, g)| a.max((w - g).abs())) / scale;
            worst = worst.max(diff);
            if m == 1 { t_gpu1 += tg; t_cpu1 += tc; } else { t_gpu64 += tg; }
            if diff > 1e-4 {
                println!("MISMATCH {name} m {m}: max rel diff {diff:.2e}");
            }
        }
    }
    println!("{} layers, {:.1} MiB on the device; worst max|diff|/max|y| = {:.2e}", layers.len(), bytes as f64 / 1048576.0, worst);
    println!("all layers once, m=1: device {:.2} ms (incl. upload+readback per layer), CPU reference {:.2} ms; m=64: device {:.2} ms", 1e3 * t_gpu1, 1e3 * t_cpu1, 1e3 * t_gpu64);
    if worst > 1e-4 { Err("parity failed".into()) } else { Ok(()) }
}
