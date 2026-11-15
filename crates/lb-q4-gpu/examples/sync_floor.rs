//! What one host sync costs through CubeCL, against the raw CUDA floor (`docs/ENGINE_SYNC_COST.md`).
//! Chains of `LAYERS` x (1 "router" launch, a 1.2 KiB read, K more launches), timed per layer for:
//! no read at all; `read_raw_keep` (what the E2 path does); `read_begin` + `read_finish` timed apart.
//! `LB_CUDA_DEVICE` picks the card (driver order). `SF_K` launches after the read (default 8), `SF_E` row width.
use lb_q4_gpu::Gpu;
use std::time::Instant;

fn stat(name: &str, v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    println!("{name:58} mean {mean:8.2}  p50 {:8.2}  p99 {:8.2}  min {:8.2} us", v[v.len() / 2], v[(v.len() as f64 * 0.99) as usize], v[0]);
}

fn main() {
    let dev: usize = std::env::var("LB_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    #[cfg(feature = "cuda")]
    let gpu = if std::env::var("LB_CUDA_DEVICE").is_ok() { Gpu::open_cuda(dev) } else { Gpu::open() }.expect("device");
    #[cfg(not(feature = "cuda"))]
    let gpu = Gpu::open().expect("device");
    let k: usize = std::env::var("SF_K").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let e: usize = std::env::var("SF_E").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let reps: usize = std::env::var("SF_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let layers = 38usize;
    let x = gpu.upload_f32(&vec![0.5f32; e]);
    let w = gpu.upload_f32(&vec![1.0f32; e]);
    for _ in 0..300 {
        let _ = gpu.add_rms_norm(&x, None, &w, e, 1e-5);
    }
    gpu.sync_recycle().unwrap();
    println!("# {} e={e} K={k} layers={layers} reps={reps}", gpu.name);
    // single launch + read round trip
    let mut v = Vec::new();
    for _ in 0..2000 {
        let t = Instant::now();
        let y = gpu.add_rms_norm(&x, None, &w, e, 1e-5);
        let _ = gpu.read_raw_keep(vec![(y, 300 * 4)]).unwrap();
        v.push(t.elapsed().as_secs_f64() * 1e6);
        if v.len() % 300 == 0 { gpu.sync_recycle().unwrap(); }
    }
    stat("1 launch + 1.2 KiB read_raw_keep (round trip)", &mut v);
    let mut v = Vec::new();
    for i in 0..2000 {
        let t = Instant::now();
        let _ = gpu.add_rms_norm(&x, None, &w, e, 1e-5);
        v.push(t.elapsed().as_secs_f64() * 1e6);
        if i % 300 == 299 { gpu.sync_recycle().unwrap(); }
    }
    gpu.sync_recycle().unwrap();
    stat("1 launch call only (client enqueue cost)", &mut v);
    let fe: usize = std::env::var("SF_FLUSH_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    for mode in 0..4 {
        let mut per_layer = Vec::new();
        let (mut begin, mut fin, mut before) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..reps {
            let t0 = Instant::now();
            for _ in 0..layers {
                let tb = Instant::now();
                let r = gpu.add_rms_norm(&x, None, &w, e, 1e-5);
                match mode {
                    1 => {
                        let _ = gpu.read_raw_keep(vec![(r, 300 * 4)]).unwrap();
                    }
                    2 => {
                        before.push(tb.elapsed().as_secs_f64() * 1e6);
                        let t1 = Instant::now();
                        let p = gpu.read_begin(vec![(r, 300 * 4)]);
                        begin.push(t1.elapsed().as_secs_f64() * 1e6);
                        let t2 = Instant::now();
                        let _ = gpu.read_finish(p).unwrap();
                        fin.push(t2.elapsed().as_secs_f64() * 1e6);
                    }
                    3 => {
                        let _ = gpu.read_raw_keep(vec![(r, 300 * 4)]).unwrap();
                    }
                    _ => {}
                }
                for i in 0..k {
                    let _ = gpu.add_rms_norm(&x, None, &w, e, 1e-5);
                    if mode == 3 && i % fe == fe - 1 {
                        gpu.flush_queue();
                    }
                }
            }
            gpu.sync_recycle().unwrap();
            per_layer.push(t0.elapsed().as_secs_f64() * 1e6 / layers as f64);
        }
        let name = ["chain, no per-layer read", "chain, read_raw_keep after the first launch", "chain, read_begin/finish after the first launch", "chain, read_raw_keep + flush_queue every SF_FLUSH_EVERY launches"][mode];
        stat(&format!("{name} (per layer)"), &mut per_layer);
        if mode == 2 {
            stat("   client time before read_begin (launch)", &mut before);
            stat("   read_begin", &mut begin);
            stat("   read_finish", &mut fin);
        }
    }
}
