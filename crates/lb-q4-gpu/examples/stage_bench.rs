//! Does a prefetch on a second stream overlap the main stream's work on this card? (cuda feature)
//!
//! The expert staging ring (`lb-lfm2` `stage.rs`) issues next-layer expert copies on a side
//! stream (`Gpu::side_stream`) while the main stream runs the current layer. This bench checks
//! the mechanism and sizes it on one card, before any end-to-end number is read:
//!
//!   demand_N        N experts uploaded on the main stream, each launched once, one read: what a
//!                   layer with N misses pays today (`MissPolicy::OnDemand`)
//!   side_dma_N      N experts uploaded on the side stream + sync: the bus alone
//!   compute         C resident-expert launches + one read on the main stream: stand-in for the
//!                   work a prefetch has to hide behind
//!   compute_with_N  the same C launches + read, with N side-stream uploads issued first: equal to
//!                   `compute` when the copies overlap (the read does not wait for the side stream)
//!   consume_N       after `compute_with_N`, launch the N prefetched experts on the main stream + one
//!                   read: ~the kernels alone if the copies landed during `compute`, more if late
//!
//! Medians of `LB_BENCH_REPS` after `LB_BENCH_WARM`; `LB_STAGE_N` (default 4) and
//! `LB_STAGE_C` (default 16) set N and C. Needs `LB_MODEL` (Q4_0 GGUF) and `LB_CUDA_DEVICE`.
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert};
use std::time::Instant;

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() -> Result<(), String> {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (reps, warm, n, c) = (env("LB_BENCH_REPS", 30), env("LB_BENCH_WARM", 20), env("LB_STAGE_N", 4), env("LB_STAGE_C", 16));
    let model = std::env::var("LB_MODEL").map_err(|_| "LB_MODEL=<gguf>")?;
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&model)).map_err(|e| format!("{e}"))?;
    let gpu = Gpu::open()?;
    // SAFETY: `g` outlives `gpu` (dropped at the end of main, after it).
    let reg = unsafe { gpu.register_mapping(g.mapping())? };
    eprintln!("registered {:.2} GB", reg as f64 / 1e9);
    let (rows, cols) = (1536usize, 2048usize);
    let mat_bytes = rows * cols / 32 * 18;
    let big = g.tensors.iter().filter(|t| t.ty == GgmlType::Q4_0).max_by_key(|t| t.n_bytes().unwrap_or(0)).ok_or("no Q4_0 tensor")?;
    let region = g.tensor_bytes(big);
    let slice = |i: usize| -> &[u8] {
        let o = (i * 7_919 * 18) % (region.len() - mat_bytes) / 18 * 18;
        &region[o..o + mat_bytes]
    };
    let expert = |i: usize| (QuantMat::new(GgmlType::Q4_0, rows, cols, slice(3 * i)), QuantMat::new(GgmlType::Q4_0, rows, cols, slice(3 * i + 1)), QuantMat::new(GgmlType::Q4_0, cols, rows, slice(3 * i + 2)));
    let up = |i: usize| -> Result<GpuExpert, String> {
        let (a, b, d) = expert(i);
        GpuExpert::upload(&gpu, a, b, d)
    };
    let side = gpu.side_stream();
    let x: Vec<f32> = (0..cols).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let xh = gpu.upload_f32(&x);
    // C resident experts for the stand-in compute
    let resident: Vec<GpuExpert> = (0..c.max(1)).map(|i| up(100_000 + i)).collect::<Result<_, _>>()?;
    let header = format!("\"host\":\"{host}\",\"device\":\"{}\",\"n\":{n},\"c\":{c},\"expert_mb\":{:.2},\"reps\":{reps}", gpu.name.replace('"', "'"), 3.0 * mat_bytes as f64 / 1e6);
    let mut rows_out: Vec<(String, Vec<f64>)> = Vec::new();
    let mut push = |k: &str, v: f64| match rows_out.iter_mut().find(|(n, _)| n == k) {
        Some((_, xs)) => xs.push(v),
        None => rows_out.push((k.to_string(), vec![v])),
    };
    // the arms interleave inside each rep, so a drift shows as a spread, not a bias
    let mut parity_checked = false;
    for rep in 0..warm + reps {
        let base = 1 + rep * 4 * n.max(1);
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        // demand
        let t = Instant::now();
        let held: Vec<GpuExpert> = (0..n).map(|i| up(base + i)).collect::<Result<_, _>>()?;
        let outs: Vec<_> = held.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
        gpu.read_all(outs)?;
        let demand = ms(t);
        drop(held);
        // side DMA alone
        let t = Instant::now();
        let held: Vec<GpuExpert> = gpu.on_stream(&side, || (0..n).map(|i| up(base + n + i)).collect::<Result<_, _>>())?;
        gpu.on_stream(&side, || gpu.sync())?;
        let side_dma = ms(t);
        drop(held);
        // compute alone
        let t = Instant::now();
        let outs: Vec<_> = resident.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
        gpu.read_all(outs)?;
        let compute = ms(t);
        // compute with a prefetch in flight, then consume it
        let t = Instant::now();
        let staged: Vec<GpuExpert> = gpu.on_stream(&side, || (0..n).map(|i| up(base + 2 * n + i)).collect::<Result<_, _>>())?;
        let issue = ms(t);
        let outs: Vec<_> = resident.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
        gpu.read_all(outs)?;
        let compute_with = ms(t);
        let t = Instant::now();
        let outs: Vec<_> = staged.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
        let ys = gpu.read_all(outs)?;
        let consume = ms(t);
        if !parity_checked {
            // the staged copy must compute exactly what a main-stream upload of the same bytes does
            let again: Vec<GpuExpert> = (0..n).map(|i| up(base + 2 * n + i)).collect::<Result<_, _>>()?;
            let outs: Vec<_> = again.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
            let ys2 = gpu.read_all(outs)?;
            if ys != ys2 {
                return Err("parity: a side-stream upload computes a different output than a main-stream one".into());
            }
            println!("{{{header},\"metric\":\"parity\",\"side_equals_main\":true}}");
            parity_checked = true;
        }
        drop(staged);
        if rep >= warm {
            push("demand", demand);
            push("side_dma", side_dma);
            push("compute", compute);
            push("issue_side", issue);
            push("compute_with", compute_with);
            push("consume", consume);
        }
    }
    for (k, v) in rows_out {
        let (med, lo, hi) = stats(v);
        let rate = match k.as_str() {
            "demand" | "side_dma" => format!(",\"per_expert_ms\":{:.3},\"gb_s\":{:.2}", med / n.max(1) as f64, n as f64 * 3.0 * mat_bytes as f64 / med / 1e6),
            _ => String::new(),
        };
        println!("{{{header},\"metric\":\"{k}\",\"median_ms\":{med:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3}{rate}}}");
    }
    drop(resident);
    drop(gpu);
    drop(g);
    Ok(())
}
