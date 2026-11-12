//! Where an on-demand expert miss spends its time, on one card, steady state (cuda feature).
//! Two ways to serve the same miss, same binary, A/B by a runtime switch:
//!   host:   repack on the host into the page-locked pool, three `create`s DMA the packed words
//!   device: the file mapping is page-locked; three `create`s DMA the raw bytes, the device repacks
//! plus the stages of each (medians of `LB_BENCH_REPS` after `LB_BENCH_WARM`, min/max shown):
//!   whole            `GpuExpert::upload` + sync (what `moe.rs` pays per miss, minus the launch)
//!   dma_only         the `create`s of the expert's bytes + sync: three raw slices (device), or six
//!                    pool buffers filled before the clock (host)
//!   host_repack_only three `repack_q4_0_into` into pool buffers, no upload
//! Needs `LB_MODEL` (a Q4_0 GGUF; slices are carved out of its largest Q4_0 tensor, which is
//! what a real miss uploads) and `LB_CUDA_DEVICE`. `LB_NO_REGISTER=1` skips the page-lock.
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert};
use std::time::Instant;

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() -> Result<(), String> {
    let reps: usize = std::env::var("LB_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let warm: usize = std::env::var("LB_BENCH_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let model = std::env::var("LB_MODEL").map_err(|_| "LB_MODEL=<gguf>")?;
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&model)).map_err(|e| format!("{e}"))?;
    let gpu = Gpu::open()?;
    let (rows, cols) = (1536usize, 2048usize);
    let nb = rows * cols / 32;
    let mat_bytes = nb * 18;
    // the largest Q4_0 tensor: expert slices are carved from it at block-aligned offsets
    let big = g.tensors.iter().filter(|t| t.ty == GgmlType::Q4_0).max_by_key(|t| t.n_bytes().unwrap_or(0)).ok_or("no Q4_0 tensor")?;
    let region = g.tensor_bytes(big);
    assert!(region.len() > 4 * mat_bytes, "tensor too small");
    let mut reg_ms = 0.0;
    if std::env::var_os("LB_NO_REGISTER").is_none() {
        let t = Instant::now();
        // SAFETY: `g` outlives `gpu` (dropped at the end of main, after it).
        let n = unsafe { gpu.register_mapping(g.mapping())? };
        reg_ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("registered {:.2} GB in {reg_ms:.0} ms", n as f64 / 1e9);
    }
    let registered = gpu.is_registered(&region[..mat_bytes]);
    let header = format!("\"host\":\"{host}\",\"device\":\"{}\",\"registered\":{registered},\"register_ms\":{reg_ms:.0},\"pinned_pool\":{},\"warm\":{warm},\"reps\":{reps}", gpu.name.replace('"', "'"), gpu.has_pinned());
    // a different expert each rep (as misses are), block aligned, spread over the tensor
    let slice = |i: usize| -> &[u8] {
        let o = (i * 7_919 * 18) % (region.len() - mat_bytes) / 18 * 18;
        &region[o..o + mat_bytes]
    };
    // `f` returns `Some(ms)` to report a span it clocked itself (work before it stays out)
    let mut time = |label: &str, f: &mut dyn FnMut(usize) -> Result<Option<f64>, String>| -> Result<f64, String> {
        let mut v = Vec::with_capacity(reps);
        for i in 0..warm + reps {
            let t = Instant::now();
            let inner = f(i)?;
            if i >= warm {
                v.push(inner.unwrap_or_else(|| t.elapsed().as_secs_f64() * 1e3));
            }
        }
        let (med, lo, hi) = stats(v);
        println!("{{{header},\"metric\":\"{label}\",\"median_ms\":{med:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"expert_mb_s\":{:.0}}}", 3.0 * mat_bytes as f64 / med / 1e3);
        Ok(med)
    };
    let expert = |i: usize| (QuantMat::new(GgmlType::Q4_0, rows, cols, slice(3 * i)), QuantMat::new(GgmlType::Q4_0, rows, cols, slice(3 * i + 1)), QuantMat::new(GgmlType::Q4_0, cols, rows, slice(3 * i + 2)));

    for (path, host_repack) in [("device", false), ("host", true), ("device_rep2", false), ("host_rep2", true)] {
        if host_repack || registered {
            gpu.set_host_repack(host_repack);
            eprintln!("section: {path}");
            time(&format!("{path}_whole"), &mut |i| {
                let (a, b, c) = expert(i);
                let ex = GpuExpert::upload(&gpu, a, b, c)?;
                gpu.sync()?;
                drop(ex);
                Ok(None)
            })?;
        }
    }
    gpu.set_host_repack(false);
    if registered {
        // DMA alone: raw bytes from the mapping, no kernel
        time("device_dma_only", &mut |i| {
            let hs: Vec<_> = (0..3).map(|k| gpu.client.create(lb_q4_gpu::registered::RegisteredSlice::bytes(gpu.registration_of(slice(3 * i + k)).expect("registered"), slice(3 * i + k)))).collect();
            gpu.sync()?;
            drop(hs);
            Ok(None)
        })?;
    }
    if gpu.has_pinned() {
        // host repack alone, into pool buffers that go straight back (no upload)
        time("host_repack_only", &mut |i| {
            for k in 0..3 {
                let (mut qs, mut sc) = (gpu.take_pinned(nb * 4).ok_or("pool")?, gpu.take_pinned(nb / 2).ok_or("pool")?);
                lb_q4_gpu::repack_q4_0_into(slice(3 * i + k), nb, qs.as_mut_u32(), sc.as_mut_u32());
            }
            Ok(None)
        })?;
        // DMA alone from pool buffers filled before the clock
        time("host_dma_only", &mut |i| {
            let mut bufs = Vec::new();
            for k in 0..3 {
                let (mut qs, mut sc) = (gpu.take_pinned(nb * 4).ok_or("pool")?, gpu.take_pinned(nb / 2).ok_or("pool")?);
                lb_q4_gpu::repack_q4_0_into(slice(3 * i + k), nb, qs.as_mut_u32(), sc.as_mut_u32());
                bufs.push((qs, sc));
            }
            let t = Instant::now();
            let hs: Vec<_> = bufs.into_iter().flat_map(|(q, s)| [gpu.upload_pinned(q), gpu.upload_pinned(s)]).collect();
            gpu.sync()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            drop(hs);
            Ok(Some(ms))
        })?;
    }
    eprintln!("pinned pool: {} MB page-locked; registered: {} MB", gpu.pinned_allocated() >> 20, gpu.registered_bytes() >> 20);
    Ok(())
}
