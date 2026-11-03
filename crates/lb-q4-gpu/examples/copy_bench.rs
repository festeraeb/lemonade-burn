//! The copy path, measured the way the engine uses it, on whichever runtime the environment
//! selects (`LB_CUDA_DEVICE=N` for cubecl-cuda, `LB_WGPU_DEVICE=discrete:N` for wgpu), so the
//! two runtimes can be compared on the same card from one binary.
//!
//! Blocker B1 of `agentnotes/fable-vs-gemini-engine-compare-2026-11-02.md`: the measured 46x
//! copy loss (264-483 MB/s against 12,260 MB/s raw `cudaMemcpy`) was the wgpu path, and the
//! cubecl-cuda path had never been measured on our cards. Gate: >= 6 GB/s at 5.3 MB blobs.
//!
//! Measurements, each a median of `LB_BENCH_REPS` (default 20) after one warm-up, with min and
//! max printed so a spread cannot hide behind the median:
//!   h2d_create_bytes_{5mb,64mb}  `client.create(Bytes)` of an owned Vec + sync: the engine's
//!                                upload path (`Gpu::upload_u32`), no host copy before the backend
//!   h2d_staged_pinned_{5mb,64mb} the same Vec passed through `Client::staging` first (pooled
//!                                pinned host memory), then `create`: the reused-pinned-staging path
//!   h2d_write_prealloc_{5mb,64mb}        `client.write` into a handle allocated once (no device
//!                                        allocation per call), fresh Vec
//!   h2d_write_prealloc_pinned_{5mb,64mb} the same through the pinned staging pool
//!   h2d_from_slice_{5mb,64mb}    `client.create_from_slice` + sync: the older path (two host
//!                                copies before the backend), kept for comparison with lb-gpu-bench
//!   h2d_expert_upload_mb_s       `GpuExpert::upload` (repack + 3 uploads) minus the measured
//!                                repack, i.e. what serving one miss on-demand pays for the copy
//!   d2h_h2d_act_m{1,4,32}_us     read back an [m, 2048] f32 activation and upload it again: the
//!                                through-host handoff the layer-pipeline would pay per card hop
//! Every blob is freshly allocated per rep (as a miss would be), so allocator and page-fault cost
//! on the host side is inside the number, deliberately.
use cubecl::bytes::Bytes;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert};
use std::time::Instant;

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() -> Result<(), String> {
    let reps: usize = std::env::var("LB_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string();
    let gpu = Gpu::open()?;
    let runtime = if std::env::var("LB_CUDA_DEVICE").is_ok() { "cuda" } else { "wgpu" };
    eprintln!("device: {} runtime={runtime} host={host} reps={reps} plane={:?} use_plane={}", gpu.name, gpu.plane_size, gpu.use_plane);
    let header = format!("\"host\":\"{host}\",\"runtime\":\"{runtime}\",\"device\":\"{}\"", gpu.name.replace('"', "'"));

    // timing helper: one warm-up (allocator, shader compile), then `reps` timed calls
    let time = |f: &mut dyn FnMut() -> Result<(), String>| -> Result<(f64, f64, f64), String> {
        f()?;
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            f()?;
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(stats(ts))
    };
    let mb_s = |bytes: usize, ms: f64| bytes as f64 / 1e6 / (ms / 1e3);

    // one expert = gate + up (1536 x 2048) + down (2048 x 1536), all Q4_0: 3 x 1,769,472 bytes
    let (rows, cols) = (1536usize, 2048usize);
    let nb = rows * (cols / 32);
    let expert_bytes = 3 * nb * 18;
    let sizes = [("5mb", expert_bytes), ("64mb", 64usize << 20)];

    for (label, n) in sizes {
        // the engine's path: an owned Vec<u32> handed to the backend without another host copy
        let (ms, lo, hi) = time(&mut || {
            let v: Vec<u32> = (0..n / 4).map(|i| i as u32).collect();
            let h = gpu.client.create(Bytes::from_elems(v));
            gpu.sync()?;
            drop(h);
            Ok(())
        })?;
        // the Vec fill is inside the timed region; measure it alone so it can be subtracted
        let (fill_ms, _, _) = time(&mut || {
            let v: Vec<u32> = (0..n / 4).map(|i| i as u32).collect();
            std::hint::black_box(v);
            Ok(())
        })?;
        let net = (ms - fill_ms).max(1e-6);
        println!("{{{header},\"metric\":\"h2d_create_bytes_{label}_mb_s\",\"value\":{:.0},\"ms\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"host_fill_ms\":{fill_ms:.3},\"net_ms\":{net:.3},\"bytes\":{n}}}", mb_s(n, net));

        // the same owned Vec, first moved into the runtime's pinned staging pool
        // (`Client::staging`: a host memcpy into pooled pinned memory, after which the
        // backend's `memcpy_htod_async` is a real DMA instead of the driver's own staging)
        let (ms, lo, hi) = time(&mut || {
            let v: Vec<u32> = (0..n / 4).map(|i| i as u32).collect();
            let mut b = Bytes::from_elems(v);
            gpu.client.staging([&mut b].into_iter(), false);
            let h = gpu.client.create(b);
            gpu.sync()?;
            drop(h);
            Ok(())
        })?;
        let net = (ms - fill_ms).max(1e-6);
        println!("{{{header},\"metric\":\"h2d_staged_pinned_{label}_mb_s\",\"value\":{:.0},\"ms\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"host_fill_ms\":{fill_ms:.3},\"net_ms\":{net:.3},\"bytes\":{n}}}", mb_s(n, net));

        // `write` into a handle allocated once: the copy without a device allocation per call,
        // from a fresh Vec and from the pinned staging pool. Separates allocator cost from DMA.
        let dst = gpu.client.empty(n);
        let (ms, lo, hi) = time(&mut || {
            let v: Vec<u32> = (0..n / 4).map(|i| i as u32).collect();
            gpu.client.write(&dst, Bytes::from_elems(v));
            gpu.sync()
        })?;
        let net = (ms - fill_ms).max(1e-6);
        println!("{{{header},\"metric\":\"h2d_write_prealloc_{label}_mb_s\",\"value\":{:.0},\"ms\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"host_fill_ms\":{fill_ms:.3},\"net_ms\":{net:.3},\"bytes\":{n}}}", mb_s(n, net));
        let (ms, lo, hi) = time(&mut || {
            let v: Vec<u32> = (0..n / 4).map(|i| i as u32).collect();
            let mut b = Bytes::from_elems(v);
            gpu.client.staging([&mut b].into_iter(), false);
            gpu.client.write(&dst, b);
            gpu.sync()
        })?;
        let net = (ms - fill_ms).max(1e-6);
        println!("{{{header},\"metric\":\"h2d_write_prealloc_pinned_{label}_mb_s\",\"value\":{:.0},\"ms\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"host_fill_ms\":{fill_ms:.3},\"net_ms\":{net:.3},\"bytes\":{n}}}", mb_s(n, net));
        drop(dst);

        let blob: Vec<u8> = (0..n).map(|i| i as u8).collect();
        let (ms, lo, hi) = time(&mut || {
            let h = gpu.client.create_from_slice(&blob);
            gpu.sync()?;
            drop(h);
            Ok(())
        })?;
        println!("{{{header},\"metric\":\"h2d_from_slice_{label}_mb_s\",\"value\":{:.0},\"ms\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"bytes\":{n}}}", mb_s(n, ms));
    }

    // a whole expert as a miss would upload it: repack on the host, then three uploads
    {
        let bytes: Vec<u8> = (0..nb * 18).map(|i| (i % 251) as u8).collect();
        let mat = || QuantMat::new(GgmlType::Q4_0, rows, cols, &bytes);
        let dmat = || QuantMat::new(GgmlType::Q4_0, cols, rows, &bytes);
        let (repack, _, _) = time(&mut || {
            std::hint::black_box(lb_q4_gpu::repack_q4_0(&bytes, nb));
            Ok(())
        })?;
        let (whole, lo, hi) = time(&mut || {
            let ex = GpuExpert::upload(&gpu, mat(), mat(), dmat())?;
            gpu.sync()?;
            drop(ex);
            Ok(())
        })?;
        let copy_ms = (whole - 3.0 * repack).max(1e-6);
        println!("{{{header},\"metric\":\"h2d_expert_upload_mb_s\",\"value\":{:.0},\"whole_ms\":{whole:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"repack_ms_x3\":{:.3},\"copy_ms\":{copy_ms:.3},\"bytes\":{expert_bytes}}}", mb_s(expert_bytes, copy_ms), 3.0 * repack);
    }

    // activation handoff through the host
    let n_embd = 2048usize;
    for m in [1usize, 4, 32] {
        let x: Vec<f32> = (0..m * n_embd).map(|i| (i % 97) as f32 * 0.01).collect();
        let xh = gpu.upload_f32(&x);
        let (ms, lo, hi) = time(&mut || {
            let v = gpu.read_all(vec![(xh.clone(), m * n_embd)])?;
            let h2 = gpu.upload_f32(&v[0]);
            gpu.sync()?;
            drop(h2);
            Ok(())
        })?;
        println!("{{{header},\"metric\":\"d2h_h2d_act_m{m}_us\",\"value\":{:.1},\"min_us\":{:.1},\"max_us\":{:.1},\"bytes_each_way\":{}}}", ms * 1e3, lo * 1e3, hi * 1e3, m * n_embd * 4);
    }
    Ok(())
}
