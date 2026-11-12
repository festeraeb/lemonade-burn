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
//!   h2d_pinned_fill_{5mb,64mb}   `Gpu::upload_u32_with`: the words are written straight into a
//!                                page-locked pool buffer and DMA'd from there (cuda runtime with
//!                                the pool; falls back to a Vec elsewhere or with LB_NO_PINNED=1)
//!   host_fill_pinned_{5mb,64mb}_ms  the same fill written into a pool buffer (cache-cold after
//!                                a DMA), timed alone, so the pinned path's host cost is its own
//!   h2d_pinned_upload_{5mb,64mb} only `create` of an already filled pool buffer + sync: the
//!                                copy path by itself, which is what the >= 6 GB/s gate is about
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
    // untimed calls before the timed ones (default 1). Pools (CubeCL's pinned staging, the
    // engine's `pinned.rs`) only reuse a slot after the drop queue has released it, which lags
    // ~2 x 64 MiB of uploads behind; LB_BENCH_WARM=40 measures that steady state instead of
    // the page-lock-per-call warm-up the default shows.
    let warm: usize = std::env::var("LB_BENCH_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string();
    let gpu = Gpu::open()?;
    let runtime = if std::env::var("LB_CUDA_DEVICE").is_ok() { "cuda" } else { "wgpu" };
    eprintln!("device: {} runtime={runtime} host={host} reps={reps} warm={warm} plane={:?} use_plane={} pinned_pool={}", gpu.name, gpu.plane_size, gpu.use_plane, gpu.has_pinned());
    let header = format!("\"host\":\"{host}\",\"runtime\":\"{runtime}\",\"device\":\"{}\",\"pinned_pool\":{},\"warm\":{warm}", gpu.name.replace('"', "'"), gpu.has_pinned());

    // timing helper: one warm-up (allocator, shader compile), then `reps` timed calls
    let time = |f: &mut dyn FnMut() -> Result<(), String>| -> Result<(f64, f64, f64), String> {
        for _ in 0..warm.max(1) {
            f()?;
        }
        eprintln!("section-timed-start");
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
        eprintln!("section: h2d_create_bytes_{label}");
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
        eprintln!("section: h2d_staged_pinned_{label}");
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
        eprintln!("section: h2d_write_prealloc_{label}");
        let (ms, lo, hi) = time(&mut || {
            let v: Vec<u32> = (0..n / 4).map(|i| i as u32).collect();
            gpu.client.write(&dst, Bytes::from_elems(v));
            gpu.sync()
        })?;
        let net = (ms - fill_ms).max(1e-6);
        println!("{{{header},\"metric\":\"h2d_write_prealloc_{label}_mb_s\",\"value\":{:.0},\"ms\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"host_fill_ms\":{fill_ms:.3},\"net_ms\":{net:.3},\"bytes\":{n}}}", mb_s(n, net));
        eprintln!("section: h2d_write_prealloc_pinned_{label}");
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

        // the words written straight into page-locked memory, then DMA'd (the fix the probe
        // pointed at); same fill as above so `fill_ms` subtracts the same way
        eprintln!("section: h2d_pinned_fill_{label}");
        let (ms, lo, hi) = time(&mut || {
            let h = gpu.upload_u32_with(n / 4, |s| s.iter_mut().enumerate().for_each(|(i, x)| *x = i as u32));
            gpu.sync()?;
            drop(h);
            Ok(())
        })?;
        let net = (ms - fill_ms).max(1e-6);
        println!("{{{header},\"metric\":\"h2d_pinned_fill_{label}_mb_s\",\"value\":{:.0},\"gross_mb_s\":{:.0},\"ms\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"host_fill_ms\":{fill_ms:.3},\"net_ms\":{net:.3},\"bytes\":{n}}}", mb_s(n, net), mb_s(n, ms));

        // the pinned path split in two: the fill alone (buffer taken, written, returned), then
        // the upload alone (buffer filled before the clock starts)
        #[cfg(feature = "cuda")]
        if gpu.has_pinned() {
            eprintln!("section: host_fill_pinned_{label}");
            let (pf_ms, pf_lo, pf_hi) = time(&mut || {
                let mut b = gpu.take_pinned(n / 4).ok_or("pinned pool at its cap")?;
                b.as_mut_u32().iter_mut().enumerate().for_each(|(i, x)| *x = i as u32);
                drop(b);
                Ok(())
            })?;
            println!("{{{header},\"metric\":\"host_fill_pinned_{label}_ms\",\"value\":{pf_ms:.3},\"min_ms\":{pf_lo:.3},\"max_ms\":{pf_hi:.3},\"vec_fill_ms\":{fill_ms:.3},\"bytes\":{n}}}");
            eprintln!("section: h2d_pinned_upload_{label}");
            let mut staged: Option<lb_q4_gpu::pinned::PinnedBuf> = None;
            // client-side split of each call: `create` returning, `sync` returning, handle drop
            let mut split: Vec<(f64, f64, f64)> = Vec::new();
            let (ms, lo, hi) = time(&mut || {
                let b = match staged.take() {
                    Some(b) => b,
                    None => {
                        let mut b = gpu.take_pinned(n / 4).ok_or("pinned pool at its cap")?;
                        b.as_mut_u32().iter_mut().enumerate().for_each(|(i, x)| *x = i as u32);
                        b
                    }
                };
                let t0 = Instant::now();
                let h = gpu.upload_pinned(b);
                let t1 = Instant::now();
                gpu.sync()?;
                let t2 = Instant::now();
                drop(h);
                let t3 = Instant::now();
                split.push(((t1 - t0).as_secs_f64() * 1e6, (t2 - t1).as_secs_f64() * 1e6, (t3 - t2).as_secs_f64() * 1e6));
                // fill the next rep's buffer now, outside the next timed call
                let mut nb = gpu.take_pinned(n / 4).ok_or("pinned pool at its cap")?;
                nb.as_mut_u32().iter_mut().enumerate().for_each(|(i, x)| *x = i as u32);
                staged = Some(nb);
                Ok(())
            })?;
            // the clock covers the next buffer's fill too; take it out with the fill just measured
            let net = (ms - pf_ms).max(1e-6);
            let tail = split.split_off(split.len().saturating_sub(reps));
            let (c_us, _, _) = stats(tail.iter().map(|t| t.0).collect());
            let (s_us, s_lo, s_hi) = stats(tail.iter().map(|t| t.1).collect());
            let (d_us, _, _) = stats(tail.iter().map(|t| t.2).collect());
            println!("{{{header},\"metric\":\"h2d_pinned_upload_{label}_split_us\",\"create_us\":{c_us:.1},\"sync_us\":{s_us:.1},\"sync_min_us\":{s_lo:.1},\"sync_max_us\":{s_hi:.1},\"drop_us\":{d_us:.1},\"bytes\":{n}}}");
            println!("{{{header},\"metric\":\"h2d_pinned_upload_{label}_mb_s\",\"value\":{:.0},\"ms_incl_next_fill\":{ms:.3},\"min_ms\":{lo:.3},\"max_ms\":{hi:.3},\"pinned_fill_ms\":{pf_ms:.3},\"net_ms\":{net:.3},\"bytes\":{n}}}", mb_s(n, net));
        }

        let blob: Vec<u8> = (0..n).map(|i| i as u8).collect();
        eprintln!("section: h2d_from_slice_{label}");
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
        // the repack as the engine pays it: into reused buffers (pool slots or the same
        // Vecs), not a fresh `Vec` per call, whose page faults were most of the old number
        // (c2: 0.22-0.24 ms a matrix allocating vs 0.06-0.10 into reused words)
        let (mut rq, mut rs) = (vec![0u32; nb * 4], vec![0u32; nb.div_ceil(2)]);
        let (repack, _, _) = time(&mut || {
            #[cfg(feature = "cuda")]
            if let (Some(mut q), Some(mut s)) = (gpu.take_pinned(nb * 4), gpu.take_pinned(nb.div_ceil(2))) {
                lb_q4_gpu::repack_q4_0_into(&bytes, nb, q.as_mut_u32(), s.as_mut_u32());
                return Ok(());
            }
            lb_q4_gpu::repack_q4_0_into(&bytes, nb, &mut rq, &mut rs);
            Ok(())
        })?;
        eprintln!("section: h2d_expert_upload");
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
        eprintln!("section: d2h_h2d_act_m{m}");
        let (ms, lo, hi) = time(&mut || {
            let v = gpu.read_all(vec![(xh.clone(), m * n_embd)])?;
            let h2 = gpu.upload_f32(&v[0]);
            gpu.sync()?;
            drop(h2);
            Ok(())
        })?;
        println!("{{{header},\"metric\":\"d2h_h2d_act_m{m}_us\",\"value\":{:.1},\"min_us\":{:.1},\"max_us\":{:.1},\"bytes_each_way\":{}}}", ms * 1e3, lo * 1e3, hi * 1e3, m * n_embd * 4);
    }
    println!("{{{header},\"metric\":\"pinned_pool_allocated_mb\",\"value\":{:.1}}}", gpu.pinned_allocated() as f64 / 1e6);
    Ok(())
}
