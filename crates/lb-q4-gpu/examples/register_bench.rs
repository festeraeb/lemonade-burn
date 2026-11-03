// Can the device DMA straight out of the GGUF's page cache? `cuMemHostRegister` with
// `CU_MEMHOSTREGISTER_READ_ONLY` on the file mapping, then `cuMemcpyHtoDAsync` of
// expert-matrix-sized slices (1536x2048 Q4_0 = 1,769,472 bytes) from inside it, against the
// same copies from the unregistered (pageable) mapping. Prints: the two driver attributes
// that gate it, the registration time per MB, the copy medians, and the unregister time.
// Needs the `cuda` feature; `LB_MODEL` (GGUF path), `LB_CUDA_DEVICE` (driver order),
// `LB_REG_MB` (region to register, default 1024; 0 = the whole file), `LB_BENCH_REPS`.
use cudarc::driver::{result, sys};
use std::ffi::c_void;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let model = std::env::var("LB_MODEL").expect("LB_MODEL=<gguf path>");
    let device: usize = std::env::var("LB_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let reg_mb: usize = std::env::var("LB_REG_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let reps: usize = std::env::var("LB_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let g = lb_gguf::Gguf::open(std::path::Path::new(&model)).expect("open gguf");
    let map = g.mapping();
    let page = 4096usize;
    // registered region: page-aligned, from the start of the mapping (page aligned by mmap)
    let region_len = if reg_mb == 0 { map.len() / page * page } else { (reg_mb << 20).min(map.len() / page * page) };
    let slice_len = 1536 * 64 * 18usize; // one expert matrix, raw
    assert!(region_len > 2 * slice_len, "region too small");

    unsafe {
        result::init().expect("cuInit");
        let dev = result::device::get(device as i32).expect("cuDeviceGet");
        let ctx = result::primary_ctx::retain(dev).expect("retain");
        result::ctx::set_current(ctx).expect("set current");
        let attr = |a: sys::CUdevice_attribute| result::device::get_attribute(dev, a).unwrap_or(-1);
        let hr = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED);
        let ro = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED);
        println!("{{\"attr_host_register_supported\":{hr},\"attr_read_only_host_register_supported\":{ro},\"region_mb\":{}}}", region_len >> 20);

        let stream = result::stream::create(result::stream::StreamKind::NonBlocking).expect("stream");
        let dst = result::malloc_sync(slice_len).expect("cuMemAlloc");
        let base = map.as_ptr();
        // offsets of the slices copied: spread over the region, 2-byte aligned like real experts
        let offs: Vec<usize> = (0..reps).map(|i| ((i * 7_919 * 18) % (region_len - slice_len)) / 2 * 2).collect();
        let copy_round = |label: &str| {
            let mut ts = Vec::with_capacity(reps);
            for &o in &offs {
                let src = std::slice::from_raw_parts(base.add(o), slice_len);
                let t = std::time::Instant::now();
                result::memcpy_htod_async(dst, src, stream).expect("memcpy_htod_async");
                result::stream::synchronize(stream).expect("sync");
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (lo, hi) = (ts.iter().cloned().fold(f64::MAX, f64::min), ts.iter().cloned().fold(0.0, f64::max));
            let med = median(&mut ts);
            println!(
                "{{\"copy\":\"{label}\",\"bytes\":{slice_len},\"median_ms\":{med:.4},\"min_ms\":{lo:.4},\"max_ms\":{hi:.4},\"median_mb_s\":{:.0}}}",
                slice_len as f64 / med / 1e3
            );
        };
        // touch the pages first so neither round pays the page-cache fault
        let mut sum = 0u64;
        for i in (0..region_len).step_by(page) {
            sum = sum.wrapping_add(*base.add(i) as u64);
        }
        std::hint::black_box(sum);
        copy_round("pageable_mmap");

        let t = std::time::Instant::now();
        let r = sys::cuMemHostRegister_v2(base as *mut c_void, region_len, sys::CU_MEMHOSTREGISTER_READ_ONLY);
        let reg_ms = t.elapsed().as_secs_f64() * 1e3;
        println!("{{\"register\":\"{:?}\",\"region_mb\":{},\"ms\":{reg_ms:.1},\"ms_per_gb\":{:.1}}}", r, region_len >> 20, reg_ms / (region_len as f64 / (1u64 << 30) as f64));
        if r == sys::CUresult::CUDA_SUCCESS {
            copy_round("registered_mmap");
            copy_round("registered_mmap_rep2");
            let t = std::time::Instant::now();
            let r = sys::cuMemHostUnregister(base as *mut c_void);
            println!("{{\"unregister\":\"{:?}\",\"ms\":{:.1}}}", r, t.elapsed().as_secs_f64() * 1e3);
            copy_round("pageable_mmap_after");
        }
        result::free_sync(dst).ok();
    }
}
