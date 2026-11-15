//! The E2 expert-cache miss path on whatever runtime is open (works on wgpu, no cuda feature):
//! `ExpertSlots::fill` (host words + `client.write` into the staging buffer + repack launch x3),
//! timed as host call time and as call + device completion, for 1 and 3 misses a sync, plus the
//! raw 5.3 MB write rate. `LB_WGPU_DEVICE=discrete:N`.
use lb_gguf::types::quantize_q4_0;
use lb_q4_gpu::{ExpertSlots, Gpu};
use std::time::Instant;

fn main() -> Result<(), String> {
    let gpu = Gpu::open()?;
    let (e, ff) = (2048usize, 1536usize);
    let nb = ff * e / 32;
    let mut s = 12345u64;
    let mut rnd = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mk = |rnd: &mut dyn FnMut() -> f32| -> Vec<u8> { (0..ff * e).map(|_| rnd()).collect::<Vec<f32>>().chunks_exact(e).flat_map(quantize_q4_0).collect() };
    let raws: Vec<Vec<u8>> = (0..3).map(|_| mk(&mut rnd)).collect();
    let n_slots: usize = std::env::var("FB_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let slots = ExpertSlots::new(&gpu, n_slots, [(ff, e), (ff, e), (e, ff)], 4)?;
    println!("# {} slots {} ({} MiB)", gpu.name, n_slots, slots.bytes >> 20);
    let reps = 20;
    for per_sync in [1usize, 3, 8] {
        let (mut host, mut total) = (Vec::new(), Vec::new());
        for r in 0..reps {
            let t = Instant::now();
            for k in 0..per_sync {
                slots.fill(&gpu, (r * per_sync + k) % n_slots, [&raws[0], &raws[1], &raws[2]])?;
            }
            host.push(t.elapsed().as_secs_f64() * 1e3 / per_sync as f64);
            gpu.sync_recycle()?;
            total.push(t.elapsed().as_secs_f64() * 1e3 / per_sync as f64);
        }
        host.sort_by(|a, b| a.partial_cmp(b).unwrap());
        total.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("fill x{per_sync} per sync: host call p50 {:.2} ms/expert, call+complete p50 {:.2} ms/expert ({:.2} GB/s of file bytes)", host[reps / 2], total[reps / 2], 3.0 * (nb * 18) as f64 / (total[reps / 2] * 1e6));
    }
    // the expert kernels reading the bank: 4 routed slots, launch + read-back, after the fills above
    {
        let x = gpu.upload_f32(&(0..e).map(|i| (i as f32 * 0.37).sin()).collect::<Vec<f32>>());
        let routes = gpu.upload_u32(vec![0, 1, 2, 3, 0, 0, 0, 0]);
        let bank = slots.bank();
        for _ in 0..10 {
            let (o, n) = bank.launch_device(&gpu, &x, 1, &routes, 4, 0, 4);
            let _ = gpu.read_all(vec![(o, n)])?;
        }
        let mut v = Vec::new();
        for _ in 0..50 {
            let t = Instant::now();
            let (o, n) = bank.launch_device(&gpu, &x, 1, &routes, 4, 0, 4);
            let _ = gpu.read_all(vec![(o, n)])?;
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("bank launch_device 4 slots + read: p50 {:.3} ms min {:.3} ms", v[25], v[0]);
        // many launches then one sync (what a layer does between reads)
        let t = Instant::now();
        for _ in 0..38 {
            let (o, _n) = bank.launch_device(&gpu, &x, 1, &routes, 4, 0, 4);
            drop(o);
        }
        gpu.sync_recycle()?;
        println!("38 x bank launch_device then one sync: {:.3} ms per launch", t.elapsed().as_secs_f64() * 1e3 / 38.0);
    }
    // raw writes of one 5.3 MB blob without the repack
    let blob: Vec<u32> = (0..nb * 18 * 3 / 4).map(|i| i as u32).collect();
    let h = gpu.client.empty(blob.len() * 4);
    let mut v = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        gpu.client.write(&h, cubecl::bytes::Bytes::from_elems(blob.clone()));
        gpu.sync_recycle()?;
        v.push(t.elapsed().as_secs_f64() * 1e3);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("client.write {:.1} MB + sync: p50 {:.2} ms = {:.2} GB/s (includes the Vec clone)", blob.len() as f64 * 4.0 / 1e6, v[reps / 2], blob.len() as f64 * 4.0 / (v[reps / 2] * 1e6));
    Ok(())
}
