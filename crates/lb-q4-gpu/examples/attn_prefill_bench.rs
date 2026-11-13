//! Prefill attention, per-query kernel vs tiled, at LFM2.5's shape (32 heads over 8 KV heads, head 64)
//! for a prompt of N tokens in chunks of 512 (each chunk attends to everything before it): the time
//! one attention layer spends over the whole prompt. Median of 3. `LB_BENCH_N` (default 512,2048,8192).
use lb_q4_gpu::Gpu;
use std::time::Instant;

fn main() {
    let gpu = Gpu::open().expect("device");
    let lens: Vec<usize> = std::env::var("LB_BENCH_N").unwrap_or_else(|_| "512,2048,8192".into()).split(',').filter_map(|v| v.parse().ok()).collect();
    let (nh, kvh, hd, ub) = (32usize, 8usize, 64usize, 512usize);
    for n in lens {
        let cap = n.next_power_of_two();
        let kc = gpu.upload_f32(&vec![0.01f32; kvh * cap * hd]);
        let vc = gpu.upload_f32(&vec![0.02f32; kvh * cap * hd]);
        let q = gpu.upload_f32(&vec![0.03f32; ub * nh * hd]);
        let run = |tiled: bool| -> f64 {
            let mut v = Vec::new();
            for _ in 0..3 {
                gpu.sync_recycle().unwrap();
                let t = Instant::now();
                let mut c0 = 0;
                while c0 < n {
                    let mc = ub.min(n - c0);
                    let _ = if tiled { gpu.attn_prefill_tiled(&q, &kc, &vc, (mc, c0, cap), (nh, kvh, hd)) } else { gpu.attn_prefill_naive(&q, &kc, &vc, (mc, c0, cap), (nh, kvh, hd)) };
                    gpu.sync_recycle().unwrap();
                    c0 += mc;
                }
                v.push(t.elapsed().as_secs_f64() * 1e3);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[1]
        };
        let (a, b) = (run(false), run(true));
        println!("{}: attention over a {n}-token prompt (one layer): per-query {a:.1} ms, tiled {b:.1} ms ({:.1}x)", gpu.name, a / b);
    }
}
