//! Measurement harness for the v2 pipeline simulator (docs/V2_PIPELINE.md): on the card named
//! by LB_WGPU_DEVICE, measures the numbers the discrete-event simulator needs, one card at a
//! time (no multi-GPU window needed, no seat evicted). Prints one JSON object per measurement
//! and a table at the end.
//!
//! Measurements (LFM2.5-24B-A2B shapes: n_embd 2048, expert n_ff 1536, dense in_proj 6144):
//!   h2d_expert_mb_s     host->device copy bandwidth, one expert's 3 matrices (5.3 MB blobs)
//!   h2d_bulk_mb_s       same with a 64 MB blob (what prestaging a layer looks like)
//!   d2h_act_us          read back an activation row block (m x 2048 f32) + re-upload it
//!   expert_ffn_ms       one resident expert, gate/up/silu/down, m = 1 and 4, launch+readback
//!   expert_ffn_batch_ms same, 4 experts launched then ONE readback (per-expert cost)
//!   dense_ms            in_proj 6144x2048, out_proj 2048x2048, k/v 512x2048, q6k head on CPU
//!   cpu_expert_ms       one expert on this host's CPU (lb-gguf reference path), all threads / 1 thread
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert, GpuQ4Mat};
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

fn q4_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<(), String> {
    let (n_embd, n_ff) = (2048usize, 1536usize);
    let reps: usize = std::env::var("LB_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string();
    let device = std::env::var("LB_WGPU_DEVICE").unwrap_or_default();
    let mut rows_out: Vec<(String, f64, String)> = Vec::new();
    let mut rng = Rng(3);
    let gate_b = q4_bytes(&mut rng, n_ff, n_embd);
    let up_b = q4_bytes(&mut rng, n_ff, n_embd);
    let down_b = q4_bytes(&mut rng, n_embd, n_ff);
    let expert_bytes = gate_b.len() + up_b.len() + down_b.len();
    let gate = QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &gate_b);
    let up = QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &up_b);
    let down = QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &down_b);

    // CPU reference timings first (no GPU needed)
    for (label, threads) in [("cpu_expert_ms_all_threads", 0usize), ("cpu_expert_ms_1_thread", 1)] {
        let pool = if threads == 0 { rayon::ThreadPoolBuilder::new().build() } else { rayon::ThreadPoolBuilder::new().num_threads(1).build() }.unwrap();
        let x: Vec<f32> = (0..n_embd).map(|_| rng.unit()).collect();
        let mut ts = Vec::new();
        for _ in 0..reps.max(5) {
            let t = Instant::now();
            pool.install(|| {
                let (mut g, mut u) = (vec![0f32; n_ff], vec![0f32; n_ff]);
                matvec_f32(gate, &x, 1, &mut g);
                matvec_f32(up, &x, 1, &mut u);
                for (gi, ui) in g.iter_mut().zip(&u) {
                    *gi = *gi / (1.0 + (-*gi).exp()) * ui;
                }
                let mut o = vec![0f32; n_embd];
                matvec_f32(down, &g, 1, &mut o);
                std::hint::black_box(o);
            });
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        rows_out.push((label.into(), median(ts), format!("threads={}", if threads == 0 { pool.current_num_threads() } else { 1 })));
    }
    {
        // Q6_K head on CPU: 128000 x 2048 (random bytes are valid blocks)
        let rows = 128000usize;
        let rb = GgmlType::Q6K.row_bytes(n_embd).unwrap();
        let mut hb: Vec<u8> = (0..rows * rb).map(|_| (rng.next() >> 56) as u8).collect();
        for blk in hb.chunks_exact_mut(210) {
            blk[208..210].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        }
        let head = QuantMat::new(GgmlType::Q6K, rows, n_embd, &hb);
        let x: Vec<f32> = (0..n_embd).map(|_| rng.unit()).collect();
        let mut out = vec![0f32; rows];
        let mut ts = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            matvec_f32(head, &x, 1, &mut out);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        rows_out.push(("cpu_q6k_head_ms".into(), median(ts), format!("threads={}", rayon::current_num_threads())));
    }

    let gpu = Gpu::open()?;
    eprintln!("device: {} ({device}) on {host}", gpu.name);
    let time = |f: &mut dyn FnMut() -> Result<(), String>| -> Result<f64, String> {
        let mut ts = Vec::new();
        f()?; // warm-up (shader compile)
        for _ in 0..reps {
            let t = Instant::now();
            f()?;
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(median(ts))
    };

    // host -> device bandwidth: expert blob and 64 MB blob
    let blob: Vec<u8> = (0..expert_bytes).map(|i| i as u8).collect();
    let ms = time(&mut || {
        let h = gpu.client.create_from_slice(&blob);
        gpu.sync()?;
        drop(h);
        Ok(())
    })?;
    rows_out.push(("h2d_expert_mb_s".into(), expert_bytes as f64 / 1e6 / (ms / 1e3), format!("{expert_bytes} bytes in {ms:.3} ms")));
    let big: Vec<u8> = (0..64 << 20).map(|i| i as u8).collect();
    let ms = time(&mut || {
        let h = gpu.client.create_from_slice(&big);
        gpu.sync()?;
        drop(h);
        Ok(())
    })?;
    rows_out.push(("h2d_bulk_mb_s".into(), big.len() as f64 / 1e6 / (ms / 1e3), format!("{} bytes in {ms:.3} ms", big.len())));

    // activation handoff through the host: read back [m, 2048] f32 then upload it again
    for m in [1usize, 4, 32] {
        let x: Vec<f32> = (0..m * n_embd).map(|_| rng.unit()).collect();
        let xh = gpu.upload_f32(&x);
        let ms = time(&mut || {
            let v = gpu.read_all(vec![(xh.clone(), m * n_embd)])?;
            let h2 = gpu.upload_f32(&v[0]);
            gpu.sync()?;
            drop(h2);
            Ok(())
        })?;
        rows_out.push((format!("d2h_h2d_act_m{m}_us"), ms * 1e3, format!("{} bytes each way", m * n_embd * 4)));
    }

    // resident expert FFN
    let e = GpuExpert::upload(&gpu, gate, up, down)?;
    for m in [1usize, 4] {
        let x: Vec<f32> = (0..m * n_embd).map(|_| rng.unit()).collect();
        let ms = time(&mut || {
            e.ffn(&gpu, &x, m)?;
            Ok(())
        })?;
        rows_out.push((format!("expert_ffn_m{m}_ms"), ms, "upload x + 4 launches + readback".into()));
        let ms = time(&mut || {
            let xh = gpu.upload_f32(&x);
            let hs: Vec<_> = (0..4).map(|_| e.launch(&gpu, &xh, m)).collect();
            gpu.read_all(hs)?;
            Ok(())
        })?;
        rows_out.push((format!("expert_ffn_batch4_m{m}_ms_per_expert"), ms / 4.0, "4 experts, one readback".into()));
    }

    // GPU-bound throughput: 24 distinct experts (no cache reuse between them), one readback.
    // At m = 1 the per-expert time is dominated by streaming the weights, so this is the number
    // to compare against the memory roofline.
    {
        let n_exp = 24usize;
        let mut es = Vec::new();
        for _ in 0..n_exp {
            let (g, u, d) = (q4_bytes(&mut rng, n_ff, n_embd), q4_bytes(&mut rng, n_ff, n_embd), q4_bytes(&mut rng, n_embd, n_ff));
            es.push(GpuExpert::upload(&gpu, QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &g), QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &u), QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &d))?);
        }
        for m in [1usize, 4] {
            let x: Vec<f32> = (0..m * n_embd).map(|_| rng.unit()).collect();
            let ms = time(&mut || {
                let xh = gpu.upload_f32(&x);
                let hs: Vec<_> = es.iter().map(|e| e.launch(&gpu, &xh, m)).collect();
                gpu.read_all(hs)?;
                Ok(())
            })?;
            let per = ms / n_exp as f64;
            let gbs = es[0].bytes_on_device() as f64 / 1e9 / (per / 1e3);
            rows_out.push((format!("expert_ffn_distinct{n_exp}_m{m}_ms_per_expert"), per, format!("{gbs:.1} GB/s weight stream")));
        }
    }

    // dense projections
    for (label, rows, cols) in [("dense_in_proj_6144x2048_ms", 6144usize, n_embd), ("dense_out_proj_2048x2048_ms", 2048, n_embd), ("dense_kv_512x2048_ms", 512, n_embd)] {
        let wb = q4_bytes(&mut rng, rows, cols);
        let w = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, rows, cols, &wb))?;
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let xh = gpu.upload_f32(&x);
        let ms = time(&mut || {
            let o = w.launch(&gpu, &xh, 1);
            gpu.read_all(vec![(o, rows)])?;
            Ok(())
        })?;
        rows_out.push((label.into(), ms, "launch + readback, m=1".into()));
    }

    println!("{{\"host\":\"{host}\",\"device\":\"{device}\",\"gpu\":\"{}\",\"expert_bytes\":{expert_bytes}}}", gpu.name);
    for (k, v, note) in &rows_out {
        println!("{{\"metric\":\"{k}\",\"value\":{v:.4},\"note\":\"{note}\"}}");
    }
    eprintln!("\n{:<42} {:>12}  note", "metric", "value");
    for (k, v, note) in &rows_out {
        eprintln!("{k:<42} {v:>12.4}  {note}");
    }
    Ok(())
}
