//! What the dense (non-expert) side of an LFM2.5-24B-A2B decode step costs, CPU vs GPU.
//!
//! After the expert side became fully resident, experts are ~13% of a token and the other
//! 87% is these shapes, all still on the CPU reference kernels. Three questions decide
//! whether moving them to the device can pay:
//!
//! 1. per shape, what does `matvec_f32` cost at m = 1?
//! 2. per shape, what does one resident GPU matvec cost including the readback?
//! 3. what does a *round trip* cost on its own? A decode step has ~80 of them if every
//!    layer syncs twice, so if the floor is 1 ms the GPU cannot win no matter how fast
//!    the kernel is, and the design has to keep the whole layer on the device.
//!
//! Needs a GPU (`LB_WGPU_DEVICE`). Synthetic weights: the timings depend on the shape and
//! the quantization type, not the values.

/// The real dense shapes of LFM2.5-24B-A2B (n_embd 2048, n_ff 11776, 40 layers:
/// 30 short-conv + 10 attention, the first 2 FFNs dense), and how many a decode step runs.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    // name, rows, cols, per decode step
    ("shortconv.in_proj", 6144, 2048, 30),
    ("shortconv.out_proj", 2048, 2048, 30),
    ("attn_q", 2048, 2048, 10),
    ("attn_k", 512, 2048, 10),
    ("attn_v", 512, 2048, 10),
    ("attn_output", 2048, 2048, 10),
    ("ffn_gate", 11776, 2048, 2),
    ("ffn_up", 11776, 2048, 2),
    ("ffn_down", 2048, 11776, 2),
];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), String> {
    const REPS: usize = 7;
    let gpu = lb_q4_gpu::Gpu::open()?;
    println!("{{\"device\":{:?},\"cube_dim\":{},\"use_plane\":{}}}", gpu.name, gpu.cube_dim, gpu.use_plane);

    // Round-trip floor: a matvec so small the kernel cannot be the cost, launched and read
    // back one at a time. This is what each extra host sync in a layer costs.
    {
        let (rows, cols) = (32usize, 32usize);
        let bytes: Vec<u8> = (0..rows * (cols / 32) * 18).map(|i| (i % 251) as u8).collect();
        let w = lb_q4_gpu::GpuQ4Mat::upload(&gpu, lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, rows, cols, &bytes))?;
        let x = vec![0.1f32; cols];
        let xh = gpu.upload_f32(&x);
        let mut rt = Vec::new();
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            for _ in 0..64 {
                let out = w.launch(&gpu, &xh, 1);
                std::hint::black_box(gpu.read_all(vec![(out, rows)])?);
            }
            rt.push(t.elapsed().as_secs_f64() * 1e3 / 64.0);
        }
        println!("{{\"metric\":\"round_trip_ms\",\"value\":{:.4}}}", median(rt));
    }

    // Is the sync the cost, or the dispatch? Chain `depth` square matvecs, each eating the
    // last one's output, and compare one readback at the end against one readback per step.
    // Identical arithmetic and identical dispatch count: the only difference is host syncs.
    // This is the number that says whether keeping a layer on the device is worth the
    // restructuring it needs, before any of that restructuring is done.
    {
        let (rows, cols) = (2048usize, 2048usize);
        let nb = rows * (cols / 32);
        let bytes: Vec<u8> = (0..nb * 18).map(|i| (i % 251) as u8).collect();
        let w = lb_q4_gpu::GpuQ4Mat::upload(&gpu, lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, rows, cols, &bytes))?;
        let x = vec![0.02f32; cols];
        for depth in [8usize, 32, 106] {
            let mut chained = Vec::new();
            let mut per_step = Vec::new();
            for _ in 0..5 {
                // one sync for the whole chain
                let t = std::time::Instant::now();
                let mut h = gpu.upload_f32(&x);
                for _ in 0..depth {
                    h = w.launch(&gpu, &h, 1);
                }
                std::hint::black_box(gpu.read_all(vec![(h, rows)])?);
                chained.push(t.elapsed().as_secs_f64() * 1e3);

                // one sync per step, which is what `Lfm2::linear` does today
                let t = std::time::Instant::now();
                let mut v = x.clone();
                for _ in 0..depth {
                    v = w.matvec(&gpu, &v, 1)?;
                }
                std::hint::black_box(&v);
                per_step.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (c, p) = (median(chained), median(per_step));
            println!(
                "{{\"metric\":\"chain_vs_sync_per_step\",\"depth\":{depth},\"chained_ms\":{c:.3},\"per_step_ms\":{p:.3},\"speedup\":{:.2},\"saved_per_sync_ms\":{:.4}}}",
                p / c,
                (p - c) / depth as f64
            );
        }
    }

    // --- AS CALLED, with the device loaded ----------------------------------------------
    //
    // Everything above measures an almost-empty device. In the model there are ~14,600 expert
    // buffers resident and wgpu builds a fresh BindGroup per dispatch, so the tracker state a
    // call walks is nothing like this loop's. If a real matvec costs the same here as it does
    // with 2 GB resident, then the model's per-call overhead is NOT allocator or tracker churn
    // and the remaining time is the Burn/thread path. If it costs appreciably more, that churn
    // is real and is worth attacking directly.
    //
    // The MoE row is the one that sizes the "one launch per projection per layer" idea: a layer
    // today is 1 upload, 4 experts x 2 dispatches, 1 read.
    {
        // ~2 GB of decoys, in the same shape the model holds most of (an expert's gate/up).
        let (drows, dcols) = (1536usize, 2048usize);
        let dnb = drows * (dcols / 32);
        let dbytes: Vec<u8> = (0..dnb * 18).map(|i| (i % 251) as u8).collect();
        let mut decoys: Vec<lb_q4_gpu::GpuQ4Mat> = Vec::new();
        let mut resident_mb = 0usize;
        while resident_mb < 2048 {
            let m = lb_q4_gpu::GpuQ4Mat::upload(
                &gpu,
                lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, drows, dcols, &dbytes),
            )?;
            resident_mb += m.bytes_on_device() / (1 << 20);
            decoys.push(m);
        }
        println!("{{\"metric\":\"decoys_resident_mb\",\"value\":{resident_mb},\"buffers\":{}}}", decoys.len());

        // One real dense shape, called the way `Lfm2::linear_vec` calls it: upload, launch, read.
        for &(name, rows, cols) in &[("attn_q", 2048usize, 2048usize), ("shortconv.in_proj", 6144, 2048)] {
            let nb = rows * (cols / 32);
            let bytes: Vec<u8> = (0..nb * 18).map(|i| (i % 251) as u8).collect();
            let w = lb_q4_gpu::GpuQ4Mat::upload(
                &gpu,
                lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, rows, cols, &bytes),
            )?;
            let x = vec![0.05f32; cols];
            let mut v = Vec::new();
            for _ in 0..REPS {
                let t = std::time::Instant::now();
                for _ in 0..32 {
                    std::hint::black_box(w.matvec(&gpu, &x, 1)?);
                }
                v.push(t.elapsed().as_secs_f64() * 1e3 / 32.0);
            }
            println!(
                "{{\"metric\":\"as_called_matvec_ms\",\"shape\":{name:?},\"loaded\":true,\"value\":{:.4}}}",
                median(v)
            );
        }

        // A whole MoE layer as it runs today: one upload of x, top-4 experts each doing
        // gate/up/silu then down (2 dispatches), one read for the layer.
        {
            let (n_embd, n_ff_exp) = (2048usize, 1536usize);
            let gnb = n_ff_exp * (n_embd / 32);
            let dnb2 = n_embd * (n_ff_exp / 32);
            let gb: Vec<u8> = (0..gnb * 18).map(|i| (i % 251) as u8).collect();
            let db: Vec<u8> = (0..dnb2 * 18).map(|i| (i % 241) as u8).collect();
            let experts: Vec<lb_q4_gpu::GpuExpert> = (0..4)
                .map(|_| {
                    lb_q4_gpu::GpuExpert::upload(
                        &gpu,
                        lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, n_ff_exp, n_embd, &gb),
                        lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, n_ff_exp, n_embd, &gb),
                        lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, n_embd, n_ff_exp, &db),
                    )
                })
                .collect::<Result<_, _>>()?;
            let x = vec![0.03f32; n_embd];
            let mut v = Vec::new();
            for _ in 0..REPS {
                let t = std::time::Instant::now();
                for _ in 0..8 {
                    let xh = gpu.upload_f32(&x);
                    let outs: Vec<(cubecl::server::Handle, usize)> =
                        experts.iter().map(|e| e.launch(&gpu, &xh, 1)).collect();
                    std::hint::black_box(gpu.read_all(outs)?);
                }
                v.push(t.elapsed().as_secs_f64() * 1e3 / 8.0);
            }
            let per_layer = median(v);
            println!(
                "{{\"metric\":\"as_called_moe_layer_ms\",\"experts\":4,\"dispatches\":8,\"value\":{per_layer:.4},\"x38_layers_ms\":{:.2}}}",
                per_layer * 38.0
            );
        }
        drop(decoys);
    }

    let mut cpu_total = 0.0;
    let mut gpu_total = 0.0;
    for &(name, rows, cols, count) in SHAPES {
        let nb = rows * (cols / 32);
        let bytes: Vec<u8> = (0..nb * 18).map(|i| (i % 251) as u8).collect();
        let qm = || lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, rows, cols, &bytes);
        let x = vec![0.05f32; cols];

        let mut cpu = Vec::new();
        let mut out = vec![0f32; rows];
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            lb_gguf::matvec_f32(qm(), &x, 1, &mut out);
            cpu.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let cpu_ms = median(cpu);

        let w = lb_q4_gpu::GpuQ4Mat::upload(&gpu, qm())?;
        let xh = gpu.upload_f32(&x);
        // warm-up: the first launch of a shape pays shader specialisation
        std::hint::black_box(gpu.read_all(vec![(w.launch(&gpu, &xh, 1), rows)])?);
        let mut g = Vec::new();
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let h = w.launch(&gpu, &xh, 1);
            std::hint::black_box(gpu.read_all(vec![(h, rows)])?);
            g.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let gpu_ms = median(g);
        cpu_total += cpu_ms * count as f64;
        gpu_total += gpu_ms * count as f64;
        println!(
            "{{\"shape\":{name:?},\"rows\":{rows},\"cols\":{cols},\"per_step\":{count},\"cpu_ms\":{cpu_ms:.4},\"gpu_ms\":{gpu_ms:.4},\"speedup\":{:.2},\"mb\":{:.2}}}",
            cpu_ms / gpu_ms,
            (nb * 18) as f64 / 1e6
        );
    }
    println!("{{\"metric\":\"dense_q4_cpu_ms_per_token\",\"value\":{cpu_total:.1}}}");
    println!("{{\"metric\":\"dense_q4_gpu_ms_per_token_one_sync_each\",\"value\":{gpu_total:.1}}}");

    // The Q6_K tied head, CPU only: 128000 x 2048, the single largest dense matrix and the
    // one with no GPU kernel. Its own path is dequant-row-then-scalar-dot, not the fused dot.
    {
        let (rows, cols) = (128000usize, 2048usize);
        let rb = lb_gguf::GgmlType::Q6K.row_bytes(cols).ok_or("Q6_K row bytes")?;
        let bytes: Vec<u8> = (0..rows * rb).map(|i| (i % 251) as u8).collect();
        let qm = lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q6K, rows, cols, &bytes);
        let x = vec![0.05f32; cols];
        let mut out = vec![0f32; rows];
        let mut head = Vec::new();
        for _ in 0..3 {
            let t = std::time::Instant::now();
            lb_gguf::matvec_f32(qm, &x, 1, &mut out);
            head.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let cpu_ms = median(head);
        println!("{{\"metric\":\"q6k_head_cpu_ms\",\"value\":{cpu_ms:.1},\"mb\":{:.1}}}", (rows * rb) as f64 / 1e6);

        let w = lb_q4_gpu::GpuQ6KMat::upload(&gpu, qm)?;
        let xh = gpu.upload_f32(&x);
        std::hint::black_box(gpu.read_all(vec![(w.launch(&gpu, &xh, 1), rows)])?);
        let mut g = Vec::new();
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let h = w.launch(&gpu, &xh, 1);
            std::hint::black_box(gpu.read_all(vec![(h, rows)])?);
            g.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let gpu_ms = median(g);
        println!(
            "{{\"metric\":\"q6k_head_gpu_ms\",\"value\":{gpu_ms:.3},\"speedup\":{:.2},\"device_mb\":{:.1}}}",
            cpu_ms / gpu_ms,
            w.bytes_on_device() as f64 / 1e6
        );
    }
    Ok(())
}
