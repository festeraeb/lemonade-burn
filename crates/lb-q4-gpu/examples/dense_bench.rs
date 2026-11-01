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
