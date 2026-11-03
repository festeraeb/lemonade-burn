//! Does repeatedly creating and dropping transient expert buffers cost more than one
//! upload in isolation? `MissPolicy::OnDemand` does exactly that - about 6 experts a
//! token, each allocating six device buffers and freeing them after the readback - and in
//! the engine it costs 40.9 ms a call where examples/upload_bench.rs measures 5.3 ms.
//! Needs a GPU (LB_WGPU_DEVICE).
fn main() -> Result<(), String> {
    let (rows, cols) = (1536usize, 2048usize);
    let nb = rows * (cols / 32);
    let bytes: Vec<u8> = (0..nb * 18).map(|i| (i % 251) as u8).collect();
    let mat = || lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, rows, cols, &bytes);
    let dmat = || lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, cols, rows, &bytes);
    let gpu = lb_q4_gpu::Gpu::open()?;

    // warm the driver's allocator so the first round is not charged for it
    drop(lb_q4_gpu::GpuExpert::upload(&gpu, mat(), mat(), dmat())?);
    gpu.sync()?;

    for n in [1usize, 8, 32, 96] {
        let t = std::time::Instant::now();
        for _ in 0..n {
            let ex = lb_q4_gpu::GpuExpert::upload(&gpu, mat(), mat(), dmat())?;
            gpu.sync()?;
            drop(ex);
        }
        let per = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!("{{\"metric\":\"upload_drop_cycle_ms\",\"n\":{n},\"value\":{per:.3}}}");
    }
    // held: allocate all of them before freeing any, the opposite extreme
    let t = std::time::Instant::now();
    let mut held = Vec::new();
    for _ in 0..96 {
        held.push(lb_q4_gpu::GpuExpert::upload(&gpu, mat(), mat(), dmat())?);
    }
    gpu.sync()?;
    let per = t.elapsed().as_secs_f64() * 1e3 / 96.0;
    println!("{{\"metric\":\"upload_96_held_then_freed_ms\",\"value\":{per:.3}}}");
    drop(held);
    Ok(())
}
