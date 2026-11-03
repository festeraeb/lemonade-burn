//! Split the cost of serving a miss by copying an expert over: repack on the host, then the
//! host-to-device transfer. Needs a GPU (LB_WGPU_DEVICE).
fn main() -> Result<(), String> {
    let (rows, cols) = (1536usize, 2048usize);
    let nb = rows * (cols / 32);
    let bytes: Vec<u8> = (0..nb * 18).map(|i| (i % 251) as u8).collect();
    let mat = || lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, rows, cols, &bytes);
    let dmat = || lb_gguf::QuantMat::new(lb_gguf::GgmlType::Q4_0, cols, rows, &bytes);
    let gpu = lb_q4_gpu::Gpu::open()?;

    let mut repack = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        std::hint::black_box(lb_q4_gpu::repack_q4_0(&bytes, nb));
        repack = repack.min(t.elapsed().as_secs_f64() * 1e3);
    }
    // one whole expert: gate + up (rows x cols) and down (cols x rows), same byte count
    let mut whole = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let ex = lb_q4_gpu::GpuExpert::upload(&gpu, mat(), mat(), dmat())?;
        gpu.sync()?;
        whole = whole.min(t.elapsed().as_secs_f64() * 1e3);
        drop(ex);
    }
    let mb = 3.0 * (nb * 18) as f64 / 1e6;
    println!("{{\"metric\":\"repack_one_matrix_ms\",\"value\":{repack:.3}}}");
    println!("{{\"metric\":\"upload_whole_expert_ms\",\"value\":{whole:.3},\"mb\":{mb:.2}}}");
    println!("{{\"metric\":\"upload_minus_repack_ms\",\"value\":{:.3}}}", whole - 3.0 * repack);
    println!("{{\"metric\":\"effective_h2d_mb_s\",\"value\":{:.0}}}", mb / ((whole - 3.0 * repack).max(1e-6) / 1e3));
    Ok(())
}
