//! Native-density Q6_K and Q8_0 kernels against the CPU reference (random and real Qwen3.6 rows) and against the existing
//! widened kernels' speed.
mod common;
use common::{assert_close, Rng};
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuQ6KMat, GpuQ6KNative, GpuQ8Mat, GpuQ8Native};

fn random_rows(rng: &mut Rng, ty: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
    let rb = ty.row_bytes(cols).unwrap();
    let bs = ty.block().unwrap().1;
    let mut v: Vec<u8> = (0..rows * rb).map(|_| rng.byte()).collect();
    for blk in v.chunks_exact_mut(bs) {
        let d = half::f16::from_f32(0.001 + 0.03 * (rng.unit() * 0.5 + 0.5)).to_le_bytes();
        if ty == GgmlType::Q6K { blk[208..210].copy_from_slice(&d) } else { blk[0..2].copy_from_slice(&d) }
    }
    v
}

fn check(gpu: &Gpu, ty: GgmlType, w: QuantMat<'_>, x: &[f32], m: usize, label: &str) {
    let mut want = vec![0f32; m * w.rows];
    matvec_f32(w, x, m, &mut want);
    let got = if ty == GgmlType::Q6K { GpuQ6KNative::upload(gpu, w).unwrap().matvec(gpu, x, m) } else { GpuQ8Native::upload(gpu, w).unwrap().matvec(gpu, x, m) }.unwrap();
    assert_close(&got, &want, w.cols, w.rows, label);
}

#[test]
fn native_q6_k_q8_0_random_and_real() {
    let gpu = Gpu::open().unwrap();
    for ty in [GgmlType::Q6K, GgmlType::Q8_0] {
        for (rows, cols, m, seed) in [(8usize, 256usize, 1usize, 1u64), (37, 512, 3, 2), (1000, 2048, 2, 3)] {
            let mut rng = Rng(seed);
            let bytes = random_rows(&mut rng, ty, rows, cols);
            let w = QuantMat::new(ty, rows, cols, &bytes);
            let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 4.0).collect();
            check(&gpu, ty, w, &x, m, &format!("{} native random ({rows},{cols},{m})", ty.name()));
        }
    }
    let path = std::path::Path::new("/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf");
    if !path.exists() { return; }
    let g = lb_gguf::Gguf::open(path).unwrap();
    for (name, ty) in [("output.weight", GgmlType::Q6K), ("blk.3.attn_q.weight", GgmlType::Q6K), ("blk.4.ffn_down_exps.weight", GgmlType::Q6K), ("blk.0.attn_qkv.weight", GgmlType::Q8_0), ("blk.0.ffn_gate_shexp.weight", GgmlType::Q8_0), ("blk.0.ssm_out.weight", GgmlType::Q8_0)] {
        let t = g.tensor(name).unwrap();
        let cols = t.dims[0];
        let rows = 3000usize.min(t.dims[1]);
        let w = QuantMat::new(ty, rows, cols, &g.tensor_bytes(t)[..rows * ty.row_bytes(cols).unwrap()]);
        let mut rng = Rng(21);
        let x: Vec<f32> = (0..2 * cols).map(|_| rng.unit()).collect();
        check(&gpu, ty, w, &x, 2, &format!("{} native real {name}", ty.name()));
    }
}

#[test]
#[ignore]
fn native_q6_q8_throughput() {
    let gpu = Gpu::open().unwrap();
    let (rows, cols, reps) = (16384usize, 2048usize, 300usize);
    for ty in [GgmlType::Q6K, GgmlType::Q8_0] {
        let mut rng = Rng(5);
        let bytes = random_rows(&mut rng, ty, rows, cols);
        let w = QuantMat::new(ty, rows, cols, &bytes);
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let xh = gpu.upload_f32(&x);
        let (wide_b, nat_b);
        let time = |f: &dyn Fn() -> cubecl::server::Handle| {
            let mut dt = 0.0;
            for round in 0..2 {
                let t0 = std::time::Instant::now();
                let mut last = None;
                for _ in 0..reps { last = Some(f()); }
                gpu.read_all(vec![(last.unwrap(), rows)]).unwrap();
                if round == 1 { dt = t0.elapsed().as_secs_f64() / reps as f64; }
            }
            dt
        };
        let (tw, tn);
        if ty == GgmlType::Q6K {
            let a = GpuQ6KMat::upload(&gpu, w).unwrap(); let b = GpuQ6KNative::upload(&gpu, w).unwrap();
            wide_b = a.bytes_on_device(); nat_b = b.bytes_on_device();
            tw = time(&|| a.launch(&gpu, &xh, 1)); tn = time(&|| b.launch(&gpu, &xh, 1));
        } else {
            let a = GpuQ8Mat::upload(&gpu, w).unwrap(); let b = GpuQ8Native::upload(&gpu, w).unwrap();
            wide_b = a.bytes_on_device(); nat_b = b.bytes_on_device();
            tw = time(&|| a.launch(&gpu, &xh, 1)); tn = time(&|| b.launch(&gpu, &xh, 1));
        }
        eprintln!("{}: existing {:.1} us ({:.1} MB), native {:.1} us ({:.1} MB, file {:.1} MB)", ty.name(), tw * 1e6, wide_b as f64 / 1e6, tn * 1e6, nat_b as f64 / 1e6, bytes.len() as f64 / 1e6);
    }
}
