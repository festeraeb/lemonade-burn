//! Rows kernels vs the tiled GEMM at LFM2.5-24B-A2B prefill shapes (random Q4_0 weights):
//! a dense 6144 x 2048 (conv in_proj) and one MoE layer of 64 experts (1536 x 2048, top-4) for a
//! chunk of `LB_BENCH_M` rows (512). Median of 5 after a warm-up; prints ms and the ratio.
//! Run with a device named (`LB_CUDA_DEVICE` / `LB_WGPU_DEVICE`).
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpertLayer, GpuQ4Mat};
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

fn q4(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

fn med(gpu: &Gpu, mut f: impl FnMut()) -> f64 {
    f();
    gpu.sync_recycle().unwrap();
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            gpu.sync_recycle().unwrap();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

fn main() {
    let gpu = Gpu::open().expect("device");
    eprintln!("{}: tensor_cores={} plane {:?}", gpu.name, gpu.tensor_cores, gpu.plane_size);
    let m: usize = std::env::var("LB_BENCH_M").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let mut rng = Rng(7);
    let (rows, cols) = (6144usize, 2048usize);
    let w = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, rows, cols, &q4(&mut rng, rows, cols))).unwrap();
    let x: Vec<f32> = (0..m * cols).map(|_| rng.unit()).collect();
    let xh = gpu.upload_f32(&x);
    let a = med(&gpu, || drop(w.launch(&gpu, &xh, m)));
    let b = med(&gpu, || drop(w.launch_gemm(&gpu, &xh, m)));
    let gflop = 2.0 * (m * rows * cols) as f64 / 1e9;
    println!("{}: dense {rows}x{cols} m={m}: rows {a:.2} ms, gemm {b:.2} ms ({:.2}x; gemm {:.0} GFLOP/s)", gpu.name, a / b, gflop / (b / 1e3));

    let (n_exp, n_ff, n_embd, k) = (64usize, 1536usize, 2048usize, 4usize);
    let layer = GpuExpertLayer::upload_stacked(
        &gpu,
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &q4(&mut rng, n_exp * n_ff, n_embd)),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &q4(&mut rng, n_exp * n_ff, n_embd)),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_embd, n_ff, &q4(&mut rng, n_exp * n_embd, n_ff)),
        n_exp,
    )
    .unwrap();
    let mut routes = Vec::with_capacity(2 * m * k);
    for t in 0..m {
        let mut used: Vec<u32> = Vec::new();
        while used.len() < k {
            let e = (rng.next() % n_exp as u64) as u32;
            if !used.contains(&e) {
                used.push(e);
                routes.extend_from_slice(&[e, t as u32]);
            }
        }
    }
    let slots = m * k;
    let rh = gpu.upload_u32(routes.clone());
    let a = med(&gpu, || drop(layer.launch_device(&gpu, &xh, m, &rh, slots, 0, slots)));
    let b = med(&gpu, || {
        let (perm, tiles) = lb_q4_gpu::gemm::group_slots(&routes, n_exp);
        let nt = tiles.len() / 3;
        drop(layer.launch_grouped(&gpu, &xh, m, &rh, slots, &gpu.upload_u32(perm), &gpu.upload_u32(tiles), nt));
    });
    let gflop = 2.0 * 3.0 * (slots * n_ff * n_embd) as f64 / 1e9;
    println!("{}: MoE layer 64x({n_ff}x{n_embd}) top-{k} m={m}: routed rows {a:.2} ms, grouped gemm {b:.2} ms ({:.2}x; gemm {:.0} GFLOP/s)", gpu.name, a / b, gflop / (b / 1e3));
}
