//! The P100 kernel variants of 2026-11-21 (docs/P100_KERNEL_TUNING.md) against the kernels they replace, bit for bit: each variant
//! claims the same operations in the same order, so any differing output bit is a failure (not a tolerance). Run on a card with
//! `LB_CUDA_DEVICE=0` (cuda feature); the knobs are flipped on the opened `Gpu`, so env settings of them do not matter.
mod common;
use common::Rng;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuQ6KNative};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn same(a: &[f32], b: &[f32], label: &str) {
    let (a, b) = (bits(a), bits(b));
    assert_eq!(a.len(), b.len(), "{label}: length");
    let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(diff, 0, "{label}: {diff} of {} outputs differ in their bits", a.len());
    eprintln!("{label}: {} outputs bit-identical", a.len());
}

fn q6_rows(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let rb = GgmlType::Q6K.row_bytes(cols).unwrap();
    let mut v: Vec<u8> = (0..rows * rb).map(|_| rng.byte()).collect();
    for blk in v.chunks_exact_mut(210) {
        blk[208..210].copy_from_slice(&half::f16::from_f32(0.001 + 0.03 * (rng.unit() * 0.5 + 0.5)).to_le_bytes());
    }
    v
}

#[test]
fn q6_lane_groups_match_rows4() {
    let mut gpu = Gpu::open().unwrap();
    let mut cases: Vec<(String, usize, usize, Vec<u8>)> = Vec::new();
    for (rows, cols, seed) in [(4096usize, 2048usize, 1u64), (1000, 1024, 2), (8, 512, 3), (64, 4096, 4)] {
        let mut rng = Rng(seed);
        cases.push((format!("random {rows}x{cols}"), rows, cols, q6_rows(&mut rng, rows, cols)));
    }
    let path = std::path::Path::new("/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf");
    if path.exists() {
        let g = lb_gguf::Gguf::open(path).unwrap();
        for name in ["output.weight", "blk.3.attn_q.weight"] {
            let t = g.tensor(name).unwrap();
            let (cols, rows) = (t.dims[0], 8192usize.min(t.dims[1]));
            cases.push((format!("real {name} {rows}x{cols}"), rows, cols, g.tensor_bytes(t)[..rows * GgmlType::Q6K.row_bytes(cols).unwrap()].to_vec()));
        }
    }
    for (label, rows, cols, bytes) in &cases {
        let w = GpuQ6KNative::upload(&gpu, QuantMat::new(GgmlType::Q6K, *rows, *cols, bytes)).unwrap();
        for m in [1usize, 3] {
            let mut rng = Rng(7 + m as u64);
            let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 3.0).collect();
            gpu.kq.q6_lg = false;
            let a = w.matvec(&gpu, &x, m).unwrap();
            gpu.kq.q6_lg = true;
            let b = w.matvec(&gpu, &x, m).unwrap();
            same(&a, &b, &format!("Q6_K lane groups, {label}, m {m}"));
        }
    }
}

#[test]
fn split_attention_matches_wide() {
    let mut gpu = Gpu::open().unwrap();
    let (nh, nkv, hd, half, cap) = (16usize, 2usize, 256usize, 32usize, 4352usize);
    let mut rng = Rng(11);
    let qn: Vec<f32> = (0..hd).map(|_| 1.0 + 0.1 * rng.unit()).collect();
    let kn: Vec<f32> = (0..hd).map(|_| 1.0 + 0.1 * rng.unit()).collect();
    // a filled cache (positions below `pos` as earlier tokens left them), then this token's q / k / v
    let kc0: Vec<f32> = (0..cap * nkv * hd).map(|_| rng.unit()).collect();
    let vc0: Vec<f32> = (0..cap * nkv * hd).map(|_| rng.unit()).collect();
    let (qnh, knh) = (gpu.upload_f32(&qn), gpu.upload_f32(&kn));
    for pos in [0usize, 1, 2, 3, 4, 5, 15, 16, 17, 63, 255, 256, 257, 1000, 1023, 4097, 4351] {
        let qf: Vec<f32> = (0..nh * 2 * hd).map(|_| rng.unit() * 2.0).collect();
        let kr: Vec<f32> = (0..nkv * hd).map(|_| rng.unit()).collect();
        let vr: Vec<f32> = (0..nkv * hd).map(|_| rng.unit()).collect();
        let mut outs = Vec::new();
        for split in [false, true] {
            gpu.kq.attn_split = split;
            let (kc, vc) = (gpu.upload_f32(&kc0), gpu.upload_f32(&vc0));
            let (qh, kh, vh, ph) = (gpu.upload_f32(&qf), gpu.upload_f32(&kr), gpu.upload_f32(&vr), gpu.upload_u32(vec![pos as u32]));
            let o = gpu.qw_attention(&qh, &kh, &vh, &qnh, &knh, &kc, &vc, &ph, (nh, nkv, hd, half, cap), 1e-6, 0.9);
            outs.push(gpu.read_all(vec![(o, nh * hd)]).unwrap().remove(0));
        }
        same(&outs[0], &outs[1], &format!("split attention, n = {}", pos + 1));
    }
}
