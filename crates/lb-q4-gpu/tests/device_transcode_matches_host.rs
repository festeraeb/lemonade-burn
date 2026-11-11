//! X7b: IQ4_XS -> Q4_0 transcode on the device (`Gpu::transcode_iq4_xs_on_device`) against the
//! host transcode (`transcode::q4_0_bytes` + `repack_q4_0`), on random IQ4_XS super-blocks.
//!
//! Not expected to be bit-identical: NVRTC contracts `a * b + c` into one FMA where the host
//! rounds twice, which can move a value across a rounding boundary or flip which of two nearly
//! equal scale candidates wins. Gate (docs/WORK_TO_WEIGHTS.md X7): <= 1 % of blocks pick a
//! different scale, and the device result's error against the IQ4_XS values is within 1 % of the
//! host result's (relative RMS). Also times the kernel on one expert layer's worth of bytes.
//! Runs only when a device is named (`LB_WGPU_DEVICE` / `LB_CUDA_DEVICE`).
mod common;
use common::Rng;
use half::f16;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{repack_q4_0, Gpu};

fn random_iq4_xs(rng: &mut Rng, n_super: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(n_super * 136);
    for _ in 0..n_super {
        let d = 0.002 + 0.01 * (rng.unit().abs());
        b.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for _ in 0..134 {
            b.push(rng.byte());
        }
    }
    b
}

fn deq_device(qs: &[u32], sc: &[u32], n_blocks: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_blocks * 32];
    for blk in 0..n_blocks {
        let d = f16::from_bits(((sc[blk / 2] >> (16 * (blk % 2))) & 0xffff) as u16).to_f32();
        for k in 0..4 {
            let w = qs[4 * blk + k];
            for j in 0..8 {
                let v = if j < 4 { 4 * j + k } else { 16 + 4 * (j - 4) + k };
                out[blk * 32 + v] = d * (((w >> (4 * j)) & 15) as f32 - 8.0);
            }
        }
    }
    out
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let e: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let n: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (e / n).sqrt()
}

#[test]
fn iq4_xs_device_transcode_matches_host() {
    if std::env::var_os("LB_WGPU_DEVICE").is_none() && std::env::var_os("LB_CUDA_DEVICE").is_none() {
        eprintln!("skipped: no device named");
        return;
    }
    let gpu = Gpu::open().unwrap();
    let mut rng = Rng(91);
    // one expert matrix of LFM2.5 (1536 x 2048) = 12,288 super-blocks
    let (rows, cols) = (1536usize, 2048usize);
    let n_super = rows * cols / 256;
    let raw = random_iq4_xs(&mut rng, n_super);
    let m = QuantMat::new(GgmlType::IQ4XS, rows, cols, &raw);
    let mut iq = vec![0f32; rows * cols];
    for r in 0..rows {
        m.dequant_row(r, &mut iq[r * cols..(r + 1) * cols]);
    }
    let n_blocks = rows * cols / 32;
    let words: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let rh = gpu.upload_u32(words);
    for search in [false, true] {
    // host reference for this encoder, from the same dequantized values
    let host: Vec<u8> = if search { lb_q4_gpu::transcode::quantize_q4_0_search(&iq) } else { iq.chunks_exact(cols).flat_map(lb_gguf::types::quantize_q4_0).collect() };
    let (hq, hs) = repack_q4_0(&host, n_blocks);
    let ((qh, nq), (sh, ns)) = gpu.transcode_iq4_xs_on_device_with(&rh, n_super, search);
    let (dq, ds) = (gpu.read_u32(&qh, nq).unwrap(), gpu.read_u32(&sh, ns).unwrap());
    let scale_diff = (0..n_blocks).filter(|&b| (ds[b / 2] >> (16 * (b % 2))) & 0xffff != (hs[b / 2] >> (16 * (b % 2))) & 0xffff).count();
    let word_diff = dq.iter().zip(&hq).filter(|(a, b)| a != b).count();
    let (e_dev, e_host) = (rel_rms(&deq_device(&dq, &ds, n_blocks), &iq), rel_rms(&deq_device(&hq, &hs, n_blocks), &iq));
    eprintln!(
        "IQ4_XS -> Q4_0 (search {search}), {n_blocks} blocks: scale differs in {scale_diff} ({:.3} %), quant words differ {word_diff}/{} ({:.3} %); rel RMS vs IQ4_XS: device {e_dev:.5}, host {e_host:.5}",
        100.0 * scale_diff as f64 / n_blocks as f64,
        hq.len(),
        100.0 * word_diff as f64 / hq.len() as f64
    );
    assert!(scale_diff * 100 <= n_blocks, "scale differs in {scale_diff} of {n_blocks} blocks");
    assert!(e_dev <= e_host * 1.01, "device transcode error {e_dev} > 1.01 x host {e_host}");
    }

    // timing: 64 such matrices (one MoE layer's gate or up for every expert), 5 reps after a warm-up
    let big = 64usize;
    let rawb: Vec<u32> = (0..big).flat_map(|_| raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()).collect();
    let rb = gpu.upload_u32(rawb);
    let _ = gpu.transcode_iq4_xs_on_device_with(&rb, big * n_super, true);
    gpu.sync().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..5 {
        let _ = gpu.transcode_iq4_xs_on_device_with(&rb, big * n_super, true);
    }
    gpu.sync().unwrap();
    let s = t.elapsed().as_secs_f64() / 5.0;
    let src = (big * n_super * 136) as f64;
    eprintln!("device transcode (search, the slower encoder): {:.1} MB IQ4_XS in {:.2} ms = {:.1} GB/s in, {:.2} ms per expert matrix", src / 1e6, s * 1e3, src / s / 1e9, s * 1e3 / big as f64);
    // host, same 1 matrix, for the comparison
    let t = std::time::Instant::now();
    let _ = lb_q4_gpu::transcode::quantize_q4_0_search(&iq);
    eprintln!("host transcode, search, single thread (rayon, this host): {:.2} ms per expert matrix", t.elapsed().as_secs_f64() * 1e3);
}
