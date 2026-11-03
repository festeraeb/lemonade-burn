//! `GpuConv` must reproduce the LFM2 short-conv block exactly, including its rolling state.
//!
//! The state is the part worth testing: it lives on the device and the kernel rewrites it in
//! place, so a single-token check would pass even if the history were wrong. Every test here
//! runs a sequence and compares token by token, which is the only way a bad shift shows up.
//!
//! The reference is written out longhand from the block definition rather than called into
//! `lb-lfm2` (which would drag Burn into this crate): `bx = b * x`, a tap accumulation over
//! the last `l_cache` rows of `bx`, scaled by `c`, between two quantized projections.
mod common;
use common::{assert_close, Rng};
use lb_gguf::{matvec_f32, GgmlType, QuantMat};

/// One conv block on the CPU for a whole sequence, carrying the state forward.
/// Returns one output row per token.
fn cpu_conv_sequence(
    in_proj: QuantMat<'_>,
    conv: &[f32],
    out_proj: QuantMat<'_>,
    xs: &[Vec<f32>],
    e: usize,
    l: usize,
) -> Vec<Vec<f32>> {
    // state[k] is the row l-1-k steps back; starts empty (zeros)
    let mut state = vec![vec![0f32; e]; l - 1];
    let mut outs = Vec::new();
    for x in xs {
        let mut bcx = vec![0f32; 3 * e];
        matvec_f32(in_proj, x, 1, &mut bcx);
        let bx: Vec<f32> = (0..e).map(|i| bcx[i] * bcx[2 * e + i]).collect();
        // z = [state ..., bx], l rows
        let mut z: Vec<&Vec<f32>> = state.iter().collect();
        z.push(&bx);
        let y: Vec<f32> = (0..e)
            .map(|i| {
                let acc: f32 = (0..l).map(|k| conv[k * e + i] * z[k][i]).sum();
                bcx[e + i] * acc
            })
            .collect();
        let mut out = vec![0f32; e];
        matvec_f32(out_proj, &y, 1, &mut out);
        outs.push(out);
        // keep the l-1 most recent rows
        state.remove(0);
        state.push(bx);
    }
    outs
}

/// Random Q4_0 rows with a sane f16 scale in each block. Fully random bytes give random
/// scales, which include inf and NaN, and then both sides are garbage rather than compared.
fn random_q4_0(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let mut v: Vec<u8> = (0..rows * (cols / 32) * 18).map(|_| rng.byte()).collect();
    for blk in v.chunks_exact_mut(18) {
        let d = half::f16::from_f32(0.002 + 0.03 * (rng.unit() * 0.5 + 0.5));
        blk[0..2].copy_from_slice(&d.to_le_bytes());
    }
    v
}

fn check(e: usize, l: usize, tokens: usize, seed: u64) {
    let mut rng = Rng(seed);
    let ip_bytes = random_q4_0(&mut rng, 3 * e, e);
    let op_bytes = random_q4_0(&mut rng, e, e);
    let in_proj = QuantMat::new(GgmlType::Q4_0, 3 * e, e, &ip_bytes);
    let out_proj = QuantMat::new(GgmlType::Q4_0, e, e, &op_bytes);
    let conv: Vec<f32> = (0..l * e).map(|_| rng.unit()).collect();
    let xs: Vec<Vec<f32>> = (0..tokens).map(|_| (0..e).map(|_| rng.unit()).collect()).collect();

    let want = cpu_conv_sequence(in_proj, &conv, out_proj, &xs, e, l);

    let gpu = lb_q4_gpu::Gpu::open().expect("no GPU (set LB_WGPU_DEVICE)");
    let block = lb_q4_gpu::GpuConv::upload(&gpu, in_proj, &conv, out_proj).expect("upload");
    for (t, x) in xs.iter().enumerate() {
        let got = block.forward(&gpu, x).expect("conv block on device");
        assert_close(&got, &want[t], e, e, &format!("conv e={e} l={l} token {t}"));
    }
}

/// The real LFM2.5 shape: n_embd 2048, l_cache 3. Several tokens, so the state has to be
/// right twice over: once while it is still filling with zeros, and once when it is full.
#[test]
fn lfm2_conv_block_over_a_sequence() {
    check(2048, 3, 5, 11);
}

/// A longer kernel exercises the state shift loop, which is a no-op at l = 3.
#[test]
fn longer_tap_window_shifts_state_correctly() {
    check(256, 5, 6, 12);
}

/// Resetting must put the block back to a cold start: the same token fed to a fresh block and
/// to a reset block has to give the same answer, or sequences would leak into each other.
#[test]
fn reset_returns_the_block_to_a_cold_start() {
    let (e, l) = (256usize, 3usize);
    let mut rng = Rng(13);
    let ip_bytes = random_q4_0(&mut rng, 3 * e, e);
    let op_bytes = random_q4_0(&mut rng, e, e);
    let in_proj = QuantMat::new(GgmlType::Q4_0, 3 * e, e, &ip_bytes);
    let out_proj = QuantMat::new(GgmlType::Q4_0, e, e, &op_bytes);
    let conv: Vec<f32> = (0..l * e).map(|_| rng.unit()).collect();
    let x: Vec<f32> = (0..e).map(|_| rng.unit()).collect();

    let gpu = lb_q4_gpu::Gpu::open().expect("no GPU (set LB_WGPU_DEVICE)");
    let block = lb_q4_gpu::GpuConv::upload(&gpu, in_proj, &conv, out_proj).expect("upload");
    let cold = block.forward(&gpu, &x).expect("first token");
    // run it on far enough to fill the state with something else
    for _ in 0..4 {
        block.forward(&gpu, &x).expect("advance");
    }
    let warm = block.forward(&gpu, &x).expect("warm token");
    assert!(warm.iter().zip(&cold).any(|(a, b)| (a - b).abs() > 1e-3), "state must change the output, else this test proves nothing");
    block.reset(&gpu);
    let after = block.forward(&gpu, &x).expect("token after reset");
    assert_close(&after, &cold, e, e, "conv block after reset");
}
