//! A whole MoE layer in two dispatches must give **exactly** what the per-expert path gives.
//!
//! `GpuExpertLayer` concatenates every expert of a layer into one buffer per projection and
//! takes the routing as data, so a top-4 layer costs 2 dispatches instead of 8. That is only
//! worth anything if the arithmetic is untouched, and it should be bit-identical rather than
//! merely close: each output row is the same row of the same expert, dotted in the same block
//! order by the same `q4_row_partial`, with only the base offset into the buffer changed.
//!
//! So these assert `==` on f32 bits, not a tolerance. A tolerance here would hide exactly the
//! bug this design can have: an expert boundary landing mid-u32 in the concatenated scale
//! buffer, which shifts every scale after it and is easy to mistake for rounding.
//!
//! Run on the card chosen by LB_WGPU_DEVICE.
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert, GpuExpertLayer};

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

/// Build `n` experts of shape (n_ff x n_embd, n_ff x n_embd, n_embd x n_ff) and return the
/// raw bytes so both paths can be built from identical weights.
fn experts(rng: &mut Rng, n: usize, n_ff: usize, n_embd: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    (0..n)
        .map(|_| (q4(rng, n_ff, n_embd), q4(rng, n_ff, n_embd), q4(rng, n_embd, n_ff)))
        .collect()
}

/// Run `routes` through the layer kernel and through the per-expert kernels, and compare.
fn check(n_experts: usize, n_ff: usize, n_embd: usize, n_tokens: usize, routes: &[u32], seed: u64) {
    let gpu = Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}"));
    let mut rng = Rng(seed);
    let raw = experts(&mut rng, n_experts, n_ff, n_embd);
    let x: Vec<f32> = (0..n_tokens * n_embd).map(|_| rng.unit() * 3.0).collect();

    let layer = GpuExpertLayer::upload(
        &gpu,
        raw.iter().map(|(g, u, d)| {
            (
                QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, g),
                QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, u),
                QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, d),
            )
        }),
    )
    .expect("layer upload");
    assert_eq!(layer.n_experts(), n_experts);

    let xh = gpu.upload_f32(&x);
    let (out, n) = layer.launch(&gpu, &xh, n_tokens, routes).expect("layer launch");
    let got = gpu.read_all(vec![(out, n)]).expect("read")[0].clone();

    // The per-expert path, one expert at a time, exactly as the model runs it today.
    for (slot, c) in routes.chunks(2).enumerate() {
        let (e, tok) = (c[0] as usize, c[1] as usize);
        let (g, u, d) = &raw[e];
        let ex = GpuExpert::upload(
            &gpu,
            QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, g),
            QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, u),
            QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, d),
        )
        .expect("expert upload");
        let want = ex.ffn(&gpu, &x[tok * n_embd..(tok + 1) * n_embd], 1).expect("expert ffn");
        let have = &got[slot * n_embd..(slot + 1) * n_embd];
        for (i, (h, w)) in have.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                h.to_bits(),
                w.to_bits(),
                "slot {slot} (expert {e}, token {tok}) element {i}: layer {h} vs per-expert {w}"
            );
        }
    }
}

/// The real decode shape: LFM2.5-24B-A2B is 64 experts, top-4, n_embd 2048, n_ff_exp 1536,
/// one token at a time. Four slots all reading token 0.
#[test]
fn lfm2_decode_layer_top4_one_token() {
    check(8, 1536, 2048, 1, &[0, 0, 3, 0, 5, 0, 7, 0], 0xA11CE);
}

/// Prefill: several tokens, each routed to its own set of experts, interleaved in slot order
/// so a kernel that confused `slot` with `token` could not pass.
#[test]
fn several_tokens_each_to_a_different_expert() {
    check(6, 128, 256, 4, &[2, 0, 5, 1, 0, 2, 3, 3, 1, 0, 4, 2], 0xBEEF1);
}

/// The same expert serving two different tokens, and the same token served by two experts --
/// both happen under real routing and both exercise the offset arithmetic differently.
#[test]
fn an_expert_may_repeat_and_a_token_may_repeat() {
    check(4, 96, 128, 3, &[1, 0, 1, 2, 3, 0, 3, 2, 0, 1], 0xC0FFEE);
}

/// The last expert, whose rows sit at the far end of the concatenated buffer: an off-by-one in
/// `e * blocks_per_expert` shows up here and nowhere else.
#[test]
fn the_last_expert_in_the_buffer_is_addressed_correctly() {
    check(5, 64, 96, 1, &[4, 0], 0xD00D);
}

/// The odd-block-count guard in `upload` is **unreachable**, and this records why rather than
/// pretending to test it.
///
/// `down` must be the transpose of gate/up, so `down.cols == n_ff` and `gate.cols == n_embd`
/// both have to satisfy Q4_0's `cols % 32 == 0`. That makes n_ff and n_embd both multiples of
/// 32, so `blocks_per_expert = n_ff * n_embd / 32` is a multiple of 32 -- never odd. Any input
/// that could trip the guard is rejected earlier by the shape check.
///
/// The guard stays anyway: it is one comparison, and it is the thing that would catch a future
/// shape or quantization whose block count is not forced even. Concatenated scales pack two f16
/// per u32 by GLOBAL block index, so an odd expert would shift every scale after it -- a bug
/// that looks exactly like rounding.
#[test]
fn block_count_per_expert_cannot_be_odd_for_any_valid_shape() {
    for n_embd in [32usize, 64, 96, 128, 2048] {
        for n_ff in [32usize, 64, 1536] {
            let blocks = n_ff * (n_embd / 32);
            assert_eq!(blocks % 2, 0, "n_ff {n_ff} x n_embd {n_embd} gives {blocks} blocks");
        }
    }
}

/// A shape where `down` is not the transpose is refused before anything is uploaded.
#[test]
fn a_down_that_is_not_the_transpose_is_refused() {
    let gpu = Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}"));
    let mut rng = Rng(7);
    let (g, u, d) = (q4(&mut rng, 64, 32), q4(&mut rng, 64, 32), q4(&mut rng, 64, 32));
    let r = GpuExpertLayer::upload(
        &gpu,
        std::iter::once((
            QuantMat::new(GgmlType::Q4_0, 64, 32, &g),
            QuantMat::new(GgmlType::Q4_0, 64, 32, &u),
            QuantMat::new(GgmlType::Q4_0, 64, 32, &d),
        )),
    );
    let err = match r {
        Ok(_) => panic!("down must be the transpose of gate/up"),
        Err(e) => e,
    };
    assert!(err.contains("transpose"), "{err}");
}

/// A route naming an expert or a token that does not exist is a caller bug; it must come back
/// as an error rather than reading somewhere else in the buffer.
#[test]
fn out_of_range_routes_are_refused() {
    let gpu = Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}"));
    let mut rng = Rng(11);
    let raw = experts(&mut rng, 2, 64, 96);
    let layer = GpuExpertLayer::upload(
        &gpu,
        raw.iter().map(|(g, u, d)| {
            (
                QuantMat::new(GgmlType::Q4_0, 64, 96, g),
                QuantMat::new(GgmlType::Q4_0, 64, 96, u),
                QuantMat::new(GgmlType::Q4_0, 96, 64, d),
            )
        }),
    )
    .expect("upload");
    let x = vec![0.1f32; 96];
    let xh = gpu.upload_f32(&x);
    let e = layer.launch(&gpu, &xh, 1, &[2, 0]).expect_err("expert 2 of 2 must be refused");
    assert!(e.contains("expert 2"), "{e}");
    let e = layer.launch(&gpu, &xh, 1, &[0, 5]).expect_err("token 5 of 1 must be refused");
    assert!(e.contains("token 5"), "{e}");
    let e = layer.launch(&gpu, &xh, 1, &[0]).expect_err("an odd routes length must be refused");
    assert!(e.contains("pairs"), "{e}");
}
