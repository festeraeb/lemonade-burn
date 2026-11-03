//! The GPU Q4_0 matvec must match the CPU reference (`lb_gguf::matvec_f32`) to f32
//! rounding: the per-block sums are integers times f32 activations, so only the order of
//! the block accumulation differs. Run on the card chosen by LB_WGPU_DEVICE.
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::q4_0_matvec_gpu;

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

fn check(rows: usize, cols: usize, m: usize, seed: u64) {
    let mut rng = Rng(seed);
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    let bytes: Vec<u8> = wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect();
    let w = QuantMat::new(GgmlType::Q4_0, rows, cols, &bytes);
    let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 4.0).collect();
    let mut want = vec![0f32; m * rows];
    matvec_f32(w, &x, m, &mut want);
    let got = q4_0_matvec_gpu(w, &x, m).unwrap_or_else(|e| panic!("GPU call failed: {e}"));
    assert_eq!(got.len(), want.len());
    let mut worst = 0f32;
    for (i, (g, e)) in got.iter().zip(&want).enumerate() {
        let tol = 1e-4 * (1.0 + e.abs()) + 1e-5 * cols as f32;
        assert!((g - e).abs() <= tol, "shape ({rows},{cols},{m}) elem {i} (token {}, row {}): gpu {g} cpu {e}", i / rows, i % rows);
        worst = worst.max((g - e).abs());
    }
    eprintln!("({rows},{cols},{m}): max |gpu-cpu| = {worst:e}");
}

#[test]
fn one_block_rows() {
    check(8, 32, 1, 1);
}

#[test]
fn odd_rows_three_tokens() {
    check(37, 96, 3, 2);
}

#[test]
fn lfm2_expert_gate_shape() {
    // one LFM2.5 expert gate/up matrix: 1536 rows x 2048 cols
    check(1536, 2048, 1, 3);
}

#[test]
fn lfm2_expert_down_four_tokens() {
    check(2048, 1536, 4, 4);
}

/// 48 tokens x 1536 rows = 73,728 outputs: more cubes than one dispatch dimension holds
/// (65,535); the launch must spill into the second dimension. Regression for the
/// 2026-11-01 failure on a 44-token prompt.
#[test]
fn prefill_batch_beyond_one_dispatch_dimension() {
    check(1536, 2048, 48, 5);
}

#[test]
fn real_q4_0_rows_from_gguf_if_present() {
    // A real expert matrix from the QAD-Q4_0 file, when it is on disk.
    let path = std::path::Path::new("/mnt/t440/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skip: no model file");
        return;
    }
    let g = lb_gguf::Gguf::open(path).unwrap();
    let t = g.tensor("blk.2.ffn_gate_exps.weight").unwrap();
    assert_eq!(t.ty, GgmlType::Q4_0);
    let (cols, rows) = (t.dims[0], t.dims[1]);
    let per = t.ty.row_bytes(cols).unwrap() * rows;
    let bytes = &g.tensor_bytes(t)[..per]; // expert 0
    let w = QuantMat::new(GgmlType::Q4_0, rows, cols, bytes);
    let mut rng = Rng(9);
    let x: Vec<f32> = (0..2 * cols).map(|_| rng.unit()).collect();
    let mut want = vec![0f32; 2 * rows];
    matvec_f32(w, &x, 2, &mut want);
    let got = q4_0_matvec_gpu(w, &x, 2).unwrap();
    for (i, (g, e)) in got.iter().zip(&want).enumerate() {
        assert!((g - e).abs() <= 1e-4 * (1.0 + e.abs()) + 1e-5 * cols as f32, "elem {i}: gpu {g} cpu {e}");
    }
}

/// A resident SwiGLU expert on the device must match the CPU composition
/// `down(silu(gate x) * up x)` built from the same CPU reference matvecs.
#[test]
fn resident_expert_ffn_matches_cpu() {
    use lb_q4_gpu::{Gpu, GpuExpert};
    let (n_embd, n_ff, m) = (256usize, 192usize, 3usize);
    let mut rng = Rng(11);
    let mk = |rng: &mut Rng, rows: usize, cols: usize| -> Vec<u8> {
        let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
        wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
    };
    let (gb, ub, db) = (mk(&mut rng, n_ff, n_embd), mk(&mut rng, n_ff, n_embd), mk(&mut rng, n_embd, n_ff));
    let gate = QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &gb);
    let up = QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &ub);
    let down = QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &db);
    let x: Vec<f32> = (0..m * n_embd).map(|_| rng.unit()).collect();
    let (mut g, mut u) = (vec![0f32; m * n_ff], vec![0f32; m * n_ff]);
    matvec_f32(gate, &x, m, &mut g);
    matvec_f32(up, &x, m, &mut u);
    for (gi, ui) in g.iter_mut().zip(&u) {
        *gi = *gi / (1.0 + (-*gi).exp()) * ui;
    }
    let mut want = vec![0f32; m * n_embd];
    matvec_f32(down, &g, m, &mut want);
    let gpu = Gpu::open().unwrap_or_else(|e| panic!("{e}"));
    let e = GpuExpert::upload(&gpu, gate, up, down).unwrap();
    let got = e.ffn(&gpu, &x, m).unwrap();
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() <= 1e-3 * (1.0 + b.abs()), "elem {i}: gpu {a} cpu {b}");
    }
}

/// The scratch pool recycles buffers after every `read_all`. If it ever handed back a buffer
/// the device had not finished with, or returned one of the wrong size, a second round would
/// differ from the first. Several experts per round, several rounds, all results identical.
#[test]
fn pooled_scratch_reuse_is_stable_across_rounds() {
    use lb_q4_gpu::{Gpu, GpuExpert};
    let (n_embd, n_ff) = (256usize, 192usize);
    let mut rng = Rng(23);
    let mk = |rng: &mut Rng, rows: usize, cols: usize| -> Vec<u8> {
        let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
        wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
    };
    let (gb, ub, db) = (mk(&mut rng, n_ff, n_embd), mk(&mut rng, n_ff, n_embd), mk(&mut rng, n_embd, n_ff));
    let gate = QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &gb);
    let up = QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &ub);
    let down = QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &db);

    let gpu = Gpu::open().unwrap_or_else(|e| panic!("{e}"));
    let e = GpuExpert::upload(&gpu, gate, up, down).unwrap();
    // two token counts, so the pool holds more than one size at once
    let mut first: Vec<Vec<f32>> = Vec::new();
    for round in 0..4 {
        let mut got_round = Vec::new();
        for &m in &[1usize, 3] {
            let x: Vec<f32> = (0..m * n_embd).map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5).collect();
            let xh = gpu.upload_f32(&x);
            // three experts in one round, as a layer would
            let hs: Vec<_> = (0..3).map(|_| e.launch(&gpu, &xh, m)).collect();
            let outs = gpu.read_all(hs).expect("read");
            for o in &outs {
                assert_eq!(o.len(), m * n_embd, "wrong output length at m={m}");
                assert!(o.iter().all(|v| v.is_finite()), "non-finite output at m={m}");
            }
            // every expert in the round is the same weights on the same x
            for o in &outs[1..] {
                assert_eq!(&outs[0], o, "experts in one round disagree at m={m}");
            }
            got_round.push(outs[0].clone());
        }
        if round == 0 {
            first = got_round;
            assert!(gpu.pooled() > 0, "nothing was recycled after the first round");
        } else {
            assert_eq!(first, got_round, "round {round} differs from round 0: pool handed back a dirty buffer");
        }
    }
}
