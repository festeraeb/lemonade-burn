//! The CPU reference arithmetic of one expert. This is the single source of it: the trunk's
//! one-process path (`lb_lfm2::moe::expert_ffn`) and the host's CPU backend both call
//! [`swiglu`], so a remote expert and a local one are the same instructions on the same bytes.

use lb_gguf::{matvec_f32, QuantMat};

/// One SwiGLU expert on `m` rows of `x`: `down(silu(gate x) * up x)`. Returns `m` rows of
/// `down.rows`. The batch is computed as one call per matrix (`matvec_f32` treats `m == 1`
/// and `m > 1` differently, so the batch an expert is given is part of the result).
pub fn swiglu(gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>, x: &[f32], m: usize) -> Vec<f32> {
    let n_ff = gate.rows;
    let n_embd = down.rows;
    let mut g = vec![0f32; m * n_ff];
    let mut u = vec![0f32; m * n_ff];
    matvec_f32(gate, x, m, &mut g);
    matvec_f32(up, x, m, &mut u);
    for (gi, ui) in g.iter_mut().zip(&u) {
        let s = *gi / (1.0 + (-*gi).exp());
        *gi = s * ui;
    }
    let mut out = vec![0f32; m * n_embd];
    matvec_f32(down, &g, m, &mut out);
    out
}
