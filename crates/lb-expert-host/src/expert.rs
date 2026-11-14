//! The CPU reference arithmetic of one expert. This is the single source of it: the trunk's
//! one-process path (`lb_lfm2::moe::expert_ffn`) and the host's CPU backend both call
//! [`swiglu`], so a remote expert and a local one are the same instructions on the same bytes.

use lb_gguf::kernels_i8::matvec_q4_0_i8;
use lb_gguf::{matvec_f32, GgmlType, QuantMat};

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

/// [`swiglu`] with the int8-activation kernel (`lb_gguf::kernels_i8`) when all three matrices
/// are Q4_0; any other weight type takes [`swiglu`] unchanged. Each row is run on its own
/// (the kernel quantizes one activation row at a time). Faster than the f32 reference, with
/// slightly different arithmetic: the activations are rounded to 8 bits per 32-value block.
pub fn swiglu_i8(gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>, x: &[f32], m: usize) -> Vec<f32> {
    if gate.ty != GgmlType::Q4_0 || up.ty != GgmlType::Q4_0 || down.ty != GgmlType::Q4_0 {
        return swiglu(gate, up, down, x, m);
    }
    let (n_in, n_ff, n_embd) = (gate.cols, gate.rows, down.rows);
    let mut out = vec![0f32; m * n_embd];
    let mut g = vec![0f32; n_ff];
    let mut u = vec![0f32; n_ff];
    for t in 0..m {
        let xr = &x[t * n_in..(t + 1) * n_in];
        matvec_q4_0_i8(gate, xr, &mut g, 0);
        matvec_q4_0_i8(up, xr, &mut u, 0);
        for (gi, ui) in g.iter_mut().zip(&u) {
            let s = *gi / (1.0 + (-*gi).exp());
            *gi = s * ui;
        }
        matvec_q4_0_i8(down, &g, &mut out[t * n_embd..(t + 1) * n_embd], 0);
    }
    out
}

/// A fingerprint of this machine's CPU expert arithmetic. Everything in [`swiglu`] is plain
/// IEEE f32 in a fixed order except `exp`, which comes from the C library and may round the
/// last bit differently between library versions or CPU-dispatched variants. Two processes with
/// the same fingerprint computed `silu` identically on a 2^20-point sweep of the range that
/// matters; a CPU host says its fingerprint in the hello, and a trunk warns when it differs from
/// its own (the outputs then agree to rounding, not bit for bit).
pub fn arith_fingerprint() -> u64 {
    static F: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for i in 0..(1u32 << 20) {
            let x = -24.0f32 + 48.0 * (i as f32 / (1u32 << 20) as f32);
            let s = x / (1.0 + (-x).exp());
            h = (h ^ s.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    })
}
