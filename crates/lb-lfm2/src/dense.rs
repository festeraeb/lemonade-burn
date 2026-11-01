//! Resident dense (non-expert) weights on the device.
//!
//! Once every expert is resident (2432 of them for LFM2.5-24B-A2B, which is all of them),
//! the experts cost ~13% of a decode token and the rest is these matrices, still on the CPU
//! reference kernels. There are nine shapes and about 106 matvecs per token:
//!
//! ```text
//! 30 short-conv layers:  in_proj [6144, 2048]   out_proj [2048, 2048]
//! 10 attention layers:   q, o [2048, 2048]      k, v [512, 2048]
//!  2 leading dense FFNs: gate, up [11776, 2048] down [2048, 11776]
//! ```
//!
//! Measured on a P100 (`cargo run --release -p lb-q4-gpu --example dense_bench`): 273 ms
//! per token on the CPU against 27.6 ms on the device when each matvec syncs on its own,
//! and a round trip costs 0.31 ms, so nearly all of that 27.6 ms *is* the 106 round trips
//! and the kernels themselves are nearly free. Keeping a whole layer on the device would
//! take the 27.6 down further; that is a later change, and this one is the 10x.
//!
//! The tied head (`token_embd`, 128000 x 2048 Q6_K, used as the LM head) is here too: one
//! matrix, 55 ms of CPU per token on its own, and 327 MB once repacked (see
//! `lb_q4_gpu::repack_q6_k`).
//!
//! Unlike experts these are needed by *every* token, so there is nothing to place: upload
//! all of them if they fit (424 MB of Q4_0 plus the 327 MB head). The budget exists only so
//! a small card can take the matrices that save the most first.

use crate::weights::{Ffn, MatRef, Operator, Weights};
#[cfg(feature = "gpu")]
use std::collections::HashMap;

/// Dense matrices uploaded to the device, keyed by their GGUF byte offset (unique per
/// tensor, and what a [`MatRef`] carries, so a call site needs no extra naming). Two maps
/// because the quantization types need different kernels, not because they are used
/// differently.
#[derive(Default)]
pub struct DenseStore {
    #[cfg(feature = "gpu")]
    pub resident: HashMap<usize, lb_q4_gpu::GpuQ4Mat>,
    #[cfg(feature = "gpu")]
    pub q6: HashMap<usize, lb_q4_gpu::GpuQ6KMat>,
    pub bytes: usize,
}

impl DenseStore {
    pub fn n_resident(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            return self.resident.len() + self.q6.len();
        }
        #[allow(unreachable_code)]
        0
    }

    /// Every dense matrix a decode step runs. Router rows, norms and the conv taps are f32
    /// and stay on the host.
    fn candidates(w: &Weights) -> Vec<&MatRef> {
        let mut v = vec![&w.embd];
        for layer in &w.layers {
            match &layer.op {
                Operator::Conv { in_proj, out_proj, .. } => v.extend([in_proj, out_proj]),
                Operator::Attn { q, k, v: vp, o, .. } => v.extend([q, k, vp, o]),
            }
            if let Ffn::Dense { gate, up, down } = &layer.ffn {
                v.extend([gate, up, down]);
            }
        }
        v
    }

    /// Upload dense matrices until `budget_bytes` is spent, largest first (the CPU kernel's
    /// cost tracks bytes read, so the largest matrix saves the most). Returns how many were
    /// uploaded. Without the `gpu` feature: an error.
    #[cfg(feature = "gpu")]
    pub fn load(&mut self, gpu: &lb_q4_gpu::Gpu, w: &Weights, budget_bytes: usize) -> Result<usize, String> {
        let mut cand = Self::candidates(w);
        cand.sort_by_key(|m| std::cmp::Reverse(m.len));
        let mut n = 0;
        for m in cand {
            if self.resident.contains_key(&m.start) || self.q6.contains_key(&m.start) {
                continue;
            }
            // the device copy of a matrix is `device_bytes`, which for Q6_K is larger than
            // the file's: check the budget against what will actually be allocated
            if self.bytes + Self::device_bytes(m) > budget_bytes {
                continue;
            }
            match m.ty {
                lb_gguf::GgmlType::Q4_0 => {
                    let g = lb_q4_gpu::GpuQ4Mat::upload(gpu, w.mat(m))?;
                    self.bytes += g.bytes_on_device();
                    self.resident.insert(m.start, g);
                }
                lb_gguf::GgmlType::Q6K => {
                    let g = lb_q4_gpu::GpuQ6KMat::upload(gpu, w.mat(m))?;
                    self.bytes += g.bytes_on_device();
                    self.q6.insert(m.start, g);
                }
                // anything else stays on the CPU reference kernel
                _ => continue,
            }
            n += 1;
        }
        Ok(n)
    }

    /// What one matrix takes on the device: Q4_0 repacks to the same size, Q6_K to 320 bytes
    /// per 256 values against the file's 210.
    fn device_bytes(m: &MatRef) -> usize {
        match m.ty {
            lb_gguf::GgmlType::Q6K => m.rows * (m.cols / 256) * 320,
            _ => m.len,
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn load(&mut self, _w: &Weights, _budget_bytes: usize) -> Result<usize, String> {
        Err("built without the `gpu` feature: dense weights cannot be made resident".to_string())
    }

    /// Bytes every dense matrix we have a kernel for would take on the device.
    pub fn candidate_bytes(w: &Weights) -> usize {
        Self::candidates(w)
            .iter()
            .filter(|m| matches!(m.ty, lb_gguf::GgmlType::Q4_0 | lb_gguf::GgmlType::Q6K))
            .map(|m| Self::device_bytes(m))
            .sum()
    }
}
