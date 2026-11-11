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
    /// Whole short-conv blocks, keyed by their `in_proj` offset. A block owns both its
    /// projections, so those two never also go in `resident` -- that would upload 283 MB
    /// twice.
    #[cfg(feature = "gpu")]
    pub conv: HashMap<usize, lb_q4_gpu::GpuConv>,
    /// Conv layers whose device state is ahead of the host's `Cache::conv` (decode has run).
    /// A later multi-token pass has to pull the state back before it can use the CPU path.
    /// Behind a mutex because `forward` takes `&self`.
    #[cfg(feature = "gpu")]
    conv_dirty: std::sync::Mutex<std::collections::HashSet<usize>>,
    pub bytes: usize,
}

/// Whole short-conv blocks on the device: **off unless `LB_CONV_BLOCK=1`**, because measured
/// interleaved it does not pay.
///
/// It removes 30 of a token's ~145 host syncs, which looked like the next lever. Measured
/// against its own absence in this same binary, 15 reps an arm interleaved over three rounds:
/// loose matrices **9.185 tok/s**, whole blocks **8.245** -- **0.90x**, no win. Both arms had
/// a spread above 2x, so the truthful statement is "no measurable gain, possibly a small
/// loss", not "a 10% regression".
///
/// The arithmetic says why, and it was available before the kernel was written: a sync costs
/// ~0.24 ms more than it needs to, so 30 of them is **7 ms of a ~110 ms token, 1.07x** --
/// inside this box's noise. Removing *every* sync in the token would be 35 ms, 1.46x, still
/// under our 1.5x bar. The chain microbenchmark that motivated this (6.71x) compared 106
/// matvecs *in isolation*, where syncs are nearly all of the 29.76 ms; in the real model the
/// dense side costs 85-105 ms and syncs are only about a quarter of it.
///
/// The code stays, with its tests, because it is correct and the trade changes on a quieter
/// host or with a model whose layers are cheaper. Turn it on and measure before believing it.
#[cfg(feature = "gpu")]
pub fn conv_blocks_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LB_CONV_BLOCK").is_ok())
}

impl DenseStore {
    /// The device state of this conv layer is now ahead of the host's.
    #[cfg(feature = "gpu")]
    pub fn mark_conv_dirty(&self, in_proj_start: usize) {
        self.conv_dirty.lock().expect("conv dirty set").insert(in_proj_start);
    }

    /// Host and device conv state agree again.
    #[cfg(feature = "gpu")]
    pub fn mark_conv_clean(&self, in_proj_start: usize) {
        self.conv_dirty.lock().expect("conv dirty set").remove(&in_proj_start);
    }

    #[cfg(feature = "gpu")]
    pub fn is_conv_dirty(&self, in_proj_start: usize) -> bool {
        self.conv_dirty.lock().expect("conv dirty set").contains(&in_proj_start)
    }

    /// Zero every resident conv block's rolling state: a new sequence has no history.
    #[cfg(feature = "gpu")]
    pub fn reset_conv(&self, gpu: &lb_q4_gpu::Gpu) {
        for block in self.conv.values() {
            block.reset(gpu);
        }
        self.conv_dirty.lock().expect("conv dirty set").clear();
    }

    pub fn n_resident(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            return self.resident.len() + self.q6.len() + self.conv.len();
        }
        #[allow(unreachable_code)]
        0
    }

    /// Every dense matrix a decode step runs, *except* the conv projections, which are
    /// uploaded as whole blocks by [`Self::load_conv`]. Router rows, norms and the conv taps
    /// are f32 and stay on the host.
    fn candidates(w: &Weights) -> Vec<&MatRef> {
        let mut v = vec![&w.embd];
        for layer in &w.layers {
            match &layer.op {
                Operator::Attn { q, k, v: vp, o, .. } => v.extend([q, k, vp, o]),
                // with conv blocks off, the projections are ordinary resident matrices again
                #[cfg(feature = "gpu")]
                Operator::Conv { in_proj, out_proj, .. } if !conv_blocks_enabled() => v.extend([in_proj, out_proj]),
                #[cfg(not(feature = "gpu"))]
                Operator::Conv { in_proj, out_proj, .. } => v.extend([in_proj, out_proj]),
                Operator::Conv { .. } => {}
            }
            if let Ffn::Dense { gate, up, down } = &layer.ffn {
                v.extend([gate, up, down]);
            }
        }
        v
    }

    /// The conv projections, as (layer, in_proj, out_proj).
    fn conv_layers(w: &Weights) -> Vec<(usize, &MatRef, &MatRef)> {
        w.layers
            .iter()
            .enumerate()
            .filter_map(|(l, layer)| match &layer.op {
                Operator::Conv { in_proj, out_proj, .. } => Some((l, in_proj, out_proj)),
                Operator::Attn { .. } => None,
            })
            .collect()
    }

    /// CPU ms saved per device MB, measured (`dense_bench`, P100, 2026-11-01): the 106 dense
    /// Q4_0 matvecs are 273-316 ms of CPU for 423 device MB, the Q6_K head 54.1 ms for 328 MB.
    ///
    /// So a Q4_0 matrix is worth about 4.2x as much per byte as the head, and ordering by
    /// size alone would have it backwards: the head is the single largest candidate, so a
    /// budget that cannot take everything would spend a third of itself on the matrix with
    /// the worst return. (Found by a 600 MB budget quietly taking the head plus 37 of the 106
    /// Q4_0 matrices instead of all 106.)
    fn saving_per_byte(ty: lb_gguf::GgmlType) -> f64 {
        match ty {
            lb_gguf::GgmlType::Q6K => 0.16,
            t if as_q4(t) => 0.70,
            _ => 0.0,
        }
    }

    /// Upload order: best return per device byte first, larger matrices first within a type.
    fn by_return(a: &MatRef, b: &MatRef) -> std::cmp::Ordering {
        Self::saving_per_byte(b.ty).total_cmp(&Self::saving_per_byte(a.ty)).then(b.len.cmp(&a.len))
    }

    /// Upload dense matrices until `budget_bytes` is spent, best return per device byte
    /// first and larger matrices before smaller within a type. Returns how many were
    /// uploaded. Without the `gpu` feature: an error.
    ///
    /// Conv blocks go first, before any budget is spent on single matrices: they are 30 of the
    /// 40 layers and a whole block on the device saves a host sync as well as the two
    /// matmuls, which a loose `in_proj` cannot.
    #[cfg(feature = "gpu")]
    pub fn load(&mut self, gpu: &lb_q4_gpu::Gpu, w: &Weights, budget_bytes: usize) -> Result<usize, String> {
        let mut n = self.load_conv(gpu, w, budget_bytes)?;
        n += self.load_mats(gpu, w, budget_bytes)?;
        Ok(n)
    }

    /// Upload whole short-conv blocks (both projections, the taps and the rolling state).
    #[cfg(feature = "gpu")]
    fn load_conv(&mut self, gpu: &lb_q4_gpu::Gpu, w: &Weights, budget_bytes: usize) -> Result<usize, String> {
        let mut n = 0;
        if !conv_blocks_enabled() {
            return Ok(0);
        }
        for (l, in_proj, out_proj) in Self::conv_layers(w) {
            if self.conv.contains_key(&in_proj.start) {
                continue;
            }
            if !as_q4(in_proj.ty) || !as_q4(out_proj.ty) {
                continue;
            }
            let Operator::Conv { conv, .. } = &w.layers[l].op else { continue };
            // the taps and the rolling state are allocated too: count them, or the budget is
            // exceeded by a little per layer (it reported 752 MB of a 751 MB budget)
            let e = in_proj.rows / 3;
            let extra = (conv.len() + (conv.len() / e.max(1)).saturating_sub(1) * e) * 4;
            if self.bytes + in_proj.len + out_proj.len + extra > budget_bytes {
                continue;
            }
            let g = lb_q4_gpu::GpuConv::upload(gpu, w.mat(in_proj), conv, w.mat(out_proj))?;
            self.bytes += g.bytes_on_device();
            self.conv.insert(in_proj.start, g);
            n += 1;
        }
        Ok(n)
    }

    #[cfg(feature = "gpu")]
    fn load_mats(&mut self, gpu: &lb_q4_gpu::Gpu, w: &Weights, budget_bytes: usize) -> Result<usize, String> {
        let mut cand = Self::candidates(w);
        cand.sort_by(|a, b| Self::by_return(a, b));
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
                // Q4_0, or transcoded to Q4_0 as it becomes resident (X7b)
                t if t != lb_gguf::GgmlType::Q6K && as_q4(t) => {
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
            // everything else is resident as Q4_0 (transcoded if it was not)
            _ => m.rows * (m.cols / 32) * 18,
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn load(&mut self, _w: &Weights, _budget_bytes: usize) -> Result<usize, String> {
        Err("built without the `gpu` feature: dense weights cannot be made resident".to_string())
    }

    /// Bytes every dense matrix we have a kernel for would take on the device, conv blocks
    /// included (their state and taps are a few hundred KB and are not counted).
    pub fn candidate_bytes(w: &Weights) -> usize {
        let mats: usize = Self::candidates(w)
            .iter()
            .filter(|m| m.ty == lb_gguf::GgmlType::Q6K || as_q4(m.ty))
            .map(|m| Self::device_bytes(m))
            .sum();
        let convs: usize = Self::conv_layers(w).iter().map(|(_, a, b)| a.len + b.len).sum();
        mats + convs
    }
}

/// Whether a dense matrix of type `ty` runs on the device as Q4_0: it is Q4_0, or (X7b) any other
/// dequantizable type with whole Q4_0 blocks, transcoded when it becomes resident
/// (`lb_q4_gpu::transcode`; `LB_TRANSCODE=0` turns that off).
fn as_q4(ty: lb_gguf::GgmlType) -> bool {
    ty == lb_gguf::GgmlType::Q4_0 || (ty.block().is_some() && !matches!(ty, lb_gguf::GgmlType::F32 | lb_gguf::GgmlType::F16) && std::env::var("LB_TRANSCODE").map(|v| v != "0").unwrap_or(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lb_gguf::GgmlType;

    fn mat(ty: GgmlType, len: usize, start: usize) -> MatRef {
        MatRef { ty, rows: 1, cols: 32, start, len }
    }

    /// A budget that cannot take everything must not spend a third of itself on the Q6_K
    /// head: it saves 0.16 ms per device MB against a dense Q4_0 matrix's 0.70, so every
    /// Q4_0 matrix comes first however much larger the head is. Ordering by size alone put
    /// the head first and silently dropped 69 of the 106 Q4_0 matrices at a 600 MB budget.
    #[test]
    fn the_head_is_taken_after_the_q4_0_matrices_however_large_it_is() {
        let head = mat(GgmlType::Q6K, 215_000_000, 0);
        let big = mat(GgmlType::Q4_0, 13_570_000, 1);
        let small = mat(GgmlType::Q4_0, 2_360_000, 2);
        let mut v = vec![&head, &small, &big];
        v.sort_by(|a, b| DenseStore::by_return(a, b));
        assert_eq!(v.iter().map(|m| m.start).collect::<Vec<_>>(), vec![1, 2, 0], "Q4_0 largest-first, then the head");
    }

    /// A type with no kernel must never be ordered ahead of one that has it, and never be
    /// counted as a candidate.
    #[test]
    fn types_without_a_kernel_sort_last_and_do_not_count() {
        let q8 = mat(GgmlType::Q8_0, 99_000_000, 0);
        let q4 = mat(GgmlType::Q4_0, 1_000, 1);
        let mut v = vec![&q8, &q4];
        v.sort_by(|a, b| DenseStore::by_return(a, b));
        assert_eq!(v[0].start, 1, "the Q4_0 matrix is worth more than any byte count of Q8_0");
        assert_eq!(DenseStore::saving_per_byte(GgmlType::Q8_0), 0.0);
    }

    /// Q6_K takes 320 device bytes per 256 values against the file's 210, so a budget check
    /// against the file size would overcommit the card by half the head.
    #[test]
    fn q6_k_is_budgeted_at_its_device_size_not_its_file_size() {
        let head = mat(GgmlType::Q6K, 215_000_000, 0);
        let head = MatRef { rows: 128_000, cols: 2048, ..head };
        assert_eq!(DenseStore::device_bytes(&head), 128_000 * 8 * 320);
        assert!(DenseStore::device_bytes(&head) > head.len, "device copy is larger than the file's");
        let q4 = mat(GgmlType::Q4_0, 13_570_000, 1);
        assert_eq!(DenseStore::device_bytes(&q4), q4.len, "Q4_0 repacks to the same size");
    }
}
