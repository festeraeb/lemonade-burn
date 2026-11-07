//! Tensor map for LFM2 / LFM2-MoE GGUF files. Quantized matrices stay in the mmap and are
//! served as [`QuantMat`] views; small f32 vectors (norms, conv kernels, router) are copied.
//!
//! GGUF names (per block `blk.N.`):
//! * `attn_norm` (operator pre-norm), `ffn_norm` (FFN pre-norm)
//! * conv layers: `shortconv.in_proj [n_embd -> 3 n_embd]`, `shortconv.conv [l_cache, n_embd]`,
//!   `shortconv.out_proj [n_embd -> n_embd]`
//! * attention layers: `attn_q/k/v/output`, `attn_q_norm`, `attn_k_norm [head_dim]`
//! * dense FFN: `ffn_gate/up [n_embd -> n_ff]`, `ffn_down [n_ff -> n_embd]`
//! * MoE FFN: `ffn_gate_inp [n_embd -> n_expert]` (f32 router), `exp_probs_b [n_expert]`,
//!   `ffn_gate_exps/ffn_up_exps [n_expert, n_ff_exp, n_embd]`, `ffn_down_exps [n_expert, n_embd, n_ff_exp]`
//! * `token_embd [n_vocab, n_embd]` (also the LM head: tied), `token_embd_norm` = final norm.

use crate::config::Config;
use anyhow::{bail, Context, Result};
use lb_gguf::{Gguf, GgmlType, QuantMat, TensorInfo};
use std::sync::Arc;

/// A quantized 2-D matrix inside the file: `rows` of `cols`.
#[derive(Clone, Debug)]
pub struct MatRef {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub start: usize,
    pub len: usize,
}

/// A stack of `n` equally shaped quantized matrices (the experts of one projection).
#[derive(Clone, Debug)]
pub struct ExpertMats {
    pub ty: GgmlType,
    pub n: usize,
    pub rows: usize,
    pub cols: usize,
    pub start: usize,
    pub per_expert: usize,
}

impl ExpertMats {
    /// The whole stack as one matrix `[n * rows, cols]`: the experts are contiguous in the file.
    pub fn all(&self) -> MatRef {
        MatRef { ty: self.ty, rows: self.n * self.rows, cols: self.cols, start: self.start, len: self.n * self.per_expert }
    }

    pub fn expert(&self, i: usize) -> MatRef {
        assert!(i < self.n);
        MatRef { ty: self.ty, rows: self.rows, cols: self.cols, start: self.start + i * self.per_expert, len: self.per_expert }
    }
}

pub enum Operator {
    Conv {
        in_proj: MatRef,
        /// `[l_cache][n_embd]` (rearranged from the file's per-channel `[n_embd][l_cache]`):
        /// tap `k` multiplies the row `l_cache - 1 - k` steps back.
        conv: Vec<f32>,
        out_proj: MatRef,
    },
    Attn {
        q: MatRef,
        k: MatRef,
        v: MatRef,
        o: MatRef,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

pub enum Ffn {
    Dense { gate: MatRef, up: MatRef, down: MatRef },
    Moe {
        /// `[n_expert][n_embd]` f32 router rows.
        router: Vec<f32>,
        /// Selection bias `[n_expert]` (added to the probabilities for top-k only).
        bias: Vec<f32>,
        gate: ExpertMats,
        up: ExpertMats,
        down: ExpertMats,
    },
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub op: Operator,
    pub ffn: Ffn,
}

pub struct Weights {
    pub gguf: Arc<Gguf>,
    pub cfg: Config,
    pub layers: Vec<LayerWeights>,
    pub embd: MatRef,
    pub final_norm: Vec<f32>,
}

impl Weights {
    pub fn mat(&self, m: &MatRef) -> QuantMat<'_> {
        let bytes = &self.gguf.tensor_bytes_at(m.start, m.len);
        QuantMat::new(m.ty, m.rows, m.cols, bytes)
    }

    pub fn load(gguf: Arc<Gguf>) -> Result<Self> {
        let cfg = Config::from_gguf(&gguf)?;
        let g = &*gguf;
        let info = |name: &str| -> Result<&TensorInfo> { g.tensor(name).with_context(|| format!("tensor {name} missing")) };
        let vec = |name: &str, n: usize| -> Result<Vec<f32>> {
            let t = info(name)?;
            if t.n_elems() != n {
                bail!("{name}: {} values, expected {n}", t.n_elems());
            }
            Ok(g.tensor_f32(t))
        };
        let mat = |name: &str, cols: usize, rows: usize| -> Result<MatRef> {
            let t = info(name)?;
            if t.dims.len() != 2 || t.dims[0] != cols || t.dims[1] != rows {
                bail!("{name}: dims {:?}, expected [{cols}, {rows}]", t.dims);
            }
            let len = t.n_bytes().with_context(|| format!("{name}: unsupported type {}", t.ty.name()))?;
            if t.ty.row_bytes(cols).is_none() {
                bail!("{name}: {cols} cols is not a whole number of {} blocks", t.ty.name());
            }
            Ok(MatRef { ty: t.ty, rows, cols, start: t.offset, len })
        };
        let exps = |name: &str, cols: usize, rows: usize, n: usize| -> Result<ExpertMats> {
            let t = info(name)?;
            if t.dims.len() != 3 || t.dims[0] != cols || t.dims[1] != rows || t.dims[2] != n {
                bail!("{name}: dims {:?}, expected [{cols}, {rows}, {n}]", t.dims);
            }
            let rb = t.ty.row_bytes(cols).with_context(|| format!("{name}: unsupported type {}", t.ty.name()))?;
            Ok(ExpertMats { ty: t.ty, n, rows, cols, start: t.offset, per_expert: rb * rows })
        };

        let (e, l, hd) = (cfg.n_embd, cfg.l_cache, cfg.head_dim());
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let op = if cfg.is_attn(i) {
                let kvd = cfg.kv_heads[i] * hd;
                Operator::Attn {
                    q: mat(&format!("{p}.attn_q.weight"), e, cfg.n_head * hd)?,
                    k: mat(&format!("{p}.attn_k.weight"), e, kvd)?,
                    v: mat(&format!("{p}.attn_v.weight"), e, kvd)?,
                    o: mat(&format!("{p}.attn_output.weight"), cfg.n_head * hd, e)?,
                    q_norm: vec(&format!("{p}.attn_q_norm.weight"), hd)?,
                    k_norm: vec(&format!("{p}.attn_k_norm.weight"), hd)?,
                }
            } else {
                // conv.weight has GGUF dims [l_cache, n_embd]: each channel's taps are contiguous.
                let raw = vec(&format!("{p}.shortconv.conv.weight"), l * e)?;
                let mut conv = vec![0f32; l * e];
                for c in 0..e {
                    for k in 0..l {
                        conv[k * e + c] = raw[c * l + k];
                    }
                }
                Operator::Conv {
                    in_proj: mat(&format!("{p}.shortconv.in_proj.weight"), e, 3 * e)?,
                    conv,
                    out_proj: mat(&format!("{p}.shortconv.out_proj.weight"), e, e)?,
                }
            };
            let ffn = if cfg.is_moe(i) {
                Ffn::Moe {
                    router: vec(&format!("{p}.ffn_gate_inp.weight"), cfg.n_expert * e)?,
                    bias: vec(&format!("{p}.exp_probs_b.bias"), cfg.n_expert)?,
                    gate: exps(&format!("{p}.ffn_gate_exps.weight"), e, cfg.n_ff_exp, cfg.n_expert)?,
                    up: exps(&format!("{p}.ffn_up_exps.weight"), e, cfg.n_ff_exp, cfg.n_expert)?,
                    down: exps(&format!("{p}.ffn_down_exps.weight"), cfg.n_ff_exp, e, cfg.n_expert)?,
                }
            } else {
                Ffn::Dense {
                    gate: mat(&format!("{p}.ffn_gate.weight"), e, cfg.n_ff)?,
                    up: mat(&format!("{p}.ffn_up.weight"), e, cfg.n_ff)?,
                    down: mat(&format!("{p}.ffn_down.weight"), cfg.n_ff, e)?,
                }
            };
            layers.push(LayerWeights {
                attn_norm: vec(&format!("{p}.attn_norm.weight"), e)?,
                ffn_norm: vec(&format!("{p}.ffn_norm.weight"), e)?,
                op,
                ffn,
            });
        }
        let embd = mat("token_embd.weight", e, cfg.n_vocab)?;
        if g.tensor("output.weight").is_some() {
            bail!("untied output.weight is not supported yet (LFM2.5 ties it)");
        }
        let final_norm = vec("token_embd_norm.weight", e)?;
        Ok(Self { gguf, cfg, layers, embd, final_norm })
    }

    /// Bytes of one expert (gate + up + down) in `layer`, 0 for dense layers.
    pub fn expert_bytes(&self, layer: usize) -> usize {
        match &self.layers[layer].ffn {
            Ffn::Moe { gate, up, down, .. } => gate.per_expert + up.per_expert + down.per_expert,
            Ffn::Dense { .. } => 0,
        }
    }

    /// Dequantized embedding row of token `t`.
    pub fn embed_row(&self, t: u32, out: &mut [f32]) {
        self.mat(&self.embd).dequant_row(t as usize, out);
    }
}
