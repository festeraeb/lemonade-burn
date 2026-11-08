//! Tensor map for Gemma 4 GGUF files (llama.cpp e14a779 `src/models/gemma4.cpp` names).
//! Quantized matrices stay in the mmap ([`MatRef`] views, the same type lb-lfm2 uses); norms,
//! the router and the small scales are copied out as f32.
//!
//! Per block `blk.N.`:
//! * `attn_norm`; `attn_q`, `attn_k`, optional `attn_v` (absent on the global layers of the
//!   26B: V is the K projection, before its norm); `attn_q_norm`, `attn_k_norm [head_dim]`;
//!   `attn_output`; `post_attention_norm`.
//! * shared dense FFN: `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, and for MoE layers
//!   `post_ffw_norm_1` after it.
//! * MoE: `pre_ffw_norm_2`; router `ffn_gate_inp.weight [n_embd -> n_expert]` (f32) and
//!   `ffn_gate_inp.scale [n_embd]`; experts either fused `ffn_gate_up_exps [n_expert,
//!   2*n_ff_exp, n_embd]` (gate rows first, then up) or split `ffn_gate_exps`/`ffn_up_exps`;
//!   `ffn_down_exps [n_expert, n_embd, n_ff_exp]` with a per-expert output scale
//!   `ffn_down_exps.scale [n_expert]`; `post_ffw_norm_2`.
//! * `post_ffw_norm` (after the FFN sum), `layer_output_scale [1]` (scales the whole residual).
//! * global: `token_embd` (also the LM head unless `output.weight` exists), `output_norm`,
//!   `rope_freqs [head_dim_full/2]` (the global layers' per-pair frequency divisors).

use crate::config::Config;
use anyhow::{bail, Context, Result};
use lb_gguf::{GgmlType, Gguf, QuantMat, TensorInfo};
pub use lb_lfm2::weights::MatRef;
use std::sync::Arc;

/// One projection of every expert in a layer: expert `e` is `rows` contiguous rows starting
/// `row_off` rows into its block of the stacked tensor (`stride` bytes per expert). For the
/// fused gate_up tensor, gate is `row_off = 0` and up `row_off = n_ff_exp` of a `2*n_ff_exp`
/// row block, so each half of each expert is still one contiguous byte range.
#[derive(Clone, Debug)]
pub struct ExpertProj {
    pub ty: GgmlType,
    pub n: usize,
    pub rows: usize,
    pub cols: usize,
    pub start: usize,
    pub stride: usize,
    pub row_off: usize,
}

impl ExpertProj {
    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.cols).expect("checked at load")
    }

    pub fn expert(&self, e: usize) -> MatRef {
        assert!(e < self.n);
        let rb = self.row_bytes();
        MatRef {
            ty: self.ty,
            rows: self.rows,
            cols: self.cols,
            start: self.start + e * self.stride + self.row_off * rb,
            len: self.rows * rb,
        }
    }

    /// Bytes of one expert's slice of this projection.
    pub fn bytes(&self) -> usize {
        self.rows * self.row_bytes()
    }
}

pub struct Attn {
    pub q: MatRef,
    pub k: MatRef,
    /// `None`: V is the K projection (`use_alternative_attention`).
    pub v: Option<MatRef>,
    pub o: MatRef,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
}

/// The dense FFN: GEGLU `down(gelu(gate x) * up x)` (tanh GELU).
pub struct Dense {
    pub norm: Vec<f32>,
    pub gate: MatRef,
    pub up: MatRef,
    pub down: MatRef,
    /// `post_ffw_norm_1` on MoE layers (the dense FFN is the shared expert there).
    pub post_norm: Option<Vec<f32>>,
}

pub struct Moe {
    /// `pre_ffw_norm_2`: the experts' input norm.
    pub pre_norm: Vec<f32>,
    /// `[n_expert][n_embd]`.
    pub router: Vec<f32>,
    /// `ffn_gate_inp.scale [n_embd]`, applied to the unweighted RMS norm of the router input.
    pub router_scale: Vec<f32>,
    pub gate: ExpertProj,
    pub up: ExpertProj,
    pub down: ExpertProj,
    /// Per-expert output scale (`ffn_down_exps.scale`), 1.0 when the file has none.
    pub down_scale: Vec<f32>,
    pub post_norm: Vec<f32>,
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn: Attn,
    pub dense: Dense,
    pub moe: Option<Moe>,
    /// `post_ffw_norm`, after dense (+ MoE).
    pub post_ffw_norm: Vec<f32>,
    /// `layer_output_scale`, 1.0 when absent.
    pub out_scale: f32,
}

pub struct Weights {
    pub gguf: Arc<Gguf>,
    pub cfg: Config,
    pub layers: Vec<LayerWeights>,
    pub embd: MatRef,
    /// LM head: `output.weight`, or `token_embd` when tied.
    pub output: MatRef,
    pub final_norm: Vec<f32>,
    /// Global-layer RoPE frequency divisors (`rope_freqs`), one per rotated pair.
    pub rope_freqs: Option<Vec<f32>>,
}

impl Weights {
    pub fn mat(&self, m: &MatRef) -> QuantMat<'_> {
        QuantMat::new(
            m.ty,
            m.rows,
            m.cols,
            self.gguf.tensor_bytes_at(m.start, m.len),
        )
    }

    pub fn load(gguf: Arc<Gguf>) -> Result<Self> {
        let cfg = Config::from_gguf(&gguf)?;
        let g = &*gguf;
        let info = |name: &str| -> Result<&TensorInfo> {
            g.tensor(name)
                .with_context(|| format!("tensor {name} missing"))
        };
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
            let len = t
                .n_bytes()
                .with_context(|| format!("{name}: unsupported type {}", t.ty.name()))?;
            if t.ty.row_bytes(cols).is_none() {
                bail!(
                    "{name}: {cols} cols is not a whole number of {} blocks",
                    t.ty.name()
                );
            }
            Ok(MatRef {
                ty: t.ty,
                rows,
                cols,
                start: t.offset,
                len,
            })
        };
        // `[n, rows_total, cols]` stack; `rows` of each expert's block from `row_off`
        let exps = |name: &str,
                    cols: usize,
                    rows_total: usize,
                    n: usize,
                    rows: usize,
                    row_off: usize|
         -> Result<ExpertProj> {
            let t = info(name)?;
            if t.dims.len() != 3 || t.dims[0] != cols || t.dims[1] != rows_total || t.dims[2] != n {
                bail!(
                    "{name}: dims {:?}, expected [{cols}, {rows_total}, {n}]",
                    t.dims
                );
            }
            let rb =
                t.ty.row_bytes(cols)
                    .with_context(|| format!("{name}: unsupported type {}", t.ty.name()))?;
            Ok(ExpertProj {
                ty: t.ty,
                n,
                rows,
                cols,
                start: t.offset,
                stride: rb * rows_total,
                row_off,
            })
        };

        let e = cfg.n_embd;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let (nh, kvh, hd) = (cfg.n_head[i], cfg.kv_heads[i], cfg.head_dim(i));
            let has = |n: &str| g.tensor(&format!("{p}.{n}")).is_some();
            if has("attn_qkv.weight") {
                bail!("{p}: fused attn_qkv is not implemented in lb-gemma4");
            }
            let attn = Attn {
                q: mat(&format!("{p}.attn_q.weight"), e, nh * hd)?,
                k: mat(&format!("{p}.attn_k.weight"), e, kvh * hd)?,
                v: if has("attn_v.weight") {
                    Some(mat(&format!("{p}.attn_v.weight"), e, kvh * hd)?)
                } else {
                    None
                },
                o: mat(&format!("{p}.attn_output.weight"), nh * hd, e)?,
                q_norm: vec(&format!("{p}.attn_q_norm.weight"), hd)?,
                k_norm: vec(&format!("{p}.attn_k_norm.weight"), hd)?,
                post_norm: vec(&format!("{p}.post_attention_norm.weight"), e)?,
            };
            let is_moe = has("ffn_gate_inp.weight");
            let nff = cfg.n_ff[i];
            let dense = Dense {
                norm: vec(&format!("{p}.ffn_norm.weight"), e)?,
                gate: mat(&format!("{p}.ffn_gate.weight"), e, nff)?,
                up: mat(&format!("{p}.ffn_up.weight"), e, nff)?,
                down: mat(&format!("{p}.ffn_down.weight"), nff, e)?,
                post_norm: if is_moe {
                    Some(vec(&format!("{p}.post_ffw_norm_1.weight"), e)?)
                } else {
                    None
                },
            };
            let moe = if is_moe {
                let (n, f) = (cfg.n_expert, cfg.n_ff_exp);
                if n == 0 || cfg.n_expert_used == 0 || f == 0 {
                    bail!("{p} has a router but expert_count/expert_used_count/expert_feed_forward_length are missing");
                }
                let (gate, up) = if has("ffn_gate_up_exps.weight") {
                    let name = format!("{p}.ffn_gate_up_exps.weight");
                    (
                        exps(&name, e, 2 * f, n, f, 0)?,
                        exps(&name, e, 2 * f, n, f, f)?,
                    )
                } else {
                    (
                        exps(&format!("{p}.ffn_gate_exps.weight"), e, f, n, f, 0)?,
                        exps(&format!("{p}.ffn_up_exps.weight"), e, f, n, f, 0)?,
                    )
                };
                let down_scale = if has("ffn_down_exps.scale") {
                    vec(&format!("{p}.ffn_down_exps.scale"), n)?
                } else {
                    vec![1.0; n]
                };
                for s in [
                    "ffn_gate_up_exps.scale",
                    "ffn_gate_exps.scale",
                    "ffn_up_exps.scale",
                ] {
                    if has(s) {
                        bail!(
                            "{p}.{s}: per-expert gate/up scales are not implemented in lb-gemma4"
                        );
                    }
                }
                Some(Moe {
                    pre_norm: vec(&format!("{p}.pre_ffw_norm_2.weight"), e)?,
                    router: vec(&format!("{p}.ffn_gate_inp.weight"), n * e)?,
                    router_scale: vec(&format!("{p}.ffn_gate_inp.scale"), e)?,
                    gate,
                    up,
                    down: exps(&format!("{p}.ffn_down_exps.weight"), f, e, n, e, 0)?,
                    down_scale,
                    post_norm: vec(&format!("{p}.post_ffw_norm_2.weight"), e)?,
                })
            } else {
                None
            };
            let out_scale = if has("layer_output_scale.weight") {
                vec(&format!("{p}.layer_output_scale.weight"), 1)?[0]
            } else {
                1.0
            };
            layers.push(LayerWeights {
                attn_norm: vec(&format!("{p}.attn_norm.weight"), e)?,
                attn,
                dense,
                moe,
                post_ffw_norm: vec(&format!("{p}.post_ffw_norm.weight"), e)?,
                out_scale,
            });
        }
        let embd = mat("token_embd.weight", e, cfg.n_vocab)?;
        let output = if g.tensor("output.weight").is_some() {
            mat("output.weight", e, cfg.n_vocab)?
        } else {
            embd.clone()
        };
        let final_norm = vec("output_norm.weight", e)?;
        let rope_freqs = match g.tensor("rope_freqs.weight") {
            Some(_) => Some(vec("rope_freqs.weight", cfg.n_rot_full / 2)?),
            None => None,
        };
        Ok(Self {
            gguf,
            cfg,
            layers,
            embd,
            output,
            final_norm,
            rope_freqs,
        })
    }

    /// Bytes of one expert (gate + up + down) in `layer`, 0 for dense layers: what the expert
    /// cache budgets per entry, as `lb_lfm2::Weights::expert_bytes`.
    pub fn expert_bytes(&self, layer: usize) -> usize {
        match &self.layers[layer].moe {
            Some(m) => m.gate.bytes() + m.up.bytes() + m.down.bytes(),
            None => 0,
        }
    }

    /// Dequantized embedding row of token `t` (unscaled).
    pub fn embed_row(&self, t: u32, out: &mut [f32]) {
        self.mat(&self.embd).dequant_row(t as usize, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_gate_up_halves_are_contiguous_ranges_of_each_expert_block() {
        // Q4_0: 18 bytes per 32 values; 64 cols = 36 bytes a row; 2 experts of 2*3 rows
        let rb = 36;
        let gate = ExpertProj {
            ty: GgmlType::Q4_0,
            n: 2,
            rows: 3,
            cols: 64,
            start: 1000,
            stride: 6 * rb,
            row_off: 0,
        };
        let up = ExpertProj {
            row_off: 3,
            ..gate.clone()
        };
        assert_eq!(gate.expert(0).start, 1000);
        assert_eq!(up.expert(0).start, 1000 + 3 * rb);
        assert_eq!(gate.expert(1).start, 1000 + 6 * rb);
        assert_eq!(up.expert(1).start, 1000 + 9 * rb);
        assert_eq!(up.expert(1).len, 3 * rb);
        assert_eq!(gate.bytes(), 3 * rb);
    }
}
