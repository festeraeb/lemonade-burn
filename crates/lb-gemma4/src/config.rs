//! Gemma 4 hyperparameters, read from GGUF metadata (`gemma4.*` keys).
//!
//! Gemma 4 interleaves two attention kinds (`attention.sliding_window_pattern`, one bool per
//! layer): sliding-window layers (head dim `key_length_swa`, RoPE base `rope.freq_base_swa`,
//! window `attention.sliding_window`) and global layers (head dim `key_length`, RoPE base
//! `rope.freq_base` with per-pair `rope_freqs` factors: "proportional" RoPE). KV heads differ
//! per layer (`attention.head_count_kv` is an array).
//!
//! Not handled (the 26B-A4B does not use them, and the loader refuses a file that does rather
//! than run it wrong): per-layer token embeddings (`embedding_length_per_layer_input`, E2B/E4B)
//! and KV sharing between layers (`attention.shared_kv_layers`).

use anyhow::{bail, Context, Result};
use lb_gguf::{Gguf, MetaValue};

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    /// Query heads per layer.
    pub n_head: Vec<usize>,
    /// KV heads per layer.
    pub kv_heads: Vec<usize>,
    /// Sliding-window layer?
    pub swa: Vec<bool>,
    /// Head dim of the global layers / of the sliding-window layers.
    pub head_dim_full: usize,
    pub head_dim_swa: usize,
    /// Rotated dims (NEOX pairs `j`, `j + n_rot/2`).
    pub n_rot_full: usize,
    pub n_rot_swa: usize,
    pub rope_base_full: f64,
    pub rope_base_swa: f64,
    pub n_swa: usize,
    /// Shared (dense) FFN width per layer.
    pub n_ff: Vec<usize>,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub rms_eps: f32,
    /// `30 * tanh(logits / 30)`; 0 = none.
    pub final_softcap: f32,
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub bos: u32,
    pub eos: u32,
}

/// A per-layer key that may be a scalar or an array.
fn per_layer_u64(g: &Gguf, key: &str, n_layer: usize) -> Result<Vec<usize>> {
    match g.meta.get(key).with_context(|| format!("{key} missing"))? {
        MetaValue::Array(a) => {
            let v: Option<Vec<usize>> = a.iter().map(|x| x.as_u64().map(|x| x as usize)).collect();
            let v = v.with_context(|| format!("{key}: not an integer array"))?;
            if v.len() != n_layer {
                bail!("{key} has {} entries for {n_layer} layers", v.len());
            }
            Ok(v)
        }
        x => Ok(vec![
            x.as_u64()
                .with_context(|| format!("{key}: not an integer"))?
                as usize;
            n_layer
        ]),
    }
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g
            .meta_str("general.architecture")
            .context("general.architecture")?;
        if arch != "gemma4" {
            bail!("architecture {arch} is not gemma4");
        }
        let key = |k: &str| format!("gemma4.{k}");
        let k = |k: &str| -> Result<u64> { g.meta_u64(&key(k)).with_context(|| key(k)) };
        let kf = |k: &str| -> Result<f64> { g.meta_f64(&key(k)).with_context(|| key(k)) };
        let n_layer = k("block_count")? as usize;
        let swa: Vec<bool> = match g.meta.get(&key("attention.sliding_window_pattern")) {
            Some(MetaValue::Array(a)) => a
                .iter()
                .map(|v| match v {
                    MetaValue::Bool(b) => Ok(*b),
                    x => x
                        .as_u64()
                        .map(|u| u != 0)
                        .context("sliding_window_pattern entry"),
                })
                .collect::<Result<_>>()?,
            _ => bail!("gemma4.attention.sliding_window_pattern missing"),
        };
        if swa.len() != n_layer {
            bail!(
                "sliding_window_pattern has {} entries for {n_layer} layers",
                swa.len()
            );
        }
        if g.meta_u64(&key("embedding_length_per_layer_input"))
            .unwrap_or(0)
            > 0
        {
            bail!("per-layer token embeddings (E2B/E4B) are not implemented in lb-gemma4");
        }
        if g.meta_u64(&key("attention.shared_kv_layers")).unwrap_or(0) > 0 {
            bail!("KV-shared layers are not implemented in lb-gemma4");
        }
        let head_dim_full = k("attention.key_length")? as usize;
        let head_dim_swa = k("attention.key_length_swa")? as usize;
        if k("attention.value_length")? as usize != head_dim_full
            || k("attention.value_length_swa")? as usize != head_dim_swa
        {
            bail!("gemma4 needs key_length == value_length");
        }
        let n_vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.dims[1])
            .context("token_embd.weight missing")?;
        Ok(Self {
            n_layer,
            n_embd: k("embedding_length")? as usize,
            n_head: per_layer_u64(g, &key("attention.head_count"), n_layer)?,
            kv_heads: per_layer_u64(g, &key("attention.head_count_kv"), n_layer)?,
            swa,
            head_dim_full,
            head_dim_swa,
            n_rot_full: k("rope.dimension_count").unwrap_or(head_dim_full as u64) as usize,
            n_rot_swa: k("rope.dimension_count_swa").unwrap_or(head_dim_swa as u64) as usize,
            rope_base_full: kf("rope.freq_base")?,
            rope_base_swa: kf("rope.freq_base_swa")?,
            n_swa: k("attention.sliding_window")? as usize,
            n_ff: per_layer_u64(g, &key("feed_forward_length"), n_layer)?,
            n_expert: k("expert_count").unwrap_or(0) as usize,
            n_expert_used: k("expert_used_count").unwrap_or(0) as usize,
            n_ff_exp: k("expert_feed_forward_length").unwrap_or(0) as usize,
            rms_eps: kf("attention.layer_norm_rms_epsilon")? as f32,
            final_softcap: kf("final_logit_softcapping").unwrap_or(0.0) as f32,
            n_vocab,
            n_ctx: k("context_length")? as usize,
            bos: g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(2) as u32,
            eos: g.meta_u64("tokenizer.ggml.eos_token_id").unwrap_or(1) as u32,
        })
    }

    pub fn head_dim(&self, l: usize) -> usize {
        if self.swa[l] {
            self.head_dim_swa
        } else {
            self.head_dim_full
        }
    }

    pub fn n_rot(&self, l: usize) -> usize {
        if self.swa[l] {
            self.n_rot_swa
        } else {
            self.n_rot_full
        }
    }

    pub fn rope_base(&self, l: usize) -> f64 {
        if self.swa[l] {
            self.rope_base_swa
        } else {
            self.rope_base_full
        }
    }

    /// Can a query at `qpos` see the key at `kpos`? Causal everywhere; sliding-window layers
    /// also drop keys `n_swa` or more positions back (llama.cpp `LLAMA_SWA_TYPE_STANDARD`).
    pub fn visible(&self, l: usize, qpos: usize, kpos: usize) -> bool {
        kpos <= qpos && (!self.swa[l] || qpos - kpos < self.n_swa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            n_layer: 2,
            n_embd: 8,
            n_head: vec![2; 2],
            kv_heads: vec![1; 2],
            swa: vec![true, false],
            head_dim_full: 8,
            head_dim_swa: 4,
            n_rot_full: 8,
            n_rot_swa: 4,
            rope_base_full: 1e6,
            rope_base_swa: 1e4,
            n_swa: 3,
            n_ff: vec![4; 2],
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 4,
            rms_eps: 1e-6,
            final_softcap: 30.0,
            n_vocab: 10,
            n_ctx: 64,
            bos: 2,
            eos: 1,
        }
    }

    #[test]
    fn sliding_window_drops_keys_n_swa_back_and_global_layers_do_not() {
        let c = cfg();
        assert!(c.visible(0, 5, 3));
        assert!(!c.visible(0, 5, 2)); // 5 - 2 = 3 = n_swa
        assert!(!c.visible(0, 5, 6)); // causal
        assert!(c.visible(1, 5, 0));
        assert!(!c.visible(1, 5, 6));
        assert_eq!((c.head_dim(0), c.head_dim(1)), (4, 8));
        assert_eq!((c.rope_base(0), c.rope_base(1)), (1e4, 1e6));
    }
}
