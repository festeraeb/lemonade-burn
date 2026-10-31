//! LFM2 / LFM2-MoE hyperparameters, read from GGUF metadata (`lfm2` or `lfm2moe` keys).

use anyhow::{bail, Context, Result};
use lb_gguf::Gguf;

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    /// Dense FFN width (the leading dense layers).
    pub n_ff: usize,
    pub n_head: usize,
    /// Per layer: number of KV heads; 0 means the layer is a gated short-conv block.
    pub kv_heads: Vec<usize>,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    /// Leading layers with a dense FFN instead of experts.
    pub n_dense_lead: usize,
    /// 1 = softmax, 2 = sigmoid (+ selection bias) router.
    pub gating: u64,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub n_vocab: usize,
    pub n_ctx: usize,
    /// Short-conv kernel length (`l_cache`); the state keeps `l_cache - 1` past rows.
    pub l_cache: usize,
    pub bos: u32,
    pub eos: u32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture").context("general.architecture")?.to_string();
        if arch != "lfm2" && arch != "lfm2moe" {
            bail!("architecture {arch} is not lfm2/lfm2moe");
        }
        let k = |key: &str| -> Result<u64> { g.meta_u64(&format!("{arch}.{key}")).with_context(|| format!("{arch}.{key}")) };
        let kf = |key: &str| -> Result<f64> { g.meta_f64(&format!("{arch}.{key}")).with_context(|| format!("{arch}.{key}")) };
        let n_layer = k("block_count")? as usize;
        let kv_heads: Vec<usize> = match g.meta_u64_array(&format!("{arch}.attention.head_count_kv")) {
            Some(v) => v.into_iter().map(|x| x as usize).collect(),
            None => vec![k("attention.head_count_kv")? as usize; n_layer],
        };
        if kv_heads.len() != n_layer {
            bail!("head_count_kv has {} entries for {n_layer} layers", kv_heads.len());
        }
        let moe = arch == "lfm2moe";
        Ok(Self {
            n_layer,
            n_embd: k("embedding_length")? as usize,
            n_ff: k("feed_forward_length")? as usize,
            n_head: k("attention.head_count")? as usize,
            kv_heads,
            n_expert: if moe { k("expert_count")? as usize } else { 0 },
            n_expert_used: if moe { k("expert_used_count")? as usize } else { 0 },
            n_ff_exp: if moe { k("expert_feed_forward_length")? as usize } else { 0 },
            n_dense_lead: if moe { k("leading_dense_block_count").unwrap_or(0) as usize } else { n_layer },
            gating: if moe { k("expert_gating_func").unwrap_or(1) } else { 0 },
            rms_eps: kf("attention.layer_norm_rms_epsilon")? as f32,
            rope_theta: kf("rope.freq_base")?,
            n_vocab: k("vocab_size")? as usize,
            n_ctx: k("context_length")? as usize,
            l_cache: k("shortconv.l_cache")? as usize,
            bos: g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(1) as u32,
            eos: g.meta_u64("tokenizer.ggml.eos_token_id").unwrap_or(2) as u32,
            arch,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
    pub fn is_attn(&self, layer: usize) -> bool {
        self.kv_heads[layer] > 0
    }
    pub fn is_moe(&self, layer: usize) -> bool {
        self.n_expert > 0 && layer >= self.n_dense_lead
    }
}
