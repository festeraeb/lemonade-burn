//! `deepseek41` CPU reference forward (DeepSeek V4.1 Flash), written to match llama.cpp's
//! `src/models/deepseek41.cpp` + `deepseek4.cpp` (commit dadb89f180, MIT) value for value up to f32/f16 rounding.
//! Facts and file:line citations: `docs/MODEL_DEEPSEEK4.md`; the scalar pieces live in [`math`].
//!
//! The pass (per token, 40 layers, hidden 5120):
//! * the residual stream is 4 hyper-connection streams, all four start as the token embedding;
//! * Engram layers (1, 14) first add `sigmoid-sqrt gate_s * value` from 24 hashed rows of the 110 GB IQ4_NL table;
//! * per sublayer, mixes from the unweighted RMS of all 4 x 5120 values (`hc_*_fn`, scale, base): `pre`, `post`, and a
//!   Sinkhorn-normalised 4x4 `comb`; the sublayer input is the stream collapse with the LAGGED pre (attention of layer il
//!   uses layer il-1's FFN pre, layer 0 uses stream 0; the FFN uses the same layer's attention pre), and the output goes
//!   back as `post * out + comb^T streams`;
//! * attention: q LoRA (q_a, RMS, q_b) to 64 heads x 512, one shared K = V head of 512 (attn_kv, RMS), NORM rope on
//!   channels 448..511 (plain base 1e4 at ratio-0 layers 0-1, yarn base 160000 elsewhere), scale 1/sqrt(512), a sink
//!   logit per head; keys are this layer's raw rows plus, at ratio 1/2 layers, the compressed rows of the kv source
//!   layer (2, 8, 14, 20) that are complete at the query position; output de-roped at the query position, grouped
//!   output projection (8 x (4096 -> 1024), then 8192 -> 5120);
//! * MoE: BF16 router, sqrt(softplus), top-6 of probs + bias, weights = probs / (sum + 1e-20) * 1.5, SwiGLU with clamp
//!   10 on 192 routed experts and the shared expert; `ffn_out = moe + shared`;
//! * head: collapse with layer 39's FFN pre, RMS * output_norm, output.
//!
//! Valid for a single sequence of at most [`MAX_CTX`] = 128 tokens (prompt + generated): there the 128-token sliding
//! window covers everything and the lightning indexer / candidate blocks select every visible compressed row
//! (top-512 of at most 128 rows), so both are left out; [`Model::forward`] refuses longer sequences.
//!
//! Numerics knobs (env): `LB_DS4_KV_F16` (default 1) stores raw and compressed K rows as f16 and rounds Q to f16 for the
//! dot, as llama.cpp's default F16 K cache with CPU flash attention does; `LB_DS4_ACT_EMU` (default 1) rounds each
//! matmul's f32 activation the way ggml's CPU matmul converts it (Q8_0 / Q8_K / BF16 per weight type, see
//! [`math::act_round`]); 0 keeps f32.

pub mod math;
pub mod tokenizer;

use anyhow::{anyhow, bail, Result};
use lb_gguf::{matvec_f32, GgmlType, GgufSet, MetaValue, QuantMat};
use math::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Longest single sequence this reference is valid for (see the crate docs).
pub const MAX_CTX: usize = 128;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub q_lora: usize,
    pub n_swa: usize,
    pub eps: f32,
    pub n_expert: usize,
    pub n_used: usize,
    pub ff_exp: usize,
    pub w_scale: f32,
    pub w_norm: bool,
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    pub o_groups: usize,
    pub o_lora: usize,
    pub rope_base: f32,
    pub comp_rope_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub n_ctx_orig: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub hc: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,
    pub ratio: Vec<usize>,
    /// latest kv source layer <= il (None at ratio-0 layers)
    pub kv_src: Vec<Option<usize>>,
    /// Engram module index of a layer
    pub engram: Vec<Option<usize>>,
    pub ngram: usize,
    pub ngram_heads: usize,
    pub ple_dim: usize,
    pub ple_mult: Vec<u64>,
    pub ple_offs: Vec<u64>,
    pub ple_primes: Vec<u64>,
    pub token_map: Vec<i64>,
    pub engram_pad: i64,
    pub vocab: usize,
}

fn meta_arr<'a>(g: &'a GgufSet, key: &str) -> Option<&'a [MetaValue]> {
    g.meta().get(key)?.as_array()
}

fn as_i64(v: &MetaValue) -> Option<i64> {
    Some(match *v {
        MetaValue::I8(x) => x as i64,
        MetaValue::I16(x) => x as i64,
        MetaValue::I32(x) => x as i64,
        MetaValue::I64(x) => x,
        _ => v.as_u64()? as i64,
    })
}

impl Config {
    pub fn from_gguf(g: &GgufSet) -> Result<Self> {
        let arch = g.meta_str("general.architecture").unwrap_or("");
        if arch != "deepseek41" {
            bail!("architecture {arch:?}, this crate loads deepseek41");
        }
        let k = |s: &str| format!("deepseek41.{s}");
        let u = |s: &str| g.meta_u64(&k(s)).map(|v| v as usize).ok_or_else(|| anyhow!("missing deepseek41.{s}"));
        let f = |s: &str| g.meta_f64(&k(s)).map(|v| v as f32).ok_or_else(|| anyhow!("missing deepseek41.{s}"));
        let ua = |s: &str| g.meta_u64_array(&k(s)).ok_or_else(|| anyhow!("missing array deepseek41.{s}"));
        let n_layer = u("block_count")?;
        // scalar or per-layer float array (D4:28-31)
        let farr = |s: &str| -> Option<Vec<f32>> {
            if let Some(a) = meta_arr(g, &k(s)) {
                return a.iter().map(|v| v.as_f64().map(|x| x as f32)).collect();
            }
            g.meta_f64(&k(s)).map(|v| vec![v as f32; n_layer])
        };
        let clamp_exp = farr("swiglu_clamp_exp").ok_or_else(|| anyhow!("missing swiglu_clamp_exp"))?;
        let clamp_shexp = farr("swiglu_clamp_shexp").unwrap_or_else(|| clamp_exp.clone());
        if clamp_exp.len() < n_layer || clamp_shexp.len() < n_layer {
            bail!("swiglu clamp arrays shorter than block_count");
        }
        if u("expert_gating_func")? != 4 {
            bail!("expert_gating_func must be 4 (SQRT_SOFTPLUS), D4:55-58");
        }
        if g.meta_u64(&k("hash_layer_count")).unwrap_or(0) != 0 {
            bail!("hash layers (ffn_gate_tid2eid) not implemented");
        }
        if g.meta_u64(&k("expert_group_count")).unwrap_or(1) > 1 {
            bail!("expert groups > 1 not implemented");
        }
        if g.meta_u64(&k("expert_shared_count")).unwrap_or(1) != 1 {
            bail!("expected one shared expert");
        }
        if u("attention.head_count_kv")? != 1 {
            bail!("expected one shared K=V head");
        }
        let head_dim = u("attention.key_length")?;
        if u("attention.value_length")? != head_dim {
            bail!("key_length != value_length");
        }
        let hc = u("hyper_connection.count")?;
        if hc != 4 {
            bail!("hyper_connection.count {hc}: the fused ops assume 4");
        }
        let ratios: Vec<i64> = meta_arr(g, &k("attention.compress_ratios")).ok_or_else(|| anyhow!("missing compress_ratios"))?.iter().map(|v| as_i64(v).unwrap_or(-1)).collect();
        if ratios.len() < n_layer {
            bail!("compress_ratios shorter than block_count (D4:47-53)");
        }
        let kv_sources = ua("attention.kv_source_layers")?;
        let mut ratio = vec![0usize; n_layer];
        let mut kv_src = vec![None; n_layer];
        let mut last_kv: Option<usize> = None;
        for il in 0..n_layer {
            let r = ratios[il];
            if !(0..=2).contains(&r) {
                bail!("layer {il}: compress ratio {r} (deepseek41 allows 0, 1, 2)");
            }
            ratio[il] = r as usize;
            if r == 0 {
                continue;
            }
            if kv_sources.contains(&(il as u64)) {
                last_kv = Some(il);
            }
            let s = last_kv.ok_or_else(|| anyhow!("layer {il}: no kv source layer before it"))?;
            if ratios[s] != r {
                bail!("layer {il}: kv source {s} has another ratio");
            }
            kv_src[il] = Some(s);
        }
        // Engram (D41:84-127)
        let ple_layers = g.meta_u64_array(&k("ple.layers")).unwrap_or_default();
        let mut engram = vec![None; n_layer];
        for (e, &l) in ple_layers.iter().enumerate() {
            if (l as usize) < n_layer {
                engram[l as usize] = Some(e);
            }
        }
        let (ngram, ngram_heads, ple_dim, ple_mult, ple_offs, ple_primes, token_map, engram_pad);
        if ple_layers.is_empty() {
            (ngram, ngram_heads, ple_dim) = (0, 0, 0);
            (ple_mult, ple_offs, ple_primes, token_map) = (vec![], vec![], vec![], vec![]);
            engram_pad = 0;
        } else {
            ngram = u("ple.ngram_size")?;
            ngram_heads = u("ple.heads_per_ngram")?;
            ple_dim = u("embedding_length_per_layer_input")?;
            ple_mult = ua("ple.layer_multipliers")?;
            ple_offs = ua("ple.head_offsets")?;
            ple_primes = ua("ple.head_vocab_sizes")?;
            let n_mod = ple_layers.len();
            let n_cols = (ngram - 1) * ngram_heads;
            if ple_mult.len() != n_mod * ngram || ple_offs.len() != n_mod * n_cols || ple_primes.len() != n_mod * n_cols {
                bail!("ple array sizes do not match {n_mod} modules x {ngram}-grams x {ngram_heads} heads");
            }
            token_map = meta_arr(g, &k("engram.token_map")).ok_or_else(|| anyhow!("missing engram.token_map"))?.iter().map(|v| as_i64(v).unwrap_or(ENGRAM_DEAD)).collect();
            let pad_id = u("engram.pad_token_id")?;
            engram_pad = *token_map.get(pad_id).ok_or_else(|| anyhow!("engram pad id {pad_id} outside token_map"))?;
        }
        // rope: llama.cpp reads scaling type/factor/orig ctx; yarn beta_fast/slow and ext_factor are NOT read from the GGUF
        // in this tree (hparam defaults 32 / 1, ext_factor -1 -> 1.0 for YARN, llama-context.cpp:265-267).
        let yarn = g.meta_str(&k("rope.scaling.type")) == Some("yarn");
        let factor = g.meta_f64(&k("rope.scaling.factor")).map(|v| v as f32).unwrap_or(1.0);
        let freq_scale = if factor > 0.0 { 1.0 / factor } else { 1.0 };
        let n_ctx_orig = g.meta_u64(&k("rope.scaling.original_context_length")).or_else(|| g.meta_u64(&k("context_length"))).unwrap_or(0) as u32;
        Ok(Config {
            n_layer,
            n_embd: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            head_dim,
            n_rot: u("rope.dimension_count")?,
            q_lora: u("attention.q_lora_rank")?,
            n_swa: u("attention.sliding_window")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            ff_exp: u("expert_feed_forward_length")?,
            w_scale: f("expert_weights_scale")?,
            w_norm: g.meta().get(&k("expert_weights_norm")).map(|v| matches!(v, MetaValue::Bool(true))).unwrap_or(false),
            clamp_exp,
            clamp_shexp,
            o_groups: u("attention.output_group_count")?,
            o_lora: u("attention.output_lora_rank")?,
            rope_base: f("rope.freq_base")?,
            comp_rope_base: f("attention.compress_rope_freq_base")?,
            freq_scale,
            ext_factor: if yarn { 1.0 } else { 0.0 },
            n_ctx_orig,
            beta_fast: 32.0,
            beta_slow: 1.0,
            hc,
            hc_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            ratio,
            kv_src,
            engram,
            ngram,
            ngram_heads,
            ple_dim,
            ple_mult,
            ple_offs,
            ple_primes,
            token_map,
            engram_pad,
            vocab: g.tensor("token_embd.weight").map(|t| t.dims[1]).ok_or_else(|| anyhow!("no token_embd"))?,
        })
    }

    /// The rope of layer `il` (D41:430-443): ratio-0 layers plain (`freq_base` 1e4, scale 1, no yarn); the others base
    /// `compress_rope_freq_base` with yarn and `attn_factor = 1/(1 + 0.1 ln(1/freq_scale))` (models.h:1223-1229).
    pub fn rope(&self, il: usize, offset: usize) -> Rope {
        if self.ratio[il] == 0 {
            Rope::new(self.n_rot, offset, 0, self.rope_base, 1.0, 0.0, 1.0, 0.0, 0.0)
        } else {
            let af = if self.ext_factor == 0.0 { 1.0 } else { 1.0 / (1.0 + 0.1 * (1.0 / self.freq_scale).ln()) };
            Rope::new(self.n_rot, offset, self.n_ctx_orig, self.comp_rope_base, self.freq_scale, self.ext_factor, af, self.beta_fast, self.beta_slow)
        }
    }
}

pub struct Compressor<'a> {
    pub kv: QuantMat<'a>,
    pub gate: Option<QuantMat<'a>>,
    pub norm: Vec<f32>,
}

pub struct Engram<'a> {
    pub kv: QuantMat<'a>,
    /// [hc][n_embd]: element (d, s) at d + n_embd * s
    pub gate: Vec<f32>,
}

pub struct Layer<'a> {
    pub hc_attn: (QuantMat<'a>, Vec<f32>, Vec<f32>),
    pub hc_ffn: (QuantMat<'a>, Vec<f32>, Vec<f32>),
    pub attn_norm: Vec<f32>,
    pub q_a: QuantMat<'a>,
    pub q_a_norm: Vec<f32>,
    pub q_b: QuantMat<'a>,
    pub kv: QuantMat<'a>,
    pub kv_norm: Vec<f32>,
    pub sinks: Vec<f32>,
    pub wo_a: QuantMat<'a>,
    pub wo_b: QuantMat<'a>,
    pub comp: Option<Compressor<'a>>,
    pub engram: Option<Engram<'a>>,
    pub ffn_norm: Vec<f32>,
    pub router: QuantMat<'a>,
    pub bias: Vec<f32>,
    pub gate_exps: QuantMat<'a>,
    pub up_exps: QuantMat<'a>,
    pub down_exps: QuantMat<'a>,
    pub gate_sh: QuantMat<'a>,
    pub up_sh: QuantMat<'a>,
    pub down_sh: QuantMat<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    pub tok_embd: QuantMat<'a>,
    pub out_norm: Vec<f32>,
    pub output: QuantMat<'a>,
    /// the Engram table, read row by row from the mmap
    pub ple: Option<QuantMat<'a>>,
    pub layers: Vec<Layer<'a>>,
    pub kv_f16: bool,
    pub act_emu: bool,
}

fn qmat<'a>(g: &'a GgufSet, name: &str) -> Result<QuantMat<'a>> {
    let (f, t) = g.find(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
    let cols = t.dims[0];
    let rows = t.n_elems() / cols;
    Ok(QuantMat::new(t.ty, rows, cols, f.tensor_bytes(t)))
}

fn qmat_shape<'a>(g: &'a GgufSet, name: &str, rows: usize, cols: usize) -> Result<QuantMat<'a>> {
    let m = qmat(g, name)?;
    if m.rows != rows || m.cols != cols {
        bail!("{name}: {} rows x {} cols, expected {rows} x {cols}", m.rows, m.cols);
    }
    Ok(m)
}

fn vecf(g: &GgufSet, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = g.tensor_f32(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
    if v.len() != n {
        bail!("{name}: {} values, expected {n}", v.len());
    }
    Ok(v)
}

fn env_on(k: &str) -> bool {
    std::env::var(k).map(|v| v != "0").unwrap_or(true)
}

impl<'a> Model<'a> {
    pub fn load(g: &'a GgufSet) -> Result<Self> {
        let c = Config::from_gguf(g)?;
        let (d, hd, nh) = (c.n_embd, c.head_dim, c.n_head);
        let mix = (2 + c.hc) * c.hc;
        let mut layers = Vec::with_capacity(c.n_layer);
        for il in 0..c.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let hcw = |w: &str| -> Result<(QuantMat<'a>, Vec<f32>, Vec<f32>)> {
                Ok((qmat_shape(g, &p(&format!("hc_{w}_fn.weight")), mix, c.hc * d)?, vecf(g, &p(&format!("hc_{w}_scale.weight")), 3)?, vecf(g, &p(&format!("hc_{w}_base.weight")), mix)?))
            };
            let comp = if c.kv_src[il] == Some(il) {
                Some(Compressor {
                    kv: qmat_shape(g, &p("attn_compressor_kv.weight"), hd, d)?,
                    gate: if c.ratio[il] > 1 { Some(qmat_shape(g, &p("attn_compressor_gate.weight"), hd, d)?) } else { None },
                    norm: vecf(g, &p("attn_compressor_norm.weight"), hd)?,
                })
            } else {
                None
            };
            let engram = if c.engram[il].is_some() {
                let n_cols = (c.ngram - 1) * c.ngram_heads;
                Some(Engram { kv: qmat_shape(g, &p("engram_kv.weight"), d * (c.hc + 1), n_cols * c.ple_dim)?, gate: vecf(g, &p("engram_gate.weight"), d * c.hc)? })
            } else {
                None
            };
            let (ne, ff) = (c.n_expert, c.ff_exp);
            layers.push(Layer {
                hc_attn: hcw("attn")?,
                hc_ffn: hcw("ffn")?,
                attn_norm: vecf(g, &p("attn_norm.weight"), d)?,
                q_a: qmat_shape(g, &p("attn_q_a.weight"), c.q_lora, d)?,
                q_a_norm: vecf(g, &p("attn_q_a_norm.weight"), c.q_lora)?,
                q_b: qmat_shape(g, &p("attn_q_b.weight"), nh * hd, c.q_lora)?,
                kv: qmat_shape(g, &p("attn_kv.weight"), hd, d)?,
                kv_norm: vecf(g, &p("attn_kv_a_norm.weight"), hd)?,
                sinks: vecf(g, &p("attn_sinks.weight"), nh)?,
                wo_a: qmat_shape(g, &p("attn_output_a.weight"), c.o_groups * c.o_lora, nh * hd / c.o_groups)?,
                wo_b: qmat_shape(g, &p("attn_output_b.weight"), d, c.o_groups * c.o_lora)?,
                comp,
                engram,
                ffn_norm: vecf(g, &p("ffn_norm.weight"), d)?,
                router: qmat_shape(g, &p("ffn_gate_inp.weight"), ne, d)?,
                bias: vecf(g, &p("exp_probs_b.bias"), ne)?,
                gate_exps: qmat_shape(g, &p("ffn_gate_exps.weight"), ne * ff, d)?,
                up_exps: qmat_shape(g, &p("ffn_up_exps.weight"), ne * ff, d)?,
                down_exps: qmat_shape(g, &p("ffn_down_exps.weight"), ne * d, ff)?,
                gate_sh: qmat_shape(g, &p("ffn_gate_shexp.weight"), ff, d)?,
                up_sh: qmat_shape(g, &p("ffn_up_shexp.weight"), ff, d)?,
                down_sh: qmat_shape(g, &p("ffn_down_shexp.weight"), d, ff)?,
            });
        }
        let ple = if c.engram.iter().any(|e| e.is_some()) {
            let m = qmat(g, "per_layer_token_embd.weight")?;
            if m.cols != c.ple_dim {
                bail!("per_layer_token_embd has {} cols, expected {}", m.cols, c.ple_dim);
            }
            let need = c.ple_offs.iter().zip(&c.ple_primes).map(|(o, p)| o + p).max().unwrap_or(0) as usize;
            if m.rows < need {
                bail!("per_layer_token_embd has {} rows, the hash reaches {need}", m.rows);
            }
            Some(m)
        } else {
            None
        };
        Ok(Model {
            tok_embd: qmat_shape(g, "token_embd.weight", c.vocab, d)?,
            out_norm: vecf(g, "output_norm.weight", d)?,
            output: qmat_shape(g, "output.weight", c.vocab, d)?,
            ple,
            layers,
            cfg: c,
            kv_f16: env_on("LB_DS4_KV_F16"),
            act_emu: env_on("LB_DS4_ACT_EMU"),
        })
    }
}

/// Everything a later token needs from earlier ones (single sequence).
pub struct Cache {
    pub len: usize,
    pub tokens: Vec<u32>,
    /// per layer: roped raw kv rows, `[pos][head_dim]` (f16-rounded with `LB_DS4_KV_F16`)
    pub raw: Vec<Vec<f32>>,
    /// per kv source layer: the compressor's per-token kv and gate scores (llama.cpp keeps this state in F32)
    pub comp_kv: Vec<Vec<f32>>,
    pub comp_score: Vec<Vec<f32>>,
    /// per kv source layer: complete compressed rows, roped at their block's first position, `[row][head_dim]`
    pub comp: Vec<Vec<f32>>,
}

impl Cache {
    pub fn new(n_layer: usize) -> Self {
        Cache { len: 0, tokens: Vec::new(), raw: vec![Vec::new(); n_layer], comp_kv: vec![Vec::new(); n_layer], comp_score: vec![Vec::new(); n_layer], comp: vec![Vec::new(); n_layer] }
    }
}

/// Named intermediates under llama.cpp's callback names (`l_out-<il>`, `result_output--1`), values `[token][...]`.
pub type Taps = HashMap<String, Vec<f32>>;

fn tap(taps: &mut Option<&mut Taps>, name: &str, il: i32, v: &[f32]) {
    if let Some(t) = taps.as_deref_mut() {
        t.insert(format!("{name}-{il}"), v.to_vec());
    }
}

/// Rows `[r0, r0 + n)` of a matrix as their own matrix view.
fn sub<'a>(w: QuantMat<'a>, r0: usize, n: usize) -> QuantMat<'a> {
    let rb = w.row_bytes();
    QuantMat::new(w.ty, n, w.cols, &w.bytes[r0 * rb..(r0 + n) * rb])
}

/// Per-sublayer hyper-connection mixes of `m` tokens: pre [m][hc], post [m][hc], comb [m][hc*hc] (dst + hc*src).
struct Mixes {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

impl Model<'_> {
    /// `out[t * rows + r] = W[r] . act(x[t])`, `act` = ggml's conversion of the activation to the weight's vec_dot type.
    fn mm(&self, w: QuantMat<'_>, x: &[f32], m: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * w.rows];
        if self.act_emu && w.ty != GgmlType::F32 {
            let mut xa = x.to_vec();
            act_round(w.ty, &mut xa, w.cols);
            matvec_f32(w, &xa, m, &mut out);
        } else {
            matvec_f32(w, x, m, &mut out);
        }
        out
    }

    /// Embedding rows of `tokens`, `[m][n_embd]` (`get_rows` dequantises exactly).
    pub fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.cfg.n_embd;
        let mut x = vec![0f32; tokens.len() * d];
        for (t, &id) in tokens.iter().enumerate() {
            self.tok_embd.dequant_row(id as usize, &mut x[t * d..(t + 1) * d]);
        }
        x
    }

    /// `build_hc_mixes` (D4:357-392) with the fused comb (OPS:11250-11330).
    fn hc_mixes(&self, x: &[f32], m: usize, w: &(QuantMat<'_>, Vec<f32>, Vec<f32>)) -> Mixes {
        let c = &self.cfg;
        let (hc, n) = (c.hc, c.hc * c.n_embd);
        let mut flat = vec![0f32; m * n];
        rms_norm(x, None, n, c.eps, &mut flat);
        let mixes = self.mm(w.0, &flat, m);
        let mix = (2 + hc) * hc;
        let (scale, base) = (&w.1, &w.2);
        let mut out = Mixes { pre: vec![0.0; m * hc], post: vec![0.0; m * hc], comb: vec![0.0; m * hc * hc] };
        for t in 0..m {
            let mt = &mixes[t * mix..(t + 1) * mix];
            for j in 0..hc {
                out.pre[t * hc + j] = sigmoid(mt[j] * scale[0] + base[j]) * 1.0 + c.hc_eps;
                out.post[t * hc + j] = sigmoid(mt[hc + j] * scale[1] + base[hc + j]) * 2.0;
            }
            let cb = hc_comb(mt, scale[2], base, c.hc_eps, c.hc_iters);
            out.comb[t * 16..(t + 1) * 16].copy_from_slice(&cb);
        }
        out
    }

    /// `ggml_dsv4_hc_pre`: `out[d] = sum_s w[s] * x[s][d]` (OPS:11392-11402).
    fn hc_pre(&self, x: &[f32], w: &[f32], m: usize) -> Vec<f32> {
        let (hc, d) = (self.cfg.hc, self.cfg.n_embd);
        let mut out = vec![0f32; m * d];
        for t in 0..m {
            let xt = &x[t * hc * d..(t + 1) * hc * d];
            let wt = &w[t * hc..(t + 1) * hc];
            for i in 0..d {
                let mut s = 0f32;
                for h in 0..hc {
                    s += xt[h * d + i] * wt[h];
                }
                out[t * d + i] = s;
            }
        }
        out
    }

    /// `ggml_dsv4_hc_post`: `Y[dst][d] = x[d] * post[dst] + sum_src res[src][d] * comb[dst + hc*src]` (OPS:11488-11504).
    fn hc_post(&self, x: &[f32], res: &[f32], mx: &Mixes, m: usize) -> Vec<f32> {
        let (hc, d) = (self.cfg.hc, self.cfg.n_embd);
        let mut out = vec![0f32; m * hc * d];
        out.par_chunks_mut(hc * d).enumerate().for_each(|(t, o)| {
            let rt = &res[t * hc * d..(t + 1) * hc * d];
            let cb = &mx.comb[t * hc * hc..(t + 1) * hc * hc];
            for dst in 0..hc {
                let pv = mx.post[t * hc + dst];
                for i in 0..d {
                    let mut s = x[t * d + i] * pv;
                    for src in 0..hc {
                        s += rt[src * d + i] * cb[dst + hc * src];
                    }
                    o[dst * d + i] = s;
                }
            }
        });
        out
    }

    /// Table rows (24 per token) of Engram module `e` for the tokens at positions `p0..`, from `cache.tokens` (which already
    /// holds them). `engram_hash` (D41:240-278).
    pub fn engram_ids(&self, e: usize, cache: &Cache, p0: usize, m: usize) -> Vec<i32> {
        let c = &self.cfg;
        let n_cols = (c.ngram - 1) * c.ngram_heads;
        let mult = &c.ple_mult[e * c.ngram..(e + 1) * c.ngram];
        let primes = &c.ple_primes[e * n_cols..(e + 1) * n_cols];
        let offs = &c.ple_offs[e * n_cols..(e + 1) * n_cols];
        let mut ids = Vec::with_capacity(m * n_cols);
        for t in 0..m {
            let pos = p0 + t;
            let toks = engram_toks(cache.tokens[pos] as i64, pos as i64, c.ngram, &c.token_map, c.engram_pad, |s| cache.tokens[pos - s] as i64);
            ids.extend(engram_rows(&toks, mult, primes, offs, c.ngram_heads));
        }
        ids
    }

    /// `build_engram` (D41:372-407) on the stream state `x` [m][hc][d], in place.
    fn engram(&self, il: usize, e: usize, x: &mut [f32], cache: &Cache, p0: usize, m: usize, taps: &mut Option<&mut Taps>) {
        let c = &self.cfg;
        let (hc, d) = (c.hc, c.n_embd);
        let eg = self.layers[il].engram.as_ref().unwrap();
        let ple = self.ple.as_ref().unwrap();
        let ids = self.engram_ids(e, cache, p0, m);
        let n_cols = ids.len() / m;
        let w = n_cols * c.ple_dim;
        let mut rows = vec![0f32; m * w];
        for (k, &id) in ids.iter().enumerate() {
            ple.dequant_row(id as u32 as usize, &mut rows[k * c.ple_dim..(k + 1) * c.ple_dim]);
        }
        let kv = self.mm(eg.kv, &rows, m);
        let kvw = (hc + 1) * d;
        // the dumped `engram_kv` is the last tensor of that name: the 4 keys (`engram_kv (view) (cont)`)
        if taps.is_some() {
            let keys: Vec<f32> = (0..m).flat_map(|t| kv[t * kvw..t * kvw + hc * d].iter().cloned()).collect();
            tap(taps, "engram_kv", il as i32, &keys);
        }
        let scale = 1.0 / (d as f32).sqrt();
        let mut gates = vec![0f32; m * hc];
        let mut xn = vec![0f32; d];
        let mut kn = vec![0f32; d];
        for t in 0..m {
            for s in 0..hc {
                let xs = &x[(t * hc + s) * d..(t * hc + s + 1) * d];
                let key = &kv[t * kvw + s * d..t * kvw + (s + 1) * d];
                rms_norm(xs, None, d, c.eps, &mut xn);
                rms_norm(key, None, d, c.eps, &mut kn);
                let mut sum = 0f64;
                for i in 0..d {
                    sum += ((xn[i] * kn[i]) * eg.gate[s * d + i]) as f64;
                }
                let dot = (sum as f32) * scale;
                let mag = dot.abs().max(1e-6).sqrt();
                let sign = if -dot > 0.0 { -1.0 } else { 1.0 };
                gates[t * hc + s] = sigmoid(mag * sign);
            }
            let value = &kv[t * kvw + hc * d..(t + 1) * kvw];
            for s in 0..hc {
                let g = gates[t * hc + s];
                let xs = &mut x[(t * hc + s) * d..(t * hc + s + 1) * d];
                for i in 0..d {
                    xs[i] += value[i] * g;
                }
            }
        }
        tap(taps, "engram_gate", il as i32, &gates);
        tap(taps, "engram_out", il as i32, x);
    }

    /// Run `tokens` (positions `cache.len ..`) through the model and return the logits of the last one.
    /// `taps` records intermediates under llama.cpp's names.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache, mut taps: Option<&mut Taps>) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (d, hc, m) = (c.n_embd, c.hc, tokens.len());
        if m == 0 {
            bail!("no tokens");
        }
        let p0 = cache.len;
        if p0 + m > MAX_CTX {
            bail!("sequence of {} tokens: this reference skips the sliding window and the lightning indexer, which is exact only up to {MAX_CTX} tokens", p0 + m);
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= c.vocab) {
            bail!("token id {bad} outside the vocabulary ({})", c.vocab);
        }
        cache.tokens.extend_from_slice(tokens);
        let emb = self.embed(tokens);
        let mut x = vec![0f32; m * hc * d];
        for t in 0..m {
            for s in 0..hc {
                x[(t * hc + s) * d..(t * hc + s + 1) * d].copy_from_slice(&emb[t * d..(t + 1) * d]);
            }
        }
        let mut pre_prev: Option<Vec<f32>> = None;
        let mut xn = vec![0f32; m * d];
        for (il, l) in self.layers.iter().enumerate() {
            let ii = il as i32;
            if let Some(e) = c.engram[il] {
                self.engram(il, e, &mut x, cache, p0, m, &mut taps);
            }
            let attn_mix = self.hc_mixes(&x, m, &l.hc_attn);
            let cur = match &pre_prev {
                Some(p) => self.hc_pre(&x, p, m),
                None => (0..m).flat_map(|t| x[t * hc * d..t * hc * d + d].iter().cloned()).collect(),
            };
            tap(&mut taps, "hc_attn_pre", ii, &cur);
            rms_norm(&cur, Some(&l.attn_norm), d, c.eps, &mut xn);
            tap(&mut taps, "attn_norm", ii, &xn);
            let a = self.attention(il, &xn, m, p0, cache, &mut taps);
            tap(&mut taps, "attn_out", ii, &a);
            x = self.hc_post(&a, &x, &attn_mix, m);

            let ffn_mix = self.hc_mixes(&x, m, &l.hc_ffn);
            let cur = self.hc_pre(&x, &attn_mix.pre, m);
            tap(&mut taps, "hc_ffn_pre", ii, &cur);
            rms_norm(&cur, Some(&l.ffn_norm), d, c.eps, &mut xn);
            tap(&mut taps, "ffn_norm", ii, &xn);
            let f = self.ffn(il, &xn, m, &mut taps);
            tap(&mut taps, "ffn_out", ii, &f);
            x = self.hc_post(&f, &x, &ffn_mix, m);
            tap(&mut taps, "l_out", ii, &x);
            pre_prev = Some(ffn_mix.pre);
        }
        cache.len += m;
        // head on the last token (llama.cpp crops to the output rows before hc_out, D41:919-922)
        let t = m - 1;
        let pp = pre_prev.unwrap();
        let hc_out = self.hc_pre(&x[t * hc * d..(t + 1) * hc * d], &pp[t * hc..(t + 1) * hc], 1);
        tap(&mut taps, "hc_out", -1, &hc_out);
        let mut xo = vec![0f32; d];
        rms_norm(&hc_out, Some(&self.out_norm), d, c.eps, &mut xo);
        tap(&mut taps, "result_norm", -1, &xo);
        let logits = self.mm(self.output, &xo, 1);
        tap(&mut taps, "result_output", -1, &logits);
        Ok(logits)
    }

    /// `build_attention41` (D41:409-766) without the indexer (every visible compressed row is selected below 512 rows).
    fn attention(&self, il: usize, x: &[f32], m: usize, p0: usize, cache: &mut Cache, taps: &mut Option<&mut Taps>) -> Vec<f32> {
        let c = &self.cfg;
        let l = &self.layers[il];
        let ii = il as i32;
        let (nh, hd, nrot) = (c.n_head, c.head_dim, c.n_rot);
        let rope = c.rope(il, hd - nrot);
        let ratio = c.ratio[il];

        let qa = self.mm(l.q_a, x, m);
        let mut qr = vec![0f32; m * c.q_lora];
        rms_norm(&qa, Some(&l.q_a_norm), c.q_lora, c.eps, &mut qr);
        tap(taps, "qr_norm", ii, &qr);
        let mut q = self.mm(l.q_b, &qr, m);
        let kva = self.mm(l.kv, x, m);
        let mut kv = vec![0f32; m * hd];
        rms_norm(&kva, Some(&l.kv_norm), hd, c.eps, &mut kv);
        let caches: Vec<Vec<(f32, f32)>> = (0..m).map(|t| rope.cache((p0 + t) as i64, 1.0)).collect();
        q.par_chunks_mut(hd).enumerate().for_each(|(th, qh)| rope.apply(qh, &caches[th / nh]));
        for t in 0..m {
            rope.apply(&mut kv[t * hd..(t + 1) * hd], &caches[t]);
        }
        tap(taps, "q", ii, &q);
        tap(taps, "kv", ii, &kv);
        for t in 0..m {
            let row = &kv[t * hd..(t + 1) * hd];
            if self.kv_f16 {
                cache.raw[il].extend(row.iter().map(|&v| f16r(v)));
            } else {
                cache.raw[il].extend_from_slice(row);
            }
        }

        // KV compression at a kv source layer, from this layer's attn_norm output (D41:465-502)
        if let Some(cp) = &l.comp {
            let ckv = self.mm(cp.kv, x, m);
            let csc = match cp.gate {
                Some(gw) => self.mm(gw, x, m),
                None => vec![0f32; m * hd],
            };
            cache.comp_kv[il].extend_from_slice(&ckv);
            cache.comp_score[il].extend_from_slice(&csc);
            let mut lat_all = Vec::new();
            for t in 0..m {
                let pos = p0 + t;
                if (pos + 1) % ratio != 0 {
                    continue;
                }
                let start = pos + 1 - ratio;
                let mut comp = vec![0f32; hd];
                for (ch, cv) in comp.iter_mut().enumerate() {
                    // ggml_soft_max over the ratio axis (max, exp, f64 sum, scale by (float)(1/sum)), then sum_rows(v * p)
                    let sc: Vec<f32> = (0..ratio).map(|j| cache.comp_score[il][(start + j) * hd + ch]).collect();
                    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = sc.iter().map(|&s| (s - mx).exp()).collect();
                    let sum: f64 = ex.iter().map(|&e| e as f64).sum();
                    let inv = (1.0 / sum) as f32;
                    let mut acc = 0f64;
                    for j in 0..ratio {
                        acc += (cache.comp_kv[il][(start + j) * hd + ch] * (ex[j] * inv)) as f64;
                    }
                    *cv = acc as f32;
                }
                let mut lat = vec![0f32; hd];
                rms_norm(&comp, Some(&cp.norm), hd, c.eps, &mut lat);
                lat_all.extend_from_slice(&lat);
                rope.apply(&mut lat, &rope.cache(start as i64, 1.0));
                if self.kv_f16 {
                    lat.iter_mut().for_each(|v| *v = f16r(*v));
                }
                cache.comp[il].extend_from_slice(&lat);
            }
            if !lat_all.is_empty() {
                tap(taps, "comp_latent", ii, &lat_all);
            }
        }

        // attention: [raw window | complete compressed rows of the kv source], CPU flash attention order (OPS:8840-8925)
        let scale = 1.0 / (hd as f32).sqrt();
        let raw = &cache.raw[il];
        let comp: &[f32] = match c.kv_src[il] {
            Some(s) if ratio != 0 => &cache.comp[s],
            _ => &[],
        };
        let kv_f16 = self.kv_f16;
        let n_swa = c.n_swa;
        let mut out = vec![0f32; m * nh * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(th, o)| {
            let (t, h) = (th / nh, th % nh);
            let pos = p0 + t;
            let qh: Vec<f32> = if kv_f16 { q[th * hd..(th + 1) * hd].iter().map(|&v| f16r(v)).collect() } else { q[th * hd..(th + 1) * hd].to_vec() };
            let n_vis = if ratio == 0 { 0 } else { (pos + 1) / ratio };
            let lo = (pos + 1).saturating_sub(n_swa);
            let keys = (lo..=pos).map(|j| &raw[j * hd..(j + 1) * hd]).chain((0..n_vis).map(|j| &comp[j * hd..(j + 1) * hd]));
            let (mut s_sum, mut mx) = (0f32, f32::NEG_INFINITY);
            o.iter_mut().for_each(|e| *e = 0.0);
            for k in keys {
                let mut dot = 0f64;
                for i in 0..hd {
                    dot += (qh[i] * k[i]) as f64;
                }
                let s = (dot as f32) * scale;
                let (mut ms, mut vs) = (1f32, 1f32);
                if s > mx {
                    let old = mx;
                    mx = s;
                    ms = (old - mx).exp();
                    o.iter_mut().for_each(|e| *e *= ms);
                } else {
                    vs = (s - mx).exp();
                }
                for i in 0..hd {
                    o[i] += k[i] * vs;
                }
                s_sum = s_sum * ms + vs;
            }
            let sk = l.sinks[h];
            let (mut ms, mut vs) = (1f32, 1f32);
            if sk > mx {
                ms = (mx - sk).exp();
                o.iter_mut().for_each(|e| *e *= ms);
            } else {
                vs = (sk - mx).exp();
            }
            s_sum = s_sum * ms + vs;
            let inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
            o.iter_mut().for_each(|e| *e *= inv);
        });
        tap(taps, if ratio == 0 { "attn_raw" } else { "attn_comp" }, ii, &out);
        // de-rope at the query position (ggml_rope_ext_back, D41:751-755)
        let back: Vec<Vec<(f32, f32)>> = (0..m).map(|t| rope.cache((p0 + t) as i64, -1.0)).collect();
        out.par_chunks_mut(hd).enumerate().for_each(|(th, oh)| rope.apply(oh, &back[th / nh]));
        tap(taps, "attn_derope", ii, &out);

        self.out_proj(il, &out, m)
    }

    /// Grouped output projection (D41:757-763): de-roped attention output `[m][n_head * 512]` -> `[m][n_embd]`.
    pub fn out_proj(&self, il: usize, out: &[f32], m: usize) -> Vec<f32> {
        let c = &self.cfg;
        let l = &self.layers[il];
        let (nh, hd) = (c.n_head, c.head_dim);
        let hpg = nh / c.o_groups;
        let mut oa = vec![0f32; m * c.o_groups * c.o_lora];
        for g in 0..c.o_groups {
            let (off, len, r0, rows) = out_group(g, hpg, hd, c.o_lora);
            let xg: Vec<f32> = (0..m).flat_map(|t| out[t * nh * hd + off..t * nh * hd + off + len].iter().cloned()).collect();
            let yg = self.mm(sub(l.wo_a, r0, rows), &xg, m);
            for t in 0..m {
                oa[t * c.o_groups * c.o_lora + g * c.o_lora..t * c.o_groups * c.o_lora + (g + 1) * c.o_lora].copy_from_slice(&yg[t * rows..(t + 1) * rows]);
            }
        }
        self.mm(l.wo_b, &oa, m)
    }

    /// Routed MoE (`build_moe_ffn`, GR:2300-2520) + the shared expert (`build_ffn`), returns `moe + shared`.
    pub fn ffn(&self, il: usize, x: &[f32], m: usize, taps: &mut Option<&mut Taps>) -> Vec<f32> {
        let c = &self.cfg;
        let l = &self.layers[il];
        let ii = il as i32;
        let (d, ne, k, ff) = (c.n_embd, c.n_expert, c.n_used, c.ff_exp);
        let logits = self.mm(l.router, x, m);
        tap(taps, "ffn_moe_logits", ii, &logits);
        let mut sel: Vec<Vec<usize>> = Vec::with_capacity(m);
        let mut w_raw = vec![0f32; m * k];
        let mut w = vec![0f32; m * k];
        for t in 0..m {
            let probs: Vec<f32> = logits[t * ne..(t + 1) * ne].iter().map(|&v| sqrt_softplus(v)).collect();
            let biased: Vec<f32> = probs.iter().zip(&l.bias).map(|(p, b)| p + b).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap().then(a.cmp(&b)));
            idx.truncate(k);
            for (j, &e) in idx.iter().enumerate() {
                w_raw[t * k + j] = probs[e];
            }
            let wr = &w_raw[t * k..(t + 1) * k];
            // weights / (sum_rows + 1e-20) (GR:2393-2409, deepseek41 branch), then x expert_weights_scale
            let norm = c.w_norm && k > 1;
            let sum = if norm { (wr.iter().map(|&v| v as f64).sum::<f64>() as f32) + 1e-20 } else { 1.0 };
            for j in 0..k {
                let mut v = if norm { wr[j] / sum } else { wr[j] };
                if c.w_scale != 0.0 && c.w_scale != 1.0 {
                    v *= c.w_scale;
                }
                w[t * k + j] = v;
            }
            sel.push(idx);
        }
        if taps.is_some() {
            let ids: Vec<f32> = sel.iter().flat_map(|v| v.iter().map(|&e| e as f32)).collect();
            tap(taps, "ffn_moe_topk", ii, &ids);
            tap(taps, "ffn_moe_weights", ii, &w_raw);
        }
        // expert outputs per (token, slot), then the weighted sum in slot order (GR: ggml_mul by weights, then adds)
        let mut by_e: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
        for (t, v) in sel.iter().enumerate() {
            for (j, &e) in v.iter().enumerate() {
                by_e[e].push((t, j));
            }
        }
        let mut ys = vec![0f32; m * k * d];
        let limit = c.clamp_exp[il];
        for (e, slots) in by_e.iter().enumerate() {
            if slots.is_empty() {
                continue;
            }
            let me = slots.len();
            let mut xe = vec![0f32; me * d];
            for (i, &(t, _)) in slots.iter().enumerate() {
                xe[i * d..(i + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
            }
            let g = self.mm(sub(l.gate_exps, e * ff, ff), &xe, me);
            let u = self.mm(sub(l.up_exps, e * ff, ff), &xe, me);
            let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| if limit > 1e-6 { swiglu_clamp(g, u, limit) } else { silu(g) * u }).collect();
            let y = self.mm(sub(l.down_exps, e * d, d), &act, me);
            for (i, &(t, j)) in slots.iter().enumerate() {
                ys[(t * k + j) * d..(t * k + j + 1) * d].copy_from_slice(&y[i * d..(i + 1) * d]);
            }
        }
        let mut moe = vec![0f32; m * d];
        for t in 0..m {
            let o = &mut moe[t * d..(t + 1) * d];
            for j in 0..k {
                let wv = w[t * k + j];
                let y = &ys[(t * k + j) * d..(t * k + j + 1) * d];
                if j == 0 {
                    for i in 0..d {
                        o[i] = y[i] * wv;
                    }
                } else {
                    for i in 0..d {
                        o[i] += y[i] * wv;
                    }
                }
            }
        }
        tap(taps, "ffn_moe_out", ii, &moe);
        let lim = c.clamp_shexp[il];
        let g = self.mm(l.gate_sh, x, m);
        let u = self.mm(l.up_sh, x, m);
        let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| if lim > 1e-6 { swiglu_clamp(g, u, lim) } else { silu(g) * u }).collect();
        let sh = self.mm(l.down_sh, &act, m);
        tap(taps, "ffn_shexp", ii, &sh);
        moe.iter().zip(&sh).map(|(a, b)| a + b).collect()
    }
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

/// The quant types a model's tensors use (for a load-time report).
pub fn types_used(g: &GgufSet) -> Vec<(GgmlType, usize)> {
    let mut m: HashMap<GgmlType, usize> = HashMap::new();
    for (_, t) in g.tensors() {
        *m.entry(t.ty).or_default() += 1;
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by_key(|(t, _)| t.name());
    v
}
