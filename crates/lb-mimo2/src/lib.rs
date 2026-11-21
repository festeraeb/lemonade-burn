//! `mimo2` CPU reference decoder (Xiaomi MiMo-V2 Flash: MiMo-V2.6-Flash and relatives).
//!
//! Every structural fact is read from the GGUF metadata and follows llama.cpp's `src/models/mimo2.cpp` and the graph helpers
//! it calls (`build_attn` with sinks, `build_moe_ffn` with sigmoid gating, `build_ffn`, NEOX `ggml_rope_ext`, `rms_norm`),
//! MIT licensed; the code is this engine's own. Facts and citations: `docs/MODEL_MIMO2.md`.
//!
//! The pass, per layer `il` (48 in V2.6-Flash):
//! * `x = rms_norm(h) * attn_norm`; one fused `attn_qkv` matmul gives Q (64 heads x 192), K (`n_head_kv[il]` x 192) and
//!   V (`n_head_kv[il]` x 128); NEOX rope on the first 64 dims of every Q/K head, base `rope.freq_base` (full layers) or
//!   `rope.freq_base_swa` (sliding layers, `sliding_window_pattern[il] == 1`).
//! * attention scaled by 1/sqrt(192), causal; sliding layers mask keys with `pos_q - pos_k >= sliding_window` and add the
//!   per-head sink logit to the softmax denominator; `attn_output`, then times `attention.value_scale`.
//! * residual; `rms_norm * ffn_norm`; layer 0 is a dense SiLU-gated FFN, the others MoE: `sigmoid(router x)`, top-8 by
//!   `prob + exp_probs_b`, weights = the chosen probs over their sum (clamped below at 6.1e-5), SiLU-gated experts
//!   (gate/up Q2_K, down MXFP4 in the ggml-org Q2_K file), weighted sum; residual.
//! * head: `rms_norm * output_norm`, `output`.
//!
//! f32 activations; weights stay quantised and are dequantised row by row (bit-identical to ggml's `to_float`, checked by
//! `lb-gguf/examples/ggml_ref_check.rs`). The K/V cache rounds to f16 by default, as llama.cpp's default cache type does
//! (`LB_MIMO_KV_F16=0` keeps f32), and matmul inputs are rounded to the weight's ggml `vec_dot_type` by default
//! (`LB_MIMO_ACT_EMU=0` keeps exact f32 inputs), so the numbers follow llama.cpp's CPU path.

pub mod tokenizer;

use anyhow::{anyhow, bail, Result};
use lb_gguf::{matvec_f32, GgmlType, GgufSet, QuantMat};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: Vec<usize>,
    pub head_k: usize,
    pub head_v: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub swa_window: usize,
    pub is_swa: Vec<bool>,
    pub eps: f32,
    pub n_expert: usize,
    pub n_used: usize,
    pub ff_exp: usize,
    pub n_ff: usize,
    pub value_scale: f32,
    pub vocab: usize,
}

impl Config {
    pub fn from_gguf(g: &GgufSet) -> Result<Self> {
        let arch = g.meta_str("general.architecture").unwrap_or("");
        if arch != "mimo2" {
            bail!("architecture {arch:?}, this crate loads mimo2");
        }
        let u = |k: &str| g.meta_u64(&format!("mimo2.{k}")).map(|v| v as usize).ok_or_else(|| anyhow!("missing mimo2.{k}"));
        let f = |k: &str| g.meta_f64(&format!("mimo2.{k}")).map(|v| v as f32).ok_or_else(|| anyhow!("missing mimo2.{k}"));
        let n_layer = u("block_count")?;
        let arr = |k: &str| -> Result<Vec<u64>> {
            let key = format!("mimo2.{k}");
            if let Some(a) = g.meta_u64_array(&key) {
                return Ok(a);
            }
            g.meta_u64(&key).map(|v| vec![v; n_layer]).ok_or_else(|| anyhow!("missing {key}"))
        };
        let kv = arr("attention.head_count_kv")?;
        let pat = arr("attention.sliding_window_pattern")?;
        if kv.len() < n_layer || pat.len() < n_layer {
            bail!("per-layer arrays shorter than block_count");
        }
        if g.meta_u64("mimo2.expert_gating_func") != Some(2) {
            bail!("expert_gating_func {:?}: mimo2.cpp hard-codes sigmoid (2)", g.meta_u64("mimo2.expert_gating_func"));
        }
        if g.meta_u64("mimo2.expert_group_count").unwrap_or(1) > 1 {
            bail!("expert groups > 1 not implemented");
        }
        let vs = g.meta_f64("mimo2.attention.value_scale").map(|v| v as f32).unwrap_or(1.0);
        Ok(Config {
            n_layer,
            n_embd: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_head_kv: kv[..n_layer].iter().map(|&v| v as usize).collect(),
            head_k: u("attention.key_length")?,
            head_v: u("attention.value_length")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")?,
            rope_base_swa: f("rope.freq_base_swa")?,
            swa_window: u("attention.sliding_window")?,
            is_swa: pat[..n_layer].iter().map(|&v| v != 0).collect(),
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            ff_exp: u("expert_feed_forward_length")?,
            n_ff: u("feed_forward_length")?,
            value_scale: vs,
            vocab: g.tensor("token_embd.weight").map(|t| t.dims[1]).ok_or_else(|| anyhow!("no token_embd"))?,
        })
    }
}

pub struct Dense<'a> {
    pub gate: QuantMat<'a>,
    pub up: QuantMat<'a>,
    pub down: QuantMat<'a>,
}

pub struct Moe<'a> {
    pub router: QuantMat<'a>, // F32 [n_expert][n_embd]
    pub bias: Vec<f32>,       // exp_probs_b [n_expert]
    pub gate_exps: QuantMat<'a>, // [n_expert * ff_exp][n_embd]
    pub up_exps: QuantMat<'a>,
    pub down_exps: QuantMat<'a>, // [n_expert * n_embd][ff_exp]
}

pub struct Layer<'a> {
    pub attn_norm: Vec<f32>,
    pub qkv: QuantMat<'a>,
    pub wo: QuantMat<'a>,
    pub sinks: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub dense: Option<Dense<'a>>,
    pub moe: Option<Moe<'a>>,
}

pub struct Model<'a> {
    pub cfg: Config,
    pub tok_embd: QuantMat<'a>,
    pub out_norm: Vec<f32>,
    pub output: QuantMat<'a>,
    pub layers: Vec<Layer<'a>>,
    pub kv_f16: bool,
}

fn qmat<'a>(g: &'a GgufSet, name: &str) -> Result<QuantMat<'a>> {
    let (f, t) = g.find(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
    let cols = t.dims[0];
    let rows = t.n_elems() / cols;
    Ok(QuantMat::new(t.ty, rows, cols, f.tensor_bytes(t)))
}

fn vecf(g: &GgufSet, name: &str) -> Result<Vec<f32>> {
    g.tensor_f32(name).ok_or_else(|| anyhow!("missing tensor {name}"))
}

impl<'a> Model<'a> {
    pub fn load(g: &'a GgufSet) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let qkv = qmat(g, &p("attn_qkv.weight"))?;
            let want = cfg.n_head * cfg.head_k + cfg.n_head_kv[il] * (cfg.head_k + cfg.head_v);
            if qkv.rows != want {
                bail!("blk.{il}.attn_qkv has {} rows, expected {want}", qkv.rows);
            }
            let dense = if g.tensor(&p("ffn_gate_inp.weight")).is_none() {
                Some(Dense { gate: qmat(g, &p("ffn_gate.weight"))?, up: qmat(g, &p("ffn_up.weight"))?, down: qmat(g, &p("ffn_down.weight"))? })
            } else {
                None
            };
            let moe = if dense.is_none() {
                Some(Moe {
                    router: qmat(g, &p("ffn_gate_inp.weight"))?,
                    bias: g.tensor_f32(&p("exp_probs_b.bias")).unwrap_or_else(|| vec![0.0; cfg.n_expert]),
                    gate_exps: qmat(g, &p("ffn_gate_exps.weight"))?,
                    up_exps: qmat(g, &p("ffn_up_exps.weight"))?,
                    down_exps: qmat(g, &p("ffn_down_exps.weight"))?,
                })
            } else {
                None
            };
            layers.push(Layer {
                attn_norm: vecf(g, &p("attn_norm.weight"))?,
                qkv,
                wo: qmat(g, &p("attn_output.weight"))?,
                sinks: g.tensor_f32(&p("attn_sinks.weight")),
                ffn_norm: vecf(g, &p("ffn_norm.weight"))?,
                dense,
                moe,
            });
        }
        let kv_f16 = std::env::var("LB_MIMO_KV_F16").map(|v| v != "0").unwrap_or(true);
        Ok(Model { tok_embd: qmat(g, "token_embd.weight")?, out_norm: vecf(g, "output_norm.weight")?, output: qmat(g, "output.weight")?, layers, cfg, kv_f16 })
    }
}

/// Per-layer K/V of every position so far (post-rope K), `[pos][n_head_kv * head]`.
pub struct Cache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub len: usize,
}

impl Cache {
    pub fn new(n_layer: usize) -> Self {
        Cache { k: vec![Vec::new(); n_layer], v: vec![Vec::new(); n_layer], len: 0 }
    }
}

/// Named intermediate tensors, recorded under llama.cpp's callback names (`l_out-<il>` etc.), `[token][dim]`.
pub type Taps = HashMap<String, Vec<f32>>;

fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = w.len();
    for (xr, or) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let ss: f64 = xr.iter().map(|&v| (v * v) as f64).sum(); // ggml: f32 square, double sum
        let scale = 1.0 / ((ss / n as f64) as f32 + eps).sqrt();
        for i in 0..n {
            or[i] = xr[i] * scale * w[i];
        }
    }
}

/// Whether matmul inputs are rounded as ggml's CPU matmul rounds them (`LB_MIMO_ACT_EMU`, default 1): Q8_K blocks for
/// Q2_K weights, Q8_0 blocks for Q8_0 and MXFP4 weights (`lb_gguf::act`). Off = exact f32 inputs.
fn act_emu() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LB_MIMO_ACT_EMU").map(|v| v != "0").unwrap_or(true))
}

pub fn mm(w: QuantMat<'_>, x: &[f32], m: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * w.rows];
    if act_emu() && w.ty != GgmlType::F32 {
        let mut xr = x.to_vec();
        lb_gguf::act::act_round(w.ty, &mut xr, w.cols);
        matvec_f32(w, &xr, m, &mut out);
    } else {
        matvec_f32(w, x, m, &mut out);
    }
    out
}

/// Rows `[r0, r0 + n)` of a matrix as their own matrix view.
fn sub<'a>(w: QuantMat<'a>, r0: usize, n: usize) -> QuantMat<'a> {
    let rb = w.row_bytes();
    QuantMat::new(w.ty, n, w.cols, &w.bytes[r0 * rb..(r0 + n) * rb])
}

/// NEOX rope on the first `n_rot` dims of one head: pairs (i, i + n_rot/2), theta_i = pos * base^(-2i/n_rot),
/// with the theta recurrence ggml uses (`theta *= theta_scale`, f32).
fn rope_neox(x: &mut [f32], pos: usize, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    let mut theta = pos as f32;
    for i in 0..half {
        let (s, c) = theta.sin_cos();
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

impl Model<'_> {
    /// Embedding rows of `tokens`, `[m][n_embd]`.
    pub fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.cfg.n_embd;
        let mut x = vec![0f32; tokens.len() * d];
        for (t, &id) in tokens.iter().enumerate() {
            self.tok_embd.dequant_row(id as usize, &mut x[t * d..(t + 1) * d]);
        }
        x
    }

    /// Run `tokens` (positions `cache.len ..`) through every layer, append their K/V, and return the final hidden
    /// states (before `output_norm`), `[m][n_embd]`. `taps` records intermediates under llama.cpp names.
    pub fn forward_hidden(&self, tokens: &[u32], cache: &mut Cache, mut taps: Option<&mut Taps>) -> Vec<f32> {
        let c = &self.cfg;
        let (d, m) = (c.n_embd, tokens.len());
        let mut h = self.embed(tokens);
        let mut x = vec![0f32; m * d];
        let p0 = cache.len;
        for (il, l) in self.layers.iter().enumerate() {
            rms_norm(&h, &l.attn_norm, c.eps, &mut x);
            if let Some(t) = taps.as_deref_mut() {
                t.insert(format!("attn_norm-{il}"), x.clone());
            }
            let attn = self.attention(il, &x, m, p0, cache);
            let mut a = mm(l.wo, &attn, m);
            if let Some(t) = taps.as_deref_mut() {
                t.insert(format!("attn_out-{il}"), a.clone());
            }
            if c.value_scale != 1.0 {
                a.iter_mut().for_each(|v| *v *= c.value_scale);
            }
            for (hv, av) in h.iter_mut().zip(&a) {
                *hv += av;
            }
            rms_norm(&h, &l.ffn_norm, c.eps, &mut x);
            if let Some(t) = taps.as_deref_mut() {
                t.insert(format!("ffn_norm-{il}"), x.clone());
            }
            let f = if let Some(dn) = &l.dense {
                let g = mm(dn.gate, &x, m);
                let u = mm(dn.up, &x, m);
                let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
                let out = mm(dn.down, &act, m);
                if let Some(t) = taps.as_deref_mut() {
                    t.insert(format!("ffn_out-{il}"), out.clone());
                }
                out
            } else {
                self.moe(l.moe.as_ref().unwrap(), &x, m, il, taps.as_deref_mut())
            };
            for (hv, fv) in h.iter_mut().zip(&f) {
                *hv += fv;
            }
            if let Some(t) = taps.as_deref_mut() {
                t.insert(format!("l_out-{il}"), h.clone());
            }
        }
        cache.len += m;
        h
    }

    /// Logits of the last of `tokens`.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache, mut taps: Option<&mut Taps>) -> Vec<f32> {
        let d = self.cfg.n_embd;
        let h = self.forward_hidden(tokens, cache, taps.as_deref_mut());
        let last = &h[(tokens.len() - 1) * d..];
        let mut x = vec![0f32; d];
        rms_norm(last, &self.out_norm, self.cfg.eps, &mut x);
        if let Some(t) = taps {
            t.insert("result_norm--1".into(), x.clone());
        }
        mm(self.output, &x, 1)
    }

    pub fn attention(&self, il: usize, x: &[f32], m: usize, p0: usize, cache: &mut Cache) -> Vec<f32> {
        let c = &self.cfg;
        let l = &self.layers[il];
        let (nh, nkv, hk, hv) = (c.n_head, c.n_head_kv[il], c.head_k, c.head_v);
        let qkv = mm(l.qkv, x, m);
        let rows = l.qkv.rows;
        let base = if c.is_swa[il] { c.rope_base_swa } else { c.rope_base };
        let mut q = vec![0f32; m * nh * hk];
        for t in 0..m {
            let r = &qkv[t * rows..(t + 1) * rows];
            let pos = p0 + t;
            q[t * nh * hk..(t + 1) * nh * hk].copy_from_slice(&r[..nh * hk]);
            for hh in 0..nh {
                rope_neox(&mut q[(t * nh + hh) * hk..(t * nh + hh + 1) * hk], pos, c.n_rot, base);
            }
            // Q stays f32: rounding Q (or the softmax weights) to f16 measured worse against llama.cpp's non-flash CPU
            // attention (isolated attn_out rel error 2e-5 -> 5e-5..1e-3), so llama.cpp does not do that here.
            let mut k = r[nh * hk..nh * hk + nkv * hk].to_vec();
            for hh in 0..nkv {
                rope_neox(&mut k[hh * hk..(hh + 1) * hk], pos, c.n_rot, base);
            }
            let mut v = r[nh * hk + nkv * hk..nh * hk + nkv * (hk + hv)].to_vec();
            if self.kv_f16 {
                k.iter_mut().for_each(|e| *e = f16r(*e));
                v.iter_mut().for_each(|e| *e = f16r(*e));
            }
            cache.k[il].extend_from_slice(&k);
            cache.v[il].extend_from_slice(&v);
        }
        let scale = 1.0 / (hk as f32).sqrt();
        let group = nh / nkv;
        let (kc, vc) = (&cache.k[il], &cache.v[il]);
        let mut out = vec![0f32; m * nh * hv];
        out.par_chunks_mut(hv).enumerate().for_each(|(th, o)| {
            let (t, hh) = (th / nh, th % nh);
            let pos = p0 + t;
            let kvh = hh / group;
            let qh = &q[th * hk..(th + 1) * hk];
            let lo = if c.is_swa[il] { (pos + 1).saturating_sub(c.swa_window) } else { 0 };
            let mut s: Vec<f32> = (lo..=pos)
                .map(|j| {
                    let kj = &kc[(j * nkv + kvh) * hk..(j * nkv + kvh + 1) * hk];
                    qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let mut mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sink = l.sinks.as_ref().map(|sk| sk[hh]);
            if let Some(sk) = sink {
                mx = mx.max(sk);
            }
            let mut sum = 0f64;
            for e in s.iter_mut() {
                *e = (*e - mx).exp();
                sum += *e as f64;
            }
            if let Some(sk) = sink {
                sum += ((sk - mx).exp()) as f64;
            }
            let inv = (1.0 / sum) as f32;
            o.iter_mut().for_each(|e| *e = 0.0);
            for (jj, &p) in s.iter().enumerate() {
                let j = lo + jj;
                let vj = &vc[(j * nkv + kvh) * hv..(j * nkv + kvh + 1) * hv];
                let w = p * inv;
                for (oe, ve) in o.iter_mut().zip(vj) {
                    *oe += w * ve;
                }
            }
        });
        out
    }

    pub fn moe(&self, w: &Moe<'_>, x: &[f32], m: usize, il: usize, mut taps: Option<&mut Taps>) -> Vec<f32> {
        let c = &self.cfg;
        let (d, ne, k, ff) = (c.n_embd, c.n_expert, c.n_used, c.ff_exp);
        let logits = mm(w.router, x, m);
        if let Some(t) = taps.as_deref_mut() {
            t.insert(format!("ffn_moe_logits-{il}"), logits.clone());
        }
        // per token: chosen experts and weights
        let mut chosen: Vec<Vec<(usize, f32)>> = Vec::with_capacity(m);
        for t in 0..m {
            let lg = &logits[t * ne..(t + 1) * ne];
            let probs: Vec<f32> = lg.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            let sel = |e: usize| probs[e] + w.bias[e];
            idx.sort_by(|&a, &b| sel(b).partial_cmp(&sel(a)).unwrap().then(a.cmp(&b)));
            idx.truncate(k);
            let sum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(6.103515625e-5);
            chosen.push(idx.iter().map(|&e| (e, probs[e] / sum)).collect());
        }
        if let Some(t) = taps.as_deref_mut() {
            t.insert(format!("ffn_moe_topk-{il}"), chosen.iter().flat_map(|v| v.iter().map(|&(e, _)| e as f32)).collect());
            t.insert(format!("ffn_moe_weights-{il}"), chosen.iter().flat_map(|v| v.iter().map(|&(_, w)| w)).collect());
        }
        // group tokens by expert
        let mut by_e: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
        for (t, v) in chosen.iter().enumerate() {
            for &(e, wt) in v {
                by_e[e].push((t, wt));
            }
        }
        let mut out = vec![0f32; m * d];
        for (e, toks) in by_e.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let me = toks.len();
            let mut xe = vec![0f32; me * d];
            for (i, &(t, _)) in toks.iter().enumerate() {
                xe[i * d..(i + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
            }
            let g = mm(sub(w.gate_exps, e * ff, ff), &xe, me);
            let u = mm(sub(w.up_exps, e * ff, ff), &xe, me);
            let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
            let y = mm(sub(w.down_exps, e * d, d), &act, me);
            for (i, &(t, wt)) in toks.iter().enumerate() {
                for (o, yv) in out[t * d..(t + 1) * d].iter_mut().zip(&y[i * d..(i + 1) * d]) {
                    *o += wt * yv;
                }
            }
        }
        if let Some(t) = taps.as_deref_mut() {
            t.insert(format!("ffn_moe_out-{il}"), out.clone());
        }
        out
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rope_neox_rotates_pairs_and_keeps_tail() {
        let mut x: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        rope_neox(&mut x, 0, 4, 10000.0);
        assert_eq!(x, (0..8).map(|i| i as f32 + 1.0).collect::<Vec<_>>(), "pos 0 is identity");
        let mut y = vec![1.0, 0.0, 0.0, 0.0, 5.0, 6.0];
        rope_neox(&mut y, 1, 4, 10000.0);
        // pair (0, 2) rotates by theta_0 = 1 rad
        assert!((y[0] - 1f32.cos()).abs() < 1e-6 && (y[2] - 1f32.sin()).abs() < 1e-6);
        assert_eq!(&y[4..], &[5.0, 6.0]);
    }
    #[test]
    fn rms_norm_unit() {
        let mut o = [0f32; 2];
        rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut o);
        let s = (12.5f32).sqrt();
        assert!((o[0] - 3.0 / s).abs() < 1e-6 && (o[1] - 8.0 / s).abs() < 1e-6);
    }
}
