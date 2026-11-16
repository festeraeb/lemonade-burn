//! `qwen35moe` CPU reference decoder (Qwen3.5 / Qwen3.6 MoE).
//!
//! Every structural fact below is taken from the GGUF metadata keys and from llama.cpp's `src/models/qwen35moe.cpp`,
//! `delta-net-base.cpp` and the ggml CPU ops (`ssm_conv`, `gated_delta_net`, `rope_multi`), MIT licensed; the code is this
//! engine's own. See `docs/ENGINE_SECOND_FAMILY.md` for the list of facts and what was checked how.
//!
//! One token at a time (decode and prefill alike: the delta-net state is recurrent), f32 activations, weights stay quantised
//! and are dequantised row by row (`lb_gguf`, bit-identical to ggml's `to_float` on every row of the Q5_K_M file).

pub mod chat;
#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
pub mod pipeline;
pub mod tokenizer;

use anyhow::{anyhow, bail, Result};
use lb_gguf::{Gguf, GgmlType, QuantMat};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,        // trunk blocks; the file has one more (nextn / MTP) which decode does not run
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,       // attention.key_length == value_length
    pub n_rot: usize,          // rope.dimension_count
    pub rope_base: f32,
    pub eps: f32,
    pub n_expert: usize,
    pub n_used: usize,
    pub ff_exp: usize,
    pub ff_shexp: usize,
    pub d_conv: usize,         // ssm.conv_kernel
    pub head_k: usize,         // ssm.state_size (per k head)
    pub n_k_heads: usize,      // ssm.group_count
    pub n_v_heads: usize,      // ssm.time_step_rank
    pub d_inner: usize,        // ssm.inner_size
    pub full_attn_interval: usize,
    pub vocab: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture").unwrap_or("");
        if arch != "qwen35moe" {
            bail!("architecture {arch:?}, this crate loads qwen35moe");
        }
        let u = |k: &str| g.meta_u64(&format!("qwen35moe.{k}")).map(|v| v as usize).ok_or_else(|| anyhow!("missing qwen35moe.{k}"));
        let f = |k: &str| g.meta_f64(&format!("qwen35moe.{k}")).map(|v| v as f32).ok_or_else(|| anyhow!("missing qwen35moe.{k}"));
        let nextn = g.meta_u64("qwen35moe.nextn_predict_layers").unwrap_or(0) as usize;
        let c = Config {
            n_layer: u("block_count")? - nextn,
            n_embd: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            ff_exp: u("expert_feed_forward_length")?,
            ff_shexp: u("expert_shared_feed_forward_length")?,
            d_conv: u("ssm.conv_kernel")?,
            head_k: u("ssm.state_size")?,
            n_k_heads: u("ssm.group_count")?,
            n_v_heads: u("ssm.time_step_rank")?,
            d_inner: u("ssm.inner_size")?,
            full_attn_interval: u("full_attention_interval").unwrap_or(4),
            vocab: g.tensor("token_embd.weight").map(|t| t.dims[1]).ok_or_else(|| anyhow!("no token_embd"))?,
        };
        if g.meta_u64("qwen35moe.attention.value_length").map(|v| v as usize) != Some(c.head_dim) {
            bail!("key_length != value_length");
        }
        if c.n_v_heads % c.n_k_heads != 0 || c.d_inner % c.n_v_heads != 0 {
            bail!("ssm head geometry");
        }
        Ok(c)
    }
    /// Layer `il` is a gated delta-net (recurrent) block unless it is every `full_attention_interval`-th.
    pub fn is_recurrent(&self, il: usize) -> bool {
        (il + 1) % self.full_attn_interval != 0
    }
    pub fn head_v(&self) -> usize {
        self.d_inner / self.n_v_heads
    }
    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_k_heads * self.head_k
    }
}

pub(crate) struct Mat<'a>(pub(crate) QuantMat<'a>);

pub(crate) struct Layer<'a> {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) post_norm: Vec<f32>,
    // attention (every 4th) or delta-net
    pub(crate) attn: Option<AttnW<'a>>,
    pub(crate) lin: Option<LinW<'a>>,
    // moe
    pub(crate) router: Vec<f32>, // [n_expert][n_embd]
    pub(crate) gate_exps: QuantMat<'a>,
    pub(crate) up_exps: QuantMat<'a>,
    pub(crate) down_exps: QuantMat<'a>,
    pub(crate) sh_gate: Mat<'a>,
    pub(crate) sh_up: Mat<'a>,
    pub(crate) sh_down: Mat<'a>,
    pub(crate) sh_gate_inp: Vec<f32>, // [n_embd]
}

pub(crate) struct AttnW<'a> {
    pub(crate) q: Mat<'a>, // [n_head * head_dim * 2] rows: per head [q | gate]
    pub(crate) k: Mat<'a>,
    pub(crate) v: Mat<'a>,
    pub(crate) o: Mat<'a>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
}

pub(crate) struct LinW<'a> {
    pub(crate) qkv: Mat<'a>,  // attn_qkv: [conv_channels]
    pub(crate) z: Mat<'a>,    // attn_gate
    pub(crate) beta: Vec<f32>,  // ssm_beta [n_v_heads][n_embd]
    pub(crate) alpha: Vec<f32>, // ssm_alpha [n_v_heads][n_embd]
    pub(crate) conv: Vec<f32>,  // ssm_conv1d [conv_channels][d_conv]
    pub(crate) a: Vec<f32>,     // ssm_a [n_v_heads] (already -exp(A_log))
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) norm: Vec<f32>,  // ssm_norm [head_v]
    pub(crate) out: Mat<'a>,    // ssm_out
}

pub struct Model<'a> {
    pub cfg: Config,
    pub(crate) tok_embd: QuantMat<'a>,
    pub(crate) out_norm: Vec<f32>,
    pub(crate) output: QuantMat<'a>,
    pub(crate) layers: Vec<Layer<'a>>,
}

pub enum LayerState {
    Lin { conv: Vec<f32>, s: Vec<f32> },
    Attn { k: Vec<f32>, v: Vec<f32> },
}

pub struct State {
    pub pos: usize,
    pub layers: Vec<LayerState>,
    /// The trunk output of the last token before the final norm (what the MTP head takes as its hidden input).
    pub last_hidden: Vec<f32>,
}

/// The nextn (multi-token-prediction) block: `eh_proj` over `[enorm(embd(tok)); hnorm(hidden)]` (order per [`Mtp::embed_first`]), one full trunk
/// layer (`blk.<n_layer>`), `shared_head_norm`, then the trunk's own output head.
pub struct Mtp<'a> {
    pub(crate) layer: Layer<'a>,
    pub(crate) eh_proj: QuantMat<'a>,
    pub(crate) enorm: Vec<f32>,
    pub(crate) hnorm: Vec<f32>,
    pub(crate) shead_norm: Vec<f32>,
}

/// KV cache of the MTP block's attention layer.
#[derive(Clone, Default)]
pub struct MtpState {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

fn qmat<'a>(g: &'a Gguf, name: &str) -> Result<QuantMat<'a>> {
    let t = g.tensor(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
    let cols = t.dims[0];
    let rows: usize = t.dims[1..].iter().product::<usize>().max(1);
    Ok(QuantMat::new(t.ty, rows, cols, g.tensor_bytes(t)))
}

fn f32v(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
    Ok(g.tensor_f32(t))
}


fn load_layer<'a>(g: &'a Gguf, cfg: &Config, il: usize, recurrent: bool) -> Result<Layer<'a>> {
            let n = |s: &str| format!("blk.{il}.{s}");
            let (attn, lin) = if recurrent {
                (None, Some(LinW {
                    qkv: Mat(qmat(g, &n("attn_qkv.weight"))?),
                    z: Mat(qmat(g, &n("attn_gate.weight"))?),
                    beta: f32v(g, &n("ssm_beta.weight"))?,
                    alpha: f32v(g, &n("ssm_alpha.weight"))?,
                    conv: f32v(g, &n("ssm_conv1d.weight"))?,
                    a: f32v(g, &n("ssm_a"))?,
                    dt_bias: f32v(g, &n("ssm_dt.bias"))?,
                    norm: f32v(g, &n("ssm_norm.weight"))?,
                    out: Mat(qmat(g, &n("ssm_out.weight"))?),
                }))
            } else {
                (Some(AttnW {
                    q: Mat(qmat(g, &n("attn_q.weight"))?),
                    k: Mat(qmat(g, &n("attn_k.weight"))?),
                    v: Mat(qmat(g, &n("attn_v.weight"))?),
                    o: Mat(qmat(g, &n("attn_output.weight"))?),
                    q_norm: f32v(g, &n("attn_q_norm.weight"))?,
                    k_norm: f32v(g, &n("attn_k_norm.weight"))?,
                }), None)
            };
            Ok(Layer {
                attn_norm: f32v(g, &n("attn_norm.weight"))?,
                post_norm: f32v(g, &n("post_attention_norm.weight"))?,
                attn,
                lin,
                router: f32v(g, &n("ffn_gate_inp.weight"))?,
                gate_exps: qmat(g, &n("ffn_gate_exps.weight"))?,
                up_exps: qmat(g, &n("ffn_up_exps.weight"))?,
                down_exps: qmat(g, &n("ffn_down_exps.weight"))?,
                sh_gate: Mat(qmat(g, &n("ffn_gate_shexp.weight"))?),
                sh_up: Mat(qmat(g, &n("ffn_up_shexp.weight"))?),
                sh_down: Mat(qmat(g, &n("ffn_down_shexp.weight"))?),
                sh_gate_inp: f32v(g, &n("ffn_gate_inp_shexp.weight"))?,
            })
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let mut layers = Vec::new();
        for il in 0..cfg.n_layer {
            layers.push(load_layer(g, &cfg, il, cfg.is_recurrent(il))?);
        }
        Ok(Model { tok_embd: qmat(g, "token_embd.weight")?, out_norm: f32v(g, "output_norm.weight")?, output: qmat(g, "output.weight")?, layers, cfg })
    }

    /// The nextn block of a file that carries one (`qwen35moe.nextn_predict_layers` > 0): `blk.<n_layer>` is a full-attention layer.
    pub fn load_mtp(&self, g: &'a Gguf) -> Result<Mtp<'a>> {
        let il = self.cfg.n_layer;
        let n = |x: &str| format!("blk.{il}.{x}");
        Ok(Mtp {
            layer: load_layer(g, &self.cfg, il, false)?,
            eh_proj: qmat(g, &n("nextn.eh_proj.weight"))?,
            enorm: f32v(g, &n("nextn.enorm.weight"))?,
            hnorm: f32v(g, &n("nextn.hnorm.weight"))?,
            shead_norm: f32v(g, &n("nextn.shared_head_norm.weight"))?,
        })
    }

    /// One MTP step: the draft logits for the token after `tok`, given the trunk hidden `h` of the token before it, with the
    /// block's attention at position `pos`. Returns `(logits, hidden)`; `hidden` feeds the next draft step.
    pub fn mtp_step(&self, m: &Mtp<'_>, st: &mut MtpState, h: &[f32], tok: u32, pos: usize, embed_first: bool) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let mut e = vec![0f32; c.n_embd];
        self.tok_embd.dequant_row(tok as usize, &mut e);
        let en = rms_norm(&e, &m.enorm, c.eps);
        let hn = rms_norm(h, &m.hnorm, c.eps);
        let cat: Vec<f32> = if embed_first { en.iter().chain(hn.iter()).cloned().collect() } else { hn.iter().chain(en.iter()).cloned().collect() };
        let mut x = vec![0f32; c.n_embd];
        matvec(&m.eh_proj, &cat, &mut x);
        let l = &m.layer;
        let hh = rms_norm(&x, &l.attn_norm, c.eps);
        let a = self.attention(l.attn.as_ref().unwrap(), &hh, pos, &mut st.k, &mut st.v);
        for (xi, ai) in x.iter_mut().zip(&a) {
            *xi += ai;
        }
        let h2 = rms_norm(&x, &l.post_norm, c.eps);
        let f = self.moe_l(l, &h2);
        for (xi, fi) in x.iter_mut().zip(&f) {
            *xi += fi;
        }
        let hn = rms_norm(&x, &m.shead_norm, c.eps);
        let mut logits = vec![0f32; c.vocab];
        matvec(&self.output, &hn, &mut logits);
        (logits, x)
    }

    /// `qwen35moe.context_length` of the file (the model's trained limit).
    pub fn cfg_context(&self, g: &Gguf) -> usize {
        g.meta_u64("qwen35moe.context_length").map(|v| v as usize).unwrap_or(4096)
    }

    pub fn new_state(&self) -> State {
        let c = &self.cfg;
        let layers = (0..c.n_layer)
            .map(|il| {
                if c.is_recurrent(il) {
                    LayerState::Lin { conv: vec![0.0; (c.d_conv - 1) * c.conv_channels()], s: vec![0.0; c.n_v_heads * c.head_v() * c.head_k] }
                } else {
                    LayerState::Attn { k: Vec::new(), v: Vec::new() }
                }
            })
            .collect();
        State { pos: 0, layers, last_hidden: Vec::new() }
    }

    /// Run one token through the trunk; returns the logits when `want_logits`.
    pub fn forward(&self, st: &mut State, tok: u32, want_logits: bool) -> Option<Vec<f32>> {
        self.forward_ex(st, tok, want_logits, &mut |_, _, _| {})
    }

    /// As [`forward`](Self::forward), reporting intermediate tensors to `tr(name, layer, values)`: `attn_out` (the mixer's
    /// output), `attn_residual`, `ffn_out`, `post_moe` (the layer output), named like llama.cpp's graph callbacks.
    pub fn forward_ex(&self, st: &mut State, tok: u32, want_logits: bool, tr: &mut dyn FnMut(&str, usize, &[f32])) -> Option<Vec<f32>> {
        let c = &self.cfg;
        let mut x = vec![0f32; c.n_embd];
        self.tok_embd.dequant_row(tok as usize, &mut x);
        for (il, l) in self.layers.iter().enumerate() {
            tr("layer_in", il, &x);
            let h = rms_norm(&x, &l.attn_norm, c.eps);
            let a = match (&l.attn, &l.lin, &mut st.layers[il]) {
                (Some(w), _, LayerState::Attn { k, v }) => self.attention(w, &h, st.pos, k, v),
                (_, Some(w), LayerState::Lin { conv, s }) => self.delta_net(w, &h, conv, s),
                _ => unreachable!(),
            };
            tr("attn_out", il, &a);
            for (xi, ai) in x.iter_mut().zip(&a) {
                *xi += ai;
            }
            tr("attn_residual", il, &x);
            let h2 = rms_norm(&x, &l.post_norm, c.eps);
            let f = self.moe(il, &h2);
            tr("ffn_out", il, &f);
            for (xi, fi) in x.iter_mut().zip(&f) {
                *xi += fi;
            }
            tr("post_moe", il, &x);
        }
        st.pos += 1;
        st.last_hidden = x.clone();
        if !want_logits {
            return None;
        }
        let h = rms_norm(&x, &self.out_norm, c.eps);
        let mut logits = vec![0f32; c.vocab];
        matvec(&self.output, &h, &mut logits);
        Some(logits)
    }

    fn attention(&self, w: &AttnW<'_>, h: &[f32], pos: usize, kc: &mut Vec<f32>, vc: &mut Vec<f32>) -> Vec<f32> {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_head, c.n_head_kv, c.head_dim);
        let mut qf = vec![0f32; nh * hd * 2];
        matvec(&w.q.0, h, &mut qf);
        let mut k = vec![0f32; nkv * hd];
        let mut v = vec![0f32; nkv * hd];
        matvec(&w.k.0, h, &mut k);
        matvec(&w.v.0, h, &mut v);
        // per head: first hd values are the query, next hd the output gate
        let mut q = vec![0f32; nh * hd];
        let mut gate = vec![0f32; nh * hd];
        for hh in 0..nh {
            q[hh * hd..(hh + 1) * hd].copy_from_slice(&qf[hh * 2 * hd..hh * 2 * hd + hd]);
            gate[hh * hd..(hh + 1) * hd].copy_from_slice(&qf[hh * 2 * hd + hd..(hh + 1) * 2 * hd]);
        }
        for hh in 0..nh {
            let s = &mut q[hh * hd..(hh + 1) * hd];
            let n = rms_norm(s, &w.q_norm, c.eps);
            s.copy_from_slice(&n);
            rope_neox(s, pos, c);
        }
        for hh in 0..nkv {
            let s = &mut k[hh * hd..(hh + 1) * hd];
            let n = rms_norm(s, &w.k_norm, c.eps);
            s.copy_from_slice(&n);
            rope_neox(s, pos, c);
        }
        if kv16_on() {
            for t in k.iter_mut().chain(v.iter_mut()) {
                *t = half::f16::from_f32(*t).to_f32();
            }
        }
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let n = kc.len() / (nkv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let mut o = vec![0f32; nh * hd];
        o.par_chunks_mut(hd).enumerate().for_each(|(hh, oh)| {
            let kvh = hh / group;
            let qh = &q[hh * hd..(hh + 1) * hd];
            let mut sc = vec![0f32; n];
            for (t, s) in sc.iter_mut().enumerate() {
                let kk = &kc[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                *s = dot(qh, kk) * scale;
            }
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in sc.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for (t, s) in sc.iter().enumerate() {
                let vv = &vc[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                let p = s / sum;
                for (od, vd) in oh.iter_mut().zip(vv) {
                    *od += p * vd;
                }
            }
        });
        for (oi, gi) in o.iter_mut().zip(&gate) {
            *oi *= sigmoid(*gi);
        }
        let mut out = vec![0f32; c.n_embd];
        matvec(&w.o.0, &o, &mut out);
        out
    }

    fn delta_net(&self, w: &LinW<'_>, h: &[f32], conv_state: &mut [f32], s: &mut [f32]) -> Vec<f32> {
        let c = &self.cfg;
        let (hk, hv, nk, nv, ch) = (c.head_k, c.head_v(), c.n_k_heads, c.n_v_heads, c.conv_channels());
        let mut qkv = vec![0f32; ch];
        matvec(&w.qkv.0, h, &mut qkv);
        let mut z = vec![0f32; c.d_inner];
        matvec(&w.z.0, h, &mut z);
        let beta: Vec<f32> = (0..nv).map(|i| sigmoid(dot(&w.beta[i * c.n_embd..(i + 1) * c.n_embd], h))).collect();
        let g: Vec<f32> = (0..nv)
            .map(|i| {
                let alpha = dot(&w.alpha[i * c.n_embd..(i + 1) * c.n_embd], h);
                softplus(alpha + w.dt_bias[i]) * w.a[i]
            })
            .collect();
        // causal conv over [state(d_conv-1 columns) | new column], then SiLU; state keeps the last d_conv-1 columns
        let kd = c.d_conv;
        let mut conv_out = vec![0f32; ch];
        for cc in 0..ch {
            let mut sum = 0f32;
            for t in 0..kd - 1 {
                sum += conv_state[t * ch + cc] * w.conv[cc * kd + t];
            }
            sum += qkv[cc] * w.conv[cc * kd + kd - 1];
            conv_out[cc] = silu(sum);
        }
        for t in 0..kd - 2 {
            let (a, b) = conv_state.split_at_mut((t + 1) * ch);
            a[t * ch..(t + 1) * ch].copy_from_slice(&b[..ch]);
        }
        conv_state[(kd - 2) * ch..(kd - 1) * ch].copy_from_slice(&qkv);
        let (qc, rest) = conv_out.split_at_mut(nk * hk);
        let (kc, vc) = rest.split_at_mut(nk * hk);
        for hh in 0..nk {
            l2_norm(&mut qc[hh * hk..(hh + 1) * hk], c.eps);
            l2_norm(&mut kc[hh * hk..(hh + 1) * hk], c.eps);
        }
        let qscale = 1.0 / (hk as f32).sqrt();
        let mut out = vec![0f32; nv * hv];
        // v head h reads k/q head h % n_k_heads (ggml repeat tiling)
        s.par_chunks_mut(hv * hk).zip(out.par_chunks_mut(hv)).enumerate().for_each(|(hh, (st, oh))| {
            let kh = hh % nk;
            let k = &kc[kh * hk..(kh + 1) * hk];
            let q = &qc[kh * hk..(kh + 1) * hk];
            let v = &vc[hh * hv..(hh + 1) * hv];
            let decay = g[hh].exp();
            for i in 0..hv {
                let row = &mut st[i * hk..(i + 1) * hk];
                for r in row.iter_mut() {
                    *r *= decay;
                }
                let sk = dot(row, k);
                let d = (v[i] - sk) * beta[hh];
                for (r, kj) in row.iter_mut().zip(k) {
                    *r += kj * d;
                }
                oh[i] = dot(row, q) * qscale;
            }
        });
        // gated RMS norm per v head, then output projection
        let mut gated = vec![0f32; c.d_inner];
        for hh in 0..nv {
            let n = rms_norm(&out[hh * hv..(hh + 1) * hv], &w.norm, c.eps);
            for i in 0..hv {
                gated[hh * hv + i] = n[i] * silu(z[hh * hv + i]);
            }
        }
        let mut y = vec![0f32; c.n_embd];
        matvec(&w.out.0, &gated, &mut y);
        y
    }

    /// Router (softmax over all experts, top-k, weights renormalised), routed experts, gated shared expert.
    pub fn route(&self, l: usize, h: &[f32]) -> (Vec<usize>, Vec<f32>) {
        self.route_l(&self.layers[l], h)
    }

    fn route_l(&self, lw: &Layer<'_>, h: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let c = &self.cfg;
        let logits: Vec<f32> = (0..c.n_expert).map(|e| dot(&lw.router[e * c.n_embd..(e + 1) * c.n_embd], h)).collect();
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let s: f32 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= s);
        let mut idx: Vec<usize> = (0..c.n_expert).collect();
        idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        idx.truncate(c.n_used);
        let sum: f32 = idx.iter().map(|&i| p[i]).sum();
        let w = idx.iter().map(|&i| p[i] / sum).collect();
        (idx, w)
    }

    fn moe(&self, l: usize, h: &[f32]) -> Vec<f32> {
        self.moe_l(&self.layers[l], h)
    }

    fn moe_l(&self, lw: &Layer<'_>, h: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let (ids, ws) = self.route_l(lw, h);
        let mut out = vec![0f32; c.n_embd];
        for (&e, &wt) in ids.iter().zip(&ws) {
            let rb_gu = lw.gate_exps.ty.row_bytes(c.n_embd).unwrap();
            let gate = QuantMat::new(lw.gate_exps.ty, c.ff_exp, c.n_embd, &lw.gate_exps.bytes[e * c.ff_exp * rb_gu..(e + 1) * c.ff_exp * rb_gu]);
            let rb_up = lw.up_exps.ty.row_bytes(c.n_embd).unwrap();
            let up = QuantMat::new(lw.up_exps.ty, c.ff_exp, c.n_embd, &lw.up_exps.bytes[e * c.ff_exp * rb_up..(e + 1) * c.ff_exp * rb_up]);
            let rb_dn = lw.down_exps.ty.row_bytes(c.ff_exp).unwrap();
            let down = QuantMat::new(lw.down_exps.ty, c.n_embd, c.ff_exp, &lw.down_exps.bytes[e * c.n_embd * rb_dn..(e + 1) * c.n_embd * rb_dn]);
            let y = swiglu(&gate, &up, &down, h);
            for (o, yi) in out.iter_mut().zip(&y) {
                *o += wt * yi;
            }
        }
        let sh = swiglu(&lw.sh_gate.0, &lw.sh_up.0, &lw.sh_down.0, h);
        let gate = sigmoid(dot(&lw.sh_gate_inp, h));
        for (o, s) in out.iter_mut().zip(&sh) {
            *o += gate * s;
        }
        out
    }
}

fn swiglu(gate: &QuantMat<'_>, up: &QuantMat<'_>, down: &QuantMat<'_>, h: &[f32]) -> Vec<f32> {
    let mut g = vec![0f32; gate.rows];
    let mut u = vec![0f32; up.rows];
    matvec(gate, h, &mut g);
    matvec(up, h, &mut u);
    for (gi, ui) in g.iter_mut().zip(&u) {
        *gi = silu(*gi) * ui;
    }
    let mut y = vec![0f32; down.rows];
    matvec(down, &g, &mut y);
    y
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = a.len().min(b.len());
    let (ca, cb) = (a[..n].chunks_exact(8), b[..n].chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * w).collect()
}

/// ggml `build_gdn_l2_norm`: `x / sqrt(sum(x^2) + eps)`.
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let s: f32 = x.iter().map(|v| v * v).sum();
    let r = 1.0 / (s + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= r);
}

/// NeoX-style rotary on the first `n_rot` values of a head: pairs `(i, i + n_rot/2)`, `theta_i = pos * base^(-2i/n_rot)`
/// accumulated in f32 as ggml does. Text positions make all four M-RoPE sections (11,11,10,0) use the same angle.
pub fn rope_neox(x: &mut [f32], pos: usize, c: &Config) {
    let half = c.n_rot / 2;
    let theta_scale = c.rope_base.powf(-2.0 / c.n_rot as f32);
    let mut theta = pos as f32;
    for i in 0..half {
        let (s, co) = theta.sin_cos();
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * co - x1 * s;
        x[i + half] = x0 * s + x1 * co;
        theta *= theta_scale;
    }
}

/// Emulate the activation quantisation llama.cpp's CPU kernels apply (`LB_QWEN_ACTQ=1`): K-quant weights are dotted with the
/// activation row rounded to Q8_K (blocks of 256, `d = max/-127`, `nearest_int`), Q8_0 weights with Q8_0 (blocks of 32, f16 `d =
/// amax/127`, `roundf`). Used only to separate "the reference rounds its activations" from "this decoder differs".
pub fn act_quant_roundtrip(ty: GgmlType, x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    match ty {
        GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q4K => {
            for blk in y.chunks_exact_mut(256) {
                let (mut amax, mut max) = (0f32, 0f32);
                for &v in blk.iter() {
                    if v.abs() > amax {
                        amax = v.abs();
                        max = v;
                    }
                }
                if amax == 0.0 {
                    continue;
                }
                let iscale = -127.0 / max;
                let d = 1.0 / iscale;
                for v in blk.iter_mut() {
                    let q = ((iscale * *v).round_ties_even() as i32).min(127);
                    *v = q as f32 * d;
                }
            }
        }
        GgmlType::Q8_0 => {
            for blk in y.chunks_exact_mut(32) {
                let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let d16 = half::f16::from_f32(d).to_f32();
                for v in blk.iter_mut() {
                    *v = (*v * id).round() * d16;
                }
            }
        }
        _ => {}
    }
    y
}

fn act_quant_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LB_QWEN_ACTQ").map(|v| v == "1").unwrap_or(false))
}

fn kv16_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LB_QWEN_KV16").map(|v| v == "1").unwrap_or(false))
}

/// `out[r] = W[r, :] . x` with `W` quantised, one dequantised row at a time.
pub fn matvec(w: &QuantMat<'_>, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.cols);
    debug_assert_eq!(out.len(), w.rows);
    let cols = w.cols;
    let xq;
    let x = if act_quant_on() { xq = act_quant_roundtrip(w.ty, x); &xq[..] } else { x };
    if w.ty == GgmlType::F32 {
        out.par_iter_mut().enumerate().for_each(|(r, o)| {
            let row = w.row(r);
            let v: Vec<f32> = row.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            *o = dot(&v, x);
        });
        return;
    }
    out.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; cols], |buf, (ci, chunk)| {
        for (j, o) in chunk.iter_mut().enumerate() {
            w.ty.dequant_row(w.row(ci * 16 + j), buf);
            *o = dot(buf, x);
        }
    });
}
