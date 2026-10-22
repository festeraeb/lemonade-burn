//! BitNet b1.58 2B4T decoder (HF `BitNetForCausalLM` semantics) as a Burn graph.
//!
//! Per layer (pre-norm residual):
//! ```text
//! h = x + o_proj(attn_sub_norm(attn(rope(q_proj(n1)), rope(k_proj(n1)), v_proj(n1))))  n1 = rms(x)
//! y = h + down_proj(ffn_sub_norm(relu(gate_proj(n2))^2 * up_proj(n2)))                n2 = rms(h)
//! ```
//! Every `*_proj` is a BitLinear (int8 activations x ternary weights) executed by the
//! `TernaryBackend` extension; everything else is ordinary Burn tensor ops on the Flex
//! backend. Note: the MLP is gated ReLU^2, not SwiGLU.

use crate::ops::{bf16_linear, bf16_linear_numa, ternary_linear, ternary_linear2, ternary_linear3, ternary_linear_numa_n};
use lb_ternary::numa::NodePools;
use anyhow::{bail, Context, Result};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorCreationOptions, TensorData};
use lb_convert::{qweight_name, wscale_name, FILE_NAME};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub bos_token_id: u32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

pub struct TernaryLinear {
    /// Packed rows; with a two-socket split, only the first rows (placed on node 0).
    pub packed: Tensor<2, Int>,
    /// Two-socket split: the remaining rows, placed on node 1 (see `lb_ternary::numa`).
    pub packed_node1: Option<Tensor<2, Int>>,
    pub scale: f32,
    pub rows: usize,
    pub cols: usize,
}

impl TernaryLinear {
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        match &self.packed_node1 {
            Some(p1) => {
                let [y] = ternary_linear_numa_n(x, [(&self.packed, p1)], [self.scale], self.cols);
                y
            }
            None => ternary_linear(x, &self.packed, self.scale, self.cols),
        }
    }
}

/// q/k/v (or any three linears sharing `x`) as one fused op: one activation quantization
/// and one parallel region. Bit-identical to calling `forward` on each.
fn forward3(x: Tensor<2>, a: &TernaryLinear, b: &TernaryLinear, c: &TernaryLinear) -> [Tensor<2>; 3] {
    assert!(a.cols == b.cols && b.cols == c.cols);
    if let (Some(a1), Some(b1), Some(c1)) = (&a.packed_node1, &b.packed_node1, &c.packed_node1) {
        return ternary_linear_numa_n(x, [(&a.packed, a1), (&b.packed, b1), (&c.packed, c1)], [a.scale, b.scale, c.scale], a.cols);
    }
    ternary_linear3(x, [&a.packed, &b.packed, &c.packed], [a.scale, b.scale, c.scale], a.cols)
}

/// gate/up as one fused op, like [`forward3`].
fn forward2(x: Tensor<2>, a: &TernaryLinear, b: &TernaryLinear) -> [Tensor<2>; 2] {
    assert_eq!(a.cols, b.cols);
    if let (Some(a1), Some(b1)) = (&a.packed_node1, &b.packed_node1) {
        return ternary_linear_numa_n(x, [(&a.packed, a1), (&b.packed, b1)], [a.scale, b.scale], a.cols);
    }
    ternary_linear2(x, [&a.packed, &b.packed], [a.scale, b.scale], a.cols)
}

/// Copy `bytes` (a `[rows, row_bytes]` matrix) into Burn tensor(s). With `LB_NUMA` set the
/// rows are split and each part is copied by a thread of its node, so its pages live
/// there (first touch); returns `(node-0 or whole, node-1 part)`.
fn place_rows(bytes: &[u8], rows: usize, row_bytes: usize, elem: usize, dtype: DType) -> (TensorData, Option<TensorData>) {
    let make = |b: Vec<u8>, r: usize| TensorData::from_bytes_vec(b, [r, row_bytes / elem], dtype);
    match NodePools::global() {
        Some(np) => {
            let s = np.split_rows(rows).clamp(1, rows - 1);
            let d0 = np.place(0, &bytes[..s * row_bytes]);
            let d1 = np.place(1, &bytes[s * row_bytes..]);
            (make(d0, s), Some(make(d1, rows - s)))
        }
        None => (make(bytes.to_vec(), rows), None),
    }
}

/// Run `f` (a tensor construction) on the thread pool of `node` when `LB_NUMA` is set, so
/// that if Burn copies the buffer the copy is also first-touched on that node.
fn on_node<R: Send>(node: usize, f: impl FnOnce() -> R + Send) -> R {
    match NodePools::global() {
        Some(np) => np.install(node, f),
        None => f(),
    }
}

pub struct Layer {
    pub input_norm: Tensor<1>,
    pub post_attn_norm: Tensor<1>,
    pub attn_sub_norm: Tensor<1>,
    pub ffn_sub_norm: Tensor<1>,
    pub q: TernaryLinear,
    pub k: TernaryLinear,
    pub v: TernaryLinear,
    pub o: TernaryLinear,
    pub gate: TernaryLinear,
    pub up: TernaryLinear,
    pub down: TernaryLinear,
}

pub struct BitNet {
    pub cfg: Config,
    pub device: Device,
    pub layers: Vec<Layer>,
    pub final_norm: Tensor<1>,
    /// Tied embedding / LM head, bf16 `[vocab, hidden]`.
    pub lm_head: Tensor<2>,
    /// Two-socket split of the LM head: rows placed on node 1 (`lm_head` holds the rest).
    pub lm_head_node1: Option<Tensor<2>>,
    /// Host copy of the embedding table (bf16 bits) for token lookup.
    embed: Vec<u16>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    rope_len: usize,
}

/// Per-layer K/V cache, `[kv_heads, cap, head_dim]`, grown by doubling.
pub struct KvCache {
    pub k: Vec<Tensor<3>>,
    pub v: Vec<Tensor<3>>,
    pub len: usize,
    pub cap: usize,
}

impl BitNet {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if cfg.hidden_act != "relu2" || !cfg.tie_word_embeddings {
            bail!("unsupported config: act={} tied={}", cfg.hidden_act, cfg.tie_word_embeddings);
        }
        let path = dir.join(FILE_NAME);
        let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;
        match st_meta(&mmap)?.as_deref() {
            Some(lb_convert::FORMAT) => {}
            other => bail!("{} is not a lemonade-burn ternary file (format={other:?})", path.display()),
        }

        let f32_vec = |name: &str| -> Result<Tensor<1>> {
            let t = st.tensor(name).with_context(|| name.to_string())?;
            if t.dtype() != Dtype::F32 {
                bail!("{name}: expected F32");
            }
            let v: Vec<f32> = t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let n = v.len();
            Ok(Tensor::from_data(TensorData::new(v, [n]), device))
        };
        let linear = |name: &str| -> Result<TernaryLinear> {
            let q = st.tensor(&qweight_name(name)).with_context(|| name.to_string())?;
            let s = st.tensor(&wscale_name(name))?;
            let scale = f32::from_le_bytes(s.data()[..4].try_into()?);
            let (rows, rb) = (q.shape()[0], q.shape()[1]);
            let cols = rb * 4; // all BitNet 2B4T dims are multiples of 128 (no padding)
            let opts = || TensorCreationOptions::new(device.clone()).with_dtype(DType::U8);
            let (d0, d1) = place_rows(q.data(), rows, rb, 1, DType::U8);
            let packed = on_node(0, || Tensor::<2, Int>::from_data(d0, opts()));
            let packed_node1 = d1.map(|d| on_node(1, || Tensor::<2, Int>::from_data(d, opts())));
            Ok(TernaryLinear { packed, packed_node1, scale, rows, cols })
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: f32_vec(&format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: f32_vec(&format!("{p}.post_attention_layernorm.weight"))?,
                attn_sub_norm: f32_vec(&format!("{p}.self_attn.attn_sub_norm.weight"))?,
                ffn_sub_norm: f32_vec(&format!("{p}.mlp.ffn_sub_norm.weight"))?,
                q: linear(&format!("{p}.self_attn.q_proj"))?,
                k: linear(&format!("{p}.self_attn.k_proj"))?,
                v: linear(&format!("{p}.self_attn.v_proj"))?,
                o: linear(&format!("{p}.self_attn.o_proj"))?,
                gate: linear(&format!("{p}.mlp.gate_proj"))?,
                up: linear(&format!("{p}.mlp.up_proj"))?,
                down: linear(&format!("{p}.mlp.down_proj"))?,
            });
        }
        let final_norm = f32_vec("model.norm.weight")?;
        let emb = st.tensor("model.embed_tokens.weight")?;
        if emb.dtype() != Dtype::BF16 || emb.shape() != [cfg.vocab_size, cfg.hidden_size] {
            bail!("embed_tokens: unexpected dtype/shape");
        }
        let embed: Vec<u16> = emb.data().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let head_opts = || TensorCreationOptions::new(device.clone()).with_dtype(DType::BF16);
        let (h0, h1) = place_rows(emb.data(), cfg.vocab_size, cfg.hidden_size * 2, 2, DType::BF16);
        let lm_head = on_node(0, || Tensor::<2>::from_data(h0, head_opts()));
        let lm_head_node1 = h1.map(|d| on_node(1, || Tensor::<2>::from_data(d, head_opts())));

        // RoPE tables as transformers computes them: inv_freq in f32, freqs = pos * inv_freq (f32)
        let hd = cfg.head_dim();
        let rope_len = cfg.max_position_embeddings;
        let inv: Vec<f32> = (0..hd / 2)
            .map(|i| (1.0 / cfg.rope_theta.powf((2 * i) as f64 / hd as f64)) as f32)
            .collect();
        let mut cos = vec![0f32; rope_len * hd];
        let mut sin = vec![0f32; rope_len * hd];
        for p in 0..rope_len {
            for i in 0..hd / 2 {
                let f = p as f32 * inv[i];
                let (s, c) = (f as f64).sin_cos();
                for j in [i, i + hd / 2] {
                    cos[p * hd + j] = c as f32;
                    sin[p * hd + j] = s as f32;
                }
            }
        }
        Ok(Self { cfg, device: device.clone(), layers, final_norm, lm_head, lm_head_node1, embed, cos, sin, rope_len })
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache { k: Vec::new(), v: Vec::new(), len: 0, cap: 0 }
    }

    fn ensure_cap(&self, cache: &mut KvCache, need: usize) {
        if need <= cache.cap {
            return;
        }
        let (kvh, hd) = (self.cfg.num_key_value_heads, self.cfg.head_dim());
        let new_cap = need.next_power_of_two().max(256).min(self.rope_len.max(need));
        let grow = |old: Option<Tensor<3>>| -> Tensor<3> {
            let z = Tensor::<3>::zeros([kvh, new_cap, hd], &self.device);
            match old {
                Some(o) if cache.len > 0 => z.slice_assign([0..kvh, 0..cache.len, 0..hd], o.slice([0..kvh, 0..cache.len, 0..hd])),
                _ => z,
            }
        };
        let n = self.layers.len();
        let mut ks = Vec::with_capacity(n);
        let mut vs = Vec::with_capacity(n);
        let old_k = std::mem::take(&mut cache.k);
        let old_v = std::mem::take(&mut cache.v);
        let mut ok = old_k.into_iter();
        let mut ov = old_v.into_iter();
        for _ in 0..n {
            ks.push(grow(ok.next()));
            vs.push(grow(ov.next()));
        }
        cache.k = ks;
        cache.v = vs;
        cache.cap = new_cap;
    }

    fn rms_norm<const D: usize>(&self, x: Tensor<D>, w: &Tensor<1>) -> Tensor<D> {
        let eps = self.cfg.rms_norm_eps as f32;
        let var = x.clone().square().mean_dim(D - 1);
        let w = w.clone().unsqueeze::<D>();
        x * (var + eps).sqrt().recip() * w
    }

    fn embed(&self, tokens: &[u32]) -> Tensor<2> {
        let h = self.cfg.hidden_size;
        let mut v = Vec::with_capacity(tokens.len() * h);
        for &t in tokens {
            let row = &self.embed[t as usize * h..(t as usize + 1) * h];
            v.extend(row.iter().map(|&b| lb_ternary::bf16::bf16_to_f32(b)));
        }
        Tensor::from_data(TensorData::new(v, [tokens.len(), h]), &self.device)
    }

    fn rope_tables(&self, pos0: usize, m: usize) -> (Tensor<3>, Tensor<3>) {
        let hd = self.cfg.head_dim();
        let c = self.cos[pos0 * hd..(pos0 + m) * hd].to_vec();
        let s = self.sin[pos0 * hd..(pos0 + m) * hd].to_vec();
        (
            Tensor::from_data(TensorData::new(c, [1, m, hd]), &self.device),
            Tensor::from_data(TensorData::new(s, [1, m, hd]), &self.device),
        )
    }

    /// `x: [heads, m, hd]`
    fn apply_rope(x: Tensor<3>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<3> {
        let [h, m, hd] = x.dims();
        let x1 = x.clone().slice([0..h, 0..m, 0..hd / 2]);
        let x2 = x.clone().slice([0..h, 0..m, hd / 2..hd]);
        let rot = Tensor::cat(vec![x2.neg(), x1], 2);
        x * cos.clone() + rot * sin.clone()
    }

    /// Run `tokens` at positions `cache.len ..`; returns logits `[m, vocab]` for every
    /// position when `all_logits`, else `[1, vocab]` for the last one.
    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache, all_logits: bool) -> Tensor<2> {
        let m = tokens.len();
        let pos0 = cache.len;
        assert!(pos0 + m <= self.rope_len, "context length {} exceeded", self.rope_len);
        self.ensure_cap(cache, pos0 + m);
        let (nh, kvh, hd) = (self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim());
        let group = nh / kvh;
        let total = pos0 + m;
        let (cos, sin) = self.rope_tables(pos0, m);
        let scaling = (hd as f32).powf(-0.5);
        // causal mask for the grouped query rows: row (g, i) may see keys 0..=pos0+i
        let mask = if m > 1 {
            let mut mv = vec![false; group * m * total];
            for g in 0..group {
                for i in 0..m {
                    for j in (pos0 + i + 1)..total {
                        mv[(g * m + i) * total + j] = true;
                    }
                }
            }
            Some(Tensor::<3, Bool>::from_data(TensorData::new(mv, [1, group * m, total]), &self.device))
        } else {
            None
        };

        let mut x = self.embed(tokens);
        for (li, l) in self.layers.iter().enumerate() {
            let n1 = self.rms_norm(x.clone(), &l.input_norm);
            let [q, k, v] = forward3(n1, &l.q, &l.k, &l.v);
            let q = q.reshape([m, nh, hd]).swap_dims(0, 1);
            let k = k.reshape([m, kvh, hd]).swap_dims(0, 1);
            let v = v.reshape([m, kvh, hd]).swap_dims(0, 1);
            let q = Self::apply_rope(q, &cos, &sin);
            let k = Self::apply_rope(k, &cos, &sin);

            let kc = std::mem::replace(&mut cache.k[li], Tensor::empty([0, 0, 0], &self.device));
            let vc = std::mem::replace(&mut cache.v[li], Tensor::empty([0, 0, 0], &self.device));
            let kc = kc.slice_assign([0..kvh, pos0..total, 0..hd], k);
            let vc = vc.slice_assign([0..kvh, pos0..total, 0..hd], v);
            let keys = kc.clone().slice([0..kvh, 0..total, 0..hd]);
            let vals = vc.clone().slice([0..kvh, 0..total, 0..hd]);
            cache.k[li] = kc;
            cache.v[li] = vc;

            // GQA: q heads [g*kvh .. ] -> group j holds heads j*group .. j*group+group-1
            let qg = q.reshape([kvh, group * m, hd]);
            let mut scores = qg.matmul(keys.swap_dims(1, 2)) * scaling; // [kvh, group*m, total]
            if let Some(mask) = &mask {
                scores = scores.mask_fill(mask.clone(), f32::NEG_INFINITY);
            }
            let probs = softmax(scores, 2);
            let attn = probs.matmul(vals); // [kvh, group*m, hd]
            let attn = attn.reshape([nh, m, hd]).swap_dims(0, 1).reshape([m, nh * hd]);
            let attn = self.rms_norm(attn, &l.attn_sub_norm);
            x = x + l.o.forward(attn);

            let n2 = self.rms_norm(x.clone(), &l.post_attn_norm);
            let [g, u] = forward2(n2, &l.gate, &l.up);
            let g = relu(g).square();
            let a = self.rms_norm(g * u, &l.ffn_sub_norm);
            x = x + l.down.forward(a);
        }
        cache.len = total;
        let x = if all_logits { x } else { x.slice([m - 1..m, 0..self.cfg.hidden_size]) };
        let x = self.rms_norm(x, &self.final_norm);
        match &self.lm_head_node1 {
            Some(h1) => bf16_linear_numa(x, &self.lm_head, h1),
            None => bf16_linear(x, &self.lm_head),
        }
    }
}

fn st_meta(buf: &[u8]) -> Result<Option<String>> {
    let (_, meta) = SafeTensors::read_metadata(buf)?;
    Ok(meta.metadata().as_ref().and_then(|m| m.get("format").cloned()))
}
