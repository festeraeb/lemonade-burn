//! LFM2 / LFM2-MoE decoder as a Burn graph (Flex CPU backend) over GGUF weights.
//!
//! Per layer (pre-norm residual, `h = rms(x)`):
//! ```text
//! conv layer:  x = x + out_proj( C * conv3( B * X ) )        [B, C, X] = split(in_proj(h), 3)
//!              conv3 is a causal depthwise convolution with l_cache taps per channel;
//!              the cache keeps the last l_cache-1 rows of B*X.
//! attn layer:  x = x + o_proj( attention( rope(rms_h(q)), rope(rms_h(k)), v ) )
//!              q/k are RMS-normed per head before RoPE (NEOX half-rotation, theta=freq_base).
//! then:        x = x + ffn( rms(x) )     dense SwiGLU for the leading layers, MoE after.
//! logits = rms(x) @ token_embd^T   (tied head)
//! ```
//! Quantized matmuls run through `lb_gguf::matvec_f32` (the CPU reference kernels);
//! everything else is ordinary Burn tensor ops.

use crate::config::Config;
use crate::moe::{moe_ffn, ExpertStore, Placement, PlacementStats, Scraper};
use crate::weights::{Ffn, MatRef, Operator, Weights};
use anyhow::Result;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Bool, Device, Tensor, TensorData};
use lb_gguf::{matvec_f32, Gguf};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub struct Lfm2 {
    pub w: Weights,
    pub device: Device,
    pub placement: Placement,
    /// Experts made resident on the GPU from `placement` (see [`Lfm2::load_residents`]).
    pub store: ExpertStore,
    /// Shared so a host (the server) can read the counters while the model runs on its
    /// own thread: this is the only view of where a token's time actually went.
    pub stats: Arc<PlacementStats>,
    pub scraper: Scraper,
    /// Wall time inside quantized matmuls (ns), for profiling.
    pub matmul_ns: std::sync::atomic::AtomicU64,
}

/// Per-layer state: K/V `[kv_heads, cap, head_dim]` for attention layers, the last
/// `l_cache - 1` rows of `B*X` `[l_cache-1, n_embd]` for conv layers.
pub struct Cache {
    pub k: Vec<Option<Tensor<3>>>,
    pub v: Vec<Option<Tensor<3>>>,
    pub conv: Vec<Option<Tensor<2>>>,
    pub len: usize,
    pub cap: usize,
}

impl Lfm2 {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let g = Arc::new(Gguf::open(path)?);
        let w = Weights::load(g)?;
        let cfg = &w.cfg;
        let placement = Placement::all_cpu(cfg.n_layer, cfg.n_expert);
        let model_id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("lfm2").to_string();
        Ok(Self {
            w,
            device: device.clone(),
            placement,
            store: ExpertStore::default(),
            stats: Arc::new(PlacementStats::default()),
            scraper: Scraper::from_env(&model_id),
            matmul_ns: Default::default(),
        })
    }

    /// Upload the `Home::Gpu` experts of the current placement to the device
    /// (`LB_WGPU_DEVICE`); needs the `gpu` feature. Returns how many were uploaded.
    pub fn load_residents(&mut self) -> Result<usize, String> {
        let Self { w, placement, store, .. } = self;
        store.load(w, placement)
    }

    pub fn cfg(&self) -> &Config {
        &self.w.cfg
    }

    pub fn new_cache(&self) -> Cache {
        let n = self.cfg().n_layer;
        Cache { k: vec![None; n], v: vec![None; n], conv: vec![None; n], len: 0, cap: 0 }
    }

    fn to_vec(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn from_vec(&self, v: Vec<f32>, rows: usize, cols: usize) -> Tensor<2> {
        Tensor::from_data(TensorData::new(v, [rows, cols]), &self.device)
    }

    /// `x [m, cols] @ W^T -> [m, rows]` through the quantized kernel.
    pub fn linear(&self, x: Tensor<2>, w: &MatRef) -> Tensor<2> {
        let [m, cols] = x.dims();
        assert_eq!(cols, w.cols, "linear: input width");
        let xv = Self::to_vec(x);
        let mut out = vec![0f32; m * w.rows];
        let t = Instant::now();
        matvec_f32(self.w.mat(w), &xv, m, &mut out);
        self.matmul_ns.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        self.from_vec(out, m, w.rows)
    }

    fn rms_norm<const D: usize>(&self, x: Tensor<D>, w: &[f32]) -> Tensor<D> {
        let eps = self.cfg().rms_eps;
        let var = x.clone().square().mean_dim(D - 1);
        let wt = Tensor::<1>::from_data(TensorData::new(w.to_vec(), [w.len()]), &self.device).unsqueeze::<D>();
        x * (var + eps).sqrt().recip() * wt
    }

    fn embed(&self, tokens: &[u32]) -> Tensor<2> {
        let e = self.cfg().n_embd;
        let mut v = vec![0f32; tokens.len() * e];
        for (i, &t) in tokens.iter().enumerate() {
            self.w.embed_row(t, &mut v[i * e..(i + 1) * e]);
        }
        self.from_vec(v, tokens.len(), e)
    }

    /// NEOX RoPE tables for positions `pos0 .. pos0 + m`: `[1, m, hd]`, value `j` and
    /// `j + hd/2` share frequency `theta^(-2j/hd)`.
    fn rope_tables(&self, pos0: usize, m: usize) -> (Tensor<3>, Tensor<3>) {
        let hd = self.cfg().head_dim();
        let theta = self.cfg().rope_theta;
        let mut cos = vec![0f32; m * hd];
        let mut sin = vec![0f32; m * hd];
        for i in 0..m {
            let p = (pos0 + i) as f64;
            for j in 0..hd / 2 {
                let inv = 1.0 / theta.powf(2.0 * j as f64 / hd as f64);
                let (s, c) = (p * inv).sin_cos();
                for jj in [j, j + hd / 2] {
                    cos[i * hd + jj] = c as f32;
                    sin[i * hd + jj] = s as f32;
                }
            }
        }
        (
            Tensor::from_data(TensorData::new(cos, [1, m, hd]), &self.device),
            Tensor::from_data(TensorData::new(sin, [1, m, hd]), &self.device),
        )
    }

    /// `x: [heads, m, hd]`, half rotation (NEOX).
    fn apply_rope(x: Tensor<3>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<3> {
        let [h, m, hd] = x.dims();
        let x1 = x.clone().slice([0..h, 0..m, 0..hd / 2]);
        let x2 = x.clone().slice([0..h, 0..m, hd / 2..hd]);
        let rot = Tensor::cat(vec![x2.neg(), x1], 2);
        x * cos.clone() + rot * sin.clone()
    }

    fn conv_block(&self, h: Tensor<2>, in_proj: &MatRef, conv: &[f32], out_proj: &MatRef, state: &mut Option<Tensor<2>>) -> Tensor<2> {
        let cfg = self.cfg();
        let (e, l) = (cfg.n_embd, cfg.l_cache);
        let [m, _] = h.dims();
        let bcx = self.linear(h, in_proj); // [m, 3e]
        let b = bcx.clone().slice([0..m, 0..e]);
        let c = bcx.clone().slice([0..m, e..2 * e]);
        let xx = bcx.slice([0..m, 2 * e..3 * e]);
        let bx = b * xx; // [m, e]
        let prev = state.take().unwrap_or_else(|| Tensor::zeros([l - 1, e], &self.device));
        let z = Tensor::cat(vec![prev, bx], 0); // [m + l - 1, e]
        let total = m + l - 1;
        *state = Some(z.clone().slice([total - (l - 1)..total, 0..e]));
        // out[t] = sum_k conv[k] * z[t + k]  (tap k multiplies the row l-1-k steps back)
        let mut acc: Option<Tensor<2>> = None;
        for k in 0..l {
            let wk = Tensor::<1>::from_data(TensorData::new(conv[k * e..(k + 1) * e].to_vec(), [e]), &self.device).unsqueeze::<2>();
            let term = z.clone().slice([k..k + m, 0..e]) * wk;
            acc = Some(match acc {
                Some(a) => a + term,
                None => term,
            });
        }
        let y = c * acc.unwrap();
        self.linear(y, out_proj)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_block(
        &self,
        h: Tensor<2>,
        li: usize,
        q_w: &MatRef,
        k_w: &MatRef,
        v_w: &MatRef,
        o_w: &MatRef,
        q_norm: &[f32],
        k_norm: &[f32],
        cache: &mut Cache,
        pos0: usize,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
        mask: &Option<Tensor<3, Bool>>,
    ) -> Tensor<2> {
        let cfg = self.cfg();
        let (nh, kvh, hd) = (cfg.n_head, cfg.kv_heads[li], cfg.head_dim());
        let group = nh / kvh;
        let [m, _] = h.dims();
        let total = pos0 + m;
        let q = self.linear(h.clone(), q_w).reshape([m, nh, hd]);
        let k = self.linear(h.clone(), k_w).reshape([m, kvh, hd]);
        let v = self.linear(h, v_w).reshape([m, kvh, hd]);
        let q = self.rms_norm(q, q_norm).swap_dims(0, 1); // [nh, m, hd]
        let k = self.rms_norm(k, k_norm).swap_dims(0, 1); // [kvh, m, hd]
        let v = v.swap_dims(0, 1);
        let q = Self::apply_rope(q, cos, sin);
        let k = Self::apply_rope(k, cos, sin);

        let kc = cache.k[li].take().unwrap_or_else(|| Tensor::zeros([kvh, cache.cap, hd], &self.device));
        let vc = cache.v[li].take().unwrap_or_else(|| Tensor::zeros([kvh, cache.cap, hd], &self.device));
        let kc = kc.slice_assign([0..kvh, pos0..total, 0..hd], k);
        let vc = vc.slice_assign([0..kvh, pos0..total, 0..hd], v);
        let keys = kc.clone().slice([0..kvh, 0..total, 0..hd]);
        let vals = vc.clone().slice([0..kvh, 0..total, 0..hd]);
        cache.k[li] = Some(kc);
        cache.v[li] = Some(vc);

        let scaling = (hd as f32).powf(-0.5);
        let qg = q.reshape([kvh, group * m, hd]);
        let mut scores = qg.matmul(keys.swap_dims(1, 2)) * scaling; // [kvh, group*m, total]
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask.clone(), f32::NEG_INFINITY);
        }
        let probs = softmax(scores, 2);
        let attn = probs.matmul(vals); // [kvh, group*m, hd]
        let attn = attn.reshape([nh, m, hd]).swap_dims(0, 1).reshape([m, nh * hd]);
        self.linear(attn, o_w)
    }

    fn ensure_cap(&self, cache: &mut Cache, need: usize) {
        if need <= cache.cap {
            return;
        }
        let new_cap = need.next_power_of_two().max(256);
        let hd = self.cfg().head_dim();
        for li in 0..self.cfg().n_layer {
            let kvh = self.cfg().kv_heads[li];
            if kvh == 0 {
                continue;
            }
            for slot in [&mut cache.k[li], &mut cache.v[li]] {
                if let Some(old) = slot.take() {
                    let z = Tensor::<3>::zeros([kvh, new_cap, hd], &self.device);
                    *slot = Some(if cache.len > 0 {
                        z.slice_assign([0..kvh, 0..cache.len, 0..hd], old.slice([0..kvh, 0..cache.len, 0..hd]))
                    } else {
                        z
                    });
                }
            }
        }
        cache.cap = new_cap;
    }

    /// Run `tokens` at positions `cache.len ..`. Returns logits `[m, vocab]` when
    /// `all_logits`, else `[1, vocab]` for the last position.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache, all_logits: bool) -> Tensor<2> {
        let cfg = self.cfg();
        let m = tokens.len();
        let pos0 = cache.len;
        let total = pos0 + m;
        self.ensure_cap(cache, total);
        let (cos, sin) = self.rope_tables(pos0, m);
        let e = cfg.n_embd;

        let mut x = self.embed(tokens);
        for (li, layer) in self.w.layers.iter().enumerate() {
            let h = self.rms_norm(x.clone(), &layer.attn_norm);
            let op_out = match &layer.op {
                Operator::Conv { in_proj, conv, out_proj } => self.conv_block(h, in_proj, conv, out_proj, &mut cache.conv[li]),
                Operator::Attn { q, k, v, o, q_norm, k_norm } => {
                    let group = cfg.n_head / cfg.kv_heads[li];
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
                    self.attn_block(h, li, q, k, v, o, q_norm, k_norm, cache, pos0, &cos, &sin, &mask)
                }
            };
            x = x + op_out;

            let h2 = self.rms_norm(x.clone(), &layer.ffn_norm);
            let ffn_out = match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let g = self.linear(h2.clone(), gate);
                    let u = self.linear(h2, up);
                    self.linear(silu(g) * u, down)
                }
                Ffn::Moe { router, bias, gate, up, down } => {
                    let hv = Self::to_vec(h2);
                    let t = Instant::now();
                    let (out, routes) = moe_ffn(&self.w, router, bias, gate, up, down, &hv, m, &self.store, li, &self.stats);
                    self.matmul_ns.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    self.scraper.record(li, pos0, &routes);
                    self.from_vec(out, m, e)
                }
            };
            x = x + ffn_out;
        }
        cache.len = total;
        let x = if all_logits { x } else { x.slice([m - 1..m, 0..e]) };
        let x = self.rms_norm(x, &self.w.final_norm);
        self.linear(x, &self.w.embd)
    }

    /// Greedy next token from a logits row.
    pub fn argmax_rows(logits: &Tensor<2>) -> Vec<u32> {
        let [m, v] = logits.dims();
        let data = logits.clone().into_data().try_to_vec::<f32>().unwrap();
        (0..m)
            .map(|i| {
                let row = &data[i * v..(i + 1) * v];
                let mut best = 0usize;
                for (j, &x) in row.iter().enumerate() {
                    if x > row[best] {
                        best = j;
                    }
                }
                best as u32
            })
            .collect()
    }
}
