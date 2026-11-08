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
use crate::dense::DenseStore;
use crate::moe::{moe_ffn, ExpertStore, Placement, PlacementStats, Route, Scraper};
use crate::weights::{Ffn, MatRef, Operator, Weights};
use anyhow::Result;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Bool, Device, Tensor, TensorData};
use lb_gguf::{matvec_f32, Gguf};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct Lfm2 {
    pub w: Weights,
    pub device: Device,
    pub placement: Placement,
    /// Experts made resident on the GPU from `placement` (see [`Lfm2::load_residents`]).
    pub store: ExpertStore,
    /// Dense weights made resident on the same device (see [`Lfm2::load_dense`]). The device
    /// itself is owned by `store`, which is the one place that opens it.
    pub dense: DenseStore,
    /// Shared so a host (the server) can read the counters while the model runs on its
    /// own thread: this is the only view of where a token's time actually went.
    pub stats: Arc<PlacementStats>,
    /// Shared with the host so scrape output can be redirected or switched off live.
    pub scraper: Arc<Scraper>,
    /// Where a forward pass spent its time. Shared for the same reason as `stats`: the
    /// model runs on its own thread and the host has to be able to read this while it does.
    pub timings: Arc<Timings>,
    /// Device copies for the device-resident decode, built on its first token.
    #[cfg(feature = "gpu")]
    pub(crate) dd: std::sync::Mutex<Option<crate::device_decode::Prep>>,
    /// Per-layer hidden-state capture for offline analysis (expert prediction, stand-in
    /// finder; see `docs/EXPERT_PREDICT.md`). Off unless [`Lfm2::set_capture`] turns it on;
    /// it only copies tensors out, so the forward pass computes the same numbers either way.
    capture_on: AtomicBool,
    capture: std::sync::Mutex<Vec<CapturedLayer>>,
}

/// One layer of one forward pass, captured on the host path for `m` tokens at `pos0..`.
///
/// * `x_mid` `[m, n_embd]`: the residual stream after the layer's operator (conv/attention),
///   i.e. what the layer's FFN pre-norm reads. `rms(x_mid, ffn_norm) ` is the router input.
/// * `x_out` `[m, n_embd]`: the residual stream after the FFN, the next layer's input.
/// * `routes`: the experts and weights the engine chose per token (empty for dense layers).
#[derive(Clone, Debug)]
pub struct CapturedLayer {
    pub layer: usize,
    pub pos0: usize,
    pub m: usize,
    pub x_mid: Vec<f32>,
    pub x_out: Vec<f32>,
    pub routes: Vec<Route>,
}

/// Cumulative wall time inside a forward pass, split at the one boundary that decides what
/// to optimise next: quantized matmuls against everything else.
///
/// `forward_ns - matmul_ns` is the norms, RoPE, softmax, residuals, the conv taps and the
/// tensor-to-vector hops in and out of every kernel — all Burn CPU tensor ops. When the
/// matmuls move to the device that remainder is what is left, so it needs its own number
/// rather than an inference from the token rate.
/// A prefill pass costs many times a decode pass, so they are counted apart: averaged
/// together, six prefills in two hundred passes move the per-token figure by more than the
/// change being measured.
#[derive(Default)]
pub struct Timings {
    pub forward_ns: AtomicU64,
    /// Dense `linear` plus `moe_ffn`.
    pub matmul_ns: AtomicU64,
    pub forwards: AtomicU64,
    /// The same three, restricted to single-token (decode) passes.
    pub decode_forward_ns: AtomicU64,
    pub decode_matmul_ns: AtomicU64,
    pub decodes: AtomicU64,
    /// Decode tokens that ran device-resident (`device_decode.rs`), their wall time, and the
    /// single-token passes the knob asked for that went to the old path instead.
    pub dd_tokens: AtomicU64,
    pub dd_ns: AtomicU64,
    pub dd_fallbacks: AtomicU64,
}

/// Mean ms per pass over the passes counted, split at the matmul boundary.
pub struct Split {
    pub passes: u64,
    pub forward_ms: f64,
    pub matmul_ms: f64,
    pub other_ms: f64,
}

impl Timings {
    /// Over every pass, prefill included.
    pub fn snapshot(&self) -> Split {
        Self::split(&self.forwards, &self.forward_ns, &self.matmul_ns)
    }

    /// Over decode passes only: what a generated token actually costs.
    pub fn decode(&self) -> Split {
        Self::split(&self.decodes, &self.decode_forward_ns, &self.decode_matmul_ns)
    }

    fn split(n: &AtomicU64, fwd: &AtomicU64, mm: &AtomicU64) -> Split {
        let passes = n.load(Ordering::Relaxed);
        let per = |x: &AtomicU64| if passes > 0 { x.load(Ordering::Relaxed) as f64 / 1e6 / passes as f64 } else { 0.0 };
        let (forward_ms, matmul_ms) = (per(fwd), per(mm));
        Split { passes, forward_ms, matmul_ms, other_ms: forward_ms - matmul_ms }
    }
}

/// Per-layer state: K/V `[kv_heads, cap, head_dim]` for attention layers, the last
/// `l_cache - 1` rows of `B*X` `[l_cache-1, n_embd]` for conv layers.
pub struct Cache {
    pub k: Vec<Option<Tensor<3>>>,
    pub v: Vec<Option<Tensor<3>>>,
    pub conv: Vec<Option<Tensor<2>>>,
    pub len: usize,
    pub cap: usize,
    /// Positions the host tensors above are valid for. Equal to `len` except after
    /// device-resident decode tokens, which advance only the device copy (`dev`); the next host
    /// pass reads it back first (`device_decode::pull_if_stale`).
    pub host_len: usize,
    /// The device copy of this sequence's KV cache and conv state (device-resident decode).
    #[cfg(feature = "gpu")]
    pub dev: Option<crate::device_decode::DevCache>,
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
            dense: DenseStore::default(),
            stats: Arc::new(PlacementStats::default()),
            scraper: Arc::new(Scraper::from_env(&model_id)),
            timings: Arc::new(Timings::default()),
            #[cfg(feature = "gpu")]
            dd: std::sync::Mutex::new(None),
            capture_on: AtomicBool::new(false),
            capture: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Turn the per-layer hidden-state capture on or off. While on, every forward pass runs
    /// the host path (the device-resident decode declines) and appends one
    /// [`CapturedLayer`] per layer; drain them with [`Lfm2::take_capture`].
    pub fn set_capture(&self, on: bool) {
        self.capture_on.store(on, Ordering::Relaxed);
        if !on {
            self.capture.lock().unwrap().clear();
        }
    }

    pub fn capture_enabled(&self) -> bool {
        self.capture_on.load(Ordering::Relaxed)
    }

    /// Everything captured since the last call, in pass and layer order.
    pub fn take_capture(&self) -> Vec<CapturedLayer> {
        std::mem::take(&mut *self.capture.lock().unwrap())
    }

    /// Upload the `Home::Gpu` experts of the current placement to the device
    /// (`LB_WGPU_DEVICE`); needs the `gpu` feature. Returns how many were uploaded.
    pub fn load_residents(&mut self) -> Result<usize, String> {
        let Self { w, placement, store, .. } = self;
        store.load(w, placement)
    }

    /// Turn the expert staging ring on (`stage.rs`), after the residents and dense weights are
    /// loaded: `cfg.ring_mib` of ring on the open device. Returns the ring's capacity in experts
    /// (0 and nothing changed when `cfg.on` is false). Restart-only, like residency.
    #[cfg(feature = "gpu")]
    pub fn enable_stage(&mut self, cfg: crate::stage::StageCfg) -> Result<usize, String> {
        let attn: Vec<bool> = (0..self.cfg().n_layer).map(|l| self.cfg().is_attn(l)).collect();
        self.stats.stage.set_attn_layers(attn);
        let eb = (0..self.cfg().n_layer).map(|l| self.w.expert_bytes(l)).max().unwrap_or(0);
        let cap = cfg.ring_slots(eb);
        self.store.stage = cfg;
        if cap == 0 {
            return Ok(0);
        }
        let gpu = self.store.gpu.as_ref().ok_or("staging needs the device open (--gpu)")?;
        *self.store.ring.lock().unwrap() = Some(crate::stage::Ring::new(gpu, cap));
        Ok(cap)
    }

    /// Upload dense Q4_0 weights to the device, up to `budget_bytes`. Returns how many.
    /// Opens the device if `load_residents` has not already.
    #[cfg(feature = "gpu")]
    pub fn load_dense(&mut self, budget_bytes: usize) -> Result<usize, String> {
        if self.store.gpu.is_none() {
            self.store.gpu = Some(crate::moe::ExpertStore::open_gpu(&self.w)?);
        }
        let Self { w, store, dense, .. } = self;
        dense.load(store.gpu.as_ref().expect("device just opened"), w, budget_bytes)
    }

    pub fn cfg(&self) -> &Config {
        &self.w.cfg
    }

    /// The live settings, shared with whoever hosts the model (see `knobs.rs`).
    pub fn knobs(&self) -> Arc<crate::knobs::Knobs> {
        self.store.knobs.clone()
    }

    /// Push the live knobs the device reads into it; called at the top of every forward pass
    /// so a change posted between two tokens applies to the next one.
    fn apply_knobs(&self) {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.store.gpu.as_ref() {
            let k = &self.store.knobs;
            if gpu.host_repack() != k.host_repack() {
                gpu.set_host_repack(k.host_repack());
            }
            if gpu.batch_miss() != k.batch_miss() {
                gpu.set_batch_miss(k.batch_miss());
            }
        }
    }

    pub fn new_cache(&self) -> Cache {
        let n = self.cfg().n_layer;
        Cache {
            k: vec![None; n],
            v: vec![None; n],
            conv: vec![None; n],
            len: 0,
            cap: 0,
            host_len: 0,
            #[cfg(feature = "gpu")]
            dev: None,
        }
    }

    fn to_vec(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn from_vec(&self, v: Vec<f32>, rows: usize, cols: usize) -> Tensor<2> {
        Tensor::from_data(TensorData::new(v, [rows, cols]), &self.device)
    }

    /// `x [m, cols] @ W^T -> [m, rows]` through the quantized kernel: the resident device
    /// copy when [`DenseStore`] holds this matrix, else the CPU reference kernel.
    pub fn linear(&self, x: Tensor<2>, w: &MatRef) -> Tensor<2> {
        let [m, cols] = x.dims();
        assert_eq!(cols, w.cols, "linear: input width");
        let xv = Self::to_vec(x);
        let t = Instant::now();
        let out = self.linear_vec(&xv, w, m);
        self.timings.matmul_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.from_vec(out, m, w.rows)
    }

    /// The matmul itself, on whichever side holds the weights.
    fn linear_vec(&self, xv: &[f32], w: &MatRef, m: usize) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.store.gpu.as_ref() {
            // a device fault here would be silent corruption if we guessed at the result:
            // say which matrix, then let the CPU kernel finish the token.
            let report = |e| eprintln!("dense matvec on device failed for [{}, {}]: {e}; using the CPU kernel", w.rows, w.cols);
            if let Some(mat) = self.dense.resident.get(&w.start) {
                match mat.matvec(gpu, xv, m) {
                    Ok(v) => return v,
                    Err(e) => report(e),
                }
            } else if let Some(mat) = self.dense.q6.get(&w.start) {
                match mat.matvec(gpu, xv, m) {
                    Ok(v) => return v,
                    Err(e) => report(e),
                }
            }
        }
        let mut out = vec![0f32; m * w.rows];
        matvec_f32(self.w.mat(w), xv, m, &mut out);
        out
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

    /// One short-conv layer. For a single token with the block resident, this is one host
    /// sync for the whole layer (`in_proj -> conv -> out_proj` chained on the device); every
    /// other case falls through to the CPU path below.
    ///
    /// The two states have to be kept consistent, because the device copy advances itself:
    /// * CPU path: afterwards, push the host state to the device and mark it clean, so the
    ///   first decode token after a prefill convolves against real history, not zeros.
    /// * device path: the host state is now stale, so mark the layer dirty. A later
    ///   multi-token pass (`m > 1`) pulls the state back first -- one sync, rare.
    #[allow(clippy::too_many_arguments)]
    fn conv_layer(&self, h: Tensor<2>, li: usize, in_proj: &MatRef, conv: &[f32], out_proj: &MatRef, state: &mut Option<Tensor<2>>) -> Tensor<2> {
        let (e, l) = (self.cfg().n_embd, self.cfg().l_cache);
        let [m, _] = h.dims();
        #[cfg(feature = "gpu")]
        {
            let resident = crate::dense::conv_blocks_enabled()
                && self.dense.conv.contains_key(&in_proj.start)
                && self.store.gpu.is_some();
            if resident && m == 1 {
                let gpu = self.store.gpu.as_ref().expect("checked");
                let block = self.dense.conv.get(&in_proj.start).expect("checked");
                let hv = Self::to_vec(h.clone());
                let t = Instant::now();
                let done = block.forward(gpu, &hv);
                self.timings.matmul_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                match done {
                    Ok(out) => {
                        self.dense.mark_conv_dirty(in_proj.start);
                        return self.from_vec(out, 1, e);
                    }
                    Err(err) => eprintln!("conv block {li} on device failed: {err}; using the CPU kernel"),
                }
            }
            if resident && m > 1 && self.dense.is_conv_dirty(in_proj.start) {
                // decode has advanced the device state past the host's: bring it back
                let gpu = self.store.gpu.as_ref().expect("checked");
                let block = self.dense.conv.get(&in_proj.start).expect("checked");
                match block.read_state(gpu) {
                    Ok(rows) => {
                        *state = Some(self.from_vec(rows, l - 1, e));
                        self.dense.mark_conv_clean(in_proj.start);
                    }
                    Err(err) => eprintln!("conv state {li} read back failed: {err}; the host state may be behind"),
                }
            }
        }
        let out = self.conv_block(h, in_proj, conv, out_proj, state);
        #[cfg(feature = "gpu")]
        if let (Some(gpu), Some(block), Some(st)) =
            (self.store.gpu.as_ref(), self.dense.conv.get(&in_proj.start), state.as_ref())
        {
            // the CPU path just moved the history on: tell the device where it got to
            if let Err(err) = block.set_state(gpu, &Self::to_vec(st.clone())) {
                eprintln!("conv state {li} push failed: {err}; the device state may be behind");
            } else {
                self.dense.mark_conv_clean(in_proj.start);
            }
        }
        let _ = (li, l);
        out
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

    pub(crate) fn ensure_cap(&self, cache: &mut Cache, need: usize) {
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
        self.apply_knobs();
        let mm0 = self.timings.matmul_ns.load(Ordering::Relaxed);
        let t_fwd = Instant::now();
        let out = self.forward_inner(tokens, cache, all_logits);
        let dt = t_fwd.elapsed().as_nanos() as u64;
        let t = &self.timings;
        t.forward_ns.fetch_add(dt, Ordering::Relaxed);
        t.forwards.fetch_add(1, Ordering::Relaxed);
        if tokens.len() == 1 {
            t.decode_forward_ns.fetch_add(dt, Ordering::Relaxed);
            t.decode_matmul_ns.fetch_add(t.matmul_ns.load(Ordering::Relaxed).wrapping_sub(mm0), Ordering::Relaxed);
            t.decodes.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn forward_inner(&self, tokens: &[u32], cache: &mut Cache, all_logits: bool) -> Tensor<2> {
        // One decode token device-resident when asked and possible (see `device_decode.rs`);
        // `None` means it declined and nothing changed, so the path below runs as before.
        #[cfg(feature = "gpu")]
        if tokens.len() == 1 && self.store.knobs.device_decode() && !self.capture_enabled() {
            if let Some(logits) = crate::device_decode::token(self, tokens[0], cache) {
                return logits;
            }
        }
        // a host pass after device tokens: bring the host's KV cache and conv state up first
        #[cfg(feature = "gpu")]
        crate::device_decode::pull_if_stale(self, cache);
        let out = self.forward_host(tokens, cache, all_logits);
        cache.host_len = cache.len;
        out
    }

    /// The forward pass on Burn tensors with resident matmuls (the path before device decode).
    fn forward_host(&self, tokens: &[u32], cache: &mut Cache, all_logits: bool) -> Tensor<2> {
        let cfg = self.cfg();
        let m = tokens.len();
        let pos0 = cache.len;
        let total = pos0 + m;
        // A pass at position 0 is a new sequence. The resident conv blocks carry their state
        // on the device, where no new `Cache` can clear it, so it has to be cleared here or
        // the first token would convolve against the previous request's history.
        #[cfg(feature = "gpu")]
        if pos0 == 0 {
            if let Some(gpu) = self.store.gpu.as_ref() {
                self.dense.reset_conv(gpu);
            }
        }
        self.ensure_cap(cache, total);
        let (cos, sin) = self.rope_tables(pos0, m);
        let e = cfg.n_embd;

        let capturing = self.capture_enabled();
        let mut x = self.embed(tokens);
        for (li, layer) in self.w.layers.iter().enumerate() {
            let h = self.rms_norm(x.clone(), &layer.attn_norm);
            let op_out = match &layer.op {
                Operator::Conv { in_proj, conv, out_proj } => self.conv_layer(h, li, in_proj, conv, out_proj, &mut cache.conv[li]),
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
            let x_mid = if capturing { Some(Self::to_vec(x.clone())) } else { None };

            let h2 = self.rms_norm(x.clone(), &layer.ffn_norm);
            let mut layer_routes: Vec<Route> = Vec::new();
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
                    self.timings.matmul_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    self.scraper.record(li, pos0, &routes);
                    if capturing {
                        layer_routes = routes;
                    }
                    self.from_vec(out, m, e)
                }
            };
            x = x + ffn_out;
            if let Some(x_mid) = x_mid {
                let x_out = Self::to_vec(x.clone());
                self.capture.lock().unwrap().push(CapturedLayer { layer: li, pos0, m, x_mid, x_out, routes: layer_routes });
            }
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
