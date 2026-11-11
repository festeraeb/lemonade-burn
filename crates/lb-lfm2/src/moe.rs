//! Mixture-of-experts pieces: the router, the expert store (where each expert's weights
//! are executed), scrape-driven placement, and the NautiBurn routing scrape writer.
//!
//! Router (LFM2-MoE, `expert_gating_func = 2`): `logits = x W_r^T`, `p = sigmoid(logits)`,
//! experts are *selected* by `p + bias` (top-k), and *weighted* by the unbiased `p` of the
//! selected experts, normalized to sum to one (sum clamped to >= 2^-14), times the
//! routed scaling factor (1.0 for LFM2.5). Each expert is a SwiGLU FFN:
//! `down(silu(gate(x)) * up(x))`.

use crate::weights::{ExpertMats, Weights};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(feature = "gpu")]
mod cubecl_handle {
    pub use lb_q4_gpu::cubecl_server::Handle;
}

/// Selected experts and their weights for one token.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Sigmoid router with selection bias and top-k weight normalization (see module doc).
/// `router` is `[n_expert][n_embd]`; `x` one token of `n_embd`.
pub fn route_sigmoid(router: &[f32], bias: &[f32], x: &[f32], n_expert: usize, k: usize) -> Route {
    let n_embd = x.len();
    debug_assert_eq!(router.len(), n_expert * n_embd);
    let probs: Vec<f32> = (0..n_expert)
        .map(|e| {
            let logit: f32 = router[e * n_embd..(e + 1) * n_embd].iter().zip(x).map(|(a, b)| a * b).sum();
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();
    let mut order: Vec<usize> = (0..n_expert).collect();
    // stable sort, descending by biased score; ties keep the lower expert id first
    order.sort_by(|&a, &b| (probs[b] + bias[b]).partial_cmp(&(probs[a] + bias[a])).unwrap_or(std::cmp::Ordering::Equal));
    let experts: Vec<usize> = order[..k].to_vec();
    let mut weights: Vec<f32> = experts.iter().map(|&e| probs[e]).collect();
    let sum: f32 = weights.iter().sum::<f32>().max(6.103515625e-5);
    for w in &mut weights {
        *w /= sum;
    }
    Route { experts, weights }
}

/// Where an expert's weights live for execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Home {
    /// Executed from the mmap on the CPU (the always-correct fallback).
    Cpu,
    /// Resident on the GPU selected by `LB_WGPU_DEVICE` (planned; see `docs/LFM2_CUBECL.md`).
    Gpu,
}

/// Per-layer expert placement, derived from routing histograms (NautiBurn) and a byte
/// budget. In step 1 every expert still *executes* on the CPU; the plan is computed and
/// logged so the GPU residency layer can be dropped in without changing the forward pass.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Placement {
    /// `homes[layer][expert]`.
    pub homes: Vec<Vec<Home>>,
    pub resident_bytes: usize,
}

/// A routing histogram: `counts[layer][expert]` (NautiBurn `histogram_counts` rows, or
/// counts accumulated from this engine's own scrapes).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Histogram {
    pub model_id: String,
    pub counts: Vec<Vec<u64>>,
}

impl Histogram {
    pub fn from_scrapes(model_id: &str, n_layer: usize, n_expert: usize, scrapes: &[Scrape]) -> Self {
        let mut counts = vec![vec![0u64; n_expert]; n_layer];
        for s in scrapes {
            for a in &s.activations {
                for &e in &a.expert_ids {
                    if (a.layer as usize) < n_layer && (e as usize) < n_expert {
                        counts[a.layer as usize][e as usize] += 1;
                    }
                }
            }
        }
        Self { model_id: model_id.to_string(), counts }
    }
}

impl Histogram {
    /// Experts the histogram saw, hottest first by their share of their layer's routings (the
    /// order `Placement::plan` spends a budget in). Seeds the dynamic cache (`cache.rs`).
    pub fn heat_order(&self) -> Vec<(usize, usize)> {
        let mut cands: Vec<(f64, usize, usize)> = Vec::new();
        for (l, row) in self.counts.iter().enumerate() {
            let total: u64 = row.iter().sum();
            for (e, &c) in row.iter().enumerate() {
                if c > 0 {
                    cands.push((c as f64 / total as f64, l, e));
                }
            }
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        cands.into_iter().map(|(_, l, e)| (l, e)).collect()
    }
}

impl Placement {
    /// Everything on the CPU.
    pub fn all_cpu(n_layer: usize, n_expert: usize) -> Self {
        Self { homes: vec![vec![Home::Cpu; n_expert]; n_layer], resident_bytes: 0 }
    }

    /// Greedy: the hottest experts across all layers (by histogram count, normalized per
    /// layer so every layer gets a fair share) become GPU-resident until `budget_bytes`
    /// is spent; `bytes_per_expert[layer]` is the size of one expert's three matrices.
    /// Experts the histogram never saw then fill whatever budget is left, round robin by
    /// layer (see [`Placement::fill_unseen`]), because an arbitrary resident expert still
    /// beats serving a miss from the model file.
    pub fn plan(hist: &Histogram, bytes_per_expert: &[usize], budget_bytes: usize) -> Self {
        let n_layer = hist.counts.len();
        let mut cands: Vec<(f64, usize, usize)> = Vec::new();
        for (l, row) in hist.counts.iter().enumerate() {
            let total: u64 = row.iter().sum();
            if total == 0 {
                continue;
            }
            for (e, &c) in row.iter().enumerate() {
                if c > 0 {
                    cands.push((c as f64 / total as f64, l, e));
                }
            }
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let n_expert = hist.counts.first().map_or(0, |r| r.len());
        let mut p = Self::all_cpu(n_layer, n_expert);
        for (_, l, e) in cands {
            let b = bytes_per_expert[l];
            if p.resident_bytes + b > budget_bytes {
                continue;
            }
            p.homes[l][e] = Home::Gpu;
            p.resident_bytes += b;
        }
        p.fill_unseen(bytes_per_expert, budget_bytes);
        p
    }

    /// Spend any budget the histogram could not spend on experts it never saw.
    ///
    /// Routing data is the binding constraint, not VRAM: two scrape runs name only 2066 of
    /// the 24B's 10240 experts, so a 12 GB and a 14.5 GB budget both planned the same 2066
    /// residents and left 3.5 GB of a 16 GB card idle. An expert chosen with no evidence is
    /// a poor bet individually, but the alternative is serving that call from the model file
    /// at a measured 21 ms, against 0.5 ms for any resident - so an idle card is never
    /// right. Round robin over layers so no layer is starved, lowest expert id first, which
    /// keeps the plan deterministic for a given budget.
    fn fill_unseen(&mut self, bytes_per_expert: &[usize], budget_bytes: usize) {
        let n_expert = self.homes.first().map_or(0, Vec::len);
        for e in 0..n_expert {
            let mut any = false;
            for (l, row) in self.homes.iter_mut().enumerate() {
                // layers with no MoE FFN report zero bytes: never stage those
                let b = bytes_per_expert.get(l).copied().unwrap_or(0);
                if b == 0 || self.resident_bytes + b > budget_bytes {
                    continue;
                }
                if row[e] == Home::Cpu {
                    row[e] = Home::Gpu;
                    self.resident_bytes += b;
                }
                any = true;
            }
            if !any {
                break;
            }
        }
    }

    /// Whole MoE layers, in layer order, until `budget_bytes` is spent: every expert of a
    /// placed layer is resident and every expert of the others is not. This is the placement a
    /// device-resident decode can use (its expert kernels need a layer's experts in one buffer,
    /// `lb_q4_gpu::GpuExpertLayer`), and on a card too small for the model it is the honest
    /// partial residency to test it with. Layers with no experts (`bytes_per_expert` 0) are
    /// skipped; a layer that does not fit ends the plan, so the placed layers are a prefix.
    pub fn whole_layers(n_expert: usize, bytes_per_expert: &[usize], budget_bytes: usize) -> Self {
        let n_layer = bytes_per_expert.len();
        let mut p = Self::all_cpu(n_layer, n_expert);
        for (l, &b) in bytes_per_expert.iter().enumerate() {
            if b == 0 {
                continue;
            }
            if p.resident_bytes + b * n_expert > budget_bytes {
                break;
            }
            p.homes[l] = vec![Home::Gpu; n_expert];
            p.resident_bytes += b * n_expert;
        }
        p
    }

    /// Whether every expert of `layer` is planned resident (false for a layer without experts).
    pub fn layer_whole(&self, layer: usize) -> bool {
        self.homes.get(layer).is_some_and(|r| !r.is_empty() && r.iter().all(|h| *h == Home::Gpu))
    }

    pub fn home(&self, layer: usize, expert: usize) -> Home {
        self.homes.get(layer).and_then(|r| r.get(expert)).copied().unwrap_or(Home::Cpu)
    }

    pub fn n_resident(&self) -> usize {
        self.homes.iter().flatten().filter(|h| **h == Home::Gpu).count()
    }
}

/// Counters: expert calls that ran on (or, without the `gpu` feature, would have hit) a
/// GPU-resident expert versus the CPU path, and time spent in each.
#[derive(Default, Debug)]
pub struct PlacementStats {
    pub gpu_hits: AtomicU64,
    pub cpu_calls: AtomicU64,
    pub gpu_ns: AtomicU64,
    pub cpu_ns: AtomicU64,
    /// Misses served by copying the expert to the device ([`MissPolicy::OnDemand`]):
    /// counted apart from `gpu_hits` so the residency plan's own hit rate stays readable.
    pub on_demand: AtomicU64,
    pub on_demand_ns: AtomicU64,
    /// Decode-only, per-layer hit/miss/prefetch counters (both with and without the staging
    /// ring, so the two compare): `stage.rs`.
    pub stage: crate::stage::StageStats,
}

/// A reading of [`PlacementStats`]. `hit_rate` is the share of expert calls the residency
/// plan caught outright; a plan that covers few of the experts the router asks for leaves
/// the misses on the critical path however fast the GPU kernel is, so this is the number to
/// read before optimizing the kernel again.
#[derive(Clone, Copy, Debug)]
pub struct StatsSnapshot {
    pub gpu_hits: u64,
    pub cpu_calls: u64,
    pub on_demand: u64,
    pub gpu_s: f64,
    pub cpu_s: f64,
    pub on_demand_s: f64,
    pub hit_rate: f64,
}

impl PlacementStats {
    pub fn snapshot(&self) -> StatsSnapshot {
        let o = Ordering::Relaxed;
        let (gpu_hits, cpu_calls, on_demand) = (self.gpu_hits.load(o), self.cpu_calls.load(o), self.on_demand.load(o));
        let total = gpu_hits + cpu_calls + on_demand;
        StatsSnapshot {
            gpu_hits,
            cpu_calls,
            on_demand,
            gpu_s: self.gpu_ns.load(o) as f64 / 1e9,
            cpu_s: self.cpu_ns.load(o) as f64 / 1e9,
            on_demand_s: self.on_demand_ns.load(o) as f64 / 1e9,
            hit_rate: if total == 0 { 0.0 } else { gpu_hits as f64 / total as f64 },
        }
    }
}

/// What to do with an expert the router asked for that is not GPU-resident.
///
/// Measured on the 24B with a 12 GB budget (2066 residents): the plan catches 97.1% of the
/// calls, and the 2.9% it misses take 71% of the expert wall time - 45.2 ms to run a miss on
/// the CPU against 0.542 ms to enqueue a hit. Copying the expert over and running it on the
/// device instead costs about 2 ms to repack plus 1.7 ms for the copy, so `OnDemand` trades
/// a 45 ms CPU call for a roughly 4 ms one. `Cpu` stays the default because it is the only
/// policy that needs no free VRAM and no device at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum MissPolicy {
    /// Run the miss on the CPU reference path.
    #[default]
    Cpu = 0,
    /// Upload the expert to the device, run it there, release it after the readback.
    OnDemand = 1,
}

impl MissPolicy {
    /// Parse a `--miss-policy` value.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "on-demand" | "ondemand" => Ok(Self::OnDemand),
            other => Err(format!("unknown miss policy {other:?} (expected cpu or on-demand)")),
        }
    }

    /// The `--miss-policy` spelling of this value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::OnDemand => "on-demand",
        }
    }
}

/// Experts resident on the GPU, keyed by (layer, expert). Empty without the `gpu` feature.
#[derive(Default)]
pub struct ExpertStore {
    #[cfg(feature = "gpu")]
    pub gpu: Option<lb_q4_gpu::Gpu>,
    #[cfg(feature = "gpu")]
    pub resident: HashMap<(usize, usize), lb_q4_gpu::GpuExpert>,
    /// Layers whose experts are all resident as one stacked buffer per projection (see
    /// [`expert_layers_enabled`]). Their entries in `resident` are views into these buffers,
    /// so the per-expert path and the device-resident decode read the same device memory.
    #[cfg(feature = "gpu")]
    pub layers: HashMap<usize, lb_q4_gpu::GpuExpertLayer>,
    pub bytes: usize,
    /// The live settings (how to serve a miss, which upload path), shared with the host.
    pub knobs: std::sync::Arc<crate::knobs::Knobs>,
    /// Expert staging (`stage.rs`), restart-only; the ring exists only when it is on.
    pub stage: crate::stage::StageCfg,
    /// Look-ahead over the expert cache (engine step E3, `stage.rs`), restart-only; off by default.
    pub la: crate::stage::LookaheadCfg,
    #[cfg(feature = "gpu")]
    pub ring: Mutex<Option<crate::stage::Ring>>,
    /// The dynamic expert cache (`cache.rs`, engine step E1). When present every expert runs from
    /// it and the static residents / ring are not used.
    #[cfg(feature = "gpu")]
    pub cache: Mutex<Option<crate::cache::ExpertCache>>,
    /// Work-to-weights (`remote.rs`): experts the placement puts on expert hosts are asked for
    /// there first; whatever a host does not answer runs on the paths above. `None` = every
    /// expert is this process's own.
    pub remote: Option<std::sync::Arc<crate::remote::RemoteTier>>,
    /// Work-to-weights inside the box (`inbox.rs`): expert cards of this process holding whole
    /// MoE layers; the device decode keeps the activation on the device for these.
    #[cfg(feature = "gpu")]
    pub inbox: Option<crate::inbox::InBox>,
}

impl ExpertStore {
    /// How to serve an expert that is not resident (live: `knobs`).
    pub fn miss(&self) -> MissPolicy {
        self.knobs.miss_policy()
    }

    pub fn is_resident(&self, layer: usize, expert: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            return self.resident.contains_key(&(layer, expert));
        }
        #[allow(unreachable_code)]
        {
            let _ = (layer, expert);
            false
        }
    }

    pub fn n_resident(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            return self.resident.len();
        }
        #[allow(unreachable_code)]
        0
    }

    /// The device for this store, opened (`LB_CUDA_DEVICE` / `LB_WGPU_DEVICE`) and, on the
    /// cuda runtime, with the model's file mapping page-locked so that expert uploads DMA
    /// the raw bytes out of the page cache and repack on the device (`lb-q4-gpu`
    /// `registered.rs`; `LB_NO_REGISTER=1` keeps the host repack). The page-lock holds the
    /// whole GGUF in RAM for the model's lifetime, which is what the mapping does anyway once
    /// it is hot.
    #[cfg(feature = "gpu")]
    pub fn open_gpu(w: &Weights) -> Result<lb_q4_gpu::Gpu, String> {
        let gpu = lb_q4_gpu::Gpu::open()?;
        #[cfg(feature = "cuda")]
        if std::env::var_os("LB_NO_REGISTER").is_none() {
            let t = std::time::Instant::now();
            // SAFETY: the mapping is `w.gguf`'s, an `Arc<Gguf>` the model holds as long as its
            // store, and so this `Gpu`, exists; nothing writes a read-only mapping.
            match unsafe { gpu.register_mapping(w.gguf.mapping()) } {
                Ok(n) => eprintln!("lb-lfm2: page-locked {:.2} GB of the GGUF mapping in {:.0} ms; expert uploads DMA raw bytes and repack on the device", n as f64 / 1e9, t.elapsed().as_secs_f64() * 1e3),
                Err(e) => eprintln!("lb-lfm2: mapping not page-locked ({e}); uploads repack on the host"),
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = w;
        Ok(gpu)
    }

    /// Upload every `Home::Gpu` expert of `placement` to the device chosen by
    /// `LB_WGPU_DEVICE`. Returns the number uploaded. Without the `gpu` feature: an error.
    #[cfg(feature = "gpu")]
    pub fn load(&mut self, w: &Weights, placement: &Placement) -> Result<usize, String> {
        use crate::weights::Ffn;
        if self.gpu.is_none() {
            self.gpu = Some(Self::open_gpu(w)?);
        }
        let gpu = self.gpu.as_ref().unwrap();
        let mut n = 0;
        for (l, layer) in w.layers.iter().enumerate() {
            let Ffn::Moe { gate, up, down, .. } = &layer.ffn else { continue };
            if expert_layers_enabled() && placement.layer_whole(l) && !self.layers.contains_key(&l) && (0..gate.n).all(|e| !self.resident.contains_key(&(l, e))) {
                let stack = lb_q4_gpu::GpuExpertLayer::upload_stacked(gpu, w.mat(&gate.all()), w.mat(&up.all()), w.mat(&down.all()), gate.n)?;
                for e in 0..gate.n {
                    self.resident.insert((l, e), stack.expert(gpu, e)?);
                }
                self.bytes += stack.bytes_on_device();
                self.layers.insert(l, stack);
                n += gate.n;
                continue;
            }
            for e in 0..gate.n {
                if placement.home(l, e) != Home::Gpu || self.resident.contains_key(&(l, e)) {
                    continue;
                }
                // residents take the per-matrix path: the batched one costs device memory the
                // all-resident config does not have (see `GpuExpert::upload_per_matrix`)
                let ex = lb_q4_gpu::GpuExpert::upload_per_matrix(gpu, w.mat(&gate.expert(e)), w.mat(&up.expert(e)), w.mat(&down.expert(e)))?;
                self.bytes += ex.bytes_on_device();
                self.resident.insert((l, e), ex);
                n += 1;
            }
        }
        Ok(n)
    }

    #[cfg(not(feature = "gpu"))]
    pub fn load(&mut self, _w: &Weights, _placement: &Placement) -> Result<usize, String> {
        Err("built without the `gpu` feature: experts cannot be made resident".to_string())
    }
}

/// Whether a MoE layer whose experts are all planned resident is uploaded as one stacked
/// [`lb_q4_gpu::GpuExpertLayer`] (three buffers, per-expert views into them) instead of 64
/// separate experts. `LB_EXPERT_LAYERS=1`/`0` decides; unset, it follows `LB_DEVICE_DECODE`,
/// whose expert kernels need it; with neither set it is on (the device decode's default since
/// 2026-11-07). On a P100 with all 2432 experts it also takes 13,775 MiB against the per-expert
/// upload's 15,291 (docs/WORK_LOG.md).
///
/// Read once: residency is fixed at load.
pub fn expert_layers_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = |k: &str| std::env::var(k).ok().map(|v| !v.is_empty() && v != "0");
        on("LB_EXPERT_LAYERS").or_else(|| on("LB_DEVICE_DECODE")).unwrap_or(true)
    })
}

/// Executes an expert FFN for a batch of tokens. One SwiGLU expert:
/// `down(silu(gate x) * up x)`. `x` is `m` rows of `n_embd`; returns `m` rows of `n_embd`.
pub fn expert_ffn(w: &Weights, gate: &ExpertMats, up: &ExpertMats, down: &ExpertMats, e: usize, x: &[f32], m: usize) -> Vec<f32> {
    // one source for the arithmetic: an expert host's CPU backend calls the same function
    lb_expert_host::expert::swiglu(w.mat(&gate.expert(e)), w.mat(&up.expert(e)), w.mat(&down.expert(e)), x, m)
}

/// MoE FFN for `m` tokens: route each token, group tokens by expert, run each expert once
/// on its tokens, and add the weighted outputs. Returns `[m, n_embd]` and the routes.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn(
    w: &Weights,
    router: &[f32],
    bias: &[f32],
    gate: &ExpertMats,
    up: &ExpertMats,
    down: &ExpertMats,
    x: &[f32],
    m: usize,
    store: &ExpertStore,
    layer: usize,
    stats: &PlacementStats,
) -> (Vec<f32>, Vec<Route>) {
    moe_ffn_with(w, router, bias, gate, up, down, x, m, store, layer, stats, None)
}

/// No staging ring without a device (keeps `moe_ffn_with`'s signature the same in every build).
#[cfg(not(feature = "gpu"))]
pub struct NoRing;

#[cfg(feature = "gpu")]
type RingArg<'a> = Option<crate::stage::RingCall<'a>>;
#[cfg(not(feature = "gpu"))]
type RingArg<'a> = Option<&'a mut NoRing>;

/// [`moe_ffn`] with the expert staging ring (`stage.rs`): experts found in the ring launch from
/// it, and once this layer's launches are queued the guess for the next layer is staged, before
/// the read that waits for this layer. Every expert's output is added in **ascending expert
/// id**, whichever of resident / ring / demand copy / CPU served it, so neither the ring nor the
/// residency plan changes the arithmetic (it was resident-group-first before, which made the
/// sum order, and so the greedy text, depend on the VRAM budget; an all-resident or all-CPU
/// layer already summed in ascending id and is unchanged).
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_with(
    w: &Weights,
    router: &[f32],
    bias: &[f32],
    gate: &ExpertMats,
    up: &ExpertMats,
    down: &ExpertMats,
    x: &[f32],
    m: usize,
    store: &ExpertStore,
    layer: usize,
    stats: &PlacementStats,
    #[allow(unused_mut, unused_variables)] mut ring: RingArg<'_>,
) -> (Vec<f32>, Vec<Route>) {
    let cfg = &w.cfg;
    let n_embd = cfg.n_embd;
    let routes: Vec<Route> =
        (0..m).map(|t| route_sigmoid(router, bias, &x[t * n_embd..(t + 1) * n_embd], cfg.n_expert, cfg.n_expert_used)).collect();
    // expert -> (token, weight)
    let mut by_expert: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    for (t, r) in routes.iter().enumerate() {
        for (&e, &wt) in r.experts.iter().zip(&r.weights) {
            by_expert.entry(e).or_default().push((t, wt));
        }
    }
    let mut experts: Vec<(usize, Vec<(usize, f32)>)> = by_expert.into_iter().collect();
    experts.sort_by_key(|(e, _)| *e);
    // an in-box expert card holds this layer whole: every expert runs there (`inbox.rs`)
    #[cfg(feature = "gpu")]
    if let Some(ib) = store.inbox.as_ref().filter(|ib| ib.card_of(layer).is_some()) {
        let outs = ib.run_host_routed(layer, &experts, x, n_embd).unwrap_or_else(|e| panic!("in-box card, layer {layer}: {e}"));
        let mut out = vec![0f32; m * n_embd];
        // ascending expert id, as below
        for ((_, toks), y) in experts.iter().zip(outs) {
            for (i, (t, wt)) in toks.iter().enumerate() {
                let o = &mut out[t * n_embd..(t + 1) * n_embd];
                for (oj, yj) in o.iter_mut().zip(&y[i * n_embd..(i + 1) * n_embd]) {
                    *oj += wt * yj;
                }
            }
        }
        return (out, routes);
    }
    // Work-to-weights: experts the placement puts on hosts are computed there. What comes back
    // is each expert's unweighted output; what a host did not answer stays in `experts` and
    // runs below as if there were no hosts. The sum at the end is over both, ascending id.
    let remote_parts: Vec<(usize, Vec<(usize, f32)>, Vec<f32>)> = match store.remote.as_ref() {
        Some(rt) => rt.serve_layer(layer, &mut experts, x, m),
        None => Vec::new(),
    };
    // ascending expert id: the sum is the same whatever served each expert
    let sum_parts = |mut parts: Vec<(usize, Vec<(usize, f32)>, Vec<f32>)>| -> Vec<f32> {
        parts.sort_by_key(|(e, _, _)| *e);
        let mut out = vec![0f32; m * n_embd];
        for (_, toks, y) in parts {
            for (i, (t, wt)) in toks.iter().enumerate() {
                let o = &mut out[t * n_embd..(t + 1) * n_embd];
                for (oj, yj) in o.iter_mut().zip(&y[i * n_embd..(i + 1) * n_embd]) {
                    *oj += wt * yj;
                }
            }
        }
        out
    };
    if experts.is_empty() {
        // every expert of this layer came from a host
        return (sum_parts(remote_parts), routes);
    }
    // the dynamic cache, when on, serves every expert of the layer (`cache.rs`)
    #[cfg(feature = "gpu")]
    if let (Some(gpu), Ok(mut guard)) = (store.gpu.as_ref(), store.cache.lock()) {
        if let Some(cache) = guard.as_mut() {
            match moe_cached(w, gate, up, down, x, m, gpu, cache, layer, stats, &experts) {
                Ok(outs) => {
                    let mut parts = remote_parts;
                    parts.extend(experts.into_iter().zip(outs).map(|((e, toks), y)| (e, toks, y)));
                    return (sum_parts(parts), routes);
                }
                Err(e) => panic!("expert cache, layer {layer}: {e}"),
            }
        }
    }
    let gather = |toks: &[(usize, f32)]| -> Vec<f32> {
        let mut xs = Vec::with_capacity(toks.len() * n_embd);
        for (t, _) in toks {
            xs.extend_from_slice(&x[t * n_embd..(t + 1) * n_embd]);
        }
        xs
    };
    let (gpu_part, mut cpu_part): (Vec<_>, Vec<_>) = experts.into_iter().partition(|(e, _)| store.is_resident(layer, *e));
    stats.gpu_hits.fetch_add(gpu_part.len() as u64, Ordering::Relaxed);
    // experts the staging ring holds: launched from it like residents
    #[cfg(feature = "gpu")]
    let staged_part: Vec<(usize, Vec<(usize, f32)>)> = match ring.as_ref() {
        Some(rc) => {
            let (st, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut cpu_part).into_iter().partition(|(e, _)| rc.ring.contains(layer, *e));
            cpu_part = rest;
            st
        }
        None => Vec::new(),
    };
    #[cfg(not(feature = "gpu"))]
    let staged_part: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
    let n_resident_part = gpu_part.len();

    // `MissPolicy::OnDemand`: take the misses away from the CPU and serve them by copying
    // the expert over. Each copy is a repack plus a host-to-device transfer, paid per call
    // and released after the readback, so this needs an expert's worth of free VRAM per
    // concurrent miss and nothing else.
    #[cfg(feature = "gpu")]
    let od_part: Vec<(usize, Vec<(usize, f32)>)> = if store.miss() == MissPolicy::OnDemand && store.gpu.is_some() { std::mem::take(&mut cpu_part) } else { Vec::new() };
    stats.cpu_calls.fetch_add(cpu_part.len() as u64, Ordering::Relaxed);

    // Resident experts: launch everything first (asynchronous), read back after the CPU work.
    #[cfg(feature = "gpu")]
    let gpu_launched: Vec<(usize, Vec<(usize, f32)>, (cubecl_handle::Handle, usize))> = {
        let t = std::time::Instant::now();
        let v = match store.gpu.as_ref() {
            Some(gpu) => {
                // Experts that drew the same tokens see byte-identical activations, so upload
                // once per distinct token set and share the handle. In decode (m = 1) every
                // expert in the layer draws token 0, so this is one upload per layer instead
                // of one per resident expert.
                let mut uploaded: HashMap<Vec<usize>, cubecl_handle::Handle> = HashMap::new();
                let mut v: Vec<(usize, Vec<(usize, f32)>, (cubecl_handle::Handle, usize))> = gpu_part
                    .iter()
                    .map(|(e, toks)| {
                        let key: Vec<usize> = toks.iter().map(|(t, _)| *t).collect();
                        let xh = uploaded.entry(key).or_insert_with(|| gpu.upload_f32(&gather(toks))).clone();
                        (*e, toks.clone(), store.resident[&(layer, *e)].launch(gpu, &xh, toks.len()))
                    })
                    .collect();
                if let Some(rc) = ring.as_mut() {
                    for (e, toks) in &staged_part {
                        let key: Vec<usize> = toks.iter().map(|(t, _)| *t).collect();
                        let xh = uploaded.entry(key).or_insert_with(|| gpu.upload_f32(&gather(toks))).clone();
                        let ex = rc.ring.take_use(layer, *e, rc.stats).expect("partitioned on contains");
                        v.push((*e, toks.clone(), ex.launch(gpu, &xh, toks.len())));
                    }
                }
                v
            }
            None => Vec::new(),
        };
        stats.gpu_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        v
    };
    // The transient experts have to outlive the readback, so they are held here and dropped
    // (freeing their device buffers) only once `read_all` has returned.
    #[cfg(feature = "gpu")]
    let (od_held, od_launched): (Vec<(usize, lb_q4_gpu::GpuExpert)>, Vec<(usize, Vec<(usize, f32)>, (cubecl_handle::Handle, usize))>) = {
        let t = std::time::Instant::now();
        let mut held = Vec::with_capacity(od_part.len());
        let mut launched = Vec::with_capacity(od_part.len());
        if let Some(gpu) = store.gpu.as_ref() {
            for (e, toks) in &od_part {
                match lb_q4_gpu::GpuExpert::upload(gpu, w.mat(&gate.expert(*e)), w.mat(&up.expert(*e)), w.mat(&down.expert(*e))) {
                    Ok(ex) => {
                        let xh = gpu.upload_f32(&gather(toks));
                        launched.push((*e, toks.clone(), ex.launch(gpu, &xh, toks.len())));
                        held.push((*e, ex));
                    }
                    // out of VRAM or an unsupported type: fall back to the CPU for this one
                    Err(_) => cpu_part.push((*e, toks.clone())),
                }
            }
        }
        stats.on_demand.fetch_add(launched.len() as u64, Ordering::Relaxed);
        stats.on_demand_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        (held, launched)
    };

    #[cfg(not(feature = "gpu"))]
    let _ = &gpu_part;

    // Decode counters, both arms (`stage.rs`): what served each of this token-layer's routings.
    // A CPU-served or copied-in expert is a demand fetch on the critical path.
    #[cfg(feature = "gpu")]
    let n_od = od_launched.len();
    #[cfg(not(feature = "gpu"))]
    let n_od = 0usize;
    if m == 1 {
        stats.stage.layer(layer, n_resident_part, staged_part.len(), n_od + cpu_part.len());
    }

    // The ring's refill for the next layer goes in now: this layer's work is already queued on
    // the main stream, so these side-stream copies overlap it and the read below does not wait
    // for them.
    #[cfg(feature = "gpu")]
    if let (Some(rc), Some(gpu)) = (ring.as_mut(), store.gpu.as_ref()) {
        if let Some(g) = rc.next.take() {
            let resident = |l: usize, e: usize| store.is_resident(l, e);
            rc.ring.prefetch(gpu, rc.w, rc.cfg, &g, &resident, rc.stats);
        }
    }

    // With every expert resident `cpu_part` is empty for all 38 MoE layers of every token, and
    // a rayon scope still costs something to set up and tear down -- 38 times a token, inside
    // the timed region. Skip it when there is nothing to run.
    // `mut` is only needed with the `gpu` feature, where the resident outputs are extended in.
    #[allow(unused_mut)]
    let mut parts: Vec<(usize, Vec<(usize, f32)>, Vec<f32>)> = if cpu_part.is_empty() {
        Vec::new()
    } else {
        let t = std::time::Instant::now();
        let v: Vec<(usize, Vec<(usize, f32)>, Vec<f32>)> = cpu_part
            .into_par_iter()
            .map(|(e, toks)| {
                let xs = gather(&toks);
                let y = expert_ffn(w, gate, up, down, e, &xs, toks.len());
                (e, toks, y)
            })
            .collect();
        stats.cpu_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        v
    };

    #[cfg(feature = "gpu")]
    if let Some(gpu) = store.gpu.as_ref() {
        let t = std::time::Instant::now();
        // One synchronisation for the residents and the copied-in experts together: the
        // scratch pool only recycles after a read, so splitting this into two reads would
        // cost a second round trip for no gain.
        let n_res = gpu_launched.len();
        let (keys, hs): (Vec<_>, Vec<_>) = gpu_launched.into_iter().chain(od_launched).map(|(e, toks, h)| ((e, toks), h)).unzip();
        let outs = gpu.read_all(hs).expect("GPU expert read");
        parts.extend(keys.into_iter().zip(outs).map(|((e, toks), y)| (e, toks, y)));
        let dt = t.elapsed().as_nanos() as u64;
        if m == 1 {
            stats.stage.moe_read_ns.fetch_add(dt, Ordering::Relaxed);
        }
        // charge the readback to whichever side actually used it
        if n_res > 0 {
            stats.gpu_ns.fetch_add(dt, Ordering::Relaxed);
        } else {
            stats.on_demand_ns.fetch_add(dt, Ordering::Relaxed);
        }
    }

    // demand misses the ring may keep (`LB_STAGE_KEEP_MISSES`), now that the read is done
    #[cfg(feature = "gpu")]
    match ring.as_mut() {
        Some(rc) if rc.cfg.keep_misses && m == 1 => {
            for (e, ex) in od_held {
                rc.ring.admit(layer, e, ex, rc.stats);
            }
        }
        _ => drop(od_held),
    }

    // ascending expert id: the sum is the same whatever served each expert (see above)
    parts.extend(remote_parts);
    (sum_parts(parts), routes)
}

/// One MoE layer through the dynamic cache: plan (hits promoted, misses take SLRU victims), queue
/// the fills (DMA + repack into fixed slots, same stream), launch every expert from its slot, one
/// read, sum in ascending expert id (the order `experts` is in). `x` is `m` rows of `n_embd`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn moe_cached(
    w: &Weights,
    gate: &ExpertMats,
    up: &ExpertMats,
    down: &ExpertMats,
    x: &[f32],
    m: usize,
    gpu: &lb_q4_gpu::Gpu,
    cache: &mut crate::cache::ExpertCache,
    layer: usize,
    stats: &PlacementStats,
    experts: &[(usize, Vec<(usize, f32)>)],
) -> Result<Vec<Vec<f32>>, String> {
    let n_embd = w.cfg.n_embd;
    let needed: Vec<usize> = experts.iter().map(|(e, _)| *e).collect();
    let t = std::time::Instant::now();
    cache.trace_pass("p", layer, 0, &needed, None);
    let plan = cache.slru.plan(layer, &needed).ok_or_else(|| format!("{} distinct experts > {} cache slots", needed.len(), cache.slru.n_slots()))?;
    for &(e, s) in &plan.fills {
        let raw = |r: &crate::weights::MatRef| w.mat(r).bytes;
        cache.slots.fill(gpu, cache.phys[s] as usize, [raw(&gate.expert(e)), raw(&up.expert(e)), raw(&down.expert(e))])?;
        cache.count_fill();
    }
    let fill_ns = t.elapsed().as_nanos() as u64;
    let (n_hit, n_fill) = (plan.hits.len(), plan.fills.len());
    stats.gpu_hits.fetch_add(n_hit as u64, Ordering::Relaxed);
    stats.on_demand.fetch_add(n_fill as u64, Ordering::Relaxed);
    stats.on_demand_ns.fetch_add(fill_ns, Ordering::Relaxed);
    if m == 1 {
        stats.stage.layer(layer, n_hit, 0, n_fill);
    }
    let t = std::time::Instant::now();
    let mut uploaded: HashMap<Vec<usize>, cubecl_handle::Handle> = HashMap::new();
    let mut launched = Vec::with_capacity(experts.len());
    for (e, toks) in experts {
        let s = cache.slru.slot(layer, *e).expect("planned");
        let key: Vec<usize> = toks.iter().map(|(t, _)| *t).collect();
        let xh = uploaded
            .entry(key)
            .or_insert_with(|| {
                let mut xs = Vec::with_capacity(toks.len() * n_embd);
                for (t, _) in toks {
                    xs.extend_from_slice(&x[t * n_embd..(t + 1) * n_embd]);
                }
                gpu.upload_f32(&xs)
            })
            .clone();
        launched.push(cache.slots.expert(cache.phys[s] as usize).launch(gpu, &xh, toks.len()));
    }
    let outs = gpu.read_all(launched)?;
    let dt = t.elapsed().as_nanos() as u64;
    stats.gpu_ns.fetch_add(dt, Ordering::Relaxed);
    if m == 1 {
        stats.stage.moe_read_ns.fetch_add(dt, Ordering::Relaxed);
    }
    // one output per expert, in the order of `experts`; the caller weights and sums them (with
    // any that came from expert hosts) in ascending expert id
    Ok(outs)
}

/// Read every `run-*.json` scrape file in `dir` (unreadable files are skipped).
pub fn read_scrape_dir(dir: &std::path::Path) -> std::io::Result<Vec<Scrape>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("run-") && name.ends_with(".json") {
            if let Ok(s) = serde_json::from_str::<Scrape>(&std::fs::read_to_string(&p)?) {
                out.push(s);
            }
        }
    }
    Ok(out)
}

// ---- NautiBurn routing scrape (schema_version 1) ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Activation {
    pub layer: u32,
    pub token_pos: u32,
    pub expert_ids: Vec<u32>,
}

/// One scrape file, the format NautiBurn's consumer ingests
/// (`run-<epoch>-<seq>.json`, `schema_version: 1`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scrape {
    pub schema_version: u32,
    pub model_id: String,
    pub request_id: String,
    pub class_id: String,
    pub source: String,
    pub phase: String,
    pub n_tokens: u32,
    pub dtype: String,
    pub token_ids: Vec<u32>,
    pub activations: Vec<Activation>,
}

/// Collects routes during a forward pass and writes a scrape file per request when
/// `NAUTIBURN_SCRAPE_DIR` is set (`NAUTIBURN_MODEL_ID`, `NAUTIBURN_CLASS_ID` optional).
pub struct Scraper {
    /// Where files go; `None` disables recording. Live: the server can point it elsewhere or
    /// switch it off between requests.
    dir: Mutex<Option<PathBuf>>,
    model_id: Mutex<String>,
    class_id: Mutex<String>,
    seq: AtomicU64,
    pending: Mutex<Vec<Activation>>,
}

impl Scraper {
    pub fn from_env(default_model_id: &str) -> Self {
        Self::new(
            std::env::var_os("NAUTIBURN_SCRAPE_DIR").map(PathBuf::from),
            std::env::var("NAUTIBURN_MODEL_ID").unwrap_or_else(|_| default_model_id.to_string()),
            std::env::var("NAUTIBURN_CLASS_ID").unwrap_or_else(|_| "lb-lfm2".to_string()),
        )
    }

    pub fn new(dir: Option<PathBuf>, model_id: String, class_id: String) -> Self {
        Self { dir: Mutex::new(dir), model_id: Mutex::new(model_id), class_id: Mutex::new(class_id), seq: AtomicU64::new(0), pending: Mutex::new(Vec::new()) }
    }

    pub fn enabled(&self) -> bool {
        self.dir.lock().unwrap().is_some()
    }

    pub fn dir(&self) -> Option<PathBuf> {
        self.dir.lock().unwrap().clone()
    }

    /// Point the scraper at `dir` (`None` switches recording off). Takes effect at the next
    /// flush; activations already pending go to the new place.
    pub fn set_dir(&self, dir: Option<PathBuf>) {
        *self.dir.lock().unwrap() = dir;
    }

    pub fn model_id(&self) -> String {
        self.model_id.lock().unwrap().clone()
    }

    pub fn set_model_id(&self, id: String) {
        *self.model_id.lock().unwrap() = id;
    }

    pub fn class_id(&self) -> String {
        self.class_id.lock().unwrap().clone()
    }

    pub fn set_class_id(&self, id: String) {
        *self.class_id.lock().unwrap() = id;
    }

    pub fn record(&self, layer: usize, pos0: usize, routes: &[Route]) {
        if !self.enabled() {
            return;
        }
        let mut p = self.pending.lock().unwrap();
        for (i, r) in routes.iter().enumerate() {
            p.push(Activation {
                layer: layer as u32,
                token_pos: (pos0 + i) as u32,
                expert_ids: r.experts.iter().map(|&e| e as u32).collect(),
            });
        }
    }

    /// Take the recorded activations (also used by tests and by the histogram builder).
    pub fn take(&self) -> Vec<Activation> {
        std::mem::take(&mut *self.pending.lock().unwrap())
    }

    /// Write the pending activations as one scrape file; returns its path when enabled.
    pub fn flush(&self, phase: &str, token_ids: &[u32]) -> std::io::Result<Option<PathBuf>> {
        let acts = self.take();
        let Some(dir) = self.dir() else { return Ok(None) };
        if acts.is_empty() {
            return Ok(None);
        }
        let epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let s = Scrape {
            schema_version: 1,
            model_id: self.model_id(),
            request_id: format!("lb-lfm2-{epoch}-{seq}"),
            class_id: self.class_id(),
            source: "scrape".into(),
            phase: phase.into(),
            n_tokens: token_ids.len() as u32,
            dtype: "gguf".into(),
            token_ids: token_ids.to_vec(),
            activations: acts,
        };
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("run-{epoch}-{seq}.json"));
        let tmp = dir.join(format!(".run-{epoch}-{seq}.json.tmp"));
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(serde_json::to_string(&s)?.as_bytes())?;
        std::fs::rename(&tmp, &path)?;
        Ok(Some(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_selects_by_biased_score_but_weights_by_prob() {
        // 3 experts, n_embd = 1. logits = x * w
        let router = [1.0f32, 2.0, -5.0];
        let bias = [0.0f32, 0.0, 10.0]; // expert 2 forced in by bias
        let r = route_sigmoid(&router, &bias, &[1.0], 3, 2);
        assert_eq!(r.experts, vec![2, 1]);
        let p1 = 1.0 / (1.0 + (-2.0f32).exp());
        let p2 = 1.0 / (1.0 + 5.0f32.exp());
        let sum = p1 + p2;
        assert!((r.weights[0] - p2 / sum).abs() < 1e-6);
        assert!((r.weights[1] - p1 / sum).abs() < 1e-6);
        assert!((r.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn placement_spends_budget_on_hottest_experts() {
        let hist = Histogram { model_id: "t".into(), counts: vec![vec![10, 0, 5, 1], vec![0, 0, 0, 0], vec![3, 3, 3, 3]] };
        let p = Placement::plan(&hist, &[100, 100, 100], 250);
        assert_eq!(p.home(0, 0), Home::Gpu);
        assert_eq!(p.home(0, 2), Home::Gpu);
        assert_eq!(p.home(1, 0), Home::Cpu); // no data: nothing resident
        assert_eq!(p.n_resident(), 2);
        assert_eq!(p.resident_bytes, 200);
    }

    #[test]
    fn leftover_budget_goes_to_experts_the_scrapes_never_saw() {
        // Only layer 0 and layer 2 have routing data, and only some of their experts, so
        // the histogram alone can spend at most 700 of the 1000 available.
        let hist = Histogram { model_id: "t".into(), counts: vec![vec![10, 0, 5, 1], vec![0, 0, 0, 0], vec![3, 3, 3, 3]] };
        let p = Placement::plan(&hist, &[100, 100, 100], 1000);
        assert_eq!(p.resident_bytes, 1000, "the whole budget should be spent");
        assert_eq!(p.n_resident(), 10);
        // the unseen layer gets residents now rather than leaving the card idle
        assert_eq!(p.home(1, 0), Home::Gpu);
        // The fill is round robin by layer, so the 3 slots the histogram left over do not
        // all land in one layer: the unseen layer takes 2 of them and none stays empty.
        let per_layer: Vec<usize> = (0..3).map(|l| (0..4).filter(|&e| p.home(l, e) == Home::Gpu).count()).collect();
        assert_eq!(per_layer, vec![4, 2, 4]);
        // a layer whose FFN is dense (zero bytes per expert) is never staged
        let dense = Placement::plan(&hist, &[100, 0, 100], 1000);
        assert_eq!(dense.home(1, 0), Home::Cpu);
        assert_eq!(dense.resident_bytes, 800);
    }

    #[test]
    fn whole_layer_placement_is_a_prefix_of_whole_layers() {
        // layer 0 dense (0 bytes), then four MoE layers of 3 experts x 100 bytes
        let bytes = [0usize, 100, 100, 100, 100];
        let p = Placement::whole_layers(3, &bytes, 700);
        assert_eq!(p.resident_bytes, 600, "two whole layers fit in 700; a third would need 900");
        assert!(!p.layer_whole(0), "a dense layer is never whole");
        assert!(p.layer_whole(1) && p.layer_whole(2));
        assert!(!p.layer_whole(3) && !p.layer_whole(4));
        assert_eq!(p.home(3, 0), Home::Cpu, "no partial layer: the leftover budget stays unspent");
        assert_eq!(Placement::whole_layers(3, &bytes, 10_000).n_resident(), 12);
        assert_eq!(Placement::whole_layers(3, &bytes, 299).n_resident(), 0);
    }

    #[test]
    fn scrape_round_trips_through_histogram() {
        let s = Scraper::new(None, "m".into(), "c".into());
        // recording is off with no directory: nothing pends, nothing costs
        s.record(3, 10, &[Route { experts: vec![1, 2], weights: vec![0.5, 0.5] }]);
        assert!(s.take().is_empty());
        s.set_dir(Some(std::path::PathBuf::from("/nonexistent-for-this-test")));
        assert!(s.enabled());
        s.record(3, 10, &[Route { experts: vec![1, 2], weights: vec![0.5, 0.5] }]);
        let acts = s.take();
        assert_eq!(acts[0].token_pos, 10);
        let sc = Scrape {
            schema_version: 1,
            model_id: "m".into(),
            request_id: "r".into(),
            class_id: "c".into(),
            source: "scrape".into(),
            phase: "prefill".into(),
            n_tokens: 1,
            dtype: "gguf".into(),
            token_ids: vec![],
            activations: acts,
        };
        let h = Histogram::from_scrapes("m", 4, 4, &[sc]);
        assert_eq!(h.counts[3], vec![0, 1, 1, 0]);
    }
}
