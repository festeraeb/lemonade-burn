//! Mixture-of-experts pieces: the router, the expert store (where each expert's weights
//! are executed), scrape-driven placement, and the NautiBurn routing scrape writer.
//!
//! Router (LFM2-MoE, `expert_gating_func = 2`): `logits = x W_r^T`, `p = sigmoid(logits)`,
//! experts are *selected* by `p + bias` (top-k), and *weighted* by the unbiased `p` of the
//! selected experts, normalized to sum to one (sum clamped to >= 2^-14), times the
//! routed scaling factor (1.0 for LFM2.5). Each expert is a SwiGLU FFN:
//! `down(silu(gate(x)) * up(x))`.

use crate::weights::{ExpertMats, Weights};
use lb_gguf::matvec_f32;
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
pub enum MissPolicy {
    /// Run the miss on the CPU reference path.
    #[default]
    Cpu,
    /// Upload the expert to the device, run it there, release it after the readback.
    OnDemand,
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
}

/// Experts resident on the GPU, keyed by (layer, expert). Empty without the `gpu` feature.
#[derive(Default)]
pub struct ExpertStore {
    #[cfg(feature = "gpu")]
    pub gpu: Option<lb_q4_gpu::Gpu>,
    #[cfg(feature = "gpu")]
    pub resident: HashMap<(usize, usize), lb_q4_gpu::GpuExpert>,
    pub bytes: usize,
    /// How to serve an expert that is not resident.
    pub miss: MissPolicy,
}

impl ExpertStore {
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
            for e in 0..gate.n {
                if placement.home(l, e) != Home::Gpu || self.resident.contains_key(&(l, e)) {
                    continue;
                }
                let ex = lb_q4_gpu::GpuExpert::upload(gpu, w.mat(&gate.expert(e)), w.mat(&up.expert(e)), w.mat(&down.expert(e)))?;
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

/// Executes an expert FFN for a batch of tokens. One SwiGLU expert:
/// `down(silu(gate x) * up x)`. `x` is `m` rows of `n_embd`; returns `m` rows of `n_embd`.
pub fn expert_ffn(w: &Weights, gate: &ExpertMats, up: &ExpertMats, down: &ExpertMats, e: usize, x: &[f32], m: usize) -> Vec<f32> {
    let n_ff = gate.rows;
    let n_embd = down.rows;
    let mut g = vec![0f32; m * n_ff];
    let mut u = vec![0f32; m * n_ff];
    matvec_f32(w.mat(&gate.expert(e)), x, m, &mut g);
    matvec_f32(w.mat(&up.expert(e)), x, m, &mut u);
    for (gi, ui) in g.iter_mut().zip(&u) {
        let s = *gi / (1.0 + (-*gi).exp());
        *gi = s * ui;
    }
    let mut out = vec![0f32; m * n_embd];
    matvec_f32(w.mat(&down.expert(e)), &g, m, &mut out);
    out
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
    let gather = |toks: &[(usize, f32)]| -> Vec<f32> {
        let mut xs = Vec::with_capacity(toks.len() * n_embd);
        for (t, _) in toks {
            xs.extend_from_slice(&x[t * n_embd..(t + 1) * n_embd]);
        }
        xs
    };
    let (gpu_part, mut cpu_part): (Vec<_>, Vec<_>) = experts.into_iter().partition(|(e, _)| store.is_resident(layer, *e));
    stats.gpu_hits.fetch_add(gpu_part.len() as u64, Ordering::Relaxed);

    // `MissPolicy::OnDemand`: take the misses away from the CPU and serve them by copying
    // the expert over. Each copy is a repack plus a host-to-device transfer, paid per call
    // and released after the readback, so this needs an expert's worth of free VRAM per
    // concurrent miss and nothing else.
    #[cfg(feature = "gpu")]
    let od_part: Vec<(usize, Vec<(usize, f32)>)> = if store.miss == MissPolicy::OnDemand && store.gpu.is_some() { std::mem::take(&mut cpu_part) } else { Vec::new() };
    stats.cpu_calls.fetch_add(cpu_part.len() as u64, Ordering::Relaxed);

    // Resident experts: launch everything first (asynchronous), read back after the CPU work.
    #[cfg(feature = "gpu")]
    let gpu_launched: Vec<(Vec<(usize, f32)>, (cubecl_handle::Handle, usize))> = {
        let t = std::time::Instant::now();
        let v = match store.gpu.as_ref() {
            Some(gpu) => {
                // Experts that drew the same tokens see byte-identical activations, so upload
                // once per distinct token set and share the handle. In decode (m = 1) every
                // expert in the layer draws token 0, so this is one upload per layer instead
                // of one per resident expert.
                let mut uploaded: HashMap<Vec<usize>, cubecl_handle::Handle> = HashMap::new();
                gpu_part
                    .iter()
                    .map(|(e, toks)| {
                        let key: Vec<usize> = toks.iter().map(|(t, _)| *t).collect();
                        let xh = uploaded.entry(key).or_insert_with(|| gpu.upload_f32(&gather(toks))).clone();
                        (toks.clone(), store.resident[&(layer, *e)].launch(gpu, &xh, toks.len()))
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        stats.gpu_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        v
    };
    // The transient experts have to outlive the readback, so they are held here and dropped
    // (freeing their device buffers) only once `read_all` has returned.
    #[cfg(feature = "gpu")]
    let (_od_held, od_launched): (Vec<lb_q4_gpu::GpuExpert>, Vec<(Vec<(usize, f32)>, (cubecl_handle::Handle, usize))>) = {
        let t = std::time::Instant::now();
        let mut held = Vec::with_capacity(od_part.len());
        let mut launched = Vec::with_capacity(od_part.len());
        if let Some(gpu) = store.gpu.as_ref() {
            for (e, toks) in &od_part {
                match lb_q4_gpu::GpuExpert::upload(gpu, w.mat(&gate.expert(*e)), w.mat(&up.expert(*e)), w.mat(&down.expert(*e))) {
                    Ok(ex) => {
                        let xh = gpu.upload_f32(&gather(toks));
                        launched.push((toks.clone(), ex.launch(gpu, &xh, toks.len())));
                        held.push(ex);
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

    // With every expert resident `cpu_part` is empty for all 38 MoE layers of every token, and
    // a rayon scope still costs something to set up and tear down -- 38 times a token, inside
    // the timed region. Skip it when there is nothing to run.
    // `mut` is only needed with the `gpu` feature, where the resident outputs are extended in.
    #[allow(unused_mut)]
    let mut parts: Vec<(Vec<(usize, f32)>, Vec<f32>)> = if cpu_part.is_empty() {
        Vec::new()
    } else {
        let t = std::time::Instant::now();
        let v: Vec<(Vec<(usize, f32)>, Vec<f32>)> = cpu_part
            .into_par_iter()
            .map(|(e, toks)| {
                let xs = gather(&toks);
                let y = expert_ffn(w, gate, up, down, e, &xs, toks.len());
                (toks, y)
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
        let (toks, hs): (Vec<_>, Vec<_>) = gpu_launched.into_iter().chain(od_launched).unzip();
        let outs = gpu.read_all(hs).expect("GPU expert read");
        parts.extend(toks.into_iter().zip(outs));
        let dt = t.elapsed().as_nanos() as u64;
        // charge the readback to whichever side actually used it
        if n_res > 0 {
            stats.gpu_ns.fetch_add(dt, Ordering::Relaxed);
        } else {
            stats.on_demand_ns.fetch_add(dt, Ordering::Relaxed);
        }
    }

    let mut out = vec![0f32; m * n_embd];
    for (toks, y) in parts {
        for (i, (t, wt)) in toks.iter().enumerate() {
            let o = &mut out[t * n_embd..(t + 1) * n_embd];
            for (oj, yj) in o.iter_mut().zip(&y[i * n_embd..(i + 1) * n_embd]) {
                *oj += wt * yj;
            }
        }
    }
    (out, routes)
}

/// Read every `run-*.json` scrape file in `dir` (unreadable files are skipped).
pub fn read_scrape_dir(dir: &std::path::Path) -> std::io::Result<Vec<Scrape>> {
    // `read_dir` order is unspecified: sort by file name so the same files always give the
    // same sequence (callers split train/held-out by index).
    let mut paths = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("run-") && name.ends_with(".json") {
            paths.push(p);
        }
    }
    paths.sort();
    let mut out = Vec::new();
    for p in paths {
        if let Ok(s) = serde_json::from_str::<Scrape>(&std::fs::read_to_string(&p)?) {
            out.push(s);
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
    dir: Option<PathBuf>,
    model_id: String,
    class_id: String,
    seq: AtomicU64,
    pending: Mutex<Vec<Activation>>,
}

impl Scraper {
    pub fn from_env(default_model_id: &str) -> Self {
        Self {
            dir: std::env::var_os("NAUTIBURN_SCRAPE_DIR").map(PathBuf::from),
            model_id: std::env::var("NAUTIBURN_MODEL_ID").unwrap_or_else(|_| default_model_id.to_string()),
            class_id: std::env::var("NAUTIBURN_CLASS_ID").unwrap_or_else(|_| "lb-lfm2".to_string()),
            seq: AtomicU64::new(0),
            pending: Mutex::new(Vec::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    pub fn record(&self, layer: usize, pos0: usize, routes: &[Route]) {
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
        self.write(phase, token_ids, acts)
    }

    /// Write one request's pending activations as up to two correctly labelled files: the
    /// positions inside the prompt as `prefill` (`token_ids` = the prompt), and the later
    /// positions as `decode` (`token_ids` = prompt plus the generated tokens that were fed
    /// back, so every row's `token_pos` is below `n_tokens`). Returns the paths written.
    pub fn flush_request(&self, prompt: &[u32], generated: &[u32]) -> std::io::Result<Vec<PathBuf>> {
        let acts = self.take();
        let n_prompt = prompt.len() as u32;
        let (pre, dec): (Vec<Activation>, Vec<Activation>) = acts.into_iter().partition(|a| a.token_pos < n_prompt);
        let mut out = Vec::new();
        out.extend(self.write("prefill", prompt, pre)?);
        if let Some(n_total) = dec.iter().map(|a| a.token_pos as usize + 1).max() {
            let history: Vec<u32> = prompt.iter().chain(generated).copied().take(n_total).collect();
            out.extend(self.write("decode", &history, dec)?);
        }
        Ok(out)
    }

    fn write(&self, phase: &str, token_ids: &[u32], acts: Vec<Activation>) -> std::io::Result<Option<PathBuf>> {
        let Some(dir) = &self.dir else { return Ok(None) };
        if acts.is_empty() {
            return Ok(None);
        }
        let epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let s = Scrape {
            schema_version: 1,
            model_id: self.model_id.clone(),
            request_id: format!("lb-lfm2-{epoch}-{seq}"),
            class_id: self.class_id.clone(),
            source: "scrape".into(),
            phase: phase.into(),
            n_tokens: token_ids.len() as u32,
            dtype: "gguf".into(),
            token_ids: token_ids.to_vec(),
            activations: acts,
        };
        std::fs::create_dir_all(dir)?;
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
    fn scrape_round_trips_through_histogram() {
        let s = Scraper { dir: None, model_id: "m".into(), class_id: "c".into(), seq: AtomicU64::new(0), pending: Mutex::new(Vec::new()) };
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

    #[test]
    fn read_scrape_dir_is_sorted_by_file_name() {
        let dir = std::env::temp_dir().join(format!("lb-scrape-order-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..40u32 {
            // written in a scrambled order; file name order is the id order
            let n = (i * 17) % 40;
            let sc = Scrape {
                schema_version: 1,
                model_id: "m".into(),
                request_id: format!("req{n:02}"),
                class_id: "c".into(),
                source: "scrape".into(),
                phase: "prefill".into(),
                n_tokens: 0,
                dtype: "gguf".into(),
                token_ids: vec![],
                activations: vec![],
            };
            std::fs::write(dir.join(format!("run-1000-{n:04}.json")), serde_json::to_string(&sc).unwrap()).unwrap();
        }
        let got: Vec<String> = read_scrape_dir(&dir).unwrap().into_iter().map(|s| s.request_id).collect();
        let want: Vec<String> = (0..40).map(|n| format!("req{n:02}")).collect();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got, want);
    }

    #[test]
    fn flush_request_splits_prefill_and_decode_with_matching_labels() {
        let dir = std::env::temp_dir().join(format!("lb-scrape-split-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = Scraper { dir: Some(dir.clone()), model_id: "m".into(), class_id: "c".into(), seq: AtomicU64::new(0), pending: Mutex::new(Vec::new()) };
        let r = Route { experts: vec![1, 2], weights: vec![0.5, 0.5] };
        // prompt of 3 tokens (positions 0..3), then two decode steps at positions 3 and 4
        s.record(2, 0, &[r.clone(), r.clone(), r.clone()]);
        s.record(2, 3, &[r.clone()]);
        s.record(2, 4, &[r.clone()]);
        let paths = s.flush_request(&[10, 11, 12], &[20, 21, 22]).unwrap();
        assert_eq!(paths.len(), 2);
        let read = |p: &PathBuf| serde_json::from_str::<Scrape>(&std::fs::read_to_string(p).unwrap()).unwrap();
        let (pre, dec) = (read(&paths[0]), read(&paths[1]));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!((pre.phase.as_str(), pre.n_tokens, pre.token_ids.clone()), ("prefill", 3, vec![10, 11, 12]));
        assert_eq!(pre.activations.len(), 3);
        // decode: positions 3 and 4 were fed tokens 20 and 21; 22 was never forwarded
        assert_eq!((dec.phase.as_str(), dec.n_tokens, dec.token_ids.clone()), ("decode", 5, vec![10, 11, 12, 20, 21]));
        assert_eq!(dec.activations.iter().map(|a| a.token_pos).collect::<Vec<_>>(), vec![3, 4]);
        for sc in [&pre, &dec] {
            assert!(sc.activations.iter().all(|a| a.token_pos < sc.n_tokens && sc.n_tokens as usize == sc.token_ids.len()));
        }
        assert!(s.take().is_empty(), "everything pending was consumed");
    }
}
