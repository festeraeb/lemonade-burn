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
    /// Layers with no routing data get no residents (CPU is the safe default).
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
        p
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
}

/// Experts resident on the GPU, keyed by (layer, expert). Empty without the `gpu` feature.
#[derive(Default)]
pub struct ExpertStore {
    #[cfg(feature = "gpu")]
    pub gpu: Option<lb_q4_gpu::Gpu>,
    #[cfg(feature = "gpu")]
    pub resident: HashMap<(usize, usize), lb_q4_gpu::GpuExpert>,
    pub bytes: usize,
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

    /// Upload every `Home::Gpu` expert of `placement` to the device chosen by
    /// `LB_WGPU_DEVICE`. Returns the number uploaded. Without the `gpu` feature: an error.
    #[cfg(feature = "gpu")]
    pub fn load(&mut self, w: &Weights, placement: &Placement) -> Result<usize, String> {
        use crate::weights::Ffn;
        if self.gpu.is_none() {
            self.gpu = Some(lb_q4_gpu::Gpu::open()?);
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
    let (gpu_part, cpu_part): (Vec<_>, Vec<_>) = experts.into_iter().partition(|(e, _)| store.is_resident(layer, *e));
    stats.gpu_hits.fetch_add(gpu_part.len() as u64, Ordering::Relaxed);
    stats.cpu_calls.fetch_add(cpu_part.len() as u64, Ordering::Relaxed);

    // Resident experts: launch everything first (asynchronous), read back after the CPU work.
    #[cfg(feature = "gpu")]
    let gpu_launched: Vec<(Vec<(usize, f32)>, (cubecl_handle::Handle, usize))> = {
        let t = std::time::Instant::now();
        let v = match store.gpu.as_ref() {
            Some(gpu) => gpu_part
                .iter()
                .map(|(e, toks)| {
                    let xh = gpu.upload_f32(&gather(toks));
                    (toks.clone(), store.resident[&(layer, *e)].launch(gpu, &xh, toks.len()))
                })
                .collect(),
            None => Vec::new(),
        };
        stats.gpu_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        v
    };
    #[cfg(not(feature = "gpu"))]
    let _ = &gpu_part;

    let t = std::time::Instant::now();
    #[allow(unused_mut)]
    let mut parts: Vec<(Vec<(usize, f32)>, Vec<f32>)> = cpu_part
        .into_par_iter()
        .map(|(e, toks)| {
            let xs = gather(&toks);
            let y = expert_ffn(w, gate, up, down, e, &xs, toks.len());
            (toks, y)
        })
        .collect();
    stats.cpu_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    #[cfg(feature = "gpu")]
    if let Some(gpu) = store.gpu.as_ref() {
        let t = std::time::Instant::now();
        let (toks, hs): (Vec<_>, Vec<_>) = gpu_launched.into_iter().unzip();
        let outs = gpu.read_all(hs).expect("GPU expert read");
        parts.extend(toks.into_iter().zip(outs));
        stats.gpu_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
}
