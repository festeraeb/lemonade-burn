//! The layout search. For each node (and optional helper node) and each prefix of its cards:
//! 1. `place`: split layers over the cards in proportion to free VRAM (this equalises the
//!    resident experts per layer, which is what starved the last card of the 12,24,35 split
//!    in the Le Chonk run), fill VRAM -> helper VRAM -> RAM tier -> disk tiers -> model file.
//! 2. `evaluate`: hand the resident sets to the simulator (sim.rs) on a synthetic routing
//!    trace, add helper round trips and tier/file reads, and report a low / expected / high
//!    tokens/s range.

use super::arch::{self, ArchInfo};
use super::model::ModelDesc;
use super::*;
use crate::sim::{simulate, MissPolicy, SimConfig, SimResult};
use crate::timings::{CardTimings, HostTimings, EXPERT_BYTES};
use crate::trace::{Request, TokenRoute, Trace};
use crate::tune::{CardAssign, StagingConfig};
use std::collections::HashMap;

/// Fraction of a card's memory bandwidth the engine's expert kernel reaches. Derived from the
/// c3 measurements of 2026-11-03 (fleet-gpu-inventory.md): RX 6700 XT 0.070 ms per 5.3 MB
/// expert = 76 GB/s of 384 (0.20); WX 5100 0.138 ms = 38 GB/s of 160 (0.24). Mean 0.22.
pub const KERNEL_EFF: f64 = 0.22;
/// CPU expert rate per thread, GB/s: 5.308 MB / 13.4 ms, 1 thread on c4 (V2_PIPELINE.md s3).
pub const CPU_GB_S_PER_THREAD: f64 = 0.396;
/// Measured parallel speedup of the CPU expert path on c4: 13.4 ms / 1.088 ms (80 threads).
pub const CPU_MAX_SPEEDUP: f64 = 12.3;
/// Host->device rate assumed when the PCIe link is unknown: Gen3 x8 (measured ~6 GB/s).
pub const PCIE_FALLBACK_MB_S: f64 = 6000.0;

/// Cumulative share of routed-expert uses covered by the hottest fraction of experts per layer.
#[derive(Clone, Debug)]
pub struct SkewCurve {
    pub name: String,
    pub pts: Vec<(f64, f64)>,
    pub source: String,
}

impl SkewCurve {
    pub fn uniform() -> Self {
        Self { name: "uniform".into(), pts: vec![(0.0, 0.0), (1.0, 1.0)], source: "assumption: every expert equally likely (pessimistic bound)".into() }
    }
    /// Le Chonk (Mistral Large 4, 128 experts/layer, top-4, 284 uses/token) pin plan:
    /// per-layer top-48 -> 24 file reads/token, top-64 -> 11, top-96 -> 1.8 (memory note
    /// le-chonk-polystrata, /mnt/fast/ml4-rank). Coverage = 1 - reads/284.
    pub fn profile() -> Self {
        Self {
            name: "profile".into(),
            pts: vec![(0.0, 0.0), (0.375, 1.0 - 24.0 / 284.0), (0.5, 1.0 - 11.0 / 284.0), (0.75, 1.0 - 1.8 / 284.0), (1.0, 1.0)],
            source: "measured on Le Chonk (one model, one prompt mix); assumed for other MoE models: replace with --scrapes".into(),
        }
    }
    /// From a scrape trace: rank experts per layer by count, average the cumulative coverage.
    pub fn from_trace(t: &Trace) -> Self {
        let h = t.histogram();
        let ne = t.n_expert.max(1);
        let mut cum = vec![0f64; ne + 1];
        let (mut layers, mut total) = (0usize, 0f64);
        for row in &h {
            let s: u64 = row.iter().sum();
            if s == 0 {
                continue;
            }
            layers += 1;
            let mut v = row.clone();
            v.sort_by(|a, b| b.cmp(a));
            let mut acc = 0u64;
            for (i, c) in v.iter().enumerate() {
                acc += c;
                cum[i + 1] += acc as f64 / s as f64;
            }
            total += 1.0;
        }
        let pts = (0..=ne).map(|i| (i as f64 / ne as f64, if total > 0.0 { cum[i] / total } else { i as f64 / ne as f64 })).collect();
        let _ = layers;
        Self { name: "trace".into(), pts, source: format!("measured: {} routed tokens from scrape traces", t.n_tokens()) }
    }
    pub fn by_name(n: &str) -> Self {
        if n == "uniform" {
            Self::uniform()
        } else {
            Self::profile()
        }
    }
    pub fn cov(&self, f: f64) -> f64 {
        let f = f.clamp(0.0, 1.0);
        for w in self.pts.windows(2) {
            if f <= w[1].0 {
                let t = if w[1].0 > w[0].0 { (f - w[0].0) / (w[1].0 - w[0].0) } else { 0.0 };
                return w[0].1 + t * (w[1].1 - w[0].1);
            }
        }
        1.0
    }
    /// Inverse: the expert fraction whose coverage is `u`.
    pub fn inv(&self, u: f64) -> f64 {
        for w in self.pts.windows(2) {
            if u <= w[1].1 {
                let t = if w[1].1 > w[0].1 { (u - w[0].1) / (w[1].1 - w[0].1) } else { 0.0 };
                return w[0].0 + t * (w[1].0 - w[0].0);
            }
        }
        1.0
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CardPlan {
    pub node: String,
    pub gpu_index: usize,
    pub name: String,
    pub arch: Option<String>,
    pub has_dp4a: Option<bool>,
    pub layer_start: usize,
    pub layer_end: usize,
    pub dense_mib: f64,
    pub kv_mib: f64,
    pub reserve_mib: f64,
    pub resident_per_layer: usize,
    pub resident_experts: usize,
    /// "measured (bench file)" or "estimate: arch table".
    pub timing_source: String,
    pub kernel_note: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HelperPlan {
    pub node: String,
    pub link_kind: LinkKind,
    pub link_gb_s: Option<f64>,
    pub link_latency_us: Option<f64>,
    pub gpus: Vec<String>,
    pub experts: usize,
    pub per_layer: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TierUse {
    pub kind: TierKind,
    pub path: Option<String>,
    pub experts: u64,
    pub gib: f64,
    pub read_mb_s: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct Expect {
    pub tok_s_low: Option<f64>,
    pub tok_s: Option<f64>,
    pub tok_s_high: Option<f64>,
    /// First requests after a load: RAM and disk tiers still empty, every non-resident expert read from the model file.
    #[serde(default)]
    pub tok_s_cold: Option<f64>,
    pub ms_sim: f64,
    pub ms_helper: f64,
    pub ms_io_serial: f64,
    pub ms_io_overlap: f64,
    pub uses_per_token: f64,
    /// Share of routed uses served by: local VRAM, helper VRAM, RAM, disk tiers, model file.
    pub share_vram: f64,
    pub share_helper: f64,
    pub share_ram: f64,
    pub share_tier: f64,
    pub share_file: f64,
    pub skew: String,
    pub basis: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Layout {
    pub node: String,
    pub cards: Vec<CardPlan>,
    pub layer_split: Vec<usize>,
    pub ram_tier_gib: f64,
    pub ram_experts: u64,
    pub tiers: Vec<TierUse>,
    pub file_experts: u64,
    pub model_store_mb_s: Option<f64>,
    pub helper: Option<HelperPlan>,
    pub expect: Expect,
    pub residency: String,
    pub explain: Vec<String>,
    pub unknowns_used: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct FleetPlan {
    #[serde(flatten)]
    pub staging: StagingConfig,
    pub layout: Layout,
}

/// Placement before simulation.
#[derive(Clone, Debug)]
pub struct Placement {
    pub node: String,
    pub gpu_idx: Vec<usize>,
    pub gpus: Vec<GpuDesc>,
    pub infos: Vec<Option<ArchInfo>>,
    pub layers: Vec<usize>,
    pub k_c: Vec<usize>,
    pub helper_node: Option<String>,
    pub helper_gpus: Vec<String>,
    pub helper_experts: usize,
    pub kh: usize,
    pub ram_experts: u64,
    pub tier_experts: Vec<TierUse>,
    pub file_experts: u64,
    pub ram_gib: f64,
    pub dl_mib: f64,
    pub kv_layer_mib: f64,
    pub notes: Vec<String>,
}

pub fn usable_gpus(n: &NodeDesc) -> Vec<(usize, GpuDesc, Option<ArchInfo>)> {
    let mut v: Vec<(usize, GpuDesc, Option<ArchInfo>)> = n
        .gpus
        .iter()
        .enumerate()
        .filter(|(_, g)| g.available && g.vram_mib > 0)
        .map(|(i, g)| {
            let (r, info) = arch::resolve(g);
            (i, r, info)
        })
        .collect();
    v.sort_by(|a, b| b.1.vram_mib.cmp(&a.1.vram_mib).then(a.0.cmp(&b.0)));
    v
}

/// Largest-remainder split of `l` layers in proportion to `free`, then repair so that no card
/// holds more than it can (`count * dl <= free`). `None` if the cards cannot hold the dense
/// layers or a card would get no layer.
pub fn split_layers(free: &[f64], dl: f64, l: usize) -> Option<Vec<usize>> {
    let tot: f64 = free.iter().sum();
    if tot <= 0.0 || free.iter().any(|&f| f <= 0.0) {
        return None;
    }
    let raw: Vec<f64> = free.iter().map(|f| l as f64 * f / tot).collect();
    let mut cnt: Vec<usize> = raw.iter().map(|r| r.floor() as usize).collect();
    let mut rem = l - cnt.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..free.len()).collect();
    order.sort_by(|&a, &b| (raw[b] - raw[b].floor()).partial_cmp(&(raw[a] - raw[a].floor())).unwrap());
    for &i in order.iter().cycle().take(rem.max(1) * free.len()) {
        if rem == 0 {
            break;
        }
        cnt[i] += 1;
        rem -= 1;
    }
    for _ in 0..l * free.len() {
        let Some(v) = (0..free.len()).find(|&c| cnt[c] as f64 * dl > free[c] + 1e-9) else { break };
        let to = (0..free.len()).filter(|&c| c != v).max_by(|&a, &b| (free[a] - cnt[a] as f64 * dl).partial_cmp(&(free[b] - cnt[b] as f64 * dl)).unwrap())?;
        if free[to] - (cnt[to] + 1) as f64 * dl < 0.0 {
            return None;
        }
        cnt[v] -= 1;
        cnt[to] += 1;
    }
    if cnt.iter().any(|&c| c == 0) || (0..free.len()).any(|c| cnt[c] as f64 * dl > free[c] + 1e-9) {
        return None;
    }
    Some(cnt)
}

pub fn ram_for_experts_gib(n: &NodeDesc, model: &ModelDesc, k: &Knobs) -> f64 {
    let Some(ram) = n.ram_gib else { return 0.0 };
    let reserve = (ram * k.ram_reserve_frac).max(k.ram_reserve_min_gib);
    (ram - reserve - model.head_bytes as f64 / GIB).max(0.0)
}

fn tier_order(n: &NodeDesc) -> Vec<&Tier> {
    let mut t: Vec<&Tier> = n.tiers.iter().filter(|t| t.read_mb_s.is_some() && !t.holds_model).collect();
    t.sort_by(|a, b| b.read_mb_s.partial_cmp(&a.read_mb_s).unwrap());
    t
}

pub fn model_store(fleet: &Fleet) -> Option<(String, Tier)> {
    fleet.nodes.iter().find_map(|n| n.tiers.iter().find(|t| t.holds_model).map(|t| (n.name.clone(), t.clone())))
}

/// Steps 1: where everything lives. `Err` carries the reason a layout is impossible.
pub fn place(fleet: &Fleet, node: &NodeDesc, sel: &[(usize, GpuDesc, Option<ArchInfo>)], helper: Option<(&NodeDesc, &Link)>, model: &ModelDesc, k: &Knobs) -> Result<Placement, String> {
    let b = model.expert_bytes as f64;
    let mut notes = Vec::new();
    let l = model.n_layer;
    let kv_layer = model.kv_bytes(k.ctx) as f64 / l as f64 / MIB;
    let dl = model.dense_layer_bytes as f64 / MIB + kv_layer;
    let free: Vec<f64> = sel.iter().map(|(_, g, _)| g.vram_mib as f64 - k.reserve_mib).collect();
    let Some(cnt) = split_layers(&free, dl, l) else {
        let need = dl * l as f64;
        return Err(format!(
            "dense layers ({:.1} GiB with {} ctx KV) do not fit on {} card(s) of {} ({:.1} GiB free after the {:.0} MiB reserve each)",
            need / 1024.0,
            k.ctx,
            sel.len(),
            node.name,
            free.iter().sum::<f64>() / 1024.0,
            k.reserve_mib
        ));
    };
    notes.push(format!("layer split {:?} over free VRAM {:?} MiB: proportional split gives every layer the same number of resident expert slots", cnt, free.iter().map(|f| f.round() as i64).collect::<Vec<_>>()));
    let mut layers = Vec::new();
    let mut k_c = Vec::new();
    let mut start = 0;
    let mut vram_experts = 0usize;
    for (c, &n) in cnt.iter().enumerate() {
        let moe_c = (start..start + n).filter(|&x| x >= model.n_dense_lead).count();
        let slots = if b > 0.0 { ((free[c] - n as f64 * dl) * MIB / b).floor().max(0.0) as usize } else { 0 };
        let per = if moe_c > 0 { (slots / moe_c).min(model.n_expert) } else { 0 };
        k_c.push(per);
        vram_experts += per * moe_c;
        layers.push(n);
        start += n;
    }
    let moe_l = model.moe_layers().max(1);
    // helper
    let (mut helper_node, mut helper_gpus, mut helper_experts, mut kh) = (None, Vec::new(), 0usize, 0usize);
    if let Some((h, _)) = helper {
        let hs = usable_gpus(h);
        let slots: usize = hs.iter().map(|(_, g, _)| (((g.vram_mib as f64 - k.reserve_mib).max(0.0)) * MIB / b.max(1.0)).floor() as usize).sum();
        let kmin = *k_c.iter().min().unwrap_or(&0);
        let room = model.n_expert.saturating_sub(*k_c.iter().max().unwrap_or(&0));
        kh = (slots / moe_l).min(room);
        helper_experts = kh * moe_l;
        helper_node = Some(h.name.clone());
        helper_gpus = hs.iter().map(|(_, g, _)| g.name.clone()).collect();
        let _ = kmin;
        if kh == 0 {
            return Err(format!("helper {} has no VRAM left for experts", h.name));
        }
    }
    let total_experts = (model.n_expert * model.moe_layers()) as u64;
    let mut left = total_experts.saturating_sub(vram_experts as u64 + helper_experts as u64);
    let ram_gib = ram_for_experts_gib(node, model, k);
    let ram_cap = if b > 0.0 { (ram_gib * GIB / b).floor() as u64 } else { 0 };
    let ram_experts = left.min(ram_cap);
    left -= ram_experts;
    let mut tier_experts = Vec::new();
    for t in tier_order(node) {
        if left == 0 {
            break;
        }
        let cap = (t.bytes as f64 / b.max(1.0)).floor() as u64;
        let used = left.min(cap);
        if used > 0 {
            tier_experts.push(TierUse { kind: t.kind, path: t.path.clone(), experts: used, gib: used as f64 * b / GIB, read_mb_s: t.read_mb_s.unwrap() });
            left -= used;
        }
    }
    for t in node.tiers.iter().filter(|t| t.read_mb_s.is_none() && !t.holds_model) {
        notes.push(format!("tier {:?} on {} has unknown read speed: not used", t.kind, node.name));
    }
    let _ = fleet;
    Ok(Placement {
        node: node.name.clone(),
        gpu_idx: sel.iter().map(|s| s.0).collect(),
        gpus: sel.iter().map(|s| s.1.clone()).collect(),
        infos: sel.iter().map(|s| s.2.clone()).collect(),
        layers,
        k_c,
        helper_node,
        helper_gpus,
        helper_experts,
        kh,
        ram_experts,
        tier_experts,
        file_experts: left,
        ram_gib: ram_experts as f64 * b / GIB,
        dl_mib: dl,
        kv_layer_mib: kv_layer,
        notes,
    })
}

/// Residency label: the deepest level this placement needs.
pub fn residency(p: &Placement) -> &'static str {
    if p.file_experts > 0 {
        "streaming (model-file reads on the decode path)"
    } else if !p.tier_experts.is_empty() {
        "resident + RAM + disk tier"
    } else if p.ram_experts > 0 {
        "resident + RAM"
    } else if p.helper_node.is_some() {
        "all resident (incl. helper VRAM)"
    } else {
        "all resident"
    }
}

pub struct TimingSet {
    pub cards: Vec<CardTimings>,
    pub host: HostTimings,
    pub sources: Vec<String>,
}

/// Bench files keyed by (node, gpu index).
pub type Benches = HashMap<(String, usize), (CardTimings, HostTimings)>;

pub fn timings_for(p: &Placement, node: &NodeDesc, model: &ModelDesc, knobs: &Knobs, benches: &Benches) -> TimingSet {
    let b = model.expert_bytes as f64;
    let dense_layer = model.dense_layer_bytes as f64;
    let mut cards = Vec::new();
    let mut sources = Vec::new();
    for (i, g) in p.gpus.iter().enumerate() {
        let idx = p.gpu_idx[i];
        let bw = g.mem_gb_s.unwrap_or(0.0);
        let h2d = arch::pcie_mb_s(g.pcie_gen, g.pcie_width).unwrap_or(PCIE_FALLBACK_MB_S);
        let free_mb = g.vram_mib as f64 * 1.048576 - knobs.reserve_mib * 1.048576;
        if let Some((ct, _)) = benches.get(&(p.node.clone(), idx)) {
            // measured on the LFM expert: effective bandwidth = bytes / time, reused for this model
            let eff_gb_s = EXPERT_BYTES as f64 / (ct.expert_ms * 1e6);
            let ms = |bytes: f64| bytes / (eff_gb_s * 1e6);
            let mut c = ct.clone();
            c.expert_ms = ms(b);
            c.expert_m4_ms = ct.expert_m4_ms * b / EXPERT_BYTES as f64;
            c.conv_layer_ms = ms(dense_layer);
            c.attn_layer_ms = ms(dense_layer);
            c.dense_ffn_ms = ms(dense_layer);
            c.vram_free_mb = free_mb;
            cards.push(c);
            sources.push(format!("gpu{idx} {}: measured bench ({:.1} GB/s effective on the LFM expert), scaled by bytes", g.name, eff_gb_s));
        } else {
            let eff_gb_s = bw * KERNEL_EFF;
            let ms = |bytes: f64| if eff_gb_s > 0.0 { bytes / (eff_gb_s * 1e6) } else { f64::INFINITY };
            cards.push(CardTimings {
                name: format!("{}:{}", p.node, g.name),
                host: p.node.clone(),
                device: idx.to_string(),
                expert_ms: ms(b),
                expert_m4_ms: ms(b),
                h2d_mb_s: h2d,
                h2d_bulk_mb_s: h2d,
                handoff_us: knobs.handoff_us,
                conv_layer_ms: ms(dense_layer),
                attn_layer_ms: ms(dense_layer),
                dense_ffn_ms: ms(dense_layer),
                vram_free_mb: free_mb,
                measured: false,
            });
            sources.push(format!(
                "gpu{idx} {}: estimate: {:.0} GB/s spec x {KERNEL_EFF} kernel efficiency (c3 measurements)",
                g.name, bw
            ));
        }
    }
    // host CPU: measured per-thread rate, capped by the measured parallel speedup
    let threads = node.cpu_threads.map(|t| t as f64 * knobs.cpu_thread_frac).unwrap_or(1.0);
    let speed = threads.min(CPU_MAX_SPEEDUP).max(1.0);
    let rate = CPU_GB_S_PER_THREAD * speed; // GB/s
    let cpu_ms = |bytes: f64| bytes / (rate * 1e6);
    if let Some((_, h)) = benches.iter().find(|((n, _), _)| *n == p.node).map(|(_, v)| v.clone()).map(|v| ((), v.1)) {
        sources.push(format!("host {}: measured bench CPU expert {:.2} ms (LFM)", p.node, h.cpu_expert_ms));
    }
    let host = HostTimings {
        host: p.node.clone(),
        threads: node.cpu_threads.unwrap_or(1),
        cpu_expert_ms: cpu_ms(b),
        cpu_head_ms: cpu_ms(model.head_bytes as f64),
        cpu_conv_layer_ms: cpu_ms(dense_layer),
        cpu_attn_layer_ms: cpu_ms(dense_layer),
        cpu_dense_ffn_ms: cpu_ms(dense_layer),
        measured: false,
    };
    sources.push(format!(
        "host {}: estimate: {:.3} GB/s per thread (c4, 1 thread, V2_PIPELINE s3) x min({} threads, {CPU_MAX_SPEEDUP}x measured speedup); head priced at the same rate (c4 measured head is ~2.4x faster)",
        p.node, CPU_GB_S_PER_THREAD, node.cpu_threads.map(|t| t.to_string()).unwrap_or("1 (threads unknown)".into())
    ));
    TimingSet { cards, host, sources }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Synthetic routing: each token picks `top_k` distinct experts per MoE layer; rank drawn by
/// inverting the skew curve; expert id = rank (so "staged = ids 0..k" is the hottest k).
pub fn synth_trace(model: &ModelDesc, curve: &SkewCurve, tokens: usize, seed: u64) -> Trace {
    let mut rng = Rng(seed | 1);
    let ne = model.n_expert;
    let mut toks = Vec::with_capacity(tokens);
    for _ in 0..tokens {
        let mut per_layer = Vec::with_capacity(model.n_layer);
        for l in 0..model.n_layer {
            if ne == 0 || l < model.n_dense_lead {
                per_layer.push(Vec::new());
                continue;
            }
            let mut ids: Vec<u16> = Vec::new();
            let mut tries = 0;
            while ids.len() < model.top_k.min(ne) {
                let f = curve.inv(rng.next());
                let mut id = ((f * ne as f64) as usize).min(ne - 1) as u16;
                tries += 1;
                if ids.contains(&id) {
                    if tries < 32 {
                        continue;
                    }
                    while ids.contains(&id) {
                        id = (id + 1) % ne as u16;
                    }
                }
                ids.push(id);
            }
            per_layer.push(ids);
        }
        toks.push(TokenRoute { experts: per_layer });
    }
    Trace { n_layer: model.n_layer, n_expert: ne, requests: vec![Request { id: "synthetic".into(), tokens: toks }] }
}

struct Scenario {
    ms_sim: f64,
    ms_helper: f64,
    io_serial: f64,
    io_overlap: f64,
    shares: [f64; 5],
    sim: SimResult,
    staged: Vec<Vec<u16>>,
    cov_resident: f64,
}

fn run_scenario(fleet: &Fleet, p: &Placement, node: &NodeDesc, model: &ModelDesc, knobs: &Knobs, ts: &TimingSet, curve: &SkewCurve, helper_link: Option<&Link>) -> Scenario {
    let b = model.expert_bytes as f64;
    let ne = model.n_expert.max(1) as f64;
    let moe_l = model.moe_layers();
    let uses = (moe_l * model.top_k) as f64;
    // per-layer resident rank counts
    let mut layer_card = Vec::with_capacity(model.n_layer);
    for (c, &n) in p.layers.iter().enumerate() {
        layer_card.extend(std::iter::repeat(c).take(n));
    }
    let k_layer: Vec<usize> = (0..model.n_layer).map(|l| if l < model.n_dense_lead { 0 } else { p.k_c[layer_card[l]] }).collect();
    let avg_cov = |extra: f64| -> f64 {
        if moe_l == 0 {
            return 1.0;
        }
        (model.n_dense_lead..model.n_layer).map(|l| curve.cov((k_layer[l] as f64 + extra) / ne)).sum::<f64>() / moe_l as f64
    };
    let per = |n: u64| if moe_l > 0 { n as f64 / moe_l as f64 } else { 0.0 };
    let cov_v = avg_cov(0.0);
    let cov_h = avg_cov(p.kh as f64);
    let mut extra = p.kh as f64 + per(p.ram_experts);
    let cov_r = avg_cov(extra);
    let mut tier_ms = Vec::new();
    let mut cov_prev = cov_r;
    let mut tier_share = 0.0;
    for t in &p.tier_experts {
        extra += per(t.experts);
        let c = avg_cov(extra);
        let n = uses * (c - cov_prev);
        tier_ms.push(n * b / (t.read_mb_s * 1e6) * 1000.0);
        tier_share += c - cov_prev;
        cov_prev = c;
    }
    let share_file = if p.file_experts > 0 { (1.0 - cov_prev).max(0.0) } else { 0.0 };
    let file_ms = match model_store(fleet) {
        Some((_, t)) if share_file > 0.0 => t.read_mb_s.map(|r| uses * share_file * b / (r * 1e6) * 1000.0).unwrap_or(f64::INFINITY),
        _ if share_file > 0.0 => f64::INFINITY,
        _ => 0.0,
    };
    // simulator on the synthetic trace
    let trace = synth_trace(model, curve, knobs.synth_tokens, knobs.seed);
    let kmin = (0..p.k_c.len()).map(|c| p.k_c[c] + p.kh).min().unwrap_or(0);
    let mut cfg = SimConfig::single_card(ts.cards[0].clone(), ts.host.clone(), 0, kmin);
    cfg.cards = ts.cards.clone();
    cfg.slots = (0..p.k_c.len()).map(|c| (p.k_c[c] + p.kh) * (p.layers[c]).max(1)).collect();
    cfg.miss_policy = MissPolicy::Cpu;
    cfg.n_dense_lead = model.n_dense_lead;
    cfg.expert_bytes = model.expert_bytes as usize;
    cfg.layer_card = Some(layer_card.clone());
    cfg.attn_first = 0;
    cfg.n_attn_every = 1;
    let staged: Vec<Vec<u16>> = (0..model.n_layer).map(|l| if l < model.n_dense_lead { vec![] } else { (0..(k_layer[l] + p.kh).min(model.n_expert) as u16).collect() }).collect();
    let sim = simulate(&cfg, &trace, &staged);
    // helper round trips: one per MoE layer that uses it
    let ms_helper = match (p.kh > 0, helper_link) {
        (true, Some(l)) => {
            let payload = model.hidden.map(|h| 2.0 * h as f64 * 4.0).unwrap_or(0.0);
            let per_layer = l.latency_us.unwrap_or(0.0) / 1000.0 + l.gb_s.map(|g| payload / (g * 1e6)).unwrap_or(0.0);
            moe_l as f64 * per_layer
        }
        _ => 0.0,
    };
    let _ = node;
    let io: Vec<f64> = tier_ms.iter().copied().chain(std::iter::once(file_ms)).collect();
    Scenario {
        ms_sim: sim.ms_per_token,
        ms_helper,
        io_serial: io.iter().sum(),
        io_overlap: io.iter().cloned().fold(0.0, f64::max),
        shares: [cov_v, cov_h - cov_v, cov_r - cov_h, tier_share, share_file],
        sim,
        staged,
        cov_resident: cov_h,
    }
}

fn tok(ms: f64) -> Option<f64> {
    if ms.is_finite() && ms > 0.0 {
        Some(1000.0 / ms)
    } else {
        None
    }
}

/// Step 2: simulate and package. `curve` is the expected-case skew; the low bound always uses uniform.
pub fn evaluate(fleet: &Fleet, p: &Placement, node: &NodeDesc, model: &ModelDesc, knobs: &Knobs, benches: &Benches, curve: &SkewCurve) -> FleetPlan {
    let ts = timings_for(p, node, model, knobs, benches);
    let link = p.helper_node.as_ref().and_then(|h| fleet.link(&p.node, h));
    let uni = SkewCurve::uniform();
    let lo = run_scenario(fleet, p, node, model, knobs, &ts, &uni, link);
    let ex = run_scenario(fleet, p, node, model, knobs, &ts, curve, link);
    let uses = (model.moe_layers() * model.top_k) as f64;
    let mut pc = p.clone();
    pc.file_experts += pc.ram_experts + pc.tier_experts.iter().map(|t| t.experts).sum::<u64>();
    pc.ram_experts = 0;
    pc.tier_experts.clear();
    let cold = run_scenario(fleet, &pc, node, model, knobs, &ts, curve, link);
    let expect = Expect {
        tok_s_cold: tok(cold.ms_sim + cold.ms_helper + cold.io_serial),
        tok_s_low: tok(lo.ms_sim + lo.ms_helper + lo.io_serial),
        tok_s: tok(ex.ms_sim + ex.ms_helper + ex.io_serial),
        tok_s_high: tok(ex.ms_sim + ex.ms_helper + ex.io_overlap),
        ms_sim: ex.ms_sim,
        ms_helper: ex.ms_helper,
        ms_io_serial: ex.io_serial,
        ms_io_overlap: ex.io_overlap,
        uses_per_token: uses,
        share_vram: ex.shares[0],
        share_helper: ex.shares[1],
        share_ram: ex.shares[2],
        share_tier: ex.shares[3],
        share_file: ex.shares[4],
        skew: format!("{} ({})", curve.name, curve.source),
        basis: {
            let mut v = ts.sources.clone();
            v.push("expected = chosen skew curve, warm tiers, disk reads not overlapped with compute; cold = same with RAM/disk tiers still empty (first requests after a load); low = uniform routing, no overlap; high = chosen skew, reads overlapped (max of devices)".into());
            v.push("decode, one stream; prefill and multi-stream pipelining not modelled (sim.rs)".into());
            v
        },
    };
    let b = model.expert_bytes as f64;
    let mut cards = Vec::new();
    let mut start = 0;
    for (i, g) in p.gpus.iter().enumerate() {
        let n = p.layers[i];
        let moe_c = (start..start + n).filter(|&x| x >= model.n_dense_lead).count();
        let info = p.infos[i].as_ref();
        let kernel_note = match g.has_dp4a {
            Some(false) => Some(format!("no int8 dot (dp4a) on {}: Q4/Q8/IQ kernels dequantize to {} before the multiply", g.name, if g.fp16_rate.as_deref() == Some("full") { "fp16 (full-rate on this card)" } else { "fp32" })),
            Some(true) if g.fp16_rate.as_deref() == Some("1/64") => Some(format!("{}: dp4a available; fp16 is 1/64 rate, so use the dp4a/fp32 path, not an fp16 dequant", g.name)),
            None => Some(format!("{}: dp4a unknown (card not in the arch table)", g.name)),
            _ => None,
        };
        cards.push(CardPlan {
            node: p.node.clone(),
            gpu_index: p.gpu_idx[i],
            name: g.name.clone(),
            arch: g.arch.clone(),
            has_dp4a: g.has_dp4a,
            layer_start: start,
            layer_end: start + n,
            dense_mib: n as f64 * model.dense_layer_bytes as f64 / MIB,
            kv_mib: n as f64 * p.kv_layer_mib,
            reserve_mib: knobs.reserve_mib,
            resident_per_layer: p.k_c[i],
            resident_experts: p.k_c[i] * moe_c,
            timing_source: ts.sources.get(i).cloned().unwrap_or_default(),
            kernel_note,
        });
        let _ = info;
        start += n;
    }
    let helper = p.helper_node.as_ref().map(|h| HelperPlan {
        node: h.clone(),
        link_kind: link.map(|l| l.kind).unwrap_or(LinkKind::Ethernet),
        link_gb_s: link.and_then(|l| l.gb_s),
        link_latency_us: link.and_then(|l| l.latency_us),
        gpus: p.helper_gpus.clone(),
        experts: p.helper_experts,
        per_layer: p.kh,
    });
    let store = model_store(fleet);
    let mut explain = p.notes.clone();
    explain.push(format!(
        "dense {:.2} GiB/layer-set + KV (ctx {}) on cards; head+embedding {:.2} GiB kept in host RAM (the engine runs the head on the CPU)",
        model.dense_total_bytes() as f64 / GIB,
        knobs.ctx,
        model.head_bytes as f64 / GIB
    ));
    explain.push(format!(
        "resident per layer by card: {:?}; helper adds {} per layer; RAM tier {:.1} GiB ({} experts); disk tiers {:?}; model-file experts {}",
        p.k_c,
        p.kh,
        p.ram_gib,
        p.ram_experts,
        p.tier_experts.iter().map(|t| format!("{:?} {:.1} GiB", t.kind, t.gib)).collect::<Vec<_>>(),
        p.file_experts
    ));
    explain.push(format!("expert size {:.1} MiB; {} routed uses per token", b / MIB, uses));
    let mut unknowns_used = Vec::new();
    if node.ram_gib.is_none() {
        unknowns_used.push(format!("{}.ram_gib unknown: no RAM tier planned", node.name));
    }
    if node.cpu_threads.is_none() {
        unknowns_used.push(format!("{}.cpu_threads unknown: CPU priced as 1 thread", node.name));
    }
    if p.file_experts > 0 && store.as_ref().and_then(|s| s.1.read_mb_s).is_none() {
        unknowns_used.push("model-store read speed unknown: tokens/s cannot be estimated".into());
    }
    if let Some(l) = link {
        if l.gb_s.is_none() {
            unknowns_used.push(format!("link {}-{} rate unknown: helper activation transfer time omitted (latency only)", l.a, l.b));
        }
    }
    if p.helper_node.is_some() {
        unknowns_used.push(format!("helper start-up (reading {:.1} GiB of experts from the helper's own storage) is not modelled", p.helper_experts as f64 * b / GIB));
    }
    for g in &p.gpus {
        if g.pcie_gen.is_none() || g.pcie_width.is_none() {
            unknowns_used.push(format!("{} PCIe link unknown: h2d assumed Gen3 x8 ({PCIE_FALLBACK_MB_S:.0} MB/s) in the simulator (only matters for on-demand copies)", g.name));
            break;
        }
    }
    let ram_gib_tier = p.ram_gib;
    let layout = Layout {
        node: p.node.clone(),
        cards,
        layer_split: p.layers.clone(),
        ram_tier_gib: ram_gib_tier,
        ram_experts: p.ram_experts,
        tiers: p.tier_experts.clone(),
        file_experts: p.file_experts,
        model_store_mb_s: store.and_then(|s| s.1.read_mb_s),
        helper,
        expect,
        residency: residency(p).into(),
        explain,
        unknowns_used,
    };
    let card_assign = layout
        .cards
        .iter()
        .enumerate()
        .map(|(i, c)| CardAssign {
            host: c.node.clone(),
            device: c.gpu_index.to_string(),
            name: c.name.clone(),
            layers: (c.layer_start..c.layer_end).filter(|&l| l >= model.n_dense_lead).collect(),
            slots: c.resident_experts + p.kh * (c.layer_start..c.layer_end).filter(|&l| l >= model.n_dense_lead).count(),
            budget_mb: (p.gpus[i].vram_mib as f64 - knobs.reserve_mib).max(0.0),
            is_static: true,
        })
        .collect();
    let staging = StagingConfig {
        version: 1,
        model: model.name.clone(),
        k_per_layer: p.k_c.iter().copied().min().unwrap_or(0) + p.kh,
        lookahead: 1,
        miss_policy: MissPolicy::Cpu,
        dynamic_admit: false,
        dense_on_gpu: true,
        cards: card_assign,
        staged: ex.staged.clone(),
        predicted: ex.sim.clone(),
        held_out_coverage: ex.cov_resident,
        train_tokens: 0,
        held_out_tokens: knobs.synth_tokens,
        measured_on: ts.cards.iter().filter(|c| c.measured).map(|c| c.name.clone()).collect(),
        notes: vec![
            "fleet tuner plan: staged = hottest ids 0..k per layer of a SYNTHETIC routing trace; the engine must rank real experts by its own profile".into(),
            "held_out_coverage = modelled share of routed uses served from VRAM (local + helper)".into(),
        ],
    };
    FleetPlan { staging, layout }
}

#[derive(Default, Debug)]
pub struct PlanSet {
    pub plans: Vec<FleetPlan>,
    pub rejected: Vec<String>,
    pub options: Vec<String>,
}

fn rank_key(p: &FleetPlan) -> (i64, usize) {
    let t = p.layout.expect.tok_s.map(|t| (t * 1e6) as i64).unwrap_or(-1);
    (-t, p.layout.cards.len() + p.layout.helper.as_ref().map_or(0, |h| h.gpus.len()))
}

/// Search every node, card prefix and helper choice; best first.
pub fn plan_fleet(fleet: &Fleet, model: &ModelDesc, knobs: &Knobs, benches: &Benches, curve: &SkewCurve) -> PlanSet {
    let mut out = PlanSet::default();
    for node in &fleet.nodes {
        let gpus = usable_gpus(node);
        if gpus.is_empty() {
            out.rejected.push(format!("{}: no usable GPU", node.name));
            continue;
        }
        let mut helpers: Vec<Option<(&NodeDesc, &Link)>> = vec![None];
        for h in &fleet.nodes {
            if h.name != node.name && !usable_gpus(h).is_empty() {
                if let Some(l) = fleet.link(&node.name, &h.name) {
                    helpers.push(Some((h, l)));
                }
            }
        }
        for n in 1..=gpus.len() {
            let sel = &gpus[..n];
            for h in &helpers {
                match place(fleet, node, sel, *h, model, knobs) {
                    Ok(pl) => out.plans.push(evaluate(fleet, &pl, node, model, knobs, benches, curve)),
                    Err(e) => {
                        if h.is_none() {
                            out.rejected.push(format!("{} x{} cards: {e}", node.name, n));
                        }
                    }
                }
            }
        }
    }
    out.plans.sort_by_key(rank_key);
    if out.plans.is_empty() {
        out.options = does_not_fit_options(fleet, model, knobs);
    }
    out
}

/// What to do when nothing fits: honest arithmetic, no promises.
pub fn does_not_fit_options(fleet: &Fleet, model: &ModelDesc, knobs: &Knobs) -> Vec<String> {
    let mut o = Vec::new();
    let dl = model.dense_layer_bytes as f64 + model.kv_bytes(knobs.ctx) as f64 / model.n_layer as f64;
    let need_gib = dl * model.n_layer as f64 / GIB;
    let best_node = fleet.nodes.iter().map(|n| (n, usable_gpus(n).iter().map(|g| (g.1.vram_mib as f64 - knobs.reserve_mib).max(0.0)).sum::<f64>() / 1024.0)).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    if let Some((n, have)) = best_node {
        o.push(format!("dense layers + KV need {:.1} GiB of VRAM on one node; the largest ({}) has {:.1} GiB free after reserves: add {:.1} GiB of cards on one node", need_gib, n.name, have, (need_gib - have).max(0.0)));
        if let Some(r) = n.ram_gib {
            let ram_free = r - (r * knobs.ram_reserve_frac).max(knobs.ram_reserve_min_gib);
            if ram_free > need_gib + model.head_bytes as f64 / GIB {
                o.push(format!("dense layers on the CPU would fit {} RAM ({:.1} GiB free) but every token then visits the host for each layer: expect a large slowdown (not planned here)", n.name, ram_free));
            }
        }
    }
    o.push("pipeline the dense layers across nodes over the RoCE/Ethernet link (one activation hop per boundary): not implemented in this tuner".into());
    o.push("use a smaller quant: see the quant fit table (--quants) for the sizes that would fit".into());
    o
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::fleet::model::ModelDesc;

    pub fn p100() -> GpuDesc {
        GpuDesc { name: "Tesla P100-PCIE-16GB".into(), vram_mib: 16384, pcie_gen: Some(3), pcie_width: Some(16), available: true, ..Default::default() }
    }
    pub fn node(name: &str, gpus: Vec<GpuDesc>, ram: f64) -> NodeDesc {
        NodeDesc { name: name.into(), ram_gib: Some(ram), cpu_threads: Some(48), gpus, tiers: vec![] }
    }
    /// A small MoE: 20 layers (2 dense lead), 32 experts, top-4, 16 MiB experts.
    pub fn small_moe() -> ModelDesc {
        ModelDesc {
            name: "small".into(),
            arch: "x".into(),
            n_layer: 20,
            n_dense_lead: 2,
            n_expert: 32,
            top_k: 4,
            expert_bytes: 16 << 20,
            dense_layer_bytes: 100 << 20,
            head_bytes: 500 << 20,
            total_bytes: 0,
            hidden: Some(2048),
            kv_bytes_per_token: Some(20_000),
            ..Default::default()
        }
    }
    fn one(g: Vec<GpuDesc>, ram: f64) -> Fleet {
        Fleet { nodes: vec![node("a", g, ram)], ..Default::default() }
    }

    #[test]
    fn split_layers_proportional_and_repaired() {
        assert_eq!(split_layers(&[15000.0, 15000.0], 100.0, 20), Some(vec![10, 10]));
        let s = split_layers(&[30000.0, 10000.0], 100.0, 20).unwrap();
        assert_eq!(s.iter().sum::<usize>(), 20);
        assert!(s[0] > s[1]);
        // card 1 can hold only 3 layers of 100 though proportion says 5
        let s = split_layers(&[7500.0, 300.0], 100.0, 20).unwrap_or_default();
        assert!(s.is_empty() || s[1] <= 3);
        assert_eq!(split_layers(&[500.0], 100.0, 20), None);
    }

    #[test]
    fn one_card_everything_resident() {
        let f = one(vec![p100()], 64.0);
        let m = small_moe(); // 18*32*16 MiB = 9 GiB experts + 2 GiB dense: fits 15 GiB
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        let best = &r.plans[0];
        assert_eq!(best.layout.residency, "all resident");
        assert_eq!(best.layout.cards.len(), 1);
        assert_eq!(best.layout.cards[0].resident_per_layer, 32);
        assert!(best.layout.expect.tok_s.unwrap() > 0.0);
        assert!((best.layout.expect.share_vram - 1.0).abs() < 1e-9);
    }

    #[test]
    fn four_p100_one_node_splits_dense_and_fills_vram_first() {
        let mut m = small_moe();
        m.n_expert = 128; // 18*128*16 MiB = 36 GiB experts + 2 GiB dense
        m.expert_bytes = 16 << 20;
        let f = one(vec![p100(), p100(), p100(), p100()], 64.0);
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        let best = &r.plans[0];
        // 4 cards hold 60 GiB: everything resident, split 5 layers each
        assert_eq!(best.layout.layer_split.iter().sum::<usize>(), 20);
        assert_eq!(best.layout.residency, "all resident");
        assert!(best.layout.cards.len() >= 3, "{:?}", best.layout.layer_split);
        // fewer-cards candidates exist and are ranked by tok/s, not card count
        assert!(r.plans.iter().any(|p| p.layout.cards.len() == 1));
    }

    #[test]
    fn dense_split_when_one_card_cannot_hold_dense() {
        let mut m = small_moe();
        m.n_layer = 20;
        m.dense_layer_bytes = 1200 << 20; // 20 x 1.2 GiB = 23 GiB dense > one 15 GiB card
        let f = one(vec![p100(), p100()], 64.0);
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        assert!(r.rejected.iter().any(|s| s.contains("x1 cards") && s.contains("do not fit")), "{:?}", r.rejected);
        let best = &r.plans[0];
        assert_eq!(best.layout.cards.len(), 2);
        assert_eq!(best.layout.layer_split, vec![10, 10]);
    }

    #[test]
    fn mixed_cards_split_by_free_vram_and_flag_no_dp4a() {
        let mut g1070 = p100();
        g1070.name = "NVIDIA GeForce GTX 1070".into();
        g1070.vram_mib = 8192;
        let f = one(vec![p100(), g1070], 64.0);
        let mut m = small_moe();
        m.n_expert = 256;
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        let two = r.plans.iter().find(|p| p.layout.cards.len() == 2).unwrap();
        assert!(two.layout.layer_split[0] > two.layout.layer_split[1]);
        assert!(two.layout.cards[0].kernel_note.as_ref().unwrap().contains("no int8 dot"));
        assert!(two.layout.cards[1].kernel_note.as_ref().unwrap().contains("1/64"));
    }

    #[test]
    fn two_nodes_with_roce_use_a_helper_when_it_helps() {
        let mut m = small_moe();
        m.n_expert = 256; // 18*256*16 MiB = 72 GiB
        let mut f = Fleet { nodes: vec![node("big", vec![p100(), p100()], 16.0), node("help", vec![p100(), p100(), p100()], 16.0)], ..Default::default() };
        f.links.push(Link { a: "big".into(), b: "help".into(), kind: LinkKind::Roce, gb_s: Some(2.84), latency_us: Some(8.0), source: "test".into() });
        f.nodes[0].tiers.push(Tier { kind: TierKind::Ssd, path: None, bytes: 500 << 30, read_mb_s: Some(300.0), holds_model: true });
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        let with = r.plans.iter().filter(|p| p.layout.node == "big" && p.layout.helper.is_some() && p.layout.cards.len() == 2).next().unwrap();
        let without = r.plans.iter().filter(|p| p.layout.node == "big" && p.layout.helper.is_none() && p.layout.cards.len() == 2).next().unwrap();
        assert!(with.layout.expect.tok_s.unwrap() > without.layout.expect.tok_s.unwrap(), "{:?} vs {:?}", with.layout.expect.tok_s, without.layout.expect.tok_s);
        assert!(with.layout.helper.as_ref().unwrap().experts > 0);
        assert!(with.layout.expect.ms_helper > 0.0);
    }

    #[test]
    fn does_not_fit_lists_options() {
        let mut m = small_moe();
        m.dense_layer_bytes = 4 << 30; // 80 GiB dense
        let f = one(vec![p100(), p100()], 64.0);
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        assert!(r.plans.is_empty());
        assert!(r.options.iter().any(|o| o.contains("add")), "{:?}", r.options);
        assert!(r.options.iter().any(|o| o.contains("smaller quant")));
    }

    #[test]
    fn tiers_and_file_fill_in_order_and_unknown_speed_flagged() {
        let mut m = small_moe();
        m.n_expert = 512; // 144 GiB of experts
        let mut f = one(vec![p100()], 40.0);
        f.nodes[0].tiers = vec![
            Tier { kind: TierKind::Ssd, path: None, bytes: 20 << 30, read_mb_s: Some(300.0), holds_model: false },
            Tier { kind: TierKind::Optane, path: None, bytes: 10 << 30, read_mb_s: Some(900.0), holds_model: false },
            Tier { kind: TierKind::Hdd, path: None, bytes: 1 << 40, read_mb_s: None, holds_model: false },
            Tier { kind: TierKind::Nfs, path: None, bytes: 1 << 40, read_mb_s: Some(150.0), holds_model: true },
        ];
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        let p = &r.plans[0];
        assert_eq!(p.layout.tiers[0].kind, TierKind::Optane, "fastest tier first");
        assert_eq!(p.layout.tiers[1].kind, TierKind::Ssd);
        assert!(p.layout.file_experts > 0);
        assert!(p.layout.residency.starts_with("streaming"));
        let lo = p.layout.expect.tok_s_low.unwrap();
        let hi = p.layout.expect.tok_s_high.unwrap();
        assert!(lo <= p.layout.expect.tok_s.unwrap() && p.layout.expect.tok_s.unwrap() <= hi);
        // no model store known: honest None
        f.nodes[0].tiers.pop();
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        assert!(r.plans[0].layout.expect.tok_s.is_none());
        assert!(r.plans[0].layout.unknowns_used.iter().any(|u| u.contains("model-store")));
    }

    #[test]
    fn plan_json_is_a_superset_of_the_staging_config_and_cold_is_slower() {
        let mut m = small_moe();
        m.n_expert = 512;
        let mut f = one(vec![p100()], 40.0);
        f.nodes[0].tiers = vec![Tier { kind: TierKind::Nfs, path: None, bytes: 1 << 40, read_mb_s: Some(300.0), holds_model: true }];
        let r = plan_fleet(&f, &m, &Knobs::default(), &Benches::new(), &SkewCurve::profile());
        let js = serde_json::to_string(&r.plans[0]).unwrap();
        let st: StagingConfig = serde_json::from_str(&js).expect("old reader still parses the new plan");
        assert_eq!(st.model, "small");
        let back: FleetPlan = serde_json::from_str(&js).unwrap();
        let e = &back.layout.expect;
        assert!(e.tok_s_cold.unwrap() <= e.tok_s.unwrap());
        assert!(e.tok_s_low.unwrap() <= e.tok_s.unwrap());
    }

    #[test]
    fn skew_curve_roundtrip() {
        let c = SkewCurve::profile();
        assert!((c.cov(0.375) - (1.0 - 24.0 / 284.0)).abs() < 1e-9);
        assert!((c.cov(c.inv(0.95)) - 0.95).abs() < 1e-9);
        assert_eq!(SkewCurve::uniform().cov(0.3), 0.3);
    }
}
