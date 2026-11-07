//! Fleet tuner: given a description of the machines (cards, RAM, disk tiers, links) and of
//! a model (layers, dense bytes, experts), propose a layout: which cards hold which layers,
//! how many experts stay resident per card, how big the RAM / Optane / SSD tiers are, which
//! node acts as expert helper, and what decode speed to expect (with a range).
//! See docs/FLEET_TUNER.md. Every constant carries a source note; unknown fields stay
//! `None` and are reported, never guessed.
//!
//! * `arch`  - per-architecture table (dp4a, fp16 rate class, memory bandwidth): estimates.
//! * `model` - model description from GGUF shards (lb-gguf) or JSON.
//! * `load`  - fleet description from fleet-manifest.json + the nauti fleet views.
//! * `plan`  - layout search + simulator hand-off + tier/helper arithmetic.
//! * `quant` - which quant files fit at which residency level; "remake the quant" lines.
//! * `report`- the human table and the `--explain` text.
pub mod arch;
pub mod load;
pub mod model;
pub mod plan;
pub mod quant;
pub mod report;

use serde::{Deserialize, Serialize};

pub const MIB: f64 = 1024.0 * 1024.0;
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One GPU as the tuner sees it. `None` = unknown (reported, not guessed).
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct GpuDesc {
    pub name: String,
    pub vram_mib: u64,
    /// "sm_60", "sm_75", "gfx1031", ...
    #[serde(default)]
    pub arch: Option<String>,
    /// CUDA compute capability, "6.0" / "6.1" / "7.5"; null for AMD.
    #[serde(default)]
    pub compute_cap: Option<String>,
    #[serde(default)]
    pub pcie_gen: Option<u32>,
    #[serde(default)]
    pub pcie_width: Option<u32>,
    /// Int8 dot product (dp4a / v_dot4). `None` = look it up in the arch table.
    #[serde(default)]
    pub has_dp4a: Option<bool>,
    /// "full" (2x via packed fp16), "same" (no gain over fp32), "1/64".
    #[serde(default)]
    pub fp16_rate: Option<String>,
    /// Memory bandwidth GB/s, only if measured or given; else the arch table estimate.
    #[serde(default)]
    pub mem_gb_s: Option<f64>,
    /// false = leave this card alone (pinned to a seat, photogrammetry, ...).
    #[serde(default = "yes")]
    pub available: bool,
    /// Which lb-gpu-bench JSON (if any) measured this card: filled by the CLI `--bench`.
    #[serde(default)]
    pub bench: Option<String>,
}
fn yes() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum TierKind {
    Optane,
    Ssd,
    Hdd,
    Nfs,
}

/// A place expert bytes can live below RAM.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Tier {
    pub kind: TierKind,
    pub path: Option<String>,
    /// Usable bytes for tier files.
    pub bytes: u64,
    /// Random-ish expert-sized read rate. `None` = unknown: the tier is listed but not used.
    pub read_mb_s: Option<f64>,
    /// Where the model file itself lives (the last stop for a miss). At most one per node.
    #[serde(default)]
    pub holds_model: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NodeDesc {
    pub name: String,
    pub ram_gib: Option<f64>,
    pub cpu_threads: Option<usize>,
    #[serde(default)]
    pub gpus: Vec<GpuDesc>,
    #[serde(default)]
    pub tiers: Vec<Tier>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LinkKind {
    Roce,
    Ethernet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Link {
    pub a: String,
    pub b: String,
    pub kind: LinkKind,
    /// Usable payload rate (measured if known, else nominal; `source` says which).
    pub gb_s: Option<f64>,
    /// Round-trip small-message latency.
    pub latency_us: Option<f64>,
    #[serde(default)]
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Fleet {
    pub nodes: Vec<NodeDesc>,
    #[serde(default)]
    pub links: Vec<Link>,
    /// Fields that were unknown when the description was built ("t440.ram_gib", ...).
    #[serde(default)]
    pub unknowns: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl Fleet {
    pub fn node(&self, name: &str) -> Option<&NodeDesc> {
        self.nodes.iter().find(|n| n.name == name)
    }
    pub fn link(&self, a: &str, b: &str) -> Option<&Link> {
        self.links.iter().find(|l| (l.a == a && l.b == b) || (l.a == b && l.b == a))
    }
    /// Unknown fields found in the description itself (for hand-written files).
    pub fn collect_unknowns(&mut self) {
        let mut u = self.unknowns.clone();
        for n in &self.nodes {
            if n.ram_gib.is_none() {
                u.push(format!("{}.ram_gib", n.name));
            }
            if n.cpu_threads.is_none() {
                u.push(format!("{}.cpu_threads", n.name));
            }
            for (i, g) in n.gpus.iter().enumerate() {
                if g.pcie_gen.is_none() || g.pcie_width.is_none() {
                    u.push(format!("{}.gpu{}.pcie_link", n.name, i));
                }
            }
            for t in &n.tiers {
                if t.read_mb_s.is_none() {
                    u.push(format!("{}.tier.{:?}.read_mb_s", n.name, t.kind));
                }
            }
        }
        u.sort();
        u.dedup();
        self.unknowns = u;
    }
}

/// Run-time knobs (the fleet rule: every cap is a knob with a documented default).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Knobs {
    /// Per-card reserve for compute buffers, CUDA context and fragmentation (MiB).
    /// Default 1024: the `--budget-mb` reserve of `lb-calibrate tune` (existing code).
    pub reserve_mib: f64,
    /// Context length the KV cache is sized for. Default 4096: estimate, a typical agent turn.
    pub ctx: usize,
    /// Fraction of node RAM left to the OS, page cache and engine heap. Default 0.30: the
    /// c4 Le Chonk run used an 80 GiB RAM tier of 117 GiB (memory note le-chonk-polystrata).
    pub ram_reserve_frac: f64,
    /// Floor for that reserve (GiB). Estimate: kernel + page cache + engine heap.
    pub ram_reserve_min_gib: f64,
    /// Synthetic decode tokens replayed through the simulator.
    pub synth_tokens: usize,
    /// Seed for the synthetic routing trace.
    pub seed: u64,
    /// Skew curve name: "profile" (measured Le Chonk pin plan) or "uniform".
    pub skew: String,
    /// Stated conversion rate for the "remake the quant" hours line; unknown unless given.
    pub requant_gb_per_hour: Option<f64>,
    /// Activation handoff through the host (us). Default 250: measured 129-599 us across the
    /// cards in docs/V2_PIPELINE.md section 3 (1070 129, WX5100 394, P100s 220/599).
    pub handoff_us: f64,
    /// Fraction of node threads that decode may use for CPU experts.
    pub cpu_thread_frac: f64,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            reserve_mib: 1024.0,
            ctx: 4096,
            ram_reserve_frac: 0.30,
            ram_reserve_min_gib: 8.0,
            synth_tokens: 64,
            seed: 0x5eed_f1ee7,
            skew: "profile".into(),
            requant_gb_per_hour: None,
            handoff_us: 250.0,
            cpu_thread_frac: 1.0,
        }
    }
}
