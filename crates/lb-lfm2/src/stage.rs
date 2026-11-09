//! Expert staging ring (MoE experiment #2, `docs/STAGE_RING.md`): prefetch the next MoE layer's
//! experts while the current layer runs, from a guess made on the current layer's hidden state.
//!
//! **The guess (`mid1`, experiment #1, `docs/EXPERT_PREDICT.md`).** At MoE layer `L`, once the
//! residual stream after `L`'s operator (`x_mid`) exists, layer `L+1`'s router is applied to it
//! with `L+1`'s FFN pre-norm: `s = sigmoid(rms(x_mid) * ffn_norm[L+1] . router[L+1]) + bias[L+1]`.
//! The top `slots` experts by `s` are the guess (8 for top-4 held 95 % of the experts needed and
//! all four on 83 % of decode token-layers on LFM2.5). `margin = s[k-1] - s[k]` is the guess's
//! confidence: low-margin guesses are the bad ones.
//!
//! **The ring.** A small set of expert copies on the device, apart from the static (heat-planned)
//! residents. At layer `L` the guessed experts of `L+1` that are neither resident nor already in
//! the ring are uploaded into it on a *second device stream* (`Gpu::side_stream`), so the copy
//! runs on the copy engine while the main stream finishes `L` and runs `L+1`'s operator. When
//! `L+1` routes, an expert found in the ring is launched from it; the main stream waits on the
//! device for that copy if it has not landed. Anything else is a demand miss, served exactly as
//! before (`MissPolicy::OnDemand`). Eviction is Least-Stale (SpecMD): entries not touched during
//! the current token go first (oldest token, then lowest margin); then entries of layers this
//! token has already passed; an entry for the layer being staged is never evicted for its own
//! layer.
//!
//! **What it must not change.** Only *when* bytes reach the device. A staged expert is uploaded
//! by the same code as a demand miss (`GpuExpert::upload`), runs the same kernel, and the layer's
//! expert outputs are summed in ascending expert id whatever served them (`moe::moe_ffn_with`),
//! so the greedy output is the same with the ring on or off and at any VRAM budget.
//!
//! Off by default (Avalon's PREGATE lesson, `docs/LESSONS_FROM_AVALON.md` 2.5: a prefetch that
//! competes for bandwidth when the hit rate is already high makes things slower). Settings are
//! read once at start: `LB_STAGE=1`, `LB_STAGE_SLOTS` (guesses per layer, default 8; 0 = none,
//! the ring only caches demand misses with `LB_STAGE_KEEP_MISSES=1`),
//! `LB_STAGE_RING_MIB` (ring size, default 256), `LB_STAGE_MIN_MARGIN` (skip guesses below this
//! margin, default 0 = off), `LB_STAGE_KEEP_MISSES=1` (demand misses also stay in the ring).

use crate::weights::{Ffn, Weights};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};

/// Most layers the per-layer counters hold (LFM2.5-24B has 40).
pub const MAX_LAYERS: usize = 128;

/// Staging settings, read once from the environment (restart-only).
#[derive(Clone, Debug, PartialEq)]
pub struct StageCfg {
    pub on: bool,
    /// Guesses per layer (`LB_STAGE_SLOTS`, default 8 = k + 4 for top-4).
    pub slots: usize,
    /// Ring size in MiB (`LB_STAGE_RING_MIB`, default 256: ~50 LFM2.5 experts).
    pub ring_mib: usize,
    /// Guesses with a margin below this are not staged (`LB_STAGE_MIN_MARGIN`; 0 = stage all).
    pub min_margin: f32,
    /// Keep demand misses in the ring after use (`LB_STAGE_KEEP_MISSES=1`), making the ring a
    /// small Least-Stale cache as well as a prefetch buffer. Off by default.
    pub keep_misses: bool,
}

impl Default for StageCfg {
    fn default() -> Self {
        Self { on: false, slots: 8, ring_mib: 256, min_margin: 0.0, keep_misses: false }
    }
}

impl StageCfg {
    pub fn from_env() -> Self {
        let d = Self::default();
        let on = |k: &str| std::env::var(k).map(|v| !v.is_empty() && v != "0").unwrap_or(false);
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<f64>().ok());
        Self {
            on: on("LB_STAGE"),
            // 0 = no prefetch at all: with `LB_STAGE_KEEP_MISSES=1` the ring is then only a
            // Least-Stale cache of demand misses (separates caching from prediction in an A/B)
            slots: num("LB_STAGE_SLOTS").map(|v| v as usize).unwrap_or(d.slots),
            ring_mib: num("LB_STAGE_RING_MIB").map(|v| v as usize).unwrap_or(d.ring_mib),
            min_margin: num("LB_STAGE_MIN_MARGIN").map(|v| v as f32).unwrap_or(d.min_margin),
            keep_misses: on("LB_STAGE_KEEP_MISSES"),
        }
    }

    /// Ring capacity in experts of `expert_bytes` each.
    pub fn ring_slots(&self, expert_bytes: usize) -> usize {
        if !self.on || expert_bytes == 0 {
            0
        } else {
            self.ring_mib * 1024 * 1024 / expert_bytes
        }
    }
}


/// Engine step E3 settings, read once from the environment (restart-only); all off by default, in
/// which case the decode path is E2's byte for byte.
#[derive(Clone, Debug, PartialEq)]
pub struct LookaheadCfg {
    /// Compute a guess at all (`LB_LOOKAHEAD=1`, implied by keep/prefetch).
    pub on: bool,
    /// Guesses per layer for the keep guard and the prefetch candidates (`LB_LA_N_GUESS`, default 8).
    pub n_guess: usize,
    /// Spare predicted experts from eviction (`LB_LA_KEEP=1`).
    pub keep: bool,
    /// Layers ahead the guard looks at (`LB_LA_KEEP_DEPTH`, default 1; the guess for L+2 reads the
    /// same state and costs another router dot).
    pub keep_depth: usize,
    /// Refresh predicted experts that are cached: 0 no, 1 to the MRU end of their segment, 2 to
    /// protected (`LB_LA_PROMOTE`, default 0).
    pub promote: u8,
    /// Prefetch at most this many predicted experts per layer that are NOT cached (0 = none,
    /// `LB_LA_PREFETCH_CAP`).
    pub prefetch_cap: usize,
    /// Only prefetch guesses with confidence `score - ref_score` at or above this (`LB_LA_MIN_CONF`,
    /// default negative infinity = no threshold).
    pub min_conf: f32,
    /// Prefetch landing buffers (`LB_LA_LANDINGS`, default 8), charged to the budget.
    pub landings: usize,
    /// Cycle-aware eviction: among the `cycle_window` least recently used probation slots, take the
    /// one whose layer comes up furthest in the fixed layer order (0 = plain LRU, `LB_LA_CYCLE_W`).
    pub cycle_window: usize,
    /// One JSON line per layer pass to this path (`LB_LA_TRACE`, offline simulator).
    pub trace: Option<std::path::PathBuf>,
    /// Layers ahead the prefetch looks at (`LB_LA_PF_DEPTH`, default 1): depth 2 also issues copies for
    /// L+2 at layer L, a longer lead time for each copy at a less accurate guess.
    pub pf_depth: usize,
    /// Prefetch into spare bank slots, repacked on the side stream, and rename the slot on use: a
    /// covered miss then costs the main stream nothing but one fence (`LB_LA_SPARE=1`). Otherwise the
    /// landing is raw bytes, repacked on the main stream on use.
    pub spare: bool,
    /// With the worker, pick its answer up just before the next layer's read instead of right after
    /// this layer's launches (`LB_LA_DEFER=1`), so the main thread does not wait for the router dots.
    pub defer: bool,
    /// Compute the guess on a worker thread so the router dots leave the critical path
    /// (`LB_LA_WORKER=1`; prefetch only, the keep guard needs the guess before planning).
    pub worker: bool,
    /// Diagnostic: a `LB_LA_TRACE` file of the same session, replayed as a perfect guess of the next
    /// layer (no x read, no router dots), to measure what overlap alone can buy (`LB_LA_ORACLE`).
    pub oracle: Option<std::path::PathBuf>,
}

impl Default for LookaheadCfg {
    fn default() -> Self {
        Self { on: false, n_guess: 8, keep: false, keep_depth: 1, promote: 0, prefetch_cap: 0, min_conf: f32::NEG_INFINITY, landings: 8, cycle_window: 0, trace: None, pf_depth: 1, spare: false, defer: false, worker: false, oracle: None }
    }
}

impl LookaheadCfg {
    pub fn from_env() -> Self {
        let d = Self::default();
        let on = |k: &str| std::env::var(k).map(|v| !v.is_empty() && v != "0").unwrap_or(false);
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<f64>().ok());
        let keep = on("LB_LA_KEEP");
        let promote = num("LB_LA_PROMOTE").map(|v| v as u8).unwrap_or(d.promote);
        let prefetch_cap = num("LB_LA_PREFETCH_CAP").map(|v| v as usize).unwrap_or(d.prefetch_cap);
        let trace = std::env::var("LB_LA_TRACE").ok().filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
        let oracle = std::env::var("LB_LA_ORACLE").ok().filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
        Self {
            on: on("LB_LOOKAHEAD") || keep || promote > 0 || prefetch_cap > 0 || trace.is_some(),
            n_guess: num("LB_LA_N_GUESS").map(|v| v as usize).unwrap_or(d.n_guess),
            keep,
            keep_depth: num("LB_LA_KEEP_DEPTH").map(|v| v as usize).unwrap_or(d.keep_depth),
            promote,
            prefetch_cap,
            min_conf: num("LB_LA_MIN_CONF").map(|v| v as f32).unwrap_or(d.min_conf),
            landings: num("LB_LA_LANDINGS").map(|v| v as usize).unwrap_or(d.landings),
            cycle_window: num("LB_LA_CYCLE_W").map(|v| v as usize).unwrap_or(d.cycle_window),
            trace,
            pf_depth: num("LB_LA_PF_DEPTH").map(|v| (v as usize).max(1)).unwrap_or(d.pf_depth),
            spare: on("LB_LA_SPARE"),
            defer: on("LB_LA_DEFER"),
            worker: on("LB_LA_WORKER"),
            oracle,
        }
    }

    /// Landing buffers to allocate (none unless prefetch is on).
    pub fn n_landings(&self) -> usize {
        if self.prefetch_cap > 0 {
            self.landings
        } else {
            0
        }
    }
}

/// The expert VRAM budget in bytes: `LB_EXPERT_VRAM_MIB` (MiB) when set, else `fallback_bytes`
/// (the server's `--gpu-budget-mb`). The staging ring is carved out of it, so both arms of an
/// A/B hold the same expert bytes on the card.
pub fn expert_vram_budget(fallback_bytes: usize) -> usize {
    std::env::var("LB_EXPERT_VRAM_MIB").ok().and_then(|v| v.trim().parse::<usize>().ok()).map(|m| m * 1024 * 1024).unwrap_or(fallback_bytes)
}

/// A ranked guess of one layer's experts.
#[derive(Clone, Debug, PartialEq)]
pub struct Guess {
    pub layer: usize,
    /// Most likely first (by biased selection score; the lower id first on ties, as the router).
    pub experts: Vec<usize>,
    /// `score[k-1] - score[k]` of the ranking (k = the model's top-k).
    pub margin: f32,
    /// Biased selection score of each expert in `experts`, same order.
    pub scores: Vec<f32>,
    /// `score[k]`, the best score left out of the real top-k: `scores[i] - ref_score` is expert `i`'s
    /// confidence (engine step E3), positive inside the top-k, the margin above the cut.
    pub ref_score: f32,
}

/// The `mid1` guess of `target`'s experts from the residual stream `x_mid` of the layer before
/// it: `target`'s FFN pre-norm and router applied to it (`route_sigmoid`'s ranking rule). `None`
/// when `target` has no MoE FFN.
pub fn predict(w: &Weights, target: usize, x_mid: &[f32], slots: usize) -> Option<Guess> {
    let layer = w.layers.get(target)?;
    let Ffn::Moe { router, bias, .. } = &layer.ffn else { return None };
    let cfg = &w.cfg;
    Some(rank(router, bias, &layer.ffn_norm, x_mid, cfg.rms_eps, cfg.n_expert, cfg.n_expert_used, slots, target))
}

/// Dot product for the guess. A plain float `sum()` cannot be vectorized (the compiler may not
/// reorder float additions), so the lanes are kept as separate accumulators and added at the end;
/// with the AVX2 path this took the guess from 15 ms to 2.5-3.1 ms a token [measured, c4 P100 host,
/// socket 0 busy]. The order of additions differs from `route_sigmoid`'s, which is fine: this only ranks a guess, the real route
/// is computed elsewhere.
#[inline(always)]
fn dot_body(a: &[f32], b: &[f32]) -> f32 {
    const W: usize = 32;
    let mut acc = [0f32; W];
    let (mut ia, mut ib) = (a.chunks_exact(W), b.chunks_exact(W));
    for (x, y) in (&mut ia).zip(&mut ib) {
        for i in 0..W {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = ia.remainder().iter().zip(ib.remainder()).map(|(p, q)| p * q).sum();
    acc.iter().sum::<f32>() + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    dot_body(a, b)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected.
        return unsafe { dot_avx2(a, b) };
    }
    dot_body(a, b)
}

/// The arithmetic of [`predict`], apart from the weights so it can be tested.
#[allow(clippy::too_many_arguments)]
pub fn rank(router: &[f32], bias: &[f32], norm: &[f32], x: &[f32], eps: f32, n_expert: usize, k: usize, slots: usize, layer: usize) -> Guess {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let r = 1.0 / (ms + eps).sqrt();
    let h: Vec<f32> = x.iter().zip(norm).map(|(a, b)| a * r * b).collect();
    let score: Vec<f32> = (0..n_expert)
        .map(|e| {
            let logit = dot(&router[e * n..(e + 1) * n], &h);
            1.0 / (1.0 + (-logit).exp()) + bias[e]
        })
        .collect();
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal));
    let margin = if k >= 1 && k < n_expert { score[order[k - 1]] - score[order[k]] } else { 0.0 };
    let ref_score = if k < n_expert { score[order[k]] } else { f32::NEG_INFINITY };
    order.truncate(slots.min(n_expert));
    let scores = order.iter().map(|&e| score[e]).collect();
    Guess { layer, experts: order, margin, scores, ref_score }
}


/// The next-layer guess on its own thread (engine step E3): the host read of `x_mid` hands the bytes
/// over and the main thread goes on planning and launching the layer while the router dots run, so
/// the ~3 ms a token of guessing leaves the critical path. It owns copies of the routers (20 MB).
pub struct GuessWorker {
    tx: std::sync::mpsc::Sender<(usize, Vec<u8>)>,
    rx: std::sync::mpsc::Receiver<Vec<Guess>>,
}

/// One target layer's router, bias and FFN pre-norm weights.
pub type RouterCopy = (Vec<f32>, Vec<f32>, Vec<f32>);

impl GuessWorker {
    /// `routers[l]` is `None` for a layer with no MoE FFN. For layer `l` the worker guesses `l + 1 ..=
    /// l + depth`.
    pub fn new(routers: Vec<Option<RouterCopy>>, eps: f32, n_expert: usize, k: usize, slots: usize, depth: usize) -> Self {
        let (tx, jobs) = std::sync::mpsc::channel::<(usize, Vec<u8>)>();
        let (out, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("lb-guess".into())
            .spawn(move || {
                while let Ok((li, bytes)) = jobs.recv() {
                    let x: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let gs: Vec<Guess> = (1..=depth)
                        .filter_map(|d| {
                            let (r, b, n) = routers.get(li + d)?.as_ref()?;
                            Some(rank(r, b, n, &x, eps, n_expert, k, slots, li + d))
                        })
                        .collect();
                    if out.send(gs).is_err() {
                        return;
                    }
                }
            })
            .expect("spawn lb-guess");
        Self { tx, rx }
    }

    /// Start the guess from layer `li`'s `x_mid` (raw little-endian bytes).
    pub fn send(&self, li: usize, x_mid: Vec<u8>) {
        // an answer nobody took (a token abandoned mid-way) must not meet this job
        while self.rx.try_recv().is_ok() {}
        let _ = self.tx.send((li, x_mid));
    }

    /// The guesses for the job last sent, nearest layer first; empty if the worker is gone.
    pub fn recv(&self) -> Vec<Guess> {
        self.rx.recv().unwrap_or_default()
    }
}

/// Decode-only counters, per layer, for both arms (ring on or off), so hit rates compare.
pub struct StageStats {
    /// Expert routings (k per decode token-layer).
    pub routings: [AtomicU64; MAX_LAYERS],
    /// ... served by a static resident.
    pub resident: [AtomicU64; MAX_LAYERS],
    /// ... served from the ring.
    pub staged: [AtomicU64; MAX_LAYERS],
    /// ... served by a demand fetch (on-demand copy, or the CPU): the stalls on the critical path.
    pub demand: [AtomicU64; MAX_LAYERS],
    /// Experts this layer's guess put in flight.
    pub prefetched: [AtomicU64; MAX_LAYERS],
    /// Prefetched experts that were used at least once before leaving the ring.
    pub useful: [AtomicU64; MAX_LAYERS],
    /// Prefetched experts evicted (or still in the ring at `snapshot`) never used.
    pub wasted: [AtomicU64; MAX_LAYERS],
    pub prefetch_bytes: AtomicU64,
    pub wasted_bytes: AtomicU64,
    pub evictions: AtomicU64,
    /// Guesses not issued: already resident, already in the ring, under the margin, ring full.
    pub guess_resident: AtomicU64,
    pub guess_in_ring: AtomicU64,
    pub guess_low_margin: AtomicU64,
    pub guess_no_room: AtomicU64,
    pub upload_errors: AtomicU64,
    /// Host time blocked in the decode MoE layers' syncs: reading the router input back, and
    /// reading the expert outputs back (where a late prefetch shows up).
    pub h2_read_ns: AtomicU64,
    pub moe_read_ns: AtomicU64,
    /// Host time issuing prefetches (predict + uploads on the side stream).
    pub issue_ns: AtomicU64,
    pub tokens: AtomicU64,
    /// Engine step E2: stream syncs the host waited on in device-routed decode tokens (one a layer
    /// plus the token's last read), the MoE layers that had at least one miss, and the on-device
    /// next-layer guess's score (experts of the real set it named, out of k per layer).
    pub host_syncs: AtomicU64,
    pub miss_layers: AtomicU64,
    pub moe_layers: AtomicU64,
    pub guess_named: AtomicU64,
    pub guess_total: AtomicU64,
    /// E3 look-ahead: LRU candidates passed over because a guess named them, experts promoted, and
    /// host time computing guesses (the router dots; `issue_ns` is the prefetch copies).
    pub la_spared: AtomicU64,
    pub la_promoted: AtomicU64,
    pub la_guess_ns: AtomicU64,
    attn: std::sync::Mutex<Vec<bool>>,
}

impl std::fmt::Debug for StageStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StageStats({})", self.snapshot())
    }
}

impl Default for StageStats {
    fn default() -> Self {
        let z = || std::array::from_fn(|_| AtomicU64::new(0));
        Self {
            routings: z(),
            resident: z(),
            staged: z(),
            demand: z(),
            prefetched: z(),
            useful: z(),
            wasted: z(),
            prefetch_bytes: AtomicU64::new(0),
            wasted_bytes: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            guess_resident: AtomicU64::new(0),
            guess_in_ring: AtomicU64::new(0),
            guess_low_margin: AtomicU64::new(0),
            guess_no_room: AtomicU64::new(0),
            upload_errors: AtomicU64::new(0),
            h2_read_ns: AtomicU64::new(0),
            moe_read_ns: AtomicU64::new(0),
            issue_ns: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
            host_syncs: AtomicU64::new(0),
            miss_layers: AtomicU64::new(0),
            moe_layers: AtomicU64::new(0),
            guess_named: AtomicU64::new(0),
            guess_total: AtomicU64::new(0),
            la_spared: AtomicU64::new(0),
            la_promoted: AtomicU64::new(0),
            la_guess_ns: AtomicU64::new(0),
            attn: std::sync::Mutex::new(Vec::new()),
        }
    }
}

fn add(a: &AtomicU64, n: u64) {
    a.fetch_add(n, Ordering::Relaxed);
}

impl StageStats {
    /// Which layers are attention layers (for the attention-vs-conv split of the snapshot).
    pub fn set_attn_layers(&self, attn: Vec<bool>) {
        *self.attn.lock().unwrap() = attn;
    }

    /// One decode token-layer's outcome.
    pub fn layer(&self, layer: usize, resident: usize, staged: usize, demand: usize) {
        if layer < MAX_LAYERS {
            add(&self.routings[layer], (resident + staged + demand) as u64);
            add(&self.resident[layer], resident as u64);
            add(&self.staged[layer], staged as u64);
            add(&self.demand[layer], demand as u64);
        }
    }

    /// Totals and the split by the target layer's operator, as JSON for `/v1/placement`.
    pub fn snapshot(&self) -> Value {
        let o = Ordering::Relaxed;
        let attn = self.attn.lock().unwrap().clone();
        let n_layers = attn.len().clamp(1, MAX_LAYERS);
        let sum = |arr: &[AtomicU64; MAX_LAYERS], pick: &dyn Fn(usize) -> bool| -> u64 { (0..n_layers).filter(|&l| pick(l)).map(|l| arr[l].load(o)).sum() };
        let group = |pick: &dyn Fn(usize) -> bool| {
            let (r, res, st, d) = (sum(&self.routings, pick), sum(&self.resident, pick), sum(&self.staged, pick), sum(&self.demand, pick));
            let (p, u, wst) = (sum(&self.prefetched, pick), sum(&self.useful, pick), sum(&self.wasted, pick));
            let rate = |x: u64| if r > 0 { x as f64 / r as f64 } else { 0.0 };
            json!({
                "routings": r, "resident": res, "staged": st, "demand": d,
                "hit_rate": rate(res + st), "resident_rate": rate(res), "staged_rate": rate(st),
                "prefetched": p, "prefetch_useful": u, "prefetch_wasted": wst,
                "prefetch_precision": if u + wst > 0 { u as f64 / (u + wst) as f64 } else { 0.0 },
            })
        };
        let is_attn = |l: usize| attn.get(l).copied().unwrap_or(false);
        let tokens = self.tokens.load(o);
        let per_tok = |x: u64| if tokens > 0 { x as f64 / tokens as f64 } else { 0.0 };
        let demand: u64 = sum(&self.demand, &|_| true);
        let guess_recall = match self.guess_total.load(o) {
            0 => 0.0,
            t => self.guess_named.load(o) as f64 / t as f64,
        };
        json!({
            "schema": "lemonade-burn.stage.v1",
            "tokens": tokens,
            "all": group(&|_| true),
            "attn_targets": group(&|l| is_attn(l)),
            "conv_targets": group(&|l| !is_attn(l)),
            "stalls_per_token": per_tok(demand),
            "prefetch_bytes": self.prefetch_bytes.load(o),
            "wasted_bytes": self.wasted_bytes.load(o),
            "prefetch_mb_per_token": per_tok(self.prefetch_bytes.load(o)) / 1e6,
            "wasted_mb_per_token": per_tok(self.wasted_bytes.load(o)) / 1e6,
            "evictions": self.evictions.load(o),
            "guess_resident": self.guess_resident.load(o),
            "guess_in_ring": self.guess_in_ring.load(o),
            "guess_low_margin": self.guess_low_margin.load(o),
            "guess_no_room": self.guess_no_room.load(o),
            "upload_errors": self.upload_errors.load(o),
            "h2_read_ms_per_token": per_tok(self.h2_read_ns.load(o)) / 1e6,
            "moe_read_ms_per_token": per_tok(self.moe_read_ns.load(o)) / 1e6,
            "issue_ms_per_token": per_tok(self.issue_ns.load(o)) / 1e6,
            "host_syncs_per_token": per_tok(self.host_syncs.load(o)),
            "miss_layers_per_token": per_tok(self.miss_layers.load(o)),
            "moe_layers_per_token": per_tok(self.moe_layers.load(o)),
            "guess_recall": guess_recall,
            "la_spared_per_token": per_tok(self.la_spared.load(o)),
            "la_promoted_per_token": per_tok(self.la_promoted.load(o)),
            "la_guess_ms_per_token": per_tok(self.la_guess_ns.load(o)) / 1e6,
        })
    }
}

/// Least-Stale victim choice, apart from the device so it can be tested. `entries` are
/// `(layer, last_token, margin)`; `tok` is the current token, `cur_layer` the layer the ring is
/// being filled for. Returns the index to evict, or `None` when only entries for `cur_layer`
/// itself are left (those were just staged for the coming layer).
pub fn least_stale(entries: &[(usize, u64, f32)], tok: u64, cur_layer: usize) -> Option<usize> {
    // stale (not touched this token): oldest token first, then the lowest margin
    let stale = entries.iter().enumerate().filter(|(_, e)| e.1 < tok).min_by(|a, b| a.1 .1.cmp(&b.1 .1).then(a.1 .2.partial_cmp(&b.1 .2).unwrap_or(std::cmp::Ordering::Equal)));
    if let Some((i, _)) = stale {
        return Some(i);
    }
    // all current: the ones this token has passed (layer < cur_layer - 1 are done; the layer
    // just before cur_layer is the one running now), furthest back first
    entries.iter().enumerate().filter(|(_, e)| e.0 + 1 < cur_layer).min_by_key(|(_, e)| e.0).map(|(i, _)| i)
}

#[cfg(feature = "gpu")]
pub use ring::{Ring, RingCall};

#[cfg(feature = "gpu")]
mod ring {
    use super::*;
    use lb_q4_gpu::{Gpu, GpuExpert, SideStream};
    use std::collections::HashMap;
    use std::time::Instant;

    struct Entry {
        ex: GpuExpert,
        bytes: usize,
        last_tok: u64,
        margin: f32,
        prefetched: bool,
        used: u32,
    }

    /// The staging ring: device copies of guessed (and, optionally, missed) experts.
    pub struct Ring {
        cap: usize,
        entries: HashMap<(usize, usize), Entry>,
        side: SideStream,
        tok: u64,
    }

    /// What `moe::moe_ffn_with` needs to use and refill the ring for one decode layer.
    pub struct RingCall<'a> {
        pub ring: &'a mut Ring,
        pub gpu: &'a Gpu,
        pub w: &'a Weights,
        pub cfg: &'a StageCfg,
        pub stats: &'a StageStats,
        /// The guess for the next MoE layer, staged after this layer's launches are queued.
        pub next: Option<Guess>,
    }

    impl Ring {
        pub fn new(gpu: &Gpu, cap: usize) -> Self {
            Self { cap, entries: HashMap::new(), side: gpu.side_stream(), tok: 0 }
        }

        pub fn capacity(&self) -> usize {
            self.cap
        }

        pub fn len(&self) -> usize {
            self.entries.len()
        }

        pub fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }

        /// A new decode token: entries not touched from here on become stale.
        pub fn begin_token(&mut self) {
            self.tok += 1;
        }

        pub fn contains(&self, layer: usize, expert: usize) -> bool {
            self.entries.contains_key(&(layer, expert))
        }

        /// The ring's copy of `(layer, expert)`, marked used this token.
        pub fn take_use(&mut self, layer: usize, expert: usize, stats: &StageStats) -> Option<&GpuExpert> {
            let tok = self.tok;
            let e = self.entries.get_mut(&(layer, expert))?;
            if e.prefetched && e.used == 0 && layer < MAX_LAYERS {
                add(&stats.useful[layer], 1);
            }
            e.used += 1;
            e.last_tok = tok;
            Some(&e.ex)
        }

        fn evict(&mut self, key: (usize, usize), stats: &StageStats) {
            if let Some(e) = self.entries.remove(&key) {
                add(&stats.evictions, 1);
                if e.prefetched && e.used == 0 {
                    add(&stats.wasted_bytes, e.bytes as u64);
                    if key.0 < MAX_LAYERS {
                        add(&stats.wasted[key.0], 1);
                    }
                }
            }
        }

        /// Free one slot for `layer`'s guesses; false when nothing may go.
        fn make_room(&mut self, layer: usize, stats: &StageStats) -> bool {
            if self.entries.len() < self.cap {
                return true;
            }
            let keys: Vec<(usize, usize)> = self.entries.keys().copied().collect();
            let view: Vec<(usize, u64, f32)> = keys.iter().map(|k| (k.0, self.entries[k].last_tok, self.entries[k].margin)).collect();
            match least_stale(&view, self.tok, layer) {
                Some(i) => {
                    self.evict(keys[i], stats);
                    true
                }
                None => false,
            }
        }

        /// Put `ex` (a demand miss just served) in the ring, if there is room by the rules.
        pub fn admit(&mut self, layer: usize, expert: usize, ex: GpuExpert, stats: &StageStats) {
            if self.cap == 0 || self.entries.contains_key(&(layer, expert)) || !self.make_room(layer + 1, stats) {
                return;
            }
            let bytes = ex.bytes_on_device();
            self.entries.insert((layer, expert), Entry { ex, bytes, last_tok: self.tok, margin: f32::INFINITY, prefetched: false, used: 1 });
        }

        /// Issue the copies for `guess` on the side stream. `resident` says which experts the
        /// static plan already holds.
        pub fn prefetch(&mut self, gpu: &Gpu, w: &Weights, cfg: &StageCfg, guess: &Guess, resident: &dyn Fn(usize, usize) -> bool, stats: &StageStats) {
            let t = Instant::now();
            let Some(Ffn::Moe { gate, up, down, .. }) = w.layers.get(guess.layer).map(|l| &l.ffn) else { return };
            let l = guess.layer;
            for &e in &guess.experts {
                if resident(l, e) {
                    add(&stats.guess_resident, 1);
                    continue;
                }
                let tok = self.tok;
                if let Some(en) = self.entries.get_mut(&(l, e)) {
                    en.last_tok = tok;
                    en.margin = en.margin.max(guess.margin);
                    add(&stats.guess_in_ring, 1);
                    continue;
                }
                if guess.margin < cfg.min_margin {
                    add(&stats.guess_low_margin, 1);
                    continue;
                }
                if !self.make_room(l, stats) {
                    add(&stats.guess_no_room, 1);
                    continue;
                }
                let up_r = gpu.on_stream(&self.side, || lb_q4_gpu::GpuExpert::upload(gpu, w.mat(&gate.expert(e)), w.mat(&up.expert(e)), w.mat(&down.expert(e))));
                match up_r {
                    Ok(ex) => {
                        let bytes = ex.bytes_on_device();
                        add(&stats.prefetch_bytes, bytes as u64);
                        if l < MAX_LAYERS {
                            add(&stats.prefetched[l], 1);
                        }
                        self.entries.insert((l, e), Entry { ex, bytes, last_tok: tok, margin: guess.margin, prefetched: true, used: 0 });
                    }
                    Err(_) => add(&stats.upload_errors, 1),
                }
            }
            add(&stats.issue_ns, t.elapsed().as_nanos() as u64);
        }

        /// Count the prefetched entries still unused as wasted (end of a measurement) and empty
        /// the ring.
        pub fn drain(&mut self, stats: &StageStats) {
            let keys: Vec<(usize, usize)> = self.entries.keys().copied().collect();
            for k in keys {
                self.evict(k, stats);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_matches_the_router_rule_and_reports_the_margin() {
        // 4 experts, n_embd 2, norm weights 1: h = x / rms(x)
        let x = [3.0f32, 4.0];
        let rms = (12.5f32 + 1e-6).sqrt();
        let h = [3.0 / rms, 4.0 / rms];
        let router = [1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.5, 0.5];
        let bias = [0.0f32, 0.0, 0.0, 0.3];
        let g = rank(&router, &bias, &[1.0, 1.0], &x, 1e-6, 4, 2, 3, 7);
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let s: Vec<f32> = (0..4).map(|e| sig(router[2 * e] * h[0] + router[2 * e + 1] * h[1]) + bias[e]).collect();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
        assert_eq!(g.experts, order[..3].to_vec());
        assert_eq!(g.layer, 7);
        assert!((g.margin - (s[order[1]] - s[order[2]])).abs() < 1e-6);
        // the bias decides selection: expert 3 is first with a 0.3 bias
        assert_eq!(g.experts[0], 3);
    }

    #[test]
    fn the_worker_gives_the_same_guess_as_the_direct_call() {
        let (router, bias, norm) = (vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.5, 0.5], vec![0.0f32; 4], vec![1.0f32; 2]);
        let x = [3.0f32, 4.0];
        let direct = rank(&router, &bias, &norm, &x, 1e-6, 4, 2, 3, 1);
        let w = GuessWorker::new(vec![None, Some((router, bias, norm))], 1e-6, 4, 2, 3, 2);
        w.send(0, x.iter().flat_map(|v| v.to_le_bytes()).collect());
        // depth 2 asks for layers 1 and 2; layer 2 does not exist, so one guess
        assert_eq!(w.recv(), vec![direct]);
    }

    #[test]
    fn the_lane_dot_agrees_with_the_plain_sum() {
        let a: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.11).cos()).collect();
        let plain: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - plain).abs() < 1e-2, "{} vs {plain}", dot(&a, &b));
    }

    #[test]
    fn rank_breaks_ties_by_the_lower_id_like_route_sigmoid() {
        let router = [0.0f32; 8];
        let g = rank(&router, &[0.0; 4], &[1.0, 1.0], &[1.0, 1.0], 1e-6, 4, 2, 4, 0);
        assert_eq!(g.experts, vec![0, 1, 2, 3]);
        assert_eq!(g.margin, 0.0);
        // and the same set route_sigmoid picks for k
        let r = crate::moe::route_sigmoid(&router, &[0.0; 4], &[1.0, 1.0], 4, 2);
        assert_eq!(r.experts, g.experts[..2].to_vec());
    }

    #[test]
    fn least_stale_takes_stale_first_then_passed_layers_never_the_current_one() {
        // (layer, last_token, margin), current token 5, staging for layer 10
        let e = [(10, 5, 0.1), (3, 4, 0.9), (7, 4, 0.2), (8, 5, 0.5)];
        // two stale (token 4): the lower margin goes
        assert_eq!(least_stale(&e, 5, 10), Some(2));
        let e = [(10, 5, 0.1), (3, 3, 0.9), (7, 4, 0.2)];
        // the oldest token goes before a lower margin
        assert_eq!(least_stale(&e, 5, 10), Some(1));
        // nothing stale: the furthest-back layer this token has passed
        let e = [(10, 5, 0.1), (4, 5, 0.9), (6, 5, 0.2), (9, 5, 0.0)];
        assert_eq!(least_stale(&e, 5, 10), Some(1));
        // only the layer being staged and the one running now (9): nothing may go
        let e = [(10, 5, 0.1), (9, 5, 0.0)];
        assert_eq!(least_stale(&e, 5, 10), None);
    }

    #[test]
    fn stats_split_by_target_operator() {
        let s = StageStats::default();
        s.set_attn_layers(vec![false, false, true, false]);
        s.layer(2, 1, 2, 1);
        s.layer(3, 4, 0, 0);
        s.tokens.store(1, Ordering::Relaxed);
        let v = s.snapshot();
        assert_eq!(v["attn_targets"]["routings"], 4);
        assert_eq!(v["attn_targets"]["staged"], 2);
        assert_eq!(v["conv_targets"]["resident"], 4);
        assert_eq!(v["all"]["demand"], 1);
        assert_eq!(v["stalls_per_token"], 1.0);
        assert!((v["all"]["hit_rate"].as_f64().unwrap() - 7.0 / 8.0).abs() < 1e-12);
    }

    #[test]
    fn ring_slots_follow_the_budget_and_the_switch() {
        let mut c = StageCfg { on: true, ring_mib: 256, ..Default::default() };
        assert_eq!(c.ring_slots(5_308_416), 256 * 1024 * 1024 / 5_308_416);
        c.on = false;
        assert_eq!(c.ring_slots(5_308_416), 0);
    }
}
