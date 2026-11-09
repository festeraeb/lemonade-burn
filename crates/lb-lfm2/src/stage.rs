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

/// The arithmetic of [`predict`], apart from the weights so it can be tested.
#[allow(clippy::too_many_arguments)]
pub fn rank(router: &[f32], bias: &[f32], norm: &[f32], x: &[f32], eps: f32, n_expert: usize, k: usize, slots: usize, layer: usize) -> Guess {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let r = 1.0 / (ms + eps).sqrt();
    let h: Vec<f32> = x.iter().zip(norm).map(|(a, b)| a * r * b).collect();
    let score: Vec<f32> = (0..n_expert)
        .map(|e| {
            let logit: f32 = router[e * n..(e + 1) * n].iter().zip(&h).map(|(a, b)| a * b).sum();
            1.0 / (1.0 + (-logit).exp()) + bias[e]
        })
        .collect();
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal));
    let margin = if k >= 1 && k < n_expert { score[order[k - 1]] - score[order[k]] } else { 0.0 };
    order.truncate(slots.min(n_expert));
    Guess { layer, experts: order, margin }
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
