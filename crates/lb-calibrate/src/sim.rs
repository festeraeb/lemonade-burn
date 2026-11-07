//! Discrete-event model of the layer pipeline for ONE decode stream, replayed over real
//! routing traces with measured timings.
//!
//! Model (per token, layers in order; MoE layers are dealt round-robin to `N` cards):
//! * dense part of the layer on the card (or the CPU with `dense_on_gpu = false`);
//! * MoE: experts in the card's pool run on the card (`expert_ms` each, batched launch,
//!   one sync per layer); the others ("misses") run on the CPU meanwhile
//!   (`cpu_expert_ms` each, serial) or are copied on demand (`EXPERT_BYTES / h2d`) and run
//!   on the card; `Best` takes the cheaper per layer. Layer MoE time = max(card, CPU).
//! * a handoff through the host when the next layer sits on another card;
//! * staging: a card holds `slots` experts. If every one of its layers' staged sets fits
//!   (`k * layers_on_card <= slots`) the sets are resident for good ("static") and nothing
//!   is copied per token. Otherwise the card rolls: before each of its layers it needs that
//!   layer's `k` experts, copied while the other cards compute (`lead = lookahead * time of
//!   the other cards' intervening layers`); the part of the copy not hidden is a stall.
//!   A single card has no "other cards": its rolling copies are fully exposed.
//! * optional dynamic admission: spare slots take missed experts after `admit_after`
//!   demands, evicting the coldest admitted one (counts age by halving every `age_every`
//!   demands of that layer).
//!
//! It is a cost model, not a cycle simulator: every number in it is a measured median or
//! an explicit assumption, and the output says which.

use crate::timings::{CardTimings, HostTimings, EXPERT_BYTES};
use crate::trace::Trace;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MissPolicy {
    /// Missed experts run on the CPU, overlapped with the card's resident ones.
    Cpu,
    /// Missed experts are copied to the card on demand and run there (serial on that card).
    OnDemand,
    /// Per layer, whichever of the two is faster (a dispatcher with perfect knowledge).
    Best,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    pub cards: Vec<CardTimings>,
    pub host: HostTimings,
    /// Expert slots per card (whole experts), from its VRAM budget.
    pub slots: Vec<usize>,
    /// Staged (predicted) experts per MoE layer.
    pub k: usize,
    pub lookahead: usize,
    pub dense_on_gpu: bool,
    pub head_on_cpu: bool,
    pub miss_policy: MissPolicy,
    pub dynamic_admit: bool,
    pub admit_after: u32,
    pub age_every: u32,
    /// Multiplier on every card's host->device bandwidth (sensitivity sweeps).
    pub bw_scale: f64,
    /// Fixed per-layer synchronisation cost on the card (ms): the one read-back per layer.
    pub layer_sync_ms: f64,
    pub n_dense_lead: usize,
    pub n_attn_every: usize,
    pub attn_first: usize,
    /// Bytes of one expert (sets the on-demand copy time). Default: the LFM2.5 Q4_0 expert.
    #[serde(default = "default_expert_bytes")]
    pub expert_bytes: usize,
    /// Explicit layer -> card map (contiguous layer splits). `None` keeps the round-robin deal.
    #[serde(default)]
    pub layer_card: Option<Vec<usize>>,
}

fn default_expert_bytes() -> usize {
    EXPERT_BYTES
}

impl SimConfig {
    pub fn single_card(card: CardTimings, host: HostTimings, slots: usize, k: usize) -> Self {
        Self {
            cards: vec![card],
            host,
            slots: vec![slots],
            k,
            lookahead: 1,
            dense_on_gpu: true,
            head_on_cpu: true,
            miss_policy: MissPolicy::Cpu,
            dynamic_admit: false,
            admit_after: 3,
            age_every: 256,
            bw_scale: 1.0,
            layer_sync_ms: 0.3,
            n_dense_lead: 2,
            n_attn_every: 4,
            attn_first: 2,
            expert_bytes: EXPERT_BYTES,
            layer_card: None,
        }
    }

    pub fn is_attn(&self, layer: usize) -> bool {
        layer >= self.attn_first && (layer - self.attn_first) % self.n_attn_every == 0
    }

    pub fn card_of(&self, layer: usize) -> usize {
        if let Some(m) = &self.layer_card {
            return m[layer.min(m.len() - 1)];
        }
        if layer < self.n_dense_lead {
            0
        } else {
            (layer - self.n_dense_lead) % self.cards.len()
        }
    }

    fn layers_on_card(&self, c: usize, n_layer: usize) -> usize {
        (self.n_dense_lead..n_layer).filter(|&l| self.card_of(l) == c).count()
    }

    /// Whether card `c` holds every staged set of its layers at once.
    pub fn is_static(&self, c: usize, n_layer: usize) -> bool {
        self.k * self.layers_on_card(c, n_layer) <= self.slots[c]
    }

    fn dense_ms(&self, layer: usize) -> f64 {
        let c = &self.cards[self.card_of(layer)];
        let (gpu, cpu) = if layer < self.n_dense_lead {
            (c.conv_layer_ms + c.dense_ffn_ms, self.host.cpu_conv_layer_ms + self.host.cpu_dense_ffn_ms)
        } else if self.is_attn(layer) {
            (c.attn_layer_ms, self.host.cpu_attn_layer_ms)
        } else {
            (c.conv_layer_ms, self.host.cpu_conv_layer_ms)
        };
        if self.dense_on_gpu {
            gpu
        } else {
            // activation must visit the host for every layer
            cpu + c.handoff_us / 1000.0
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SimResult {
    pub tokens: usize,
    pub ms_per_token: f64,
    pub tok_s: f64,
    pub hit_rate: f64,
    pub stall_ms_per_token: f64,
    pub stall_pct: f64,
    pub cpu_ms_per_token: f64,
    pub gpu_ms_per_token: f64,
    pub dense_ms_per_token: f64,
    pub handoff_ms_per_token: f64,
    pub staged_mb_per_token: f64,
    pub ondemand_mb_per_token: f64,
    /// Per card: true when its staged sets never move.
    pub static_cards: Vec<bool>,
    pub resident_experts_total: usize,
    pub notes: Vec<String>,
}

struct LayerPool {
    staged: Vec<bool>,
    admitted: Vec<bool>,
    counts: Vec<u32>,
    spare: usize,
    n_admitted: usize,
    demands: u32,
}

/// Replay `trace` (held-out) with the staged sets `staged[layer]` (from the train split).
pub fn simulate(cfg: &SimConfig, trace: &Trace, staged: &[Vec<u16>]) -> SimResult {
    let n_layer = trace.n_layer;
    let n_cards = cfg.cards.len();
    let mut res = SimResult { static_cards: (0..n_cards).map(|c| cfg.is_static(c, n_layer)).collect(), ..Default::default() };
    let b_mb = cfg.expert_bytes as f64 / 1e6;
    // spare slots per layer for dynamic admission (static cards only)
    let mut pools: Vec<LayerPool> = (0..n_layer)
        .map(|l| {
            let c = cfg.card_of(l);
            let on_card = cfg.layers_on_card(c, n_layer).max(1);
            let spare = if l >= cfg.n_dense_lead && res.static_cards[c] && cfg.dynamic_admit {
                (cfg.slots[c].saturating_sub(cfg.k * on_card)) / on_card
            } else {
                0
            };
            let mut st = vec![false; trace.n_expert];
            if l >= cfg.n_dense_lead {
                for &e in staged.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
                    st[e as usize] = true;
                }
            }
            LayerPool { staged: st, admitted: vec![false; trace.n_expert], counts: vec![0; trace.n_expert], spare, n_admitted: 0, demands: 0 }
        })
        .collect();
    res.resident_experts_total = pools.iter().map(|p| p.staged.iter().filter(|&&s| s).count()).sum::<usize>()
        .min(cfg.slots.iter().sum());

    // steady-state layer times for the lead-time estimate: filled on the first token pass
    let mut layer_ms_est = vec![0f64; n_layer];
    let (mut hits_total, mut demands_total) = (0u64, 0u64);
    let (mut sum_ms, mut sum_stall, mut sum_cpu, mut sum_gpu, mut sum_dense, mut sum_hand, mut sum_staged, mut sum_od) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut tokens = 0usize;
    for r in &trace.requests {
        for t in &r.tokens {
            tokens += 1;
            let mut tok_ms = 0f64;
            for l in 0..n_layer {
                let c = cfg.card_of(l);
                let card = &cfg.cards[c];
                let bw = card.h2d_mb_s * cfg.bw_scale;
                let dense = cfg.dense_ms(l);
                sum_dense += dense;
                let mut layer_ms = dense;
                if l >= cfg.n_dense_lead {
                    let es = t.experts.get(l).map(|v| v.as_slice()).unwrap_or(&[]);
                    let pool = &mut pools[l];
                    let mut hits = 0usize;
                    let mut miss_ids: Vec<u16> = Vec::new();
                    for &e in es {
                        demands_total += 1;
                        pool.demands += 1;
                        pool.counts[e as usize] = pool.counts[e as usize].saturating_add(1);
                        if pool.staged[e as usize] || pool.admitted[e as usize] {
                            hits += 1;
                        } else {
                            miss_ids.push(e);
                        }
                    }
                    hits_total += hits as u64;
                    let misses = miss_ids.len();
                    let gpu_ms = hits as f64 * card.expert_ms + cfg.layer_sync_ms;
                    let cpu_ms = misses as f64 * cfg.host.cpu_expert_ms;
                    let od_ms = misses as f64 * (b_mb / bw * 1000.0 + card.expert_ms);
                    let (moe_ms, od_bytes) = match cfg.miss_policy {
                        MissPolicy::Cpu => (gpu_ms.max(cpu_ms), 0.0),
                        MissPolicy::OnDemand => (gpu_ms + od_ms, misses as f64 * b_mb),
                        MissPolicy::Best => {
                            if cpu_ms <= od_ms {
                                (gpu_ms.max(cpu_ms), 0.0)
                            } else {
                                (gpu_ms + od_ms, misses as f64 * b_mb)
                            }
                        }
                    };
                    sum_gpu += gpu_ms;
                    sum_cpu += if od_bytes > 0.0 { 0.0 } else { cpu_ms };
                    sum_od += od_bytes;
                    layer_ms += moe_ms;
                    // dynamic admission on spare slots (deferred copies: cost counted as staging bytes)
                    if pool.spare > 0 {
                        for &e in &miss_ids {
                            if pool.counts[e as usize] >= cfg.admit_after {
                                if pool.n_admitted < pool.spare {
                                    pool.admitted[e as usize] = true;
                                    pool.n_admitted += 1;
                                    sum_staged += b_mb;
                                } else {
                                    // evict the coldest admitted expert if this one is hotter
                                    let coldest = (0..trace.n_expert).filter(|&i| pool.admitted[i]).min_by_key(|&i| pool.counts[i]);
                                    if let Some(ci) = coldest {
                                        if pool.counts[ci] < pool.counts[e as usize] {
                                            pool.admitted[ci] = false;
                                            pool.admitted[e as usize] = true;
                                            sum_staged += b_mb;
                                        }
                                    }
                                }
                            }
                        }
                        if pool.demands >= cfg.age_every {
                            pool.demands = 0;
                            for v in pool.counts.iter_mut() {
                                *v /= 2;
                            }
                        }
                    }
                    // rolling staging for non-static cards
                    if !res.static_cards[c] {
                        let stage_ms = cfg.k as f64 * b_mb / bw * 1000.0;
                        sum_staged += cfg.k as f64 * b_mb;
                        let lead = if n_cards == 1 {
                            0.0
                        } else {
                            // time the other cards spend between this card's consecutive layers
                            let mut lead = 0.0;
                            for i in 1..n_cards {
                                let other = (l + i).min(n_layer - 1);
                                lead += layer_ms_est[other];
                            }
                            lead * cfg.lookahead as f64
                        };
                        let stall = (stage_ms - lead).max(0.0);
                        sum_stall += stall;
                        layer_ms += stall;
                    }
                }
                // handoff to the next card
                if l + 1 < n_layer && cfg.card_of(l + 1) != c {
                    let h = card.handoff_us / 1000.0;
                    sum_hand += h;
                    layer_ms += h;
                }
                layer_ms_est[l] = if layer_ms_est[l] == 0.0 { layer_ms } else { 0.9 * layer_ms_est[l] + 0.1 * layer_ms };
                tok_ms += layer_ms;
            }
            if cfg.head_on_cpu {
                tok_ms += cfg.host.cpu_head_ms + cfg.cards[cfg.card_of(n_layer - 1)].handoff_us / 2000.0;
            }
            sum_ms += tok_ms;
        }
    }
    let n = tokens.max(1) as f64;
    res.tokens = tokens;
    res.ms_per_token = sum_ms / n;
    res.tok_s = if sum_ms > 0.0 { 1000.0 / res.ms_per_token } else { 0.0 };
    res.hit_rate = if demands_total > 0 { hits_total as f64 / demands_total as f64 } else { 0.0 };
    res.stall_ms_per_token = sum_stall / n;
    res.stall_pct = if sum_ms > 0.0 { 100.0 * sum_stall / sum_ms } else { 0.0 };
    res.cpu_ms_per_token = sum_cpu / n;
    res.gpu_ms_per_token = sum_gpu / n;
    res.dense_ms_per_token = sum_dense / n;
    res.handoff_ms_per_token = sum_hand / n;
    res.staged_mb_per_token = sum_staged / n;
    res.ondemand_mb_per_token = sum_od / n;
    if !cfg.cards.iter().all(|c| c.measured) {
        res.notes.push("some card timings are hypothetical (not measured)".into());
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{top_k, Request, TokenRoute};

    fn card(expert_ms: f64, bw: f64) -> CardTimings {
        CardTimings {
            name: "t".into(),
            host: "h".into(),
            device: "d".into(),
            expert_ms,
            expert_m4_ms: expert_ms,
            h2d_mb_s: bw,
            h2d_bulk_mb_s: bw,
            handoff_us: 100.0,
            conv_layer_ms: 1.0,
            attn_layer_ms: 1.0,
            dense_ffn_ms: 2.0,
            vram_free_mb: 1e9,
            measured: true,
        }
    }
    fn host() -> HostTimings {
        HostTimings { host: "h".into(), threads: 8, cpu_expert_ms: 2.0, cpu_head_ms: 10.0, cpu_conv_layer_ms: 3.0, cpu_attn_layer_ms: 3.0, cpu_dense_ffn_ms: 6.0, measured: true }
    }
    fn trace(n_layer: usize) -> Trace {
        // every token routes to experts 0..4 in every MoE layer
        let tok = TokenRoute { experts: (0..n_layer).map(|l| if l < 2 { vec![] } else { vec![0, 1, 2, 3] }).collect() };
        Trace { n_layer, n_expert: 8, requests: vec![Request { id: "r".into(), tokens: vec![tok; 10] }] }
    }

    #[test]
    fn all_resident_single_card_has_no_stall_and_full_hits() {
        let t = trace(6);
        let cfg = SimConfig::single_card(card(0.5, 5000.0), host(), 1000, 4);
        let staged = top_k(&t.histogram(), 4);
        let r = simulate(&cfg, &t, &staged);
        assert_eq!(r.hit_rate, 1.0);
        assert_eq!(r.stall_ms_per_token, 0.0);
        assert!(r.static_cards[0]);
        // 2 dense lead (1+2 each) + 4 moe layers (1 + 4*0.5 + 0.3) + head 10 + half handoff
        let want = 2.0 * 3.0 + 4.0 * (1.0 + 2.0 + 0.3) + 10.0 + 0.05;
        assert!((r.ms_per_token - want).abs() < 1e-6, "{} vs {want}", r.ms_per_token);
    }

    #[test]
    fn single_card_rolling_is_fully_exposed_and_two_cards_hide_it() {
        let t = trace(6);
        let mut cfg = SimConfig::single_card(card(0.5, 1000.0), host(), 4, 4); // 4 slots: one layer set at a time
        let staged = top_k(&t.histogram(), 4);
        let one = simulate(&cfg, &t, &staged);
        assert!(!one.static_cards[0]);
        let stage_ms = 4.0 * EXPERT_BYTES as f64 / 1e6 / 1000.0 * 1000.0; // 21 ms per layer
        assert!((one.stall_ms_per_token - 4.0 * stage_ms).abs() < 1e-6);
        cfg.cards.push(card(0.5, 1000.0));
        cfg.slots.push(4);
        let two = simulate(&cfg, &t, &staged);
        assert!(two.stall_ms_per_token < one.stall_ms_per_token);
        assert!(two.handoff_ms_per_token > 0.0);
    }

    #[test]
    fn misses_go_to_cpu_or_on_demand() {
        let t = trace(4);
        let mut cfg = SimConfig::single_card(card(0.5, 100_000.0), host(), 1000, 2); // only experts 0,1 staged
        let staged = top_k(&t.histogram(), 2);
        let cpu = simulate(&cfg, &t, &staged);
        assert!((cpu.hit_rate - 0.5).abs() < 1e-9);
        cfg.miss_policy = MissPolicy::OnDemand;
        let od = simulate(&cfg, &t, &staged);
        assert!(od.ondemand_mb_per_token > 0.0 && od.cpu_ms_per_token == 0.0);
        cfg.miss_policy = MissPolicy::Best;
        let best = simulate(&cfg, &t, &staged);
        assert!(best.ms_per_token <= cpu.ms_per_token + 1e-9 && best.ms_per_token <= od.ms_per_token + 1e-9);
    }

    #[test]
    fn dynamic_admission_turns_misses_into_hits() {
        let t = trace(4);
        let mut cfg = SimConfig::single_card(card(0.5, 100_000.0), host(), 1000, 2);
        cfg.dynamic_admit = true;
        cfg.admit_after = 2;
        let staged = top_k(&t.histogram(), 2);
        let r = simulate(&cfg, &t, &staged);
        assert!(r.hit_rate > 0.5, "{}", r.hit_rate);
    }
}
