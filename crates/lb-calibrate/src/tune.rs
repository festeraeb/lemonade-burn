//! The tuner: enumerate staging configurations the box can hold, simulate each on the
//! held-out trace with staged sets fitted on the train trace, keep the best, and write
//! the engine's staging config. Also the sensitivity sweeps for the design doc.

use crate::sim::{simulate, MissPolicy, SimConfig, SimResult};
use crate::timings::{CardTimings, HostTimings, EXPERT_BYTES};
use crate::trace::{coverage, top_k, Trace};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardAssign {
    pub host: String,
    pub device: String,
    pub name: String,
    pub layers: Vec<usize>,
    pub slots: usize,
    pub budget_mb: f64,
    pub is_static: bool,
}

/// What the engine loads: how many experts to stage per layer, where, and how to treat misses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StagingConfig {
    pub version: u32,
    pub model: String,
    pub k_per_layer: usize,
    pub lookahead: usize,
    pub miss_policy: MissPolicy,
    pub dynamic_admit: bool,
    pub dense_on_gpu: bool,
    pub cards: Vec<CardAssign>,
    /// Staged expert ids per layer (the predictor's output, fitted on the train trace).
    pub staged: Vec<Vec<u16>>,
    pub predicted: SimResult,
    pub held_out_coverage: f64,
    pub train_tokens: usize,
    pub held_out_tokens: usize,
    pub measured_on: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub n_cards: usize,
    pub k: usize,
    pub lookahead: usize,
    pub miss_policy: MissPolicy,
    pub dynamic_admit: bool,
    pub dense_on_gpu: bool,
    pub result: SimResult,
}

pub struct TuneInput<'a> {
    pub cards: Vec<CardTimings>,
    /// VRAM budget per card for the expert pool (MB), after the dense weights and reserve.
    pub budgets_mb: Vec<f64>,
    pub host: HostTimings,
    pub train: &'a Trace,
    pub test: &'a Trace,
    pub ks: Vec<usize>,
    pub lookaheads: Vec<usize>,
    pub dense_options: Vec<bool>,
}

pub fn slots_for(budget_mb: f64) -> usize {
    (budget_mb * 1e6 / EXPERT_BYTES as f64).floor().max(0.0) as usize
}

/// Run every candidate; returns them sorted best first.
pub fn enumerate(inp: &TuneInput<'_>) -> Vec<Candidate> {
    let hist = inp.train.histogram();
    let mut out = Vec::new();
    for n in 1..=inp.cards.len() {
        let cards: Vec<CardTimings> = inp.cards[..n].to_vec();
        let slots: Vec<usize> = inp.budgets_mb[..n].iter().map(|&b| slots_for(b)).collect();
        for &k in &inp.ks {
            if slots.iter().any(|&s| s < k) {
                continue; // a card must hold at least one layer's set
            }
            let staged = top_k(&hist, k);
            for &la in &inp.lookaheads {
                for &dense in &inp.dense_options {
                    for mp in [MissPolicy::Cpu, MissPolicy::Best] {
                        for dyn_admit in [false, true] {
                            let mut cfg = SimConfig::single_card(cards[0].clone(), inp.host.clone(), slots[0], k);
                            cfg.cards = cards.clone();
                            cfg.slots = slots.clone();
                            cfg.lookahead = la;
                            cfg.dense_on_gpu = dense;
                            cfg.miss_policy = mp;
                            cfg.dynamic_admit = dyn_admit;
                            let result = simulate(&cfg, inp.test, &staged);
                            out.push(Candidate { n_cards: n, k, lookahead: la, miss_policy: mp, dynamic_admit: dyn_admit, dense_on_gpu: dense, result });
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| b.result.tok_s.partial_cmp(&a.result.tok_s).unwrap());
    out
}

pub fn best_config(inp: &TuneInput<'_>, model: &str) -> Option<(StagingConfig, Vec<Candidate>)> {
    let cands = enumerate(inp);
    let best = cands.first()?.clone();
    let hist = inp.train.histogram();
    let staged = top_k(&hist, best.k);
    let n_layer = inp.train.n_layer;
    let mut cfg = SimConfig::single_card(inp.cards[0].clone(), inp.host.clone(), slots_for(inp.budgets_mb[0]), best.k);
    cfg.cards = inp.cards[..best.n_cards].to_vec();
    cfg.slots = inp.budgets_mb[..best.n_cards].iter().map(|&b| slots_for(b)).collect();
    let cards = (0..best.n_cards)
        .map(|c| CardAssign {
            host: inp.cards[c].host.clone(),
            device: inp.cards[c].device.clone(),
            name: inp.cards[c].name.clone(),
            layers: (0..n_layer).filter(|&l| cfg.card_of(l) == c && l >= cfg.n_dense_lead).collect(),
            slots: cfg.slots[c],
            budget_mb: inp.budgets_mb[c],
            is_static: cfg.is_static(c, n_layer),
        })
        .collect();
    let mut notes = vec![
        "decode, one stream; prefill and multi-stream pipelining not modelled".into(),
        "staged sets = top-k per layer of the train trace histogram; judged on the held-out trace".into(),
    ];
    if inp.cards.iter().any(|c| !c.measured) {
        notes.push("includes hypothetical cards".into());
    }
    Some((
        StagingConfig {
            version: 1,
            model: model.to_string(),
            k_per_layer: best.k,
            lookahead: best.lookahead,
            miss_policy: best.miss_policy,
            dynamic_admit: best.dynamic_admit,
            dense_on_gpu: best.dense_on_gpu,
            cards,
            held_out_coverage: coverage(inp.test, &staged),
            staged,
            predicted: best.result.clone(),
            train_tokens: inp.train.n_tokens(),
            held_out_tokens: inp.test.n_tokens(),
            measured_on: inp.cards.iter().map(|c| c.name.clone()).collect(),
            notes,
        },
        cands,
    ))
}

/// Bandwidth sensitivity of one configuration: tok/s and stall% per bandwidth multiplier.
pub fn bw_sweep(cfg: &SimConfig, test: &Trace, staged: &[Vec<u16>], scales: &[f64]) -> Vec<(f64, SimResult)> {
    scales
        .iter()
        .map(|&s| {
            let mut c = cfg.clone();
            c.bw_scale = s;
            (s, simulate(&c, test, staged))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{Request, TokenRoute};

    #[test]
    fn tuner_prefers_more_resident_experts_when_vram_allows() {
        let card = CardTimings {
            name: "c".into(),
            host: "h".into(),
            device: "d".into(),
            expert_ms: 0.3,
            expert_m4_ms: 0.3,
            h2d_mb_s: 3000.0,
            h2d_bulk_mb_s: 3000.0,
            handoff_us: 200.0,
            conv_layer_ms: 0.5,
            attn_layer_ms: 0.5,
            dense_ffn_ms: 1.0,
            vram_free_mb: 8000.0,
            measured: true,
        };
        let host = HostTimings { host: "h".into(), threads: 32, cpu_expert_ms: 1.0, cpu_head_ms: 30.0, cpu_conv_layer_ms: 2.0, cpu_attn_layer_ms: 1.5, cpu_dense_ffn_ms: 6.0, measured: true };
        // 6 layers, experts spread over 8 ids
        let mut reqs = Vec::new();
        for r in 0..8u16 {
            let toks: Vec<TokenRoute> = (0..20u16)
                .map(|t| TokenRoute { experts: (0..6).map(|l| if l < 2 { vec![] } else { vec![(t + r) % 8, (t * 3 + l as u16) % 8, (r + l as u16) % 8, (t + 1) % 8] }).collect() })
                .collect();
            reqs.push(Request { id: format!("{r}"), tokens: toks });
        }
        let trace = Trace { n_layer: 6, n_expert: 8, requests: reqs };
        let (train, test) = trace.split(4);
        let inp = TuneInput { cards: vec![card], budgets_mb: vec![1000.0], host, train: &train, test: &test, ks: vec![2, 4, 8], lookaheads: vec![1], dense_options: vec![true] };
        let (best, cands) = best_config(&inp, "test").unwrap();
        assert!(!cands.is_empty());
        assert_eq!(best.k_per_layer, 8, "all 8 experts fit, so the best k is 8");
        assert_eq!(best.predicted.hit_rate, 1.0);
        assert!(best.cards[0].is_static);
    }
}
