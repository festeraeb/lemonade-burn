//! Router maths and set-overlap scores.
//!
//! The router here must pick exactly what `lb_lfm2::moe::route_sigmoid` picks: sigmoid of the
//! logits, selection by `p + bias` in a stable descending sort (ties keep the lower expert id),
//! top `k`. The *margin* is how far the k-th selected expert's selection score sits above the
//! best unselected one: a small margin means the gate nearly chose differently.

use serde::{Deserialize, Serialize};

/// `x / rms(x) * w`, the LFM2 pre-norm (`eps` inside the root).
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * inv * b).collect()
}

/// Sigmoid router probabilities `p` and selection scores `p + bias` for one token.
/// `router` is `[n_expert][n_embd]`.
pub fn router_scores(router: &[f32], bias: &[f32], h: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n_embd = h.len();
    let n_expert = bias.len();
    assert_eq!(router.len(), n_expert * n_embd);
    let probs: Vec<f32> = (0..n_expert)
        .map(|e| {
            let logit: f32 = router[e * n_embd..(e + 1) * n_embd].iter().zip(h).map(|(a, b)| a * b).sum();
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();
    let sel = probs.iter().zip(bias).map(|(p, b)| p + b).collect();
    (probs, sel)
}

/// Top `k` by selection score (same order rule as the engine's router) and the margin
/// `sel[k-th chosen] - sel[best unchosen]` (0 when `k == n`).
pub fn topk_margin(sel: &[f32], k: usize) -> (Vec<usize>, f32) {
    let mut order: Vec<usize> = (0..sel.len()).collect();
    order.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap_or(std::cmp::Ordering::Equal));
    let chosen = order[..k].to_vec();
    let margin = if k < sel.len() { sel[order[k - 1]] - sel[order[k]] } else { 0.0 };
    (chosen, margin)
}

/// Experts in both sets (sets have no repeats).
pub fn overlap(a: &[usize], b: &[usize]) -> usize {
    a.iter().filter(|x| b.contains(x)).count()
}

/// One predicted set scored against the actual set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SetScore {
    /// Actual experts named by the prediction.
    pub hits: usize,
    /// Size of the actual set (the model's top-k).
    pub k: usize,
    pub jaccard: f64,
    /// Every actual expert was predicted.
    pub all_hit: bool,
}

pub fn score_sets(pred: &[usize], actual: &[usize]) -> SetScore {
    let hits = overlap(pred, actual);
    let union = pred.len() + actual.len() - hits;
    SetScore { hits, k: actual.len(), jaccard: if union == 0 { 1.0 } else { hits as f64 / union as f64 }, all_hit: hits == actual.len() }
}

/// Running sums of [`SetScore`]s. `recall()` is the mean of hits/k (NautiBurn's
/// "4.21/8 = 53%" metric), `jaccard()` the mean Jaccard, `all_k()` the share of tokens whose
/// whole set was named.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Acc {
    pub n: u64,
    pub hits: u64,
    pub k_total: u64,
    pub jaccard_sum: f64,
    pub all_hit: u64,
}

impl Acc {
    pub fn add(&mut self, s: SetScore) {
        self.n += 1;
        self.hits += s.hits as u64;
        self.k_total += s.k as u64;
        self.jaccard_sum += s.jaccard;
        self.all_hit += s.all_hit as u64;
    }
    pub fn merge(&mut self, o: &Acc) {
        self.n += o.n;
        self.hits += o.hits;
        self.k_total += o.k_total;
        self.jaccard_sum += o.jaccard_sum;
        self.all_hit += o.all_hit;
    }
    pub fn recall(&self) -> f64 {
        if self.k_total == 0 { f64::NAN } else { self.hits as f64 / self.k_total as f64 }
    }
    pub fn jaccard(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.jaccard_sum / self.n as f64 }
    }
    pub fn all_k(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.all_hit as f64 / self.n as f64 }
    }
}

/// The `k` most frequent experts in `counts` (ties: lower id first).
pub fn top_by_count(counts: &[u64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    order.truncate(k);
    order
}

/// Share of all routings taken by the `top` most-used experts of a layer.
pub fn top_share(counts: &[u64], top: usize) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return f64::NAN;
    }
    let mut c = counts.to_vec();
    c.sort_unstable_by(|a, b| b.cmp(a));
    c.iter().take(top).sum::<u64>() as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_scores_on_hand_made_sets() {
        let s = score_sets(&[1, 2, 3, 4], &[4, 3, 2, 1]);
        assert_eq!((s.hits, s.k, s.all_hit), (4, 4, true));
        assert!((s.jaccard - 1.0).abs() < 1e-12);
        let s = score_sets(&[1, 2, 5, 6], &[1, 2, 3, 4]);
        assert_eq!((s.hits, s.all_hit), (2, false));
        assert!((s.jaccard - 2.0 / 6.0).abs() < 1e-12);
        let s = score_sets(&[7, 8, 9, 10], &[1, 2, 3, 4]);
        assert_eq!((s.hits, s.jaccard, s.all_hit), (0, 0.0, false));
    }

    #[test]
    fn accumulator_means() {
        let mut a = Acc::default();
        a.add(score_sets(&[1, 2, 3, 4], &[1, 2, 3, 4])); // 4/4, J 1
        a.add(score_sets(&[1, 2, 5, 6], &[1, 2, 3, 4])); // 2/4, J 1/3
        assert!((a.recall() - 0.75).abs() < 1e-12);
        assert!((a.jaccard() - (1.0 + 1.0 / 3.0) / 2.0).abs() < 1e-12);
        assert!((a.all_k() - 0.5).abs() < 1e-12);
        let mut b = Acc::default();
        b.add(score_sets(&[9, 8, 7, 6], &[1, 2, 3, 4]));
        a.merge(&b);
        assert_eq!(a.n, 3);
        assert!((a.recall() - 6.0 / 12.0).abs() < 1e-12);
        assert!(Acc::default().recall().is_nan());
    }

    #[test]
    fn topk_matches_engine_router_including_ties_and_margin() {
        // 4 experts, 2-dim hidden; bias lifts expert 3 into the top 2.
        let router = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -1.0, 0.0];
        let bias = vec![0.0, 0.0, 0.0, 0.6];
        let h = vec![2.0, 1.0];
        let (probs, sel) = router_scores(&router, &bias, &h);
        let (chosen, margin) = topk_margin(&sel, 2);
        let engine = lb_lfm2::moe::route_sigmoid(&router, &bias, &h, 4, 2);
        assert_eq!(chosen, engine.experts);
        // sigmoid(2)=0.8808, sigmoid(1)=0.7311, sigmoid(1.5)=0.8176, sigmoid(-2)+0.6=0.7192
        assert_eq!(chosen, vec![0, 2]);
        assert!((margin - (probs[2] - probs[1])).abs() < 1e-6);
        // exact tie: the lower id wins, as in the engine's stable sort
        let (c, m) = topk_margin(&[0.5, 0.5, 0.1], 1);
        assert_eq!((c, m), (vec![0], 0.0));
    }

    #[test]
    fn rms_norm_unit_rows() {
        let y = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0);
        let r = (12.5f32).sqrt();
        assert!((y[0] - 3.0 / r).abs() < 1e-6 && (y[1] - 8.0 / r).abs() < 1e-6);
    }

    #[test]
    fn concentration_and_frequency() {
        let counts = [10, 0, 30, 60];
        assert_eq!(top_by_count(&counts, 2), vec![3, 2]);
        assert!((top_share(&counts, 1) - 0.6).abs() < 1e-12);
        assert!((top_share(&counts, 2) - 0.9).abs() < 1e-12);
        assert!((top_share(&counts, 9) - 1.0).abs() < 1e-12);
        assert_eq!(top_by_count(&[1, 1, 1], 2), vec![0, 1]);
    }
}
