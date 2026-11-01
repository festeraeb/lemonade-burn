//! Routing traces from NautiBurn scrape files: per request, per token, per MoE layer, the
//! set of experts the router chose. The simulator replays them token by token; the
//! predictor is fitted on one subset (train) and judged on another (held out).

use anyhow::Result;
use lb_lfm2::moe::{read_scrape_dir, Scrape};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// One token's routing: `experts[layer]` (empty for dense layers).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenRoute {
    pub experts: Vec<Vec<u16>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    pub id: String,
    pub tokens: Vec<TokenRoute>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    pub n_layer: usize,
    pub n_expert: usize,
    pub requests: Vec<Request>,
}

impl Trace {
    pub fn from_scrapes(scrapes: &[Scrape], n_layer: usize, n_expert: usize) -> Self {
        let mut requests = Vec::new();
        for s in scrapes {
            let mut by_tok: BTreeMap<u32, Vec<Vec<u16>>> = BTreeMap::new();
            for a in &s.activations {
                if (a.layer as usize) >= n_layer {
                    continue;
                }
                let t = by_tok.entry(a.token_pos).or_insert_with(|| vec![Vec::new(); n_layer]);
                t[a.layer as usize] = a.expert_ids.iter().map(|&e| e as u16).collect();
            }
            if by_tok.is_empty() {
                continue;
            }
            requests.push(Request { id: s.request_id.clone(), tokens: by_tok.into_values().map(|experts| TokenRoute { experts }).collect() });
        }
        Self { n_layer, n_expert, requests }
    }

    pub fn load_dir(dir: &Path, n_layer: usize, n_expert: usize) -> Result<Self> {
        let scrapes = read_scrape_dir(dir)?;
        Ok(Self::from_scrapes(&scrapes, n_layer, n_expert))
    }

    pub fn n_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.tokens.len()).sum()
    }

    /// Deterministic split by request index: every `k`-th request is held out.
    pub fn split(&self, k: usize) -> (Trace, Trace) {
        let mut a = Trace { n_layer: self.n_layer, n_expert: self.n_expert, requests: Vec::new() };
        let mut b = a.clone();
        for (i, r) in self.requests.iter().enumerate() {
            if k > 0 && i % k == k - 1 {
                b.requests.push(r.clone());
            } else {
                a.requests.push(r.clone());
            }
        }
        (a, b)
    }

    /// `counts[layer][expert]` over every token.
    pub fn histogram(&self) -> Vec<Vec<u64>> {
        let mut h = vec![vec![0u64; self.n_expert]; self.n_layer];
        for r in &self.requests {
            for t in &r.tokens {
                for (l, es) in t.experts.iter().enumerate() {
                    for &e in es {
                        h[l][e as usize] += 1;
                    }
                }
            }
        }
        h
    }
}

/// Top-`k` experts per layer by count (ties: lower id first). Layers with no data: empty.
pub fn top_k(hist: &[Vec<u64>], k: usize) -> Vec<Vec<u16>> {
    hist.iter()
        .map(|row| {
            let mut idx: Vec<usize> = (0..row.len()).filter(|&e| row[e] > 0).collect();
            idx.sort_by(|&a, &b| row[b].cmp(&row[a]).then(a.cmp(&b)));
            idx.into_iter().take(k).map(|e| e as u16).collect()
        })
        .collect()
}

/// Fraction of routed expert demands of `trace` that fall in `staged[layer]`.
pub fn coverage(trace: &Trace, staged: &[Vec<u16>]) -> f64 {
    let (mut hit, mut total) = (0u64, 0u64);
    for r in &trace.requests {
        for t in &r.tokens {
            for (l, es) in t.experts.iter().enumerate() {
                for e in es {
                    total += 1;
                    if staged.get(l).map_or(false, |s| s.contains(e)) {
                        hit += 1;
                    }
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace() -> Trace {
        let mk = |ids: Vec<Vec<u16>>| Request { id: "r".into(), tokens: ids.into_iter().map(|e| TokenRoute { experts: vec![vec![], e] }).collect() };
        Trace { n_layer: 2, n_expert: 4, requests: vec![mk(vec![vec![0, 1], vec![0, 2]]), mk(vec![vec![0, 3]])] }
    }

    #[test]
    fn histogram_topk_coverage() {
        let t = trace();
        let h = t.histogram();
        assert_eq!(h[1], vec![3, 1, 1, 1]);
        let s = top_k(&h, 2);
        assert_eq!(s[1], vec![0, 1]);
        assert!((coverage(&t, &s) - 4.0 / 6.0).abs() < 1e-9);
        let (a, b) = t.split(2);
        assert_eq!((a.requests.len(), b.requests.len()), (1, 1));
    }
}
