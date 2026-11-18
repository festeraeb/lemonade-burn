//! The device MTP speculative loop: draft with the nextn block, verify on the trunk as one batched chunk, keep the longest matching prefix,
//! roll the recurrent state back with the checkpoint/replay of `Pipeline::run_rows` / `spec_rollback`. Greedy only.
//!
//! Draft depth: fixed (`SpecDev::new`), or adaptive (`SpecDev::adaptive`): each round picks the depth `k` in `0..=max` that maximises expected tokens
//! per second, `E_k / r_k`, where `E_k = 1 + sum_{j<=k} prod_{i<=j} q_i` from running acceptance rates `q_i` (draft `i` accepted given drafts `< i`
//! were) and `r_k` is the round time at depth `k` (draft steps + a verify of `k + 1` rows; `k = 0` is a plain step through the 1-row verify graph,
//! which still feeds the head's cache). Round times come from one timed pass per depth at the first rounds, scaled by how the current depth's
//! observed time drifts from its own baseline (context growth). Depths above the current one get a probe round every `PROBE_EVERY` rounds so their
//! acceptance does not go stale. The output is the trunk's greedy tokens either way; only speed changes.
//!
//! Invariant between calls: the trunk has consumed `trunk_len` tokens = the caller's `cache.len` + `stash.len()`; `stash` holds tokens the verify
//! already produced but the caller has not been handed yet; `h_pending` is the trunk hidden of position `trunk_len - 1`.

use crate::pipeline::Pipeline;
use anyhow::Result;
use std::collections::VecDeque;

#[derive(Default, Debug, Clone)]
pub struct SpecCounters {
    pub rounds: u64,
    /// `prefix_ok[i]` = rounds where drafts `1..=i+1` were all accepted
    pub prefix_ok: [u64; 8],
    pub emitted: u64,
    pub draft_s: f64,
    pub verify_s: f64,
    pub fixup_s: f64,
    /// rounds run at each depth (index = depth)
    pub at_depth: [u64; 8],
}

/// Rounds between probes of the next deeper depth when adaptive.
const PROBE_EVERY: u64 = 16;
/// Timed rounds per depth before the picker starts choosing (the fastest is the baseline: the first replay of a graph can be slow).
const BASE_SAMPLES: u32 = 2;
/// Weight of a new observation in the acceptance and drift averages.
const EMA: f64 = 0.1;

/// The adaptive depth rule, separate from the device so it can be tested.
#[derive(Debug, Clone)]
pub struct DepthPicker {
    pub max: usize,
    /// `q[i]` = rate draft `i + 1` is accepted given drafts `1..=i` were
    pub q: Vec<f64>,
    /// baseline round seconds per depth (`None` until measured)
    pub base: Vec<Option<f64>>,
    /// timed samples taken per depth for its baseline (the baseline is the fastest of `BASE_SAMPLES`)
    pub base_n: Vec<u32>,
    /// observed / baseline time of the depth in use (context growth makes every depth slower alike)
    pub drift: f64,
    pub rounds: u64,
}

impl DepthPicker {
    pub fn new(max: usize) -> Self {
        DepthPicker { max, q: vec![0.75; max], base: vec![None; max + 1], base_n: vec![0; max + 1], drift: 1.0, rounds: 0 }
    }

    pub fn expected_tokens(&self, k: usize) -> f64 {
        let (mut e, mut p) = (1.0, 1.0);
        for i in 0..k {
            p *= self.q[i];
            e += p;
        }
        e
    }

    /// Next depth: an unmeasured depth first (shallow to deep), then a probe of `best + 1` every `PROBE_EVERY` rounds, else the best rate.
    pub fn pick(&self) -> usize {
        if let Some(k) = self.base_n.iter().position(|&n| n < BASE_SAMPLES) {
            return k;
        }
        let best = self.best();
        if best < self.max && self.rounds % PROBE_EVERY == PROBE_EVERY - 1 {
            return best + 1;
        }
        best
    }

    pub fn best(&self) -> usize {
        let mut best = (0usize, f64::MIN);
        for k in 0..=self.max {
            if let Some(b) = self.base[k] {
                let rate = self.expected_tokens(k) / (b * self.drift);
                if rate > best.1 {
                    best = (k, rate);
                }
            }
        }
        best.0
    }

    /// A round at depth `k` took `secs` and accepted `a` drafts.
    pub fn observe(&mut self, k: usize, a: usize, secs: f64) {
        self.rounds += 1;
        for i in 0..k.min(self.max) {
            if i < a {
                self.q[i] += EMA * (1.0 - self.q[i]);
            } else {
                self.q[i] += EMA * (0.0 - self.q[i]);
                break;
            }
        }
        if self.base_n[k] < BASE_SAMPLES {
            self.base_n[k] += 1;
            self.base[k] = Some(self.base[k].map_or(secs, |b| b.min(secs)));
        } else if let Some(b) = self.base[k] {
            self.drift += EMA * (secs / b - self.drift);
        }
    }
}

pub struct SpecDev {
    /// fixed depth, or the largest depth when adaptive (the pipeline's verify chunk is `depth + 1` rows)
    pub depth: usize,
    pub picker: Option<DepthPicker>,
    round_m: usize,
    pub stash: VecDeque<u32>,
    pub h_pending: Option<Vec<f32>>,
    pub trunk_len: usize,
    round_p: usize,
    round_hs: Vec<f32>,
    pub stats: SpecCounters,
}

impl SpecDev {
    pub fn new(depth: usize) -> Self {
        let depth = depth.clamp(1, 7);
        SpecDev { depth, picker: None, round_m: 0, stash: VecDeque::new(), h_pending: None, trunk_len: 0, round_p: 0, round_hs: Vec::new(), stats: SpecCounters::default() }
    }

    /// Adaptive depth in `0..=max` (see the module notes).
    pub fn adaptive(max: usize) -> Self {
        let mut s = Self::new(max);
        s.picker = Some(DepthPicker::new(s.depth));
        s
    }

    /// Forget everything (a new sequence at position 0, or a plain-decode excursion).
    pub fn invalidate(&mut self) {
        self.stash.clear();
        self.h_pending = None;
    }

    pub fn ready_for(&self, len: usize) -> bool {
        self.h_pending.is_some() && self.trunk_len == len + self.stash.len()
    }

    /// After the trunk consumed `tokens` at `pos0..` with hidden rows `rows` (`[n, n_embd]`): build the head's cache entries for them and keep the last hidden.
    pub fn after_prefill(&mut self, pipe: &Pipeline<'_, '_>, tokens: &[u32], pos0: usize, rows: &[f32]) -> Result<()> {
        let ne = rows.len() / tokens.len();
        // entry j pairs H_j with token j+1: entries pos0-1 (if the previous hidden is known) .. pos0+n-2
        let mut hs: Vec<f32> = Vec::new();
        let mut ts: Vec<u32> = Vec::new();
        let mut j0 = pos0;
        if let (Some(h), true) = (&self.h_pending, pos0 > 0 && self.trunk_len == pos0) {
            hs.extend_from_slice(h);
            ts.push(tokens[0]);
            j0 = pos0 - 1;
        }
        let n = tokens.len();
        hs.extend_from_slice(&rows[..(n - 1) * ne]);
        ts.extend_from_slice(&tokens[1..]);
        if !ts.is_empty() {
            pipe.mtp_prefill(&hs, &ts, j0)?;
        }
        self.h_pending = Some(rows[(n - 1) * ne..].to_vec());
        self.trunk_len = pos0 + n;
        self.stash.clear();
        Ok(())
    }

    /// The caller feeds `x` (the token at position `p = cache.len`) and wants the greedy token after it.
    pub fn next(&mut self, pipe: &Pipeline<'_, '_>, x: u32, p: usize) -> Result<u32> {
        if let Some(t) = self.stash.pop_front() {
            return Ok(t);
        }
        debug_assert_eq!(self.trunk_len, p);
        let k = self.picker.as_ref().map(|pk| pk.pick()).unwrap_or(self.depth);
        let t0 = std::time::Instant::now();
        let h = self.h_pending.take().expect("hidden of the previous position");
        let mut toks = vec![x];
        if k == 0 {
            // a plain step: the head only needs its cache entry for (H_{p-1}, x)
            pipe.mtp_kv(&h, x, p - 1);
        } else {
            let (d1, mut hid) = pipe.mtp_draft(&h, x, p - 1)?;
            toks.push(d1);
            for d in 1..k {
                let (dd, hn) = pipe.mtp_draft(&hid, toks[d], p - 1 + d)?;
                hid = hn;
                toks.push(dd);
            }
        }
        let t1 = std::time::Instant::now();
        let (g, hs) = pipe.run_rows(&toks, p)?;
        let t2 = std::time::Instant::now();
        let ne = hs.len() / toks.len();
        let mut a = 0;
        while a < k && toks[a + 1] == g[a] {
            a += 1;
        }
        if a < k {
            pipe.spec_rollback(p, a + 1);
        }
        for j in 0..a {
            pipe.mtp_kv(&hs[j * ne..(j + 1) * ne], toks[j + 1], p + j);
        }
        let st = &mut self.stats;
        st.rounds += 1;
        for i in 0..a {
            st.prefix_ok[i] += 1;
        }
        st.emitted += a as u64 + 1;
        st.draft_s += (t1 - t0).as_secs_f64();
        st.verify_s += (t2 - t1).as_secs_f64();
        st.fixup_s += t2.elapsed().as_secs_f64();
        st.at_depth[k] += 1;
        if let Some(pk) = self.picker.as_mut() {
            pk.observe(k, a, t0.elapsed().as_secs_f64());
        }
        self.round_m = k + 1;
        self.h_pending = Some(hs[a * ne..(a + 1) * ne].to_vec());
        self.trunk_len = p + a + 1;
        self.stash = g[1..=a].iter().cloned().collect();
        self.round_p = p;
        self.round_hs = hs;
        Ok(g[0])
    }

    /// The caller stops consuming at `len` (the end of a request): put the trunk back to exactly `len` tokens.
    pub fn trim(&mut self, pipe: &Pipeline<'_, '_>, len: usize) {
        if self.stash.is_empty() || self.trunk_len == len {
            return;
        }
        let ne = self.round_hs.len() / self.round_m.max(1);
        let keep = len - self.round_p;
        pipe.spec_rollback(self.round_p, keep);
        self.h_pending = Some(self.round_hs[(keep - 1) * ne..keep * ne].to_vec());
        self.trunk_len = len;
        self.stash.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::DepthPicker;

    #[test]
    fn expected_tokens_is_one_plus_prefix_products() {
        let mut p = DepthPicker::new(3);
        p.q = vec![0.9, 0.5, 0.2];
        assert!((p.expected_tokens(0) - 1.0).abs() < 1e-12);
        assert!((p.expected_tokens(1) - 1.9).abs() < 1e-12);
        assert!((p.expected_tokens(3) - (1.0 + 0.9 + 0.45 + 0.09)).abs() < 1e-12);
    }

    #[test]
    fn measures_every_depth_once_then_picks_by_rate() {
        let mut p = DepthPicker::new(2);
        assert_eq!(p.pick(), 0);
        p.observe(0, 0, 0.019);
        assert_eq!(p.pick(), 0);
        p.observe(0, 0, 0.015);
        assert_eq!(p.pick(), 1);
        p.observe(1, 1, 0.025);
        p.observe(1, 1, 0.025);
        assert_eq!(p.pick(), 2);
        p.observe(2, 2, 0.031);
        p.observe(2, 2, 0.031);
        assert_eq!(p.base[0], Some(0.015));
        // high acceptance: depth 2 gives (1 + q1 + q1 q2) / 31 ms, beats 1 / 15 ms
        assert_eq!(p.best(), 2);
    }

    #[test]
    fn low_acceptance_falls_back_to_plain() {
        let mut p = DepthPicker::new(2);
        for _ in 0..2 {
            p.observe(0, 0, 0.015);
            p.observe(1, 0, 0.026);
            p.observe(2, 0, 0.032);
        }
        for _ in 0..60 {
            let k = p.pick();
            let t = [0.015, 0.026, 0.032][k];
            p.observe(k, 0, t);
        }
        assert_eq!(p.best(), 0);
        assert!(p.q[0] < 0.5, "q1 = {}", p.q[0]);
    }

    #[test]
    fn probes_deeper_so_acceptance_recovers() {
        let mut p = DepthPicker::new(1);
        for _ in 0..2 {
            p.observe(0, 0, 0.015);
            p.observe(1, 0, 0.026);
        }
        p.q[0] = 0.1;
        let mut probes = 0;
        for _ in 0..400 {
            let k = p.pick();
            if k == 1 {
                probes += 1;
            }
            // the text turned predictable: every draft is accepted when tried
            p.observe(k, k, [0.015, 0.026][k]);
        }
        assert!(probes > 0);
        assert_eq!(p.best(), 1, "q1 = {}", p.q[0]);
    }

    #[test]
    fn drift_scales_all_depths_alike() {
        let mut p = DepthPicker::new(1);
        for _ in 0..2 {
            p.observe(0, 0, 0.015);
            p.observe(1, 1, 0.025);
        }
        let b = p.best();
        for _ in 0..100 {
            p.observe(b, if b == 1 { 1 } else { 0 }, [0.030, 0.050][b]);
        }
        assert!(p.drift > 1.8);
        assert_eq!(p.best(), b);
    }
}
