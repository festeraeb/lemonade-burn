//! The device MTP speculative loop: draft with the nextn block, verify on the trunk as one batched chunk, keep the longest matching prefix,
//! roll the recurrent state back with the checkpoint/replay of `Pipeline::run_rows` / `spec_rollback`. Greedy only.
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
}

pub struct SpecDev {
    pub depth: usize,
    pub stash: VecDeque<u32>,
    pub h_pending: Option<Vec<f32>>,
    pub trunk_len: usize,
    round_p: usize,
    round_hs: Vec<f32>,
    pub stats: SpecCounters,
}

impl SpecDev {
    pub fn new(depth: usize) -> Self {
        SpecDev { depth: depth.clamp(1, 7), stash: VecDeque::new(), h_pending: None, trunk_len: 0, round_p: 0, round_hs: Vec::new(), stats: SpecCounters::default() }
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
        let k = self.depth;
        let t0 = std::time::Instant::now();
        let h = self.h_pending.take().expect("hidden of the previous position");
        let (d1, mut hid) = pipe.mtp_draft(&h, x, p - 1)?;
        let mut toks = vec![x, d1];
        for d in 1..k {
            let (dd, hn) = pipe.mtp_draft(&hid, toks[d], p - 1 + d)?;
            hid = hn;
            toks.push(dd);
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
        let ne = self.round_hs.len() / (self.depth + 1);
        let keep = len - self.round_p;
        pipe.spec_rollback(self.round_p, keep);
        self.h_pending = Some(self.round_hs[(keep - 1) * ne..keep * ne].to_vec());
        self.trunk_len = len;
        self.stash.clear();
    }
}
