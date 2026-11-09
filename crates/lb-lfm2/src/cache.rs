//! Dynamic expert cache (engine step E1, `docs/ENGINE_E1_CACHE.md`): a segmented LRU over a fixed
//! bank of device slots, in place of the static heat plan.
//!
//! **Policy** (the one llama.cpp's `--moe-cache-mib` uses, `src/llama-moe-cache.cpp`): every slot is
//! in one of two LRU lists, *probation* and *protected*. A demand miss takes the least recently used
//! slot of probation (then of protected), never one holding an expert this layer pass needs, and goes
//! in at the probation end; a hit moves its slot to the protected end; protected is capped at half the
//! slots, its least recently used slot falling back to probation. One-off experts so pass through
//! probation without pushing out the experts that keep coming back. The cache is global over layers
//! (all LFM2.5 MoE layers share one expert shape), so a layer's share follows its demand.
//!
//! **Bytes.** The slots are [`lb_q4_gpu::ExpertSlots`]: allocated once, filled by DMA + one repack
//! launch, never reallocated. The budget is counted in the bank's real device bytes (slots plus the
//! staging buffers), so the cache cannot grow device memory (the kept-miss bug of exp. #2).
//!
//! **Arithmetic.** A cached expert runs the same kernel on the same repacked words as a resident or
//! a copied-in one, and `moe::moe_ffn_with` adds expert outputs in ascending expert id whatever served
//! them, so the greedy text does not depend on the budget or on what the cache holds.
//!
//! **E2 note.** `slot_of` is a dense `[layer][expert] -> slot` table; a device-side router can read
//! the same table from a device buffer to launch slots without a host round trip.

/// Segmented LRU over `n_slots` slots, keyed by `(layer, expert)`. Pure bookkeeping (testable
/// without a device); [`ExpertCache`] pairs it with the device slots.
#[derive(Clone, Debug)]
pub struct Slru {
    n_expert: usize,
    /// `[layer * n_expert + expert]` -> slot, -1 when not cached.
    slot_of: Vec<i32>,
    /// slot -> key (`layer * n_expert + expert`), -1 when empty.
    key_of: Vec<i64>,
    prev: Vec<i32>,
    next: Vec<i32>,
    /// `[probation, protected]` heads (least recent) and tails (most recent).
    head: [i32; 2],
    tail: [i32; 2],
    protected: Vec<bool>,
    n_protected: usize,
}

/// What a layer pass needs from the cache: `(expert, slot)` already held, and `(expert, slot)` to fill.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub hits: Vec<(usize, usize)>,
    pub fills: Vec<(usize, usize)>,
}

impl Slru {
    pub fn new(n_layer: usize, n_expert: usize, n_slots: usize) -> Self {
        let n = n_slots as i32;
        Self {
            n_expert,
            slot_of: vec![-1; n_layer * n_expert],
            key_of: vec![-1; n_slots],
            prev: (0..n).map(|s| s - 1).collect(),
            next: (0..n).map(|s| if s + 1 < n { s + 1 } else { -1 }).collect(),
            head: [if n > 0 { 0 } else { -1 }, -1],
            tail: [n - 1, -1],
            protected: vec![false; n_slots],
            n_protected: 0,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.key_of.len()
    }

    /// The slot holding `(layer, expert)`, if cached.
    pub fn slot(&self, layer: usize, expert: usize) -> Option<usize> {
        let s = *self.slot_of.get(layer * self.n_expert + expert)?;
        (s >= 0).then_some(s as usize)
    }

    /// The `[layer][expert] -> slot` table (-1 = not cached), for a device-side router.
    pub fn slot_table(&self) -> &[i32] {
        &self.slot_of
    }

    pub fn n_protected(&self) -> usize {
        self.n_protected
    }

    fn unlink(&mut self, s: i32) {
        let seg = self.protected[s as usize] as usize;
        let (p, n) = (self.prev[s as usize], self.next[s as usize]);
        if p >= 0 { self.next[p as usize] = n } else { self.head[seg] = n }
        if n >= 0 { self.prev[n as usize] = p } else { self.tail[seg] = p }
    }

    /// Move slot `s` to the most recently used end of segment `protect`.
    fn move_to(&mut self, s: usize, protect: bool) {
        let si = s as i32;
        self.unlink(si);
        let seg = protect as usize;
        self.prev[s] = self.tail[seg];
        self.next[s] = -1;
        if self.tail[seg] >= 0 { self.next[self.tail[seg] as usize] = si } else { self.head[seg] = si }
        self.tail[seg] = si;
        let was = self.protected[s];
        self.protected[s] = protect;
        self.n_protected = self.n_protected + protect as usize - was as usize;
    }

    fn touch(&mut self, s: usize) {
        self.move_to(s, true);
        if self.n_protected > self.n_slots() / 2 {
            let h = self.head[1];
            self.move_to(h as usize, false);
        }
    }

    /// Least recently used slot, probation first, whose expert is not one `layer` needs now.
    fn victim(&self, layer: usize, needed: &[usize]) -> Option<usize> {
        for seg in 0..2 {
            let mut s = self.head[seg];
            while s >= 0 {
                let k = self.key_of[s as usize];
                let busy = k >= 0 && (k as usize) / self.n_expert == layer && needed.contains(&((k as usize) % self.n_expert));
                if !busy {
                    return Some(s as usize);
                }
                s = self.next[s as usize];
            }
        }
        None
    }

    /// Give every expert `layer` needs a slot: hits are promoted, misses take victims and go in on
    /// probation. `needed` is the layer pass's distinct experts in ascending id. `None` when they
    /// do not all fit (more distinct experts than slots).
    pub fn plan(&mut self, layer: usize, needed: &[usize]) -> Option<Plan> {
        if needed.len() > self.n_slots() {
            return None;
        }
        let mut plan = Plan::default();
        for &e in needed {
            if let Some(s) = self.slot(layer, e) {
                self.touch(s);
                plan.hits.push((e, s));
            }
        }
        for &e in needed {
            if self.slot(layer, e).is_some() {
                continue;
            }
            let s = self.victim(layer, needed)?;
            self.assign(s, layer, e);
            self.move_to(s, false);
            plan.fills.push((e, s));
        }
        Some(plan)
    }

    /// [`Slru::victim`], also sparing the experts the look-ahead expects the next layers to need
    /// (`keep`, E3); the second value is how many LRU candidates it walked past for them. Falls back
    /// to the unguarded victim when nothing else can go, so a pass never fails because of a guess.
    fn victim_guarded(&self, layer: usize, needed: &[usize], keep: &[(usize, usize)]) -> (Option<usize>, u64) {
        let mut skipped = 0;
        for seg in 0..2 {
            let mut s = self.head[seg];
            while s >= 0 {
                let k = self.key_of[s as usize];
                if k < 0 {
                    return (Some(s as usize), skipped);
                }
                let (l, e) = ((k as usize) / self.n_expert, (k as usize) % self.n_expert);
                if !(l == layer && needed.contains(&e)) {
                    if !keep.contains(&(l, e)) {
                        return (Some(s as usize), skipped);
                    }
                    skipped += 1;
                }
                s = self.next[s as usize];
            }
        }
        (self.victim(layer, needed), skipped)
    }

    /// See [`Slru::plan_la`]: the furthest-in-cycle of the first `window` probation candidates that are
    /// neither needed now nor predicted; `None` lets the caller fall back. An empty slot goes first.
    fn cycle_victim(&self, layer: usize, needed: &[usize], keep: &[(usize, usize)], window: usize, n_layer: usize) -> Option<usize> {
        let (mut best, mut best_d) = (None, 0usize);
        let (mut s, mut seen) = (self.head[0], 0usize);
        while s >= 0 && seen < window {
            let k = self.key_of[s as usize];
            if k < 0 {
                return Some(s as usize);
            }
            let (l, e) = ((k as usize) / self.n_expert, (k as usize) % self.n_expert);
            if !(l == layer && needed.contains(&e)) && !keep.contains(&(l, e)) {
                // how many layers until this one runs again (0 = the next one)
                let d = (l + 2 * n_layer - layer - 1) % n_layer;
                if best.is_none() || d > best_d {
                    best = Some(s as usize);
                    best_d = d;
                }
                seen += 1;
            }
            s = self.next[s as usize];
        }
        best
    }

    /// [`Slru::plan`] with look-ahead: misses take victims that spare `keep`, the `(layer, expert)`
    /// keys predicted for the coming layers (engine step E3, no extra copies). The second value counts
    /// LRU candidates passed over for them. With `keep` empty this is `plan` exactly.
    pub fn plan_guarded(&mut self, layer: usize, needed: &[usize], keep: &[(usize, usize)]) -> Option<(Plan, u64)> {
        self.plan_la(layer, needed, keep, 0, 0)
    }

    /// [`Slru::plan_guarded`], and with `cycle_window > 0` a cycle-aware victim: a token visits the
    /// layers in a fixed order, so among the `cycle_window` least recently used probation slots the one
    /// whose layer comes up furthest away (of `n_layer` in the cycle) is the better victim than plain
    /// LRU's pick, which is often the layer about to run.
    pub fn plan_la(&mut self, layer: usize, needed: &[usize], keep: &[(usize, usize)], cycle_window: usize, n_layer: usize) -> Option<(Plan, u64)> {
        if needed.len() > self.n_slots() {
            return None;
        }
        let mut plan = Plan::default();
        for &e in needed {
            if let Some(s) = self.slot(layer, e) {
                self.touch(s);
                plan.hits.push((e, s));
            }
        }
        let mut spared = 0;
        for &e in needed {
            if self.slot(layer, e).is_some() {
                continue;
            }
            let (v, sk) = match (cycle_window > 0).then(|| self.cycle_victim(layer, needed, keep, cycle_window, n_layer)).flatten() {
                Some(s) => (Some(s), 0),
                None => self.victim_guarded(layer, needed, keep),
            };
            spared += sk;
            let s = v?;
            self.assign(s, layer, e);
            self.move_to(s, false);
            plan.fills.push((e, s));
        }
        Some((plan, spared))
    }

    /// A predicted expert that is cached: refresh it so the fills before its layer do not take it.
    /// `to_protected` moves it to the protected segment as a hit would; otherwise it goes to the most
    /// recently used end of the segment it is in. No-op when it is not cached.
    pub fn promote(&mut self, layer: usize, expert: usize, to_protected: bool) -> bool {
        let Some(s) = self.slot(layer, expert) else { return false };
        if to_protected {
            self.touch(s);
        } else {
            let seg = self.protected[s];
            self.move_to(s, seg);
        }
        true
    }

    fn assign(&mut self, s: usize, layer: usize, e: usize) {
        let old = self.key_of[s];
        if old >= 0 {
            self.slot_of[old as usize] = -1;
        }
        let k = layer * self.n_expert + e;
        self.key_of[s] = k as i64;
        self.slot_of[k] = s as i32;
    }

    /// Seed: put `(layer, expert)` in the next free slot (or the probation LRU), on probation.
    /// Seeding hottest-last leaves the hottest seeds most recently used. Returns the slot to fill.
    pub fn seed(&mut self, layer: usize, expert: usize) -> Option<usize> {
        if self.slot(layer, expert).is_some() {
            return None;
        }
        let s = self.victim(usize::MAX, &[])?;
        self.assign(s, layer, expert);
        self.move_to(s, false);
        Some(s)
    }

    /// Consistency check for tests: both lists cover every slot once, maps agree.
    pub fn check(&self) -> Result<(), String> {
        let mut seen = vec![false; self.n_slots()];
        let mut n_prot = 0;
        for seg in 0..2 {
            let (mut s, mut last) = (self.head[seg], -1);
            while s >= 0 {
                let su = s as usize;
                if seen[su] { return Err(format!("slot {s} twice")) }
                seen[su] = true;
                if self.protected[su] != (seg == 1) { return Err(format!("slot {s} in the wrong list")) }
                if self.prev[su] != last { return Err(format!("slot {s} prev link")) }
                n_prot += seg;
                last = s;
                s = self.next[su];
            }
            if self.tail[seg] != last { return Err(format!("segment {seg} tail")) }
        }
        if seen.iter().any(|x| !x) { return Err("a slot is in no list".into()) }
        if n_prot != self.n_protected { return Err("protected count".into()) }
        for (s, &k) in self.key_of.iter().enumerate() {
            if k >= 0 && self.slot_of[k as usize] != s as i32 { return Err(format!("slot {s} map")) }
        }
        Ok(())
    }
}

/// Where a landing buffer stands (`LandingBook`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    Free,
    /// A prefetch copy of `(layer, expert)` issued, or already landed; not consumed yet.
    Flight(usize, usize),
    /// Consumed by a repack this layer; free once that repack ran (after the next sync).
    Consumed,
}

/// Bookkeeping of the prefetch landing buffers (E3 miss-only prefetch), apart from the device so it
/// can be tested. An expert copied speculatively lands in a landing buffer on a side stream; the
/// slot it goes into is taken from the cache only when the layer really routes it, so a wrong guess
/// costs bus bytes and never evicts anything.
#[derive(Clone, Debug)]
pub struct LandingBook {
    st: Vec<Landing>,
}

impl LandingBook {
    pub fn new(n: usize) -> Self {
        Self { st: vec![Landing::Free; n] }
    }

    pub fn len(&self) -> usize {
        self.st.len()
    }

    pub fn is_empty(&self) -> bool {
        self.st.is_empty()
    }

    pub fn state(&self, i: usize) -> Landing {
        self.st[i]
    }

    /// The buffer holding (or receiving) this expert, if any.
    pub fn find(&self, layer: usize, expert: usize) -> Option<usize> {
        self.st.iter().position(|s| *s == Landing::Flight(layer, expert))
    }

    /// A free buffer, not yet claimed.
    pub fn free_one(&self) -> Option<usize> {
        self.st.iter().position(|s| *s == Landing::Free)
    }

    /// Claim free buffer `i` for a prefetch of `(layer, expert)`.
    pub fn issue(&mut self, i: usize, layer: usize, expert: usize) {
        debug_assert_eq!(self.st[i], Landing::Free);
        self.st[i] = Landing::Flight(layer, expert);
    }

    /// Consume the landing of `(layer, expert)` for a repack; returns its buffer.
    pub fn consume(&mut self, layer: usize, expert: usize) -> Option<usize> {
        let i = self.find(layer, expert)?;
        self.st[i] = Landing::Consumed;
        Some(i)
    }

    /// Landings for layers up to `layer` that nothing consumed: the guesses were wrong. They go
    /// straight back (no repack read them). Returns how many.
    pub fn expire_through(&mut self, layer: usize) -> usize {
        let mut n = 0;
        for s in &mut self.st {
            if matches!(s, Landing::Flight(l, _) if *l <= layer) {
                *s = Landing::Free;
                n += 1;
            }
        }
        n
    }

    /// Free the buffers consumed before this sync (their repacks have run).
    pub fn reclaim(&mut self) {
        for s in &mut self.st {
            if *s == Landing::Consumed {
                *s = Landing::Free;
            }
        }
    }
}

#[cfg(feature = "gpu")]
pub use dev::{ExpertCache, Landings};

#[cfg(feature = "gpu")]
mod dev {
    use super::*;
    use lb_q4_gpu::cubecl_server::Handle;
    use lb_q4_gpu::{ExpertSlots, SideStream};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// The prefetch landing buffers of E3: the bookkeeping plus their device side.
    pub struct Landings {
        pub book: LandingBook,
        pub side: SideStream,
        pub buffers: Vec<Handle>,
        /// Per buffer: the sentinel the repack binds so it waits for that buffer's copy (see
        /// [`ExpertSlots::land`]).
        pub sentinels: Vec<Option<Handle>>,
    }

    /// The SLRU plus its device slots.
    pub struct ExpertCache {
        pub slru: Slru,
        pub slots: ExpertSlots,
        /// Fills since start (demand misses plus seeds).
        pub fills: AtomicU64,
        pub seeds: usize,
        /// Prefetch landing buffers (E3 miss-only prefetch); `None` when prefetch is off.
        pub landings: Option<Landings>,
        /// `LB_LA_TRACE`: one JSON line per layer pass for the offline simulator.
        pub trace: Option<std::sync::Mutex<std::io::LineWriter<std::fs::File>>>,
        /// `LB_LA_ORACLE`: the decode passes of a recorded session and the replay position.
        pub oracle: Option<(Vec<(usize, Vec<usize>)>, usize)>,
    }

    impl ExpertCache {
        pub fn new(slru: Slru, slots: ExpertSlots) -> Self {
            Self { slru, slots, fills: AtomicU64::new(0), seeds: 0, landings: None, trace: None, oracle: None }
        }

        /// Record one layer pass for the offline simulator (`LB_LA_TRACE`, off by default).
        pub fn trace_pass(&self, kind: &str, layer: usize, pos: usize, needed: &[usize], guess: Option<&crate::stage::Guess>) {
            let Some(w) = &self.trace else { return };
            use std::io::Write;
            let mut w = w.lock().expect("look-ahead trace");
            let line = serde_json::json!({
                "k": kind, "l": layer, "p": pos, "need": needed,
                "g": guess.map(|g| serde_json::json!({"t": g.layer, "e": g.experts, "s": g.scores, "r": g.ref_score})),
            });
            let _ = writeln!(w, "{line}");
        }

        pub fn count_fill(&self) {
            self.fills.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misses_go_in_on_probation_hits_are_promoted() {
        let mut c = Slru::new(2, 8, 4);
        let p = c.plan(0, &[1, 2]).unwrap();
        assert_eq!(p.hits, vec![]);
        assert_eq!(p.fills.len(), 2);
        assert_eq!(c.n_protected(), 0);
        let p = c.plan(0, &[1, 3]).unwrap();
        assert_eq!(p.hits.len(), 1);
        assert_eq!(p.fills.len(), 1);
        assert_eq!(c.n_protected(), 1);
        c.check().unwrap();
    }

    #[test]
    fn a_one_off_scan_does_not_evict_protected_experts() {
        let mut c = Slru::new(1, 64, 4);
        c.plan(0, &[1]).unwrap();
        c.plan(0, &[1]).unwrap(); // 1 is protected now
        for e in 10..30 {
            c.plan(0, &[e]).unwrap(); // one-offs churn probation
            c.check().unwrap();
        }
        assert!(c.slot(0, 1).is_some(), "the re-used expert survived the scan");
    }

    #[test]
    fn protected_is_capped_at_half_the_slots() {
        let mut c = Slru::new(1, 64, 6);
        for e in 0..6 {
            c.plan(0, &[e]).unwrap();
        }
        for e in 0..6 {
            c.plan(0, &[e]).unwrap(); // all hits: each promoted
        }
        assert_eq!(c.n_protected(), 3);
        c.check().unwrap();
    }

    #[test]
    fn a_layer_never_evicts_what_it_needs_now() {
        let mut c = Slru::new(1, 64, 4);
        let p = c.plan(0, &[0, 1, 2, 3]).unwrap();
        assert_eq!(p.fills.len(), 4);
        // all four slots hold experts this pass needs; a fifth does not fit
        assert!(c.plan(0, &[0, 1, 2, 3, 4]).is_none());
        // a different set evicts only what it does not need
        let p = c.plan(0, &[2, 3, 7, 8]).unwrap();
        assert_eq!(p.hits.len(), 2);
        for (_, s) in &p.fills {
            assert!(c.slot(0, 2) != Some(*s) && c.slot(0, 3) != Some(*s));
        }
        c.check().unwrap();
    }

    #[test]
    fn layers_share_the_slots_and_maps_stay_consistent() {
        let mut c = Slru::new(4, 16, 10);
        let mut x = 12345u64;
        for t in 0..500 {
            let l = t % 4;
            let mut need: Vec<usize> = (0..4).map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 33) % 16) as usize
            }).collect();
            need.sort();
            need.dedup();
            let p = c.plan(l, &need).unwrap();
            assert_eq!(p.hits.len() + p.fills.len(), need.len());
            for &e in &need {
                assert!(c.slot(l, e).is_some());
            }
            c.check().unwrap();
        }
    }

    #[test]
    fn guarded_plan_spares_predicted_experts_and_equals_plan_without_a_guess() {
        let mk = || {
            let mut c = Slru::new(2, 8, 4);
            c.seed(0, 1).unwrap();
            c.seed(0, 2).unwrap();
            c.seed(1, 3).unwrap();
            c.seed(1, 4).unwrap();
            c
        };
        // no guess: same as plan
        let (mut a, mut b) = (mk(), mk());
        let (p, skipped) = a.plan_guarded(0, &[5], &[]).unwrap();
        assert_eq!(p, b.plan(0, &[5]).unwrap());
        assert_eq!(skipped, 0);
        // the LRU slot (0,1) is predicted: the next one goes instead
        let mut a = mk();
        let (p, skipped) = a.plan_guarded(0, &[5], &[(0, 1)]).unwrap();
        assert_eq!(p.fills.len(), 1);
        assert!(a.slot(0, 1).is_some(), "the predicted expert survived");
        assert!(a.slot(0, 2).is_none(), "the next LRU expert went instead");
        assert_eq!(skipped, 1);
        a.check().unwrap();
        // everything predicted: falls back to the unguarded victim, never fails
        let mut a = mk();
        let (p, _) = a.plan_guarded(0, &[5], &[(0, 1), (0, 2), (1, 3), (1, 4)]).unwrap();
        assert_eq!(p.fills.len(), 1);
        a.check().unwrap();
    }

    #[test]
    fn promote_refreshes_or_protects_a_predicted_expert_and_ignores_absent_ones() {
        let mut c = Slru::new(1, 8, 3);
        c.seed(0, 1).unwrap();
        c.seed(0, 2).unwrap();
        c.seed(0, 3).unwrap();
        assert!(c.promote(0, 1, false), "cached");
        assert_eq!(c.n_protected(), 0);
        // 1 is most recent now, so a miss takes 2
        c.plan(0, &[7]).unwrap();
        assert!(c.slot(0, 2).is_none());
        assert!(c.slot(0, 1).is_some());
        assert!(c.promote(0, 1, true));
        assert_eq!(c.n_protected(), 1);
        assert!(!c.promote(0, 5, true), "not cached");
        c.check().unwrap();
    }

    #[test]
    fn cycle_aware_victim_takes_the_layer_furthest_in_the_cycle() {
        let mut c = Slru::new(8, 8, 3);
        c.seed(5, 0).unwrap(); // LRU
        c.seed(1, 0).unwrap();
        c.seed(3, 0).unwrap();
        // running layer 4: (5,0) is next up (d=0), (1,0) d=4, (3,0) d=6 = furthest
        let (p, _) = c.plan_la(4, &[7], &[], 3, 8).unwrap();
        assert_eq!(p.fills.len(), 1);
        assert!(c.slot(3, 0).is_none(), "the layer that just ran goes");
        assert!(c.slot(5, 0).is_some(), "the layer about to run stays");
        c.check().unwrap();
        // window 1 is plain LRU
        let mut c = Slru::new(8, 8, 3);
        c.seed(5, 0).unwrap();
        c.seed(1, 0).unwrap();
        c.seed(3, 0).unwrap();
        c.plan_la(4, &[7], &[], 1, 8).unwrap();
        assert!(c.slot(5, 0).is_none());
    }

    #[test]
    fn landings_track_issue_consume_expire_reclaim() {
        let mut b = LandingBook::new(2);
        let i = b.free_one().unwrap();
        b.issue(i, 5, 3);
        assert_eq!(b.find(5, 3), Some(i));
        let j = b.free_one().unwrap();
        b.issue(j, 5, 4);
        assert!(b.free_one().is_none(), "both taken");
        // layer 5 routes expert 3 only: 4 was a wrong guess
        assert_eq!(b.consume(5, 3), Some(i));
        assert_eq!(b.expire_through(5), 1);
        assert_eq!(b.state(j), Landing::Free);
        assert_eq!(b.state(i), Landing::Consumed, "held until the repack ran");
        assert!(b.free_one().is_some());
        b.reclaim();
        assert_eq!(b.state(i), Landing::Free);
        // a landing for a later layer is not expired by an earlier one
        b.issue(0, 9, 1);
        assert_eq!(b.expire_through(8), 0);
        assert_eq!(b.state(0), Landing::Flight(9, 1));
    }

    #[test]
    fn seeds_fill_free_slots_then_recycle_probation() {
        let mut c = Slru::new(2, 8, 3);
        assert!(c.seed(0, 1).is_some());
        assert!(c.seed(0, 1).is_none(), "already cached");
        c.seed(1, 2).unwrap();
        c.seed(1, 3).unwrap();
        c.seed(0, 4).unwrap(); // evicts the oldest seed
        assert!(c.slot(0, 1).is_none());
        assert!(c.slot(0, 4).is_some());
        c.check().unwrap();
    }
}
