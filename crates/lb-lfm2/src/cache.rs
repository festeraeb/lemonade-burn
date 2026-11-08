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

#[cfg(feature = "gpu")]
pub use dev::ExpertCache;

#[cfg(feature = "gpu")]
mod dev {
    use super::*;
    use lb_q4_gpu::ExpertSlots;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// The SLRU plus its device slots.
    pub struct ExpertCache {
        pub slru: Slru,
        pub slots: ExpertSlots,
        /// Fills since start (demand misses plus seeds).
        pub fills: AtomicU64,
        pub seeds: usize,
    }

    impl ExpertCache {
        pub fn new(slru: Slru, slots: ExpertSlots) -> Self {
            Self { slru, slots, fills: AtomicU64::new(0), seeds: 0 }
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
