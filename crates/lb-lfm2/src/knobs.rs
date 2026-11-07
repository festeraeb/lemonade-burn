//! Live knobs: the settings the forward pass reads on every call, held behind atomics so a
//! host (the server's `/v1/tunables`) can change them while the model runs on its own thread.
//!
//! Only what can take effect between two forward passes without touching device memory lives
//! here. Residency (GPU budget, dense budget, placement), the device itself and the thread pool
//! are set once at load and stay restart-only; the server lists those too, but refuses to
//! change them.

use crate::moe::MissPolicy;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Settings the model reads live. Defaults match the engine's behaviour before this existed:
/// `cpu` misses, device repack, batched miss uploads.
#[derive(Debug)]
pub struct Knobs {
    miss_policy: AtomicU8,
    /// Repack registered bytes on the host (A/B switch for the device repack).
    host_repack: AtomicBool,
    /// Upload a miss's three matrices as one batch (A/B switch for the per-matrix path).
    batch_miss: AtomicBool,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            miss_policy: AtomicU8::new(MissPolicy::Cpu as u8),
            host_repack: AtomicBool::new(std::env::var_os("LB_HOST_REPACK").is_some()),
            batch_miss: AtomicBool::new(std::env::var_os("LB_NO_BATCH_MISS").is_none()),
        }
    }
}

impl Knobs {
    pub fn miss_policy(&self) -> MissPolicy {
        match self.miss_policy.load(Ordering::Relaxed) {
            1 => MissPolicy::OnDemand,
            _ => MissPolicy::Cpu,
        }
    }

    pub fn set_miss_policy(&self, p: MissPolicy) {
        self.miss_policy.store(p as u8, Ordering::Relaxed);
    }

    pub fn host_repack(&self) -> bool {
        self.host_repack.load(Ordering::Relaxed)
    }

    pub fn set_host_repack(&self, on: bool) {
        self.host_repack.store(on, Ordering::Relaxed);
    }

    pub fn batch_miss(&self) -> bool {
        self.batch_miss.load(Ordering::Relaxed)
    }

    pub fn set_batch_miss(&self, on: bool) {
        self.batch_miss.store(on, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knobs_round_trip() {
        let k = Knobs::default();
        assert_eq!(k.miss_policy(), MissPolicy::Cpu);
        k.set_miss_policy(MissPolicy::OnDemand);
        assert_eq!(k.miss_policy(), MissPolicy::OnDemand);
        k.set_miss_policy(MissPolicy::Cpu);
        assert_eq!(k.miss_policy(), MissPolicy::Cpu);
        let b = k.batch_miss();
        k.set_batch_miss(!b);
        assert_eq!(k.batch_miss(), !b);
        let h = k.host_repack();
        k.set_host_repack(!h);
        assert_eq!(k.host_repack(), !h);
    }
}
