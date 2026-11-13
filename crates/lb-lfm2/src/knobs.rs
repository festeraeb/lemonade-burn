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
    /// Run a decode token device-resident (`device_decode.rs`): one upload, one read-back.
    /// **On by default** since the P100 measurement of 2026-11-07 (3.9x the old path, every
    /// parity gate passed, docs/WORK_LOG.md); `LB_DEVICE_DECODE=0` starts with the old path.
    /// A token whose weights are not all resident takes the old path either way.
    device_decode: AtomicBool,
    /// Inside a device-resident token, run a MoE layer whose experts are not all resident on
    /// the host path (one extra sync for that layer) instead of giving the whole token to the
    /// old path. `LB_DD_HOST_MOE=1` at start; off by default (the plan's rule: a token takes
    /// the device path only when everything it needs is resident).
    dd_host_moe: AtomicBool,
    /// With the dynamic expert cache, route each MoE layer on the device and run its experts from
    /// the slot bank, the host seeing only the routes (engine step E2, one sync a layer instead of
    /// two, nothing uploaded or summed on the host). `LB_E2=0` keeps E1's host-routed layer.
    e2_device_router: AtomicBool,
    /// Router logits as one sequential dot per expert (the host's summation order) instead
    /// of a plane reduce: slower, for parity checks. `LB_DD_ROUTER_SEQ=1`; off by default.
    dd_router_seq: AtomicBool,
    /// Run a multi-token pass (a prompt) device-resident (`device_prefill.rs`): batched kernels,
    /// the KV cache and conv state written on the device, one read at the end of each chunk.
    /// `LB_DEVICE_PREFILL=1` at start; off by default until its parity gate has passed on a
    /// full model (`lfm2-check --ab-prefill`).
    device_prefill: AtomicBool,
}

fn env_on(k: &str) -> bool {
    env_or(k, false)
}

/// `k` set to anything but "" or "0" is on, set to those is off, unset is `default`.
fn env_or(k: &str, default: bool) -> bool {
    std::env::var(k).map(|v| !v.is_empty() && v != "0").unwrap_or(default)
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            miss_policy: AtomicU8::new(MissPolicy::Cpu as u8),
            host_repack: AtomicBool::new(std::env::var_os("LB_HOST_REPACK").is_some()),
            batch_miss: AtomicBool::new(std::env::var_os("LB_NO_BATCH_MISS").is_none()),
            device_decode: AtomicBool::new(env_or("LB_DEVICE_DECODE", true)),
            dd_host_moe: AtomicBool::new(env_on("LB_DD_HOST_MOE")),
            e2_device_router: AtomicBool::new(env_or("LB_E2", true)),
            dd_router_seq: AtomicBool::new(env_on("LB_DD_ROUTER_SEQ")),
            device_prefill: AtomicBool::new(env_on("LB_DEVICE_PREFILL")),
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

    pub fn device_decode(&self) -> bool {
        self.device_decode.load(Ordering::Relaxed)
    }

    pub fn set_device_decode(&self, on: bool) {
        self.device_decode.store(on, Ordering::Relaxed);
    }

    pub fn dd_host_moe(&self) -> bool {
        self.dd_host_moe.load(Ordering::Relaxed)
    }

    pub fn set_dd_host_moe(&self, on: bool) {
        self.dd_host_moe.store(on, Ordering::Relaxed);
    }

    pub fn e2_device_router(&self) -> bool {
        self.e2_device_router.load(Ordering::Relaxed)
    }

    pub fn set_e2_device_router(&self, on: bool) {
        self.e2_device_router.store(on, Ordering::Relaxed);
    }

    pub fn device_prefill(&self) -> bool {
        self.device_prefill.load(Ordering::Relaxed)
    }

    pub fn set_device_prefill(&self, on: bool) {
        self.device_prefill.store(on, Ordering::Relaxed);
    }

    pub fn dd_router_seq(&self) -> bool {
        self.dd_router_seq.load(Ordering::Relaxed)
    }

    pub fn set_dd_router_seq(&self, on: bool) {
        self.dd_router_seq.store(on, Ordering::Relaxed);
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
        for (get, set) in [
            (Knobs::device_decode as fn(&Knobs) -> bool, Knobs::set_device_decode as fn(&Knobs, bool)),
            (Knobs::dd_host_moe, Knobs::set_dd_host_moe),
            (Knobs::dd_router_seq, Knobs::set_dd_router_seq),
        ] {
            let b = get(&k);
            set(&k, !b);
            assert_eq!(get(&k), !b);
            set(&k, b);
        }
    }

    #[test]
    fn device_decode_is_on_unless_switched_off() {
        if std::env::var_os("LB_DEVICE_DECODE").is_none() {
            assert!(Knobs::default().device_decode());
        }
        assert!(!env_on("LB_SURELY_UNSET_FOR_THIS_TEST"));
        assert!(env_or("LB_SURELY_UNSET_FOR_THIS_TEST", true));
    }
}
