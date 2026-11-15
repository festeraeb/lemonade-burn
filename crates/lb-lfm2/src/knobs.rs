//! Live knobs: the settings the forward pass reads on every call, held behind atomics so a
//! host (the server's `/v1/tunables`) can change them while the model runs on its own thread.
//!
//! Only what can take effect between two forward passes without touching device memory lives
//! here. Residency (GPU budget, dense budget, placement), the device itself and the thread pool
//! are set once at load and stay restart-only; the server lists those too, but refuses to
//! change them.

use crate::moe::MissPolicy;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

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
    /// Compute CPU experts (misses) with the int8-activation kernel (`lb_gguf::kernels_i8`)
    /// instead of the f32 reference: faster, slightly different arithmetic. `LB_CPU_I8=1` at
    /// start; off by default until its accuracy gate has passed.
    cpu_i8: AtomicBool,
    /// Replay a device decode token as one captured CUDA graph (`device_decode.rs`, cubecl-cuda's
    /// `start_capture` / `Graph::replay`) instead of issuing its ~390 launches one by one. Same
    /// kernels, same arguments, same order: the logits are bit-identical. `LB_DD_GRAPH=1` at
    /// start; off by default until its gates have passed. Live.
    dd_graph: AtomicBool,
    /// Device-time cuts in the decode token that keep every result bit for bit (`device_decode.rs`):
    /// the router spread over one cube per expert, and the expert-sum order worked out once per cube.
    /// `LB_DD_FAST=1` at start; off by default until its gates have passed. Live.
    dd_fast: AtomicBool,
    /// Engine step E2, miss policy: a cache miss is computed on the CPU from the host-resident
    /// GGUF bytes and its output joins the device token, instead of copying the expert to the card.
    /// The kernel is the exact f32 SIMD one, or the int8 one with `cpu_i8`. `LB_CPU_MISS=1` at
    /// start; off by default. Live.
    cpu_miss: AtomicBool,
    /// With `cpu_miss`: a missed expert is copied into the cache on its Nth miss (second touch by
    /// default); before that it is computed on the CPU. 0 = never copy (cache frozen at its seed),
    /// 1 = always copy (no CPU work). `LB_CPU_MISS_ADMIT=N`. Live.
    cpu_miss_admit: AtomicU32,
    /// Engine step E2, speculative layers: the experts of a MoE layer are launched from a device-side
    /// (layer, expert) -> slot table right after the router, before the host has read the routes, so the card
    /// runs the hits while the host reads and plans; only a layer with a miss is repaired afterwards. Bit-identical
    /// to the synced layer. Needs the look-ahead off (`LB_LA_*` unset). `LB_E2_SPEC=1` at start; off by default. Live.
    e2_spec: AtomicBool,
    /// E2 path: hand the queued launches to CubeCL's device thread after every Nth launch group instead of
    /// only at the routes read, so the thread issues them while the host is still building the rest of the
    /// layer (`docs/ENGINE_SYNC_COST.md`). 0 = off, 1 = after every group (the default). Bit-identical.
    /// `LB_E2_FLUSH=N`. Live.
    e2_flush: AtomicU32,
    /// E2 path: fold a MoE layer's weighted expert sum into the next pre-norm's launch (as the all-resident
    /// path does): one launch fewer a layer, bit-identical. On by default; `LB_E2_FUSE=0` turns it off. Live.
    e2_fuse: AtomicBool,
    /// Predictor (`LB_PRED=1`): a guessed expert is prefetched when its probability is at least this many thousandths.
    /// `LB_PRED_THETA=0.4`. Live.
    pred_theta_milli: AtomicU32,
    /// Predictor: it runs only while the moving average of demand misses a token is at least this many (0 = always).
    /// `LB_PRED_MIN_MISS=6`. Live.
    pred_min_miss: AtomicU32,
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
            cpu_i8: AtomicBool::new(env_on("LB_CPU_I8")),
            dd_graph: AtomicBool::new(env_on("LB_DD_GRAPH")),
            dd_fast: AtomicBool::new(env_or("LB_DD_FAST", true)),
            cpu_miss: AtomicBool::new(env_on("LB_CPU_MISS")),
            e2_spec: AtomicBool::new(env_on("LB_E2_SPEC")),
            e2_fuse: AtomicBool::new(env_or("LB_E2_FUSE", true)),
            e2_flush: AtomicU32::new(std::env::var("LB_E2_FLUSH").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1)),
            pred_theta_milli: AtomicU32::new(std::env::var("LB_PRED_THETA").ok().and_then(|v| v.trim().parse::<f32>().ok()).map(|t| (t * 1000.0).round().clamp(0.0, 1000.0) as u32).unwrap_or(400)),
            pred_min_miss: AtomicU32::new(std::env::var("LB_PRED_MIN_MISS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(6)),
            cpu_miss_admit: AtomicU32::new(std::env::var("LB_CPU_MISS_ADMIT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2)),
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

    pub fn cpu_i8(&self) -> bool {
        self.cpu_i8.load(Ordering::Relaxed)
    }

    pub fn set_cpu_i8(&self, on: bool) {
        self.cpu_i8.store(on, Ordering::Relaxed);
    }

    pub fn dd_router_seq(&self) -> bool {
        self.dd_router_seq.load(Ordering::Relaxed)
    }

    pub fn set_dd_router_seq(&self, on: bool) {
        self.dd_router_seq.store(on, Ordering::Relaxed);
    }

    pub fn dd_graph(&self) -> bool {
        self.dd_graph.load(Ordering::Relaxed)
    }

    pub fn set_dd_graph(&self, on: bool) {
        self.dd_graph.store(on, Ordering::Relaxed);
    }

    pub fn dd_fast(&self) -> bool {
        self.dd_fast.load(Ordering::Relaxed)
    }

    pub fn set_dd_fast(&self, on: bool) {
        self.dd_fast.store(on, Ordering::Relaxed);
    }

    pub fn e2_spec(&self) -> bool {
        self.e2_spec.load(Ordering::Relaxed)
    }

    pub fn set_e2_spec(&self, on: bool) {
        self.e2_spec.store(on, Ordering::Relaxed);
    }

    pub fn e2_fuse(&self) -> bool {
        self.e2_fuse.load(Ordering::Relaxed)
    }

    pub fn set_e2_fuse(&self, on: bool) {
        self.e2_fuse.store(on, Ordering::Relaxed);
    }

    pub fn e2_flush(&self) -> u32 {
        self.e2_flush.load(Ordering::Relaxed)
    }

    pub fn set_e2_flush(&self, n: u32) {
        self.e2_flush.store(n, Ordering::Relaxed);
    }

    pub fn pred_min_miss(&self) -> u32 {
        self.pred_min_miss.load(Ordering::Relaxed)
    }

    pub fn set_pred_min_miss(&self, n: u32) {
        self.pred_min_miss.store(n, Ordering::Relaxed);
    }

    pub fn pred_theta(&self) -> f32 {
        self.pred_theta_milli.load(Ordering::Relaxed) as f32 / 1000.0
    }

    pub fn set_pred_theta_milli(&self, n: u32) {
        self.pred_theta_milli.store(n.min(1000), Ordering::Relaxed);
    }

    pub fn cpu_miss(&self) -> bool {
        self.cpu_miss.load(Ordering::Relaxed)
    }

    pub fn set_cpu_miss(&self, on: bool) {
        self.cpu_miss.store(on, Ordering::Relaxed);
    }

    pub fn cpu_miss_admit(&self) -> u32 {
        self.cpu_miss_admit.load(Ordering::Relaxed)
    }

    pub fn set_cpu_miss_admit(&self, n: u32) {
        self.cpu_miss_admit.store(n, Ordering::Relaxed);
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
            (Knobs::dd_graph, Knobs::set_dd_graph),
            (Knobs::dd_fast, Knobs::set_dd_fast),
            (Knobs::cpu_miss, Knobs::set_cpu_miss),
            (Knobs::e2_spec, Knobs::set_e2_spec),
            (Knobs::e2_fuse, Knobs::set_e2_fuse),
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
