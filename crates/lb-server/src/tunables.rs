//! `/v1/tunables`: every knob the engine reads, with its current value, and a way to change
//! the live ones without a restart.
//!
//! An agent tuning the engine reads `GET /v1/tunables` (one entry per knob: `name`, `value`,
//! `type`, `range` or `choices`, `scope` = `live` or `restart`, `source` = the flag or
//! environment variable that sets it at start, `description`) and posts `{"name": value, ...}`
//! to `POST /v1/tunables`. A post is all-or-nothing: every key is checked first, and one
//! unknown key, one restart-only knob or one value out of range refuses the whole post with
//! the reason, so a half-applied change can never be mistaken for a measurement.
//!
//! Live knobs take effect at the next forward pass (the model reads them at the top of every
//! pass, see `lb_lfm2::knobs`), so a change posted during a request applies from the next
//! token on; between requests it applies to the whole next request.

use lb_lfm2::knobs::Knobs;
use lb_lfm2::moe::{MissPolicy, Scraper};
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Where the live knobs are read from and written to.
#[derive(Default)]
pub struct LiveTargets {
    /// LFM2 only: the model's live settings.
    pub knobs: Option<Arc<Knobs>>,
    /// LFM2 only: the routing scraper.
    pub scraper: Option<Arc<Scraper>>,
    /// Server: `max_tokens` when a request gives none.
    pub default_max_tokens: Arc<AtomicUsize>,
    /// Server: the context the model was loaded with (the ceiling for `default_max_tokens`).
    pub ctx: usize,
}

/// A knob that is set once at start: listed so an agent sees the whole picture, refused on a
/// post with the exact flag or variable to use instead.
#[derive(Clone, Debug)]
pub struct Fixed {
    pub name: &'static str,
    pub value: Value,
    pub ty: &'static str,
    pub source: &'static str,
    pub description: &'static str,
}

/// Why a post was refused: the whole post is dropped, this names the first offending key.
#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Not a knob at all.
    Unknown(String),
    /// A knob, but restart-only: the message says how to set it.
    RestartOnly { name: String, how: String },
    /// A live knob given a value it cannot take.
    Invalid { name: String, why: String },
}

impl Refusal {
    pub fn message(&self) -> String {
        match self {
            Self::Unknown(n) => format!("{n:?} is not a tunable (GET /v1/tunables lists them)"),
            Self::RestartOnly { name, how } => format!("{name} is restart-only: {how}; nothing was changed"),
            Self::Invalid { name, why } => format!("{name}: {why}; nothing was changed"),
        }
    }

    /// 400 for a bad request, 409 for a knob that exists but cannot move live.
    pub fn status(&self) -> u16 {
        match self {
            Self::RestartOnly { .. } => 409,
            _ => 400,
        }
    }
}

pub struct Tunables {
    live: LiveTargets,
    fixed: Vec<Fixed>,
}

/// One validated, not yet applied change.
enum Change {
    MissPolicy(MissPolicy),
    HostRepack(bool),
    BatchMiss(bool),
    DeviceDecode(bool),
    DdHostMoe(bool),
    DdRouterSeq(bool),
    ScrapeDir(Option<PathBuf>),
    ScrapeModelId(String),
    ScrapeClassId(String),
    DefaultMaxTokens(usize),
}

impl Tunables {
    pub fn new(live: LiveTargets, fixed: Vec<Fixed>) -> Self {
        Self { live, fixed }
    }

    /// The restart-only knobs as the server started: CLI flags and the environment variables
    /// the kernels read. `env` is read here so the listing shows what the process actually saw.
    #[allow(clippy::too_many_arguments)]
    pub fn fixed_from_start(
        model: &str,
        gpu: bool,
        gpu_budget_mb: usize,
        gpu_dense_mb: usize,
        placement_from_scrapes: Option<&str>,
        ctx: usize,
        threads: usize,
        model_id: &str,
    ) -> Vec<Fixed> {
        let env = |k: &str| std::env::var(k).ok().map(Value::from).unwrap_or(Value::Null);
        let env_flag = |k: &str| Value::from(std::env::var_os(k).is_some());
        vec![
            Fixed { name: "model", value: json!(model), ty: "path", source: "--model", description: "the GGUF (LFM2) or converted BitNet directory being served" },
            Fixed { name: "model_id", value: json!(model_id), ty: "string", source: "--model-id", description: "the id /v1/models reports" },
            Fixed { name: "gpu", value: json!(gpu), ty: "bool", source: "--gpu", description: "resident experts and dense weights on the device (needs the gpu/cuda feature)" },
            Fixed { name: "gpu_budget_mb", value: json!(gpu_budget_mb), ty: "int", source: "--gpu-budget-mb", description: "device bytes for resident experts (MB); the placement is planned and uploaded at load" },
            Fixed { name: "gpu_dense_mb", value: json!(gpu_dense_mb), ty: "int", source: "--gpu-dense-mb", description: "device bytes for the dense weights (MB): conv/attention projections, leading dense FFNs, the tied head" },
            Fixed { name: "placement_from_scrapes", value: placement_from_scrapes.map(Value::from).unwrap_or(Value::Null), ty: "path", source: "--placement-from-scrapes", description: "directory of NautiBurn scrape files the residency plan is built from" },
            Fixed { name: "ctx", value: json!(ctx), ty: "int", source: "--ctx", description: "max prompt + generated tokens" },
            Fixed { name: "threads", value: json!(threads), ty: "int", source: "--threads / LB_THREADS / RAYON_NUM_THREADS", description: "rayon compute threads (the global pool is built once)" },
            Fixed { name: "cuda_device", value: env("LB_CUDA_DEVICE"), ty: "int", source: "LB_CUDA_DEVICE", description: "CUDA device index (driver order) for the cubecl-cuda runtime; unset = wgpu" },
            Fixed { name: "wgpu_device", value: env("LB_WGPU_DEVICE"), ty: "string", source: "LB_WGPU_DEVICE", description: "wgpu adapter, e.g. discrete:0 (its order is not nvidia-smi's)" },
            Fixed { name: "no_register", value: env_flag("LB_NO_REGISTER"), ty: "bool", source: "LB_NO_REGISTER", description: "skip page-locking the GGUF mapping (misses then repack on the host from the pinned pool)" },
            Fixed { name: "pinned_pool_mb", value: env("LB_PINNED_POOL_MB"), ty: "int", source: "LB_PINNED_POOL_MB", description: "page-locked staging pool cap for host repacks (default 256)" },
            Fixed { name: "no_pinned", value: env_flag("LB_NO_PINNED"), ty: "bool", source: "LB_NO_PINNED", description: "no pinned staging pool: host repacks upload from pageable memory" },
            Fixed { name: "q4_cube", value: env("LB_Q4_CUBE"), ty: "int", source: "LB_Q4_CUBE", description: "units per cube for the Q4_0 matvec kernels (default: two planes)" },
            Fixed { name: "q4_rows", value: env("LB_Q4_ROWS"), ty: "int", source: "LB_Q4_ROWS", description: "output rows per plane in the Q4_0 kernels: 1 (unset: the one-row kernels), 4 or 8 (the D2 rows kernels; need plane ops)" },
            Fixed { name: "q4_xshared", value: env_flag("LB_Q4_XSHARED"), ty: "bool", source: "LB_Q4_XSHARED", description: "rows kernels stage the activation row in shared memory (one input row, up to q4_xshared_max_cols)" },
            Fixed { name: "q4_xshared_max_cols", value: env("LB_Q4_XSHARED_MAX_COLS"), ty: "int", source: "LB_Q4_XSHARED_MAX_COLS", description: "widest matrix whose activation row goes to shared memory (default 4096 columns = 16 KB)" },
            Fixed { name: "no_plane", value: env_flag("LB_NO_PLANE"), ty: "bool", source: "LB_NO_PLANE", description: "reduce with a shared-memory tree instead of plane_sum" },
            Fixed { name: "expert_layers", value: env("LB_EXPERT_LAYERS"), ty: "bool", source: "LB_EXPERT_LAYERS (unset: follows LB_DEVICE_DECODE)", description: "upload a MoE layer whose experts are all planned resident as one stacked buffer per projection (per-expert views into it); the device-resident decode needs it" },
            Fixed { name: "conv_block", value: env_flag("LB_CONV_BLOCK"), ty: "bool", source: "LB_CONV_BLOCK", description: "whole short-conv blocks resident on the device (off by default; measured slower)" },
            Fixed { name: "repack_scalar", value: env_flag("LB_REPACK_SCALAR"), ty: "bool", source: "LB_REPACK_SCALAR", description: "host repack with the scalar loop instead of SSSE3" },
            Fixed { name: "no_batch_miss_at_start", value: env_flag("LB_NO_BATCH_MISS"), ty: "bool", source: "LB_NO_BATCH_MISS", description: "the start value of the live knob batch_miss" },
            Fixed { name: "host_repack_at_start", value: env_flag("LB_HOST_REPACK"), ty: "bool", source: "LB_HOST_REPACK", description: "the start value of the live knob host_repack" },
            Fixed { name: "kernel", value: env("LB_KERNEL"), ty: "string", source: "LB_KERNEL", description: "BitNet only: ternary kernel (avx2, avx512, ...)" },
            Fixed { name: "tasks_per_thread", value: env("LB_TASKS_PER_THREAD"), ty: "int", source: "LB_TASKS_PER_THREAD", description: "BitNet only: row-chunk tasks per thread (default 8)" },
            Fixed { name: "numa", value: env("LB_NUMA"), ty: "string", source: "LB_NUMA / LB_NUMA_SHARE", description: "BitNet only: two-socket split, e.g. 0-19;20-39" },
        ]
    }

    fn live_entries(&self) -> Vec<Value> {
        let mut v = Vec::new();
        if let Some(k) = &self.live.knobs {
            v.push(json!({"name": "miss_policy", "value": k.miss_policy().name(), "type": "enum", "choices": ["cpu", "on-demand"], "scope": "live", "source": "--miss-policy",
                "description": "an expert the router asked for that is not resident: run it on the CPU, or copy it to the device for this call"}));
            v.push(json!({"name": "host_repack", "value": k.host_repack(), "type": "bool", "scope": "live", "source": "LB_HOST_REPACK",
                "description": "repack a miss's Q4_0 blocks on the host (pinned pool) instead of on the device from the page-locked mapping; no effect without the cuda runtime and a registered mapping"}));
            v.push(json!({"name": "batch_miss", "value": k.batch_miss(), "type": "bool", "scope": "live", "source": "LB_NO_BATCH_MISS (inverted)",
                "description": "upload a miss's three matrices in three device calls (one create, one allocation, one repack) instead of twelve; cuda + registered mapping only"}));
            v.push(json!({"name": "device_decode", "value": k.device_decode(), "type": "bool", "scope": "live", "source": "LB_DEVICE_DECODE",
                "description": "run a decode token device-resident (one upload, one read-back) when every dense matrix, the head and every MoE layer (as a stacked layer) is resident; otherwise the token takes the old path"}));
            v.push(json!({"name": "dd_host_moe", "value": k.dd_host_moe(), "type": "bool", "scope": "live", "source": "LB_DD_HOST_MOE",
                "description": "inside a device-resident token, run a MoE layer that is not resident on the host path (one sync for that layer) instead of giving the whole token to the old path"}));
            v.push(json!({"name": "dd_router_seq", "value": k.dd_router_seq(), "type": "bool", "scope": "live", "source": "LB_DD_ROUTER_SEQ",
                "description": "device router logits as one sequential dot per expert (the host's summation order) instead of a plane reduce; slower, for parity checks"}));
        }
        if let Some(s) = &self.live.scraper {
            v.push(json!({"name": "scrape_dir", "value": s.dir().map(|p| p.to_string_lossy().into_owned()), "type": "path|null", "scope": "live", "source": "NAUTIBURN_SCRAPE_DIR",
                "description": "where NautiBurn routing scrape files go, one per request; null switches recording off"}));
            v.push(json!({"name": "scrape_model_id", "value": s.model_id(), "type": "string", "scope": "live", "source": "NAUTIBURN_MODEL_ID", "description": "model_id written into scrape files"}));
            v.push(json!({"name": "scrape_class_id", "value": s.class_id(), "type": "string", "scope": "live", "source": "NAUTIBURN_CLASS_ID", "description": "class_id written into scrape files"}));
        }
        v.push(json!({"name": "default_max_tokens", "value": self.live.default_max_tokens.load(Ordering::Relaxed), "type": "int", "range": [1, self.live.ctx], "scope": "live", "source": "--default-max-tokens",
            "description": "max_tokens for a request that gives none"}));
        v
    }

    /// Every knob, live ones first.
    pub fn describe(&self) -> Value {
        let mut all = self.live_entries();
        for f in &self.fixed {
            all.push(json!({"name": f.name, "value": f.value, "type": f.ty, "scope": "restart", "source": f.source, "description": f.description}));
        }
        json!({"schema": "lemonade-burn.tunables.v1", "tunables": all})
    }

    fn validate(&self, name: &str, value: &Value) -> Result<Change, Refusal> {
        let invalid = |why: String| Refusal::Invalid { name: name.to_string(), why };
        let as_bool = |v: &Value| v.as_bool().ok_or_else(|| invalid(format!("expected true or false, got {v}")));
        let as_str = |v: &Value| v.as_str().map(str::to_string).ok_or_else(|| invalid(format!("expected a string, got {v}")));
        let knobs = self.live.knobs.as_ref();
        let scraper = self.live.scraper.as_ref();
        let not_lfm2 = || Refusal::RestartOnly { name: name.to_string(), how: "only an LFM2 (GGUF) model has this knob".to_string() };
        match name {
            "miss_policy" => {
                knobs.ok_or_else(not_lfm2)?;
                let s = as_str(value)?;
                MissPolicy::parse(&s).map(Change::MissPolicy).map_err(invalid)
            }
            "host_repack" => {
                knobs.ok_or_else(not_lfm2)?;
                as_bool(value).map(Change::HostRepack)
            }
            "batch_miss" => {
                knobs.ok_or_else(not_lfm2)?;
                as_bool(value).map(Change::BatchMiss)
            }
            "device_decode" => {
                knobs.ok_or_else(not_lfm2)?;
                as_bool(value).map(Change::DeviceDecode)
            }
            "dd_host_moe" => {
                knobs.ok_or_else(not_lfm2)?;
                as_bool(value).map(Change::DdHostMoe)
            }
            "dd_router_seq" => {
                knobs.ok_or_else(not_lfm2)?;
                as_bool(value).map(Change::DdRouterSeq)
            }
            "scrape_dir" => {
                scraper.ok_or_else(not_lfm2)?;
                match value {
                    Value::Null => Ok(Change::ScrapeDir(None)),
                    Value::String(s) if !s.is_empty() => Ok(Change::ScrapeDir(Some(PathBuf::from(s)))),
                    v => Err(invalid(format!("expected a directory path or null, got {v}"))),
                }
            }
            "scrape_model_id" => {
                scraper.ok_or_else(not_lfm2)?;
                as_str(value).map(Change::ScrapeModelId)
            }
            "scrape_class_id" => {
                scraper.ok_or_else(not_lfm2)?;
                as_str(value).map(Change::ScrapeClassId)
            }
            "default_max_tokens" => {
                let n = value.as_u64().ok_or_else(|| invalid(format!("expected a positive integer, got {value}")))? as usize;
                if n == 0 || n > self.live.ctx {
                    return Err(invalid(format!("must be in 1..={} (the context), got {n}", self.live.ctx)));
                }
                Ok(Change::DefaultMaxTokens(n))
            }
            other => match self.fixed.iter().find(|f| f.name == other) {
                Some(f) => Err(Refusal::RestartOnly { name: other.to_string(), how: format!("restart the server with {}", f.source) }),
                None => Err(Refusal::Unknown(other.to_string())),
            },
        }
    }

    /// Apply `req` (`{"name": value, ...}`) all at once. On `Err` nothing was changed. On
    /// `Ok` the returned object holds the names that changed with their new values.
    pub fn apply(&self, req: &Map<String, Value>) -> Result<Value, Refusal> {
        let mut changes = Vec::with_capacity(req.len());
        for (name, value) in req {
            changes.push((name.clone(), self.validate(name, value)?));
        }
        let mut applied = Map::new();
        for (name, c) in changes {
            let now = match c {
                Change::MissPolicy(p) => {
                    self.live.knobs.as_ref().expect("validated").set_miss_policy(p);
                    json!(p.name())
                }
                Change::HostRepack(b) => {
                    self.live.knobs.as_ref().expect("validated").set_host_repack(b);
                    json!(b)
                }
                Change::BatchMiss(b) => {
                    self.live.knobs.as_ref().expect("validated").set_batch_miss(b);
                    json!(b)
                }
                Change::DeviceDecode(b) => {
                    self.live.knobs.as_ref().expect("validated").set_device_decode(b);
                    json!(b)
                }
                Change::DdHostMoe(b) => {
                    self.live.knobs.as_ref().expect("validated").set_dd_host_moe(b);
                    json!(b)
                }
                Change::DdRouterSeq(b) => {
                    self.live.knobs.as_ref().expect("validated").set_dd_router_seq(b);
                    json!(b)
                }
                Change::ScrapeDir(d) => {
                    let s = self.live.scraper.as_ref().expect("validated");
                    s.set_dir(d);
                    json!(s.dir().map(|p| p.to_string_lossy().into_owned()))
                }
                Change::ScrapeModelId(id) => {
                    self.live.scraper.as_ref().expect("validated").set_model_id(id.clone());
                    json!(id)
                }
                Change::ScrapeClassId(id) => {
                    self.live.scraper.as_ref().expect("validated").set_class_id(id.clone());
                    json!(id)
                }
                Change::DefaultMaxTokens(n) => {
                    self.live.default_max_tokens.store(n, Ordering::Relaxed);
                    json!(n)
                }
            };
            applied.insert(name, now);
        }
        Ok(json!({"applied": applied, "applies": "from the next forward pass"}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lfm2_tunables() -> (Tunables, Arc<Knobs>, Arc<Scraper>) {
        let knobs = Arc::new(Knobs::default());
        let scraper = Arc::new(Scraper::new(None, "m".into(), "c".into()));
        let live = LiveTargets { knobs: Some(knobs.clone()), scraper: Some(scraper.clone()), default_max_tokens: Arc::new(AtomicUsize::new(512)), ctx: 2048 };
        let fixed = Tunables::fixed_from_start("/m/x.gguf", true, 14500, 900, Some("/s"), 2048, 80, "x");
        (Tunables::new(live, fixed), knobs, scraper)
    }

    fn entry<'a>(d: &'a Value, name: &str) -> &'a Value {
        d["tunables"].as_array().unwrap().iter().find(|e| e["name"] == name).unwrap_or_else(|| panic!("{name} missing"))
    }

    #[test]
    fn describe_lists_every_knob_with_its_scope_and_source() {
        let (t, _, _) = lfm2_tunables();
        let d = t.describe();
        assert_eq!(d["schema"], "lemonade-burn.tunables.v1");
        for e in d["tunables"].as_array().unwrap() {
            for k in ["name", "value", "type", "scope", "source", "description"] {
                assert!(e.get(k).is_some(), "{} lacks {k}", e["name"]);
            }
            assert!(e["scope"] == "live" || e["scope"] == "restart", "{}", e["name"]);
        }
        // defaults, untouched: the engine behaves as before anything is posted
        assert_eq!(entry(&d, "miss_policy")["value"], "cpu");
        assert_eq!(entry(&d, "batch_miss")["value"], true);
        assert_eq!(entry(&d, "scrape_dir")["value"], Value::Null);
        assert_eq!(entry(&d, "default_max_tokens")["value"], 512);
        assert_eq!(entry(&d, "gpu_budget_mb")["value"], 14500);
        assert_eq!(entry(&d, "gpu_budget_mb")["scope"], "restart");
        assert_eq!(entry(&d, "threads")["value"], 80);
        if std::env::var_os("LB_DEVICE_DECODE").is_none() {
            assert_eq!(entry(&d, "device_decode")["value"], false, "off unless asked");
        }
        for live in ["miss_policy", "host_repack", "batch_miss", "device_decode", "dd_host_moe", "dd_router_seq", "scrape_dir", "scrape_model_id", "scrape_class_id", "default_max_tokens"] {
            assert_eq!(entry(&d, live)["scope"], "live", "{live}");
        }
        for fixed in ["model", "gpu", "gpu_dense_mb", "placement_from_scrapes", "ctx", "cuda_device", "wgpu_device", "no_register", "pinned_pool_mb", "q4_cube", "q4_rows", "q4_xshared", "q4_xshared_max_cols", "expert_layers", "conv_block"] {
            assert_eq!(entry(&d, fixed)["scope"], "restart", "{fixed}");
        }
    }

    #[test]
    fn live_knobs_change_and_show_the_new_value() {
        let (t, knobs, scraper) = lfm2_tunables();
        let req: Map<String, Value> = serde_json::from_value(json!({
            "miss_policy": "on-demand", "host_repack": true, "batch_miss": false, "device_decode": true, "dd_host_moe": true,
            "scrape_dir": "/tmp/scr", "scrape_class_id": "live-test", "default_max_tokens": 64
        }))
        .unwrap();
        let r = t.apply(&req).expect("all live");
        assert_eq!(r["applied"]["miss_policy"], "on-demand");
        assert_eq!(r["applied"]["batch_miss"], false);
        assert_eq!(knobs.miss_policy(), MissPolicy::OnDemand);
        assert!(knobs.host_repack());
        assert!(!knobs.batch_miss());
        assert!(knobs.device_decode() && knobs.dd_host_moe());
        assert_eq!(scraper.dir(), Some(PathBuf::from("/tmp/scr")));
        assert_eq!(scraper.class_id(), "live-test");
        let d = t.describe();
        assert_eq!(entry(&d, "miss_policy")["value"], "on-demand");
        assert_eq!(entry(&d, "scrape_dir")["value"], "/tmp/scr");
        assert_eq!(entry(&d, "default_max_tokens")["value"], 64);
        // and back off
        let req: Map<String, Value> = serde_json::from_value(json!({"scrape_dir": null, "miss_policy": "cpu"})).unwrap();
        t.apply(&req).unwrap();
        assert_eq!(scraper.dir(), None);
        assert_eq!(knobs.miss_policy(), MissPolicy::Cpu);
    }

    #[test]
    fn restart_only_is_refused_and_nothing_in_the_post_is_applied() {
        let (t, knobs, _) = lfm2_tunables();
        let req: Map<String, Value> = serde_json::from_value(json!({"miss_policy": "on-demand", "gpu_budget_mb": 3200})).unwrap();
        let e = t.apply(&req).unwrap_err();
        assert_eq!(e, Refusal::RestartOnly { name: "gpu_budget_mb".into(), how: "restart the server with --gpu-budget-mb".into() });
        assert_eq!(e.status(), 409);
        assert!(e.message().contains("--gpu-budget-mb"), "{}", e.message());
        // the live key in the same post did not move either
        assert_eq!(knobs.miss_policy(), MissPolicy::Cpu);
    }

    #[test]
    fn bad_values_and_unknown_keys_are_refused() {
        let (t, knobs, _) = lfm2_tunables();
        for (req, status) in [
            (json!({"miss_policy": "sometimes"}), 400),
            (json!({"host_repack": "yes"}), 400),
            (json!({"device_decode": 1}), 400),
            (json!({"default_max_tokens": 0}), 400),
            (json!({"default_max_tokens": 4096}), 400),
            (json!({"scrape_dir": 7}), 400),
            (json!({"nonsense": 1}), 400),
        ] {
            let req: Map<String, Value> = serde_json::from_value(req).unwrap();
            let e = t.apply(&req).unwrap_err();
            assert_eq!(e.status(), status, "{e:?}");
        }
        assert_eq!(knobs.miss_policy(), MissPolicy::Cpu);
        assert!(knobs.batch_miss());
    }

    #[test]
    fn a_model_without_experts_only_has_the_server_knobs() {
        let live = LiveTargets { knobs: None, scraper: None, default_max_tokens: Arc::new(AtomicUsize::new(512)), ctx: 4096 };
        let t = Tunables::new(live, Tunables::fixed_from_start("/m/bitnet", false, 0, 0, None, 4096, 8, "bitnet"));
        let d = t.describe();
        let live: Vec<&str> = d["tunables"].as_array().unwrap().iter().filter(|e| e["scope"] == "live").map(|e| e["name"].as_str().unwrap()).collect();
        assert_eq!(live, vec!["default_max_tokens"]);
        let req: Map<String, Value> = serde_json::from_value(json!({"miss_policy": "cpu"})).unwrap();
        assert_eq!(t.apply(&req).unwrap_err().status(), 409);
        let req: Map<String, Value> = serde_json::from_value(json!({"default_max_tokens": 1000})).unwrap();
        t.apply(&req).unwrap();
        assert_eq!(t.live.default_max_tokens.load(Ordering::Relaxed), 1000);
    }
}
