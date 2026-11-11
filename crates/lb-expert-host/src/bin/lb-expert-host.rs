//! An expert host process: opens the model file, holds the `(layer, expert)` set a placement
//! file assigns to `--name` (or a `--hold` spec), and answers `Run` requests.
//!
//! ```text
//! lb-expert-host --model m.gguf --placement placement.json --name a --listen tcp://0.0.0.0:7411
//! lb-expert-host --model m.gguf --hold "2-39:0-63" --name all --listen rdma://10.0.10.204:7411 --wait spin:200
//! ```
//! With `--placement` the file is watched: when it changes the host loads what it gained,
//! releases what it lost, and answers for the new set. With `--backend gpu` (features `gpu` or
//! `cuda`) the experts are uploaded to the device named by `LB_CUDA_DEVICE` / `LB_WGPU_DEVICE`;
//! pin a card by UUID with `CUDA_VISIBLE_DEVICES=GPU-<uuid> LB_CUDA_DEVICE=0`.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use lb_expert_host::host::{self, Backend, CpuBackend, Host};
use lb_expert_host::model::hex;
use lb_expert_host::placement::{HeldSet, PlacementMap, Wait};
use lb_expert_host::{transport, ModelExperts};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, env = "LB_MODEL")]
    model: PathBuf,
    /// This host's name in the placement file (and in the hello).
    #[arg(long)]
    name: String,
    /// Placement file; the host holds what it assigns to `--name`.
    #[arg(long, conflicts_with = "hold")]
    placement: Option<PathBuf>,
    /// Held set without a file: `layers:experts,...` (`2-39:0-63`, `*`).
    #[arg(long)]
    hold: Option<String>,
    /// Address to listen on; repeatable (`tcp://0.0.0.0:7411`, `unix:///run/x.sock`,
    /// `rdma://10.0.10.204:7412`).
    #[arg(long, required = true)]
    listen: Vec<String>,
    /// How a connection waits for the next request: `block`, `busy` (one core per
    /// connection, spinning) or `spin:<us>` (spin that long after each request, then block).
    #[arg(long, default_value = "block")]
    wait: String,
    /// `cpu`, or `gpu` (needs the `gpu` or `cuda` feature).
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// CPU backend threads (0 = rayon's default, every core).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Print a stats line this often (seconds; 0 = never).
    #[arg(long, default_value_t = 60)]
    stats_s: u64,
}

fn held_from(placement: &Path, name: &str, m: &ModelExperts) -> Result<HeldSet> {
    let text = std::fs::read_to_string(placement).with_context(|| format!("read {}", placement.display()))?;
    let map = PlacementMap::parse(&text, m.n_layer(), m.n_expert).map_err(|e| anyhow!(e))?;
    map.held_by(name).map_err(|e| anyhow!(e))
}

fn backend(kind: &str) -> Result<Box<dyn Backend>> {
    match kind {
        "cpu" => Ok(Box::new(CpuBackend)),
        #[cfg(feature = "gpu")]
        "gpu" => Ok(Box::new(lb_expert_host::host::GpuBackend::open().map_err(|e| anyhow!(e))?)),
        #[cfg(not(feature = "gpu"))]
        "gpu" => bail!("this binary was built without the `gpu` / `cuda` feature"),
        other => bail!("unknown backend {other:?} (cpu or gpu)"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().context("thread pool")?;
    }
    let wait = Wait::parse(&args.wait).map_err(|e| anyhow!(e))?;
    let t0 = Instant::now();
    let model = ModelExperts::open(&args.model)?;
    let held = match (&args.placement, &args.hold) {
        (Some(p), _) => held_from(p, &args.name, &model)?,
        (None, Some(spec)) => HeldSet::parse_spec(spec, model.n_layer(), model.n_expert).map_err(|e| anyhow!(e))?,
        (None, None) => bail!("give --placement or --hold"),
    };
    let bytes: usize = held.iter().map(|(l, _)| model.expert_bytes(l)).sum();
    let n_held = held.len();
    let host = Arc::new(Host::new(&args.name, model, held, backend(&args.backend)?).map_err(|e| anyhow!(e))?);
    eprintln!(
        "lb-expert-host {}: model {} (fingerprint {}), {} experts held ({:.0} MB of file), backend {}, loaded in {:.1}s",
        args.name,
        args.model.display(),
        hex(&host.model.fingerprint),
        n_held,
        bytes as f64 / 1e6,
        host.info().backend,
        t0.elapsed().as_secs_f32()
    );
    let mut threads = Vec::new();
    for addr in &args.listen {
        let l = transport::listen(addr, wait).with_context(|| format!("listen {addr}"))?;
        eprintln!("lb-expert-host {}: listening on {} (wait {})", args.name, l.local(), wait.name());
        threads.push(host::spawn(host.clone(), l));
    }
    // placement watcher and stats
    let mut mtime = args.placement.as_ref().and_then(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    let mut last_stats = Instant::now();
    let mut last_runs = 0u64;
    loop {
        std::thread::sleep(Duration::from_millis(500));
        if let Some(p) = &args.placement {
            let now = std::fs::metadata(p).and_then(|m| m.modified()).ok();
            if now.is_some() && now != mtime {
                mtime = now;
                match held_from(p, &args.name, &host.model).and_then(|h| {
                    let n = h.len();
                    host.set_held(h).map(|b| (n, b)).map_err(|e| anyhow!(e))
                }) {
                    Ok((n, b)) => eprintln!("lb-expert-host {}: placement reloaded, {n} experts held ({:.0} MB on device)", args.name, b as f64 / 1e6),
                    Err(e) => eprintln!("lb-expert-host {}: placement reload failed, the old set stays: {e:#}", args.name),
                }
            }
        }
        if args.stats_s > 0 && last_stats.elapsed() >= Duration::from_secs(args.stats_s) {
            last_stats = Instant::now();
            let s = &host.stats;
            let runs = s.runs.load(Ordering::Relaxed);
            if runs != last_runs {
                last_runs = runs;
                eprintln!(
                    "lb-expert-host {}: {} runs, {} experts, {:.3} ms compute a run, {} refused, {} errors",
                    args.name,
                    runs,
                    s.experts.load(Ordering::Relaxed),
                    s.compute_ns.load(Ordering::Relaxed) as f64 / 1e6 / runs.max(1) as f64,
                    s.refused.load(Ordering::Relaxed),
                    s.errors.load(Ordering::Relaxed)
                );
            }
        }
        if threads.iter().all(|t| t.is_finished()) {
            bail!("every listener stopped");
        }
    }
}
