//! `lb-tune` - suggest a layout (layer split, resident experts, RAM/Optane/SSD tiers, expert
//! helper) for a model on a fleet, with the expected tokens/s and its range. See docs/FLEET_TUNER.md.
//!
//!   lb-tune --model M.gguf --manifest fleet-manifest.json --matrix /var/lib/nauti/fleet-matrix.json \
//!           --overlay tiers.json [--bench node:gpu=bench.jsonl] [--scrapes DIR] [--explain] [--out plan.json]
//!   lb-tune --model-json m.json --fleet fleet.json ...
//!   lb-tune --manifest ... --dump-fleet fleet.json          (write the loaded fleet, no model needed)
use anyhow::{bail, Context, Result};
use clap::Parser;
use lb_calibrate::fleet::{self, load, model, plan, quant, report, Fleet, Knobs};
use lb_calibrate::timings::load_bench;
use lb_calibrate::trace::Trace;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// GGUF file (first shard of a split model is enough).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Model description JSON instead of a GGUF.
    #[arg(long)]
    model_json: Option<PathBuf>,
    /// Write the model description read from the GGUF.
    #[arg(long)]
    dump_model: Option<PathBuf>,
    /// Fleet description JSON (see FLEET_TUNER.md).
    #[arg(long)]
    fleet: Option<PathBuf>,
    /// Build the fleet from fleet-manifest.json ...
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// ... and the nauti fleet matrix (live RAM, GPUs, links).
    #[arg(long)]
    matrix: Option<PathBuf>,
    /// Partial fleet JSON merged over the loaded one (disk tiers, measured links, re-carded GPUs).
    #[arg(long)]
    overlay: Vec<PathBuf>,
    #[arg(long)]
    dump_fleet: Option<PathBuf>,
    /// Measured timings of one card: NODE:GPU_INDEX=bench.jsonl (from lb-gpu-bench).
    #[arg(long)]
    bench: Vec<String>,
    /// NautiBurn scrape dir: the routing skew is then measured instead of assumed.
    #[arg(long)]
    scrapes: Option<PathBuf>,
    /// Skew curve when no scrapes: profile | uniform.
    #[arg(long)]
    skew: Option<String>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    explain: bool,
    /// Only layouts whose primary node is NODE.
    #[arg(long)]
    node: Option<String>,
    /// Only layouts using this helper node ("none" = no helper).
    #[arg(long)]
    helper: Option<String>,
    /// Print every variant of a node, not just its best.
    #[arg(long)]
    variants: bool,
    /// Layouts to print.
    #[arg(long, default_value_t = 3)]
    top: usize,
    // knobs
    #[arg(long)]
    reserve_mib: Option<f64>,
    #[arg(long)]
    ctx: Option<usize>,
    #[arg(long)]
    ram_reserve_frac: Option<f64>,
    #[arg(long)]
    ram_reserve_min_gib: Option<f64>,
    #[arg(long)]
    handoff_us: Option<f64>,
    #[arg(long)]
    cpu_thread_frac: Option<f64>,
    #[arg(long)]
    synth_tokens: Option<usize>,
    /// Quantize rate for the "remake the quant" hours line (GB of source per hour). Unset = unknown.
    #[arg(long = "requant-gb-h")]
    requant_gb_h: Option<f64>,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let mut knobs = Knobs::default();
    macro_rules! set { ($f:ident) => { if let Some(v) = a.$f.clone() { knobs.$f = v; } }; }
    set!(reserve_mib);
    set!(ctx);
    set!(ram_reserve_frac);
    set!(ram_reserve_min_gib);
    set!(handoff_us);
    set!(cpu_thread_frac);
    set!(synth_tokens);
    if let Some(s) = &a.skew {
        knobs.skew = s.clone();
    }
    knobs.requant_gb_per_hour = a.requant_gb_h;
    eprintln!("knobs: {}", serde_json::to_string(&knobs)?);

    let mut fl: Fleet = if let Some(p) = &a.fleet {
        let mut f: Fleet = serde_json::from_str(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?)?;
        f.collect_unknowns();
        f
    } else if let Some(m) = &a.manifest {
        load::from_files(m, a.matrix.as_deref())?
    } else {
        bail!("need --fleet or --manifest");
    };
    for o in &a.overlay {
        let ov: Fleet = serde_json::from_str(&std::fs::read_to_string(o).with_context(|| o.display().to_string())?)?;
        load::apply_overlay(&mut fl, ov);
    }
    if let Some(p) = &a.dump_fleet {
        std::fs::write(p, serde_json::to_string_pretty(&fl)?)?;
        eprintln!("fleet written to {}", p.display());
    }
    let md = match (&a.model, &a.model_json) {
        (Some(p), _) => model::from_gguf(p)?,
        (None, Some(p)) => model::ModelDesc::load_json(p)?,
        _ if a.dump_fleet.is_some() => return Ok(()),
        _ => bail!("need --model or --model-json"),
    };
    if let Some(p) = &a.dump_model {
        std::fs::write(p, serde_json::to_string_pretty(&md)?)?;
    }
    let mut benches = plan::Benches::new();
    for b in &a.bench {
        let (key, path) = b.split_once('=').context("--bench NODE:GPU=FILE")?;
        let (node, idx) = key.rsplit_once(':').context("--bench NODE:GPU=FILE")?;
        benches.insert((node.to_string(), idx.parse()?), load_bench(&PathBuf::from(path))?);
    }
    let curve = match &a.scrapes {
        Some(d) => plan::SkewCurve::from_trace(&Trace::load_dir(d, md.n_layer, md.n_expert)?),
        None => plan::SkewCurve::by_name(&knobs.skew),
    };
    let mut set = plan::plan_fleet(&fl, &md, &knobs, &benches, &curve);
    if let Some(n) = &a.node {
        set.plans.retain(|p| &p.layout.node == n);
    }
    if let Some(h) = &a.helper {
        set.plans.retain(|p| match &p.layout.helper {
            Some(x) => &x.node == h,
            None => h == "none",
        });
    }
    let rows = quant::table(&fl, &md, &knobs);
    let remake = quant::remake_advice(&md, &rows, &knobs);
    print!("{}", report::render(&md, &set, &rows, &remake, a.top, a.explain, &fl, a.variants));
    if let Some(p) = &a.out {
        let best = set.plans.first().context("no layout to write")?;
        std::fs::write(p, serde_json::to_string_pretty(best)?)?;
        eprintln!("plan written to {}", p.display());
    }
    let _ = fleet::GIB;
    Ok(())
}
