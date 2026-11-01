//! `lb-calibrate topology` — what cards this box has (driver counters).
//! `lb-calibrate tune --bench a.json [--bench b.json ...] --scrapes DIR [--budget-mb N ...] --out staging.json`
//!     fit the staged sets on the train split, simulate every candidate on the held-out split, write the best.
//! `lb-calibrate sweep --bench a.json --scrapes DIR --k 24 [--cards 2]` — bandwidth / k sensitivity tables.
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lb_calibrate::sim::{simulate, MissPolicy, SimConfig};
use lb_calibrate::timings::{load_bench, CardTimings, HostTimings, EXPERT_BYTES};
use lb_calibrate::trace::{coverage, top_k, Trace};
use lb_calibrate::tune::{best_config, bw_sweep, slots_for, TuneInput};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Topology,
    Tune {
        #[arg(long = "bench", required = true)]
        benches: Vec<PathBuf>,
        #[arg(long)]
        scrapes: PathBuf,
        /// Expert-pool budget per card (MB), same order as --bench; default: that card's free VRAM minus 1024 reserve.
        #[arg(long = "budget-mb")]
        budgets: Vec<f64>,
        #[arg(long, default_value_t = 40)]
        n_layer: usize,
        #[arg(long, default_value_t = 64)]
        n_expert: usize,
        /// Every k-th request is held out.
        #[arg(long, default_value_t = 4)]
        holdout_every: usize,
        #[arg(long, default_value = "LFM2.5-24B-A2B-QAD-Q4_0")]
        model: String,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Add a hypothetical card: NAME:COMPUTE_SCALE:H2D_MB_S:VRAM_MB (compute relative to the first bench card).
        #[arg(long = "hypothetical")]
        hypothetical: Vec<String>,
        #[arg(long, default_value_t = 12)]
        top: usize,
    },
    Sweep {
        #[arg(long = "bench", required = true)]
        benches: Vec<PathBuf>,
        #[arg(long)]
        scrapes: PathBuf,
        #[arg(long = "budget-mb")]
        budgets: Vec<f64>,
        #[arg(long, default_value_t = 24)]
        k: usize,
        #[arg(long, default_value_t = 1)]
        cards: usize,
        #[arg(long, default_value_t = 40)]
        n_layer: usize,
        #[arg(long, default_value_t = 64)]
        n_expert: usize,
        #[arg(long, default_value_t = 4)]
        holdout_every: usize,
        #[arg(long = "hypothetical")]
        hypothetical: Vec<String>,
    },
}

fn load_all(benches: &[PathBuf], hyps: &[String]) -> Result<(Vec<CardTimings>, HostTimings)> {
    let mut cards = Vec::new();
    let mut host = None;
    for b in benches {
        let (c, h) = load_bench(b)?;
        cards.push(c);
        host.get_or_insert(h);
    }
    for h in hyps {
        let f: Vec<&str> = h.split(':').collect();
        anyhow::ensure!(f.len() == 4, "--hypothetical NAME:COMPUTE_SCALE:H2D_MB_S:VRAM_MB");
        cards.push(CardTimings::hypothetical(f[0], &cards[0], f[1].parse()?, f[2].parse()?, f[3].parse()?));
    }
    Ok((cards, host.context("at least one --bench")?))
}

fn budgets_for(cards: &[CardTimings], given: &[f64]) -> Vec<f64> {
    cards.iter().enumerate().map(|(i, c)| given.get(i).copied().unwrap_or((c.vram_free_mb - 1024.0).max(0.0))).collect()
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Topology => {
            for c in lb_calibrate::topology::probe() {
                println!("{}", serde_json::to_string(&c)?);
            }
        }
        Cmd::Tune { benches, scrapes, budgets, n_layer, n_expert, holdout_every, model, out, hypothetical, top } => {
            let (cards, host) = load_all(&benches, &hypothetical)?;
            let budgets_mb = budgets_for(&cards, &budgets);
            let trace = Trace::load_dir(&scrapes, n_layer, n_expert)?;
            let (train, test) = trace.split(holdout_every);
            eprintln!("trace: {} requests / {} tokens; train {} tokens, held out {} tokens", trace.requests.len(), trace.n_tokens(), train.n_tokens(), test.n_tokens());
            for (c, b) in cards.iter().zip(&budgets_mb) {
                eprintln!("card {}: expert {:.3} ms, h2d {:.0} MB/s, budget {b:.0} MB = {} slots{}", c.name, c.expert_ms, c.h2d_mb_s, slots_for(*b), if c.measured { "" } else { " (hypothetical)" });
            }
            eprintln!("host: cpu expert {:.3} ms, head {:.1} ms", host.cpu_expert_ms, host.cpu_head_ms);
            let inp = TuneInput { cards, budgets_mb, host, train: &train, test: &test, ks: vec![8, 12, 16, 24, 32, 48, 64], lookaheads: vec![1, 2], dense_options: vec![true, false] };
            let (best, cands) = best_config(&inp, &model).context("no feasible configuration")?;
            println!("{:>5} {:>3} {:>3} {:>8} {:>5} {:>5} {:>8} {:>7} {:>7} {:>8} {:>8}", "cards", "k", "la", "miss", "dyn", "dense", "tok/s", "hit%", "stall%", "cpu ms", "stage MB");
            for c in cands.iter().take(top) {
                println!(
                    "{:>5} {:>3} {:>3} {:>8} {:>5} {:>5} {:>8.2} {:>7.1} {:>7.1} {:>8.1} {:>8.1}",
                    c.n_cards, c.k, c.lookahead, format!("{:?}", c.miss_policy), c.dynamic_admit, if c.dense_on_gpu { "gpu" } else { "cpu" },
                    c.result.tok_s, 100.0 * c.result.hit_rate, c.result.stall_pct, c.result.cpu_ms_per_token, c.result.staged_mb_per_token
                );
            }
            println!("\nbest: {} card(s), k={} per layer, lookahead {}, {:?} misses, dynamic admit {}, dense on {} -> {:.2} tok/s predicted ({:.1}% hits, {:.1}% stall); held-out coverage of the staged sets {:.1}%",
                best.cards.len(), best.k_per_layer, best.lookahead, best.miss_policy, best.dynamic_admit, if best.dense_on_gpu { "gpu" } else { "cpu" },
                best.predicted.tok_s, 100.0 * best.predicted.hit_rate, best.predicted.stall_pct, 100.0 * best.held_out_coverage);
            if let Some(p) = out {
                std::fs::write(&p, serde_json::to_string_pretty(&best)?)?;
                eprintln!("wrote {}", p.display());
            }
        }
        Cmd::Sweep { benches, scrapes, budgets, k, cards: n, n_layer, n_expert, holdout_every, hypothetical } => {
            let (cards, host) = load_all(&benches, &hypothetical)?;
            let budgets_mb = budgets_for(&cards, &budgets);
            let trace = Trace::load_dir(&scrapes, n_layer, n_expert)?;
            let (train, test) = trace.split(holdout_every);
            let hist = train.histogram();
            println!("held-out coverage by k (top-k of the train histogram):");
            for kk in [8, 12, 16, 24, 32, 48, 64] {
                println!("  k={kk:>2}: {:.1}%  ({:.0} MB per layer, {:.1} GB for 38 layers)", 100.0 * coverage(&test, &top_k(&hist, kk)), kk as f64 * EXPERT_BYTES as f64 / 1e6, 38.0 * kk as f64 * EXPERT_BYTES as f64 / 1e9);
            }
            let n = n.min(cards.len()).max(1);
            let mut cfg = SimConfig::single_card(cards[0].clone(), host, slots_for(budgets_mb[0]), k);
            cfg.cards = cards[..n].to_vec();
            cfg.slots = budgets_mb[..n].iter().map(|&b| slots_for(b)).collect();
            let staged = top_k(&hist, k);
            for mp in [MissPolicy::Cpu, MissPolicy::Best] {
                cfg.miss_policy = mp;
                println!("\n{n} card(s), k={k}, misses {mp:?}: bandwidth sensitivity (x measured h2d)");
                println!("{:>6} {:>8} {:>7} {:>7} {:>9} {:>9}", "bw x", "tok/s", "hit%", "stall%", "stage MB", "od MB");
                for (s, r) in bw_sweep(&cfg, &test, &staged, &[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]) {
                    println!("{s:>6.2} {:>8.2} {:>7.1} {:>7.1} {:>9.1} {:>9.1}", r.tok_s, 100.0 * r.hit_rate, r.stall_pct, r.staged_mb_per_token, r.ondemand_mb_per_token);
                }
            }
            cfg.miss_policy = MissPolicy::Cpu;
            let r = simulate(&cfg, &test, &staged);
            println!("\nbreakdown per token at bw x1 (Cpu misses): dense {:.1} ms, gpu experts {:.1} ms, cpu experts {:.1} ms, handoff {:.2} ms, stall {:.1} ms, total {:.1} ms; static cards {:?}", r.dense_ms_per_token, r.gpu_ms_per_token, r.cpu_ms_per_token, r.handoff_ms_per_token, r.stall_ms_per_token, r.ms_per_token, r.static_cards);
        }
    }
    Ok(())
}
