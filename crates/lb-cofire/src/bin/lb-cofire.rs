//! `lb-cofire`: neighbouring-expert report (and co-fire placement) from scrape dirs or the
//! histogram DB. See the crate doc.
//!
//! ```text
//! lb-cofire --scrape-dir /data/lfm2-scrapes --out report.json
//! lb-cofire --db /mnt/fast/nautiburn/histogram/sidechannel.sqlite --class live-tiel \
//!           --max-requests 20000 --placement-cards 4 --placement-out tiel-4cards.json --out tiel.json
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use lb_cofire::{analyse, Corpus, Options, SplitMode};
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Co-firing, layer transitions, runner-up promotion and a co-fire card placement from MoE routing scrapes")]
struct Args {
    /// Scrape directory (run-*.json, schema 1 or 2), searched recursively; repeatable.
    #[arg(long = "scrape-dir")]
    scrape_dirs: Vec<PathBuf>,
    /// Histogram DB (opened read-only).
    #[arg(long)]
    db: Option<PathBuf>,
    /// Request class in the DB (e.g. live-tiel). Use live-* classes, not the sidechannel-* ones.
    #[arg(long)]
    class: Option<String>,
    /// Model id in the DB (default: the class's most frequent).
    #[arg(long)]
    model: Option<String>,
    /// Evenly spaced sample of at most this many DB requests (0 = all).
    #[arg(long, default_value_t = 20000)]
    max_requests: usize,
    /// Number of experts, when the scrapes alone may not show the highest id.
    #[arg(long, default_value_t = 0)]
    n_expert: usize,
    /// Held-out requests, one id or file name per line (overrides --split).
    #[arg(long)]
    test_requests: Option<PathBuf>,
    /// `time` (default): the newest --test-fraction of requests held out; `hash`: fnv1a(id) % 2.
    #[arg(long, default_value = "time")]
    split: String,
    #[arg(long, default_value_t = 0.3)]
    test_fraction: f64,
    #[arg(long, value_delimiter = ',', default_value = "4,8,16")]
    guesses: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
    cards: Vec<usize>,
    #[arg(long, default_value_t = 32)]
    top_pairs: usize,
    #[arg(long, default_value_t = 8)]
    top_next: usize,
    /// Chain single-token decode files in id order for runner-up promotion (one-slot servers only).
    #[arg(long, default_value_t = false)]
    chain_decode: bool,
    /// Write a co-fire placement for this many cards (lb-expert-host format).
    #[arg(long)]
    placement_cards: Option<usize>,
    #[arg(long)]
    placement_out: Option<PathBuf>,
    /// Report path (stdout when absent).
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let corpus = match (&a.db, a.scrape_dirs.is_empty()) {
        (Some(db), true) => {
            let class = a.class.as_deref().context("--db needs --class")?;
            Corpus::load_db(db, class, a.model.as_deref(), a.max_requests)?
        }
        (None, false) => {
            let (c, skipped) = Corpus::load_scrape_dirs(&a.scrape_dirs, a.n_expert)?;
            if skipped > 0 {
                eprintln!("lb-cofire: skipped {skipped} unreadable scrape files");
            }
            c
        }
        _ => bail!("give either --scrape-dir (repeatable) or --db with --class"),
    };
    eprintln!(
        "lb-cofire: {} requests, {} tokens, {} experts, top-{}, runners-up: {}",
        corpus.requests.len(),
        corpus.n_tokens(),
        corpus.n_expert,
        corpus.top_k,
        corpus.has_near()
    );
    let test_ids: Option<HashSet<String>> = match &a.test_requests {
        Some(p) => Some(std::fs::read_to_string(p)?.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()),
        None => None,
    };
    let opts = Options {
        guesses: a.guesses.clone(),
        cards: a.cards.clone(),
        top_pairs: a.top_pairs,
        top_next: a.top_next,
        chain_decode: a.chain_decode,
        placement_cards: a.placement_cards,
        split: match a.split.as_str() {
            "time" => SplitMode::Time(a.test_fraction),
            "hash" => SplitMode::Hash,
            other => bail!("--split {other:?}: expected time or hash"),
        },
    };
    let (mut report, placement) = analyse(&corpus, test_ids.as_ref(), &opts)?;
    if let (Some(p), Some(path)) = (placement, &a.placement_out) {
        std::fs::write(path, serde_json::to_string_pretty(&p)?)?;
        report["placement_file"] = serde_json::json!(path);
    }
    let text = serde_json::to_string_pretty(&report)?;
    match &a.out {
        Some(p) => std::fs::write(p, text)?,
        None => println!("{text}"),
    }
    // the headline numbers on stderr
    for r in report["next_layer_coverage"].as_array().into_iter().flatten() {
        eprintln!(
            "next layer, guess {:>2}: from this layer {:.1}%, from heat {:.1}%{}",
            r["guesses"],
            r["from_this_layer"].as_f64().unwrap_or(0.0) * 100.0,
            r["from_heat"].as_f64().unwrap_or(0.0) * 100.0,
            r["from_this_layer_plus_runners_up"].as_f64().map(|v| format!(", + runners-up {:.1}%", v * 100.0)).unwrap_or_default()
        );
    }
    for r in report["cards_touched_per_token_layer"].as_array().into_iter().flatten() {
        eprintln!(
            "{} cards: touched per token-layer by id {:.2}, by co-fire {:.2} (random {:.2})",
            r["cards"],
            r["by_expert_id"].as_f64().unwrap_or(0.0),
            r["by_cofire"].as_f64().unwrap_or(0.0),
            r["random_expected"].as_f64().unwrap_or(0.0)
        );
    }
    if let Some(p) = report["runner_up_promotion"].as_object() {
        eprintln!(
            "runner-up fires next token {:.1}% (fired fires again {:.1}%, other experts {:.2}%)",
            p["runner_up_fires_next_token"].as_f64().unwrap_or(0.0) * 100.0,
            p["fired_fires_again_next_token"].as_f64().unwrap_or(0.0) * 100.0,
            p["other_expert_fires_next_token"].as_f64().unwrap_or(0.0) * 100.0
        );
    }
    Ok(())
}
