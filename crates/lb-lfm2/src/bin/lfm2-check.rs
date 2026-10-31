//! Teacher-forced check of the LFM2 forward pass against a llama.cpp server used as the
//! reference oracle (same GGUF). For each prompt: tokenize through the oracle
//! (`/tokenize`), run OUR forward over the whole token sequence once, and for every
//! prefix ask the oracle for its greedy next token (`/completion`, n_predict 1,
//! temperature 0). Prints one line per position: ours, oracle, the real next token,
//! and whether ours == oracle; then the aggregate.
//!
//! Example: `lfm2-check --model m.gguf --oracle http://127.0.0.1:18080 --prompt "..."`.

use anyhow::{Context, Result};
use burn::tensor::Device;
use clap::Parser;
use lb_lfm2::model::Lfm2;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Base URL of a llama-server serving the same GGUF (the reference oracle).
    #[arg(long)]
    oracle: String,
    /// Prompt text; repeatable.
    #[arg(long = "prompt")]
    prompts: Vec<String>,
    /// Prepend the BOS token (LFM2's chat template starts with it).
    #[arg(long, default_value_t = true)]
    bos: bool,
    /// Also greedily generate this many tokens from our engine after each prompt
    /// (exercises the decode path with the conv/KV caches) and print their ids.
    #[arg(long, default_value_t = 0)]
    generate: usize,
    /// Write a JSON report here.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Directory of NautiBurn scrape files (`run-*.json`) to build the routing histogram
    /// from; with `--gpu-budget-mb` this plans the expert placement (shadow only in step 1).
    #[arg(long)]
    placement_from_scrapes: Option<PathBuf>,
    /// GPU byte budget for resident experts, in MB.
    #[arg(long, default_value_t = 0)]
    gpu_budget_mb: usize,
}

fn read_scrapes(dir: &std::path::Path) -> Result<Vec<lb_lfm2::moe::Scrape>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("run-") && name.ends_with(".json") {
            if let Ok(s) = serde_json::from_str::<lb_lfm2::moe::Scrape>(&std::fs::read_to_string(&p)?) {
                out.push(s);
            }
        }
    }
    Ok(out)
}

fn oracle_tokenize(base: &str, text: &str) -> Result<Vec<u32>> {
    let r: Value = ureq::post(&format!("{base}/tokenize"))
        .send_json(json!({ "content": text, "add_special": false, "with_pieces": false }))?
        .into_json()?;
    r["tokens"].as_array().context("tokens")?.iter().map(|v| v.as_u64().map(|x| x as u32).context("id")).collect()
}

fn oracle_detokenize(base: &str, ids: &[u32]) -> Result<String> {
    let r: Value = ureq::post(&format!("{base}/detokenize")).send_json(json!({ "tokens": ids }))?.into_json()?;
    Ok(r["content"].as_str().unwrap_or("").to_string())
}

/// Greedy next token of the oracle for a token prefix.
fn oracle_next(base: &str, prefix: &[u32]) -> Result<u32> {
    let r: Value = ureq::post(&format!("{base}/completion"))
        .send_json(json!({
            "prompt": prefix, "n_predict": 1, "temperature": 0.0, "top_k": 1, "samplers": ["top_k"],
            "cache_prompt": true, "return_tokens": true, "n_probs": 1
        }))?
        .into_json()?;
    if let Some(t) = r["tokens"].as_array().and_then(|a| a.first()).and_then(|v| v.as_u64()) {
        return Ok(t as u32);
    }
    // older servers: completion_probabilities[0].top_probs[0].id
    r["completion_probabilities"][0]["top_probs"][0]["id"].as_u64().map(|x| x as u32).context("oracle gave no token")
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::default();
    let t0 = Instant::now();
    let mut model = Lfm2::load(&args.model, &device)?;
    let cfg = model.cfg().clone();
    if let Some(dir) = &args.placement_from_scrapes {
        use lb_lfm2::moe::{Histogram, Placement};
        let scrapes = read_scrapes(dir)?;
        let hist = Histogram::from_scrapes("lfm2.5-24b-a2b", cfg.n_layer, cfg.n_expert, &scrapes);
        let bytes_per_expert: Vec<usize> = (0..cfg.n_layer)
            .map(|l| match &model.w.layers[l].ffn {
                lb_lfm2::weights::Ffn::Moe { gate, up, down, .. } => gate.per_expert + up.per_expert + down.per_expert,
                _ => 0,
            })
            .collect();
        model.placement = Placement::plan(&hist, &bytes_per_expert, args.gpu_budget_mb * 1_000_000);
        let seen: u64 = hist.counts.iter().flatten().sum();
        eprintln!(
            "placement from {} scrape files ({seen} expert activations): {} of {} experts GPU-resident in {:.0} MB (budget {} MB)",
            scrapes.len(),
            model.placement.n_resident(),
            cfg.n_layer.saturating_sub(cfg.n_dense_lead) * cfg.n_expert,
            model.placement.resident_bytes as f64 / 1e6,
            args.gpu_budget_mb
        );
    }
    eprintln!(
        "loaded {} in {:.1}s: {} layers, {} experts x top-{}, attn layers {:?}",
        args.model.display(),
        t0.elapsed().as_secs_f32(),
        cfg.n_layer,
        cfg.n_expert,
        cfg.n_expert_used,
        (0..cfg.n_layer).filter(|&l| cfg.is_attn(l)).collect::<Vec<_>>()
    );

    let mut report = Vec::new();
    let (mut agree, mut agree_truth, mut oracle_truth, mut total) = (0usize, 0usize, 0usize, 0usize);
    for prompt in &args.prompts {
        let mut ids = oracle_tokenize(&args.oracle, prompt)?;
        if args.bos {
            ids.insert(0, cfg.bos);
        }
        let n = ids.len();
        let t = Instant::now();
        let mut cache = model.new_cache();
        let logits = model.forward(&ids, &mut cache, true);
        let ours = Lfm2::argmax_rows(&logits);
        let dt = t.elapsed().as_secs_f32();
        let mm = model.matmul_ns.swap(0, std::sync::atomic::Ordering::Relaxed) as f32 / 1e9;
        eprintln!("prompt {n} tokens: our forward {dt:.1}s ({mm:.1}s in quantized matmuls)");
        if let Some(p) = model.scraper.flush("prefill", &ids)? {
            eprintln!("scrape written: {}", p.display());
        }

        println!("\n== {:?}", prompt);
        println!("{:>4} {:>8} {:>8} {:>8}  {:<4} {}", "pos", "ours", "oracle", "truth", "ok", "oracle-piece");
        let mut rows = Vec::new();
        for i in 0..n {
            let orc = oracle_next(&args.oracle, &ids[..=i])?;
            let truth = ids.get(i + 1).copied();
            let ok = ours[i] == orc;
            total += 1;
            agree += ok as usize;
            if let Some(tr) = truth {
                agree_truth += (ours[i] == tr) as usize;
                oracle_truth += (orc == tr) as usize;
            }
            let piece = oracle_detokenize(&args.oracle, &[orc]).unwrap_or_default();
            println!(
                "{:>4} {:>8} {:>8} {:>8}  {:<4} {:?}",
                i,
                ours[i],
                orc,
                truth.map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
                if ok { "yes" } else { "NO" },
                piece
            );
            rows.push(json!({ "pos": i, "ours": ours[i], "oracle": orc, "truth": truth, "agree": ok }));
        }
        let mut gen = Vec::new();
        if args.generate > 0 {
            let mut next = ours[n - 1];
            for _ in 0..args.generate {
                gen.push(next);
                let l = model.forward(&[next], &mut cache, false);
                next = Lfm2::argmax_rows(&l)[0];
                if next == cfg.eos {
                    break;
                }
            }
            println!("ours greedy: {:?}", oracle_detokenize(&args.oracle, &gen).unwrap_or_default());
        }
        report.push(json!({ "prompt": prompt, "tokens": ids, "forward_s": dt, "positions": rows, "generated": gen }));
    }
    println!(
        "\nagreement ours==oracle: {agree}/{total} ({:.1}%); ours==truth {agree_truth}, oracle==truth {oracle_truth} (of {} with a next token)",
        100.0 * agree as f32 / total.max(1) as f32,
        total - args.prompts.len()
    );
    println!(
        "placement shadow: {} expert calls would hit GPU residents, {} CPU (all executed on CPU in step 1)",
        model.stats.gpu_hits.load(std::sync::atomic::Ordering::Relaxed),
        model.stats.cpu_calls.load(std::sync::atomic::Ordering::Relaxed)
    );
    if let Some(p) = &args.report {
        std::fs::write(p, serde_json::to_string_pretty(&json!({ "model": args.model, "agree": agree, "total": total, "prompts": report }))?)?;
    }
    Ok(())
}
