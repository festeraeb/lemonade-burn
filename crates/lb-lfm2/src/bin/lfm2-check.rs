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
    /// Base URL of a llama-server serving the same GGUF (the reference oracle). Not needed
    /// with `--ab-device` and `--tokenizer`.
    #[arg(long)]
    oracle: Option<String>,
    /// Tokenize locally with this `tokenizer.json` instead of through the oracle.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// A/B the device-resident decode against the old path in this one process, same
    /// residency for both arms: every prompt token by token (decode passes, teacher-forced),
    /// top-1 and max |logit| difference per position and the routes each arm chose; then a
    /// greedy completion of `--greedy-n` tokens per arm from `--greedy-prompt`.
    #[arg(long, default_value_t = false)]
    ab_device: bool,
    /// A/B the device-resident prefill (`device_prefill.rs`) against the host prompt pass, same
    /// process, same residency: every `--prompt` in one pass per arm with all logits (top-1 and
    /// max |logit| difference per position), then the greedy continuation of `--greedy-n` tokens
    /// from each arm's prefilled state (device decode in both), then last-row prefill timings
    /// over `--prefill-lens` synthetic prompts (the greedy prompt repeated to each length).
    #[arg(long, default_value_t = false)]
    ab_prefill: bool,
    /// Prompt lengths (tokens) for the `--ab-prefill` timing rows.
    #[arg(long, value_delimiter = ',', default_value = "128,512,2048")]
    prefill_lens: Vec<usize>,
    /// X7a: compare the device decode in each listed Q4_0 math (`f32m,f16,dp4a`) against the f32
    /// device decode, same process, same upload (the math is a launch parameter): teacher-forced
    /// top-1 and KLD(f32 || math) per position over `--prompt`s, then a greedy completion of
    /// `--greedy-n` tokens from `--greedy-prompt` per math (sha + tok/s), `--ab-rounds` times.
    #[arg(long, value_delimiter = ',')]
    ab_math: Vec<String>,
    #[arg(long, default_value_t = 1)]
    ab_rounds: usize,
    /// Prompt for the greedy part of `--ab-device` (c4_e2e.sh's by default).
    #[arg(long, default_value = "Explain in two sentences why a sonar return from a hard sand bottom differs from one over silt.")]
    greedy_prompt: String,
    #[arg(long, default_value_t = 32)]
    greedy_n: usize,
    /// Plan the expert placement as whole MoE layers in layer order within `--gpu-budget-mb`
    /// (`Placement::whole_layers`), instead of from scrapes.
    #[arg(long, default_value_t = false)]
    whole_layers: bool,
    /// With `--gpu`: device budget (MB) for the dense weights and the head (`load_dense`).
    #[arg(long, default_value_t = 0)]
    gpu_dense_mb: usize,
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
    /// Work-to-weights: a placement file naming expert hosts (`lb_expert_host::placement`); the
    /// experts it places on hosts are computed there for every arm of the run.
    #[arg(long, env = "LB_W2W_PLACEMENT")]
    w2w_placement: Option<PathBuf>,
    /// Actually upload the planned residents to the GPU (LB_WGPU_DEVICE) and run them there
    /// (needs the `gpu` feature); without it the placement is only a counted shadow.
    #[arg(long, default_value_t = false)]
    gpu: bool,
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
    if args.whole_layers {
        let bytes_per_expert: Vec<usize> = (0..cfg.n_layer).map(|l| model.w.expert_bytes(l)).collect();
        model.placement = lb_lfm2::moe::Placement::whole_layers(cfg.n_expert, &bytes_per_expert, args.gpu_budget_mb * 1_000_000);
        let whole: Vec<usize> = (0..cfg.n_layer).filter(|&l| model.placement.layer_whole(l)).collect();
        eprintln!("placement: whole layers {:?} ({} experts, {:.0} MB)", whole, model.placement.n_resident(), model.placement.resident_bytes as f64 / 1e6);
        if args.gpu {
            let t = Instant::now();
            let n = model.load_residents().map_err(|e| anyhow::anyhow!(e))?;
            eprintln!("uploaded {n} experts ({:.0} MB, {} stacked layers) in {:.1}s", model.store.bytes as f64 / 1e6, stacked_layers(&model), t.elapsed().as_secs_f32());
        }
    }
    if let Some(dir) = &args.placement_from_scrapes {
        use lb_lfm2::moe::{Histogram, Placement};
        let scrapes = lb_lfm2::moe::read_scrape_dir(dir)?;
        let hist = Histogram::from_scrapes("lfm2.5-24b-a2b", cfg.n_layer, cfg.n_expert, &scrapes);
        let bytes_per_expert: Vec<usize> = (0..cfg.n_layer).map(|l| model.w.expert_bytes(l)).collect();
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
        if args.gpu {
            let t = Instant::now();
            let n = model.load_residents().map_err(|e| anyhow::anyhow!(e))?;
            eprintln!("uploaded {n} experts ({:.0} MB) to the GPU in {:.1}s", model.store.bytes as f64 / 1e6, t.elapsed().as_secs_f32());
        }
    }
    if args.gpu && args.gpu_dense_mb > 0 {
        #[cfg(feature = "gpu")]
        {
            let n = model.load_dense(args.gpu_dense_mb * 1_000_000).map_err(|e| anyhow::anyhow!(e))?;
            eprintln!("dense: {n} matrices resident ({:.0} MB)", model.dense.bytes as f64 / 1e6);
        }
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("--gpu-dense-mb needs the gpu feature");
    }
    #[cfg(feature = "gpu")]
    let in_box = match &args.w2w_placement {
        Some(p) => match model.attach_inbox(p).map_err(|e| anyhow::anyhow!("--w2w-placement {}: {e}", p.display()))? {
            Some(st) => {
                eprintln!("w2w in-box: {st}");
                true
            }
            None => false,
        },
        None => false,
    };
    #[cfg(not(feature = "gpu"))]
    let in_box = false;
    let _w2w = match args.w2w_placement.as_ref().filter(|_| !in_box) {
        Some(p) => {
            let rt = model.attach_remote(p).map_err(|e| anyhow::anyhow!("--w2w-placement {}: {e}", p.display()))?;
            eprintln!("w2w: {} experts placed on {} host(s); {}", rt.n_placed(), rt.hosts().len(), rt.status());
            Some(rt)
        }
        None => None,
    };
    #[cfg(feature = "gpu")]
    if let Some(msg) = model.attach_prefill_card_from_env().map_err(|e| anyhow::anyhow!("prefill card: {e}"))? {
        eprintln!("{msg}");
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

    let codec = match &args.tokenizer {
        Some(p) => Some(tokenizers::Tokenizer::from_file(p).map_err(|e| anyhow::anyhow!("tokenizer {}: {e}", p.display()))?),
        None => None,
    };
    let tokenize = |text: &str| -> Result<Vec<u32>> {
        match (&codec, &args.oracle) {
            (Some(t), _) => Ok(t.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().to_vec()),
            (None, Some(o)) => oracle_tokenize(o, text),
            (None, None) => anyhow::bail!("give --tokenizer or --oracle"),
        }
    };
    if args.ab_device {
        return ab_device(&args, &model, &tokenize, codec.as_ref());
    }
    #[cfg(feature = "gpu")]
    if args.ab_prefill {
        return ab_prefill(&args, &model, &tokenize, codec.as_ref());
    }
    #[cfg(feature = "gpu")]
    if !args.ab_math.is_empty() {
        return ab_math(&args, &model, &tokenize, codec.as_ref());
    }
    let oracle = args.oracle.clone().context("--oracle is needed unless --ab-device")?;
    let oracle_url = oracle;

    let mut report = Vec::new();
    let (mut agree, mut agree_truth, mut oracle_truth, mut total) = (0usize, 0usize, 0usize, 0usize);
    for prompt in &args.prompts {
        let mut ids = tokenize(prompt)?;
        if args.bos {
            ids.insert(0, cfg.bos);
        }
        let n = ids.len();
        let t = Instant::now();
        let mut cache = model.new_cache();
        let logits = model.forward(&ids, &mut cache, true);
        let ours = Lfm2::argmax_rows(&logits);
        let dt = t.elapsed().as_secs_f32();
        let mm = model.timings.matmul_ns.swap(0, std::sync::atomic::Ordering::Relaxed) as f32 / 1e9;
        eprintln!("prompt {n} tokens: our forward {dt:.1}s ({mm:.1}s in quantized matmuls)");
        if let Some(p) = model.scraper.flush("prefill", &ids)? {
            eprintln!("scrape written: {}", p.display());
        }

        println!("\n== {:?}", prompt);
        println!("{:>4} {:>8} {:>8} {:>8}  {:<4} {}", "pos", "ours", "oracle", "truth", "ok", "oracle-piece");
        let mut rows = Vec::new();
        for i in 0..n {
            let orc = oracle_next(&oracle_url, &ids[..=i])?;
            let truth = ids.get(i + 1).copied();
            let ok = ours[i] == orc;
            total += 1;
            agree += ok as usize;
            if let Some(tr) = truth {
                agree_truth += (ours[i] == tr) as usize;
                oracle_truth += (orc == tr) as usize;
            }
            let piece = oracle_detokenize(&oracle_url, &[orc]).unwrap_or_default();
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
        let mut decode_s = 0f32;
        if args.generate > 0 {
            let mut next = ours[n - 1];
            let t = Instant::now();
            for _ in 0..args.generate {
                gen.push(next);
                let l = model.forward(&[next], &mut cache, false);
                next = Lfm2::argmax_rows(&l)[0];
                if next == cfg.eos {
                    break;
                }
            }
            decode_s = t.elapsed().as_secs_f32();
            println!(
                "ours greedy: {:?}  ({} decode steps in {decode_s:.1}s = {:.2} tok/s)",
                oracle_detokenize(&oracle_url, &gen).unwrap_or_default(),
                gen.len(),
                gen.len() as f32 / decode_s.max(1e-6)
            );
        }
        report.push(json!({ "prompt": prompt, "tokens": ids, "forward_s": dt, "decode_s": decode_s, "positions": rows, "generated": gen }));
    }
    println!(
        "\nagreement ours==oracle: {agree}/{total} ({:.1}%); ours==truth {agree_truth}, oracle==truth {oracle_truth} (of {} with a next token)",
        100.0 * agree as f32 / total.max(1) as f32,
        total - args.prompts.len()
    );
    let o = std::sync::atomic::Ordering::Relaxed;
    println!(
        "placement: {} expert calls on GPU residents ({:.1}s), {} on CPU ({:.1}s); {} experts resident{}",
        model.stats.gpu_hits.load(o),
        model.stats.gpu_ns.load(o) as f64 / 1e9,
        model.stats.cpu_calls.load(o),
        model.stats.cpu_ns.load(o) as f64 / 1e9,
        model.store.n_resident(),
        if args.gpu { " (executed on the GPU)" } else { " (shadow only: all executed on CPU)" }
    );
    if let Some(p) = &args.report {
        std::fs::write(p, serde_json::to_string_pretty(&json!({ "model": args.model, "agree": agree, "total": total, "prompts": report }))?)?;
    }
    if let Some(rt) = model.store.remote.as_ref() {
        println!("w2w: {}", rt.status());
    }
    Ok(())
}

fn stacked_layers(model: &Lfm2) -> usize {
    #[cfg(feature = "gpu")]
    {
        model.store.layers.len()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = model;
        0
    }
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (j, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = j;
        }
    }
    best as u32
}

fn logits_row(t: burn::tensor::Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

/// What one arm produced: per position logits' top-1, the full logits, and the routes.
struct Arm {
    top1: Vec<u32>,
    logits: Vec<Vec<f32>>,
    routes: Vec<(u32, u32, Vec<u32>)>,
    secs: f64,
    dd_tokens: u64,
    dd_fallbacks: u64,
}

/// Every token of `ids` as its own decode pass (teacher-forced). `device`: the knob for every
/// token; `alternate`: device on for even positions and off for odd ones, so every token
/// crosses between the two states (a host pass reads the device copy back, a device pass
/// uploads the host's).
fn run_arm(model: &Lfm2, ids: &[u32], device: bool, alternate: bool) -> Arm {
    use std::sync::atomic::Ordering::Relaxed;
    model.knobs().set_device_decode(device);
    let (t0, f0) = (model.timings.dd_tokens.load(Relaxed), model.timings.dd_fallbacks.load(Relaxed));
    let _ = model.scraper.take();
    let mut cache = model.new_cache();
    let t = Instant::now();
    let mut logits = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        if alternate {
            model.knobs().set_device_decode(i % 2 == 0);
        }
        logits.push(logits_row(model.forward(&[id], &mut cache, false)));
    }
    let secs = t.elapsed().as_secs_f64();
    let routes = model.scraper.take().into_iter().map(|a| (a.token_pos, a.layer, a.expert_ids)).collect();
    Arm {
        top1: logits.iter().map(|l| argmax(l)).collect(),
        logits,
        routes,
        secs,
        dd_tokens: model.timings.dd_tokens.load(Relaxed) - t0,
        dd_fallbacks: model.timings.dd_fallbacks.load(Relaxed) - f0,
    }
}

/// Prefill `ids` on the host path, then `n` greedy decode tokens with the device knob as given.
fn greedy_arm(model: &Lfm2, ids: &[u32], n: usize, device: bool, eos: u32) -> (Vec<u32>, Vec<Vec<f32>>, f64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    model.knobs().set_device_decode(device);
    let t0 = model.timings.dd_tokens.load(Relaxed);
    let mut cache = model.new_cache();
    let l = model.forward(ids, &mut cache, false);
    let mut next = argmax(&logits_row(l));
    let (mut out, mut rows) = (Vec::new(), Vec::new());
    let t = Instant::now();
    for _ in 0..n {
        out.push(next);
        if next == eos {
            break;
        }
        let row = logits_row(model.forward(&[next], &mut cache, false));
        next = argmax(&row);
        rows.push(row);
    }
    (out, rows, t.elapsed().as_secs_f64(), model.timings.dd_tokens.load(Relaxed) - t0)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn sha16(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(text.as_bytes());
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// `--ab-device`: the parity gates of `docs/DEVICE_DECODE_PLAN.md` D1, old path vs device path,
/// same process, same residency.
fn ab_device(args: &Args, model: &Lfm2, tokenize: &dyn Fn(&str) -> Result<Vec<u32>>, codec: Option<&tokenizers::Tokenizer>) -> Result<()> {
    let cfg = model.cfg();
    // the scraper records only when it has somewhere to write; nothing is flushed here
    if !model.scraper.enabled() {
        model.scraper.set_dir(Some(std::env::temp_dir().join("lfm2-check-ab-unused")));
    }
    let k = model.knobs();
    println!(
        "ab-device: dd_host_moe={} dd_router_seq={} (old arm = device_decode off, device arm = on)",
        k.dd_host_moe(),
        k.dd_router_seq()
    );
    let (mut positions, mut agree, mut route_diff) = (0usize, 0usize, 0usize);
    let mut worst = 0f32;
    let mut alt_total = (0usize, 0f32);
    let mut report = Vec::new();
    for prompt in &args.prompts {
        let mut ids = tokenize(prompt)?;
        if args.bos {
            ids.insert(0, cfg.bos);
        }
        let old = run_arm(model, &ids, false, false);
        let dev = run_arm(model, &ids, true, false);
        let alt = run_arm(model, &ids, true, true);
        let alt_agree = (0..ids.len()).filter(|&i| alt.top1[i] == old.top1[i]).count();
        let alt_worst = (0..ids.len()).map(|i| max_abs_diff(&alt.logits[i], &old.logits[i])).fold(0f32, f32::max);
        alt_total.0 += alt_agree;
        alt_total.1 = alt_total.1.max(alt_worst);
        println!("\n== {:?} ({} positions)", prompt, ids.len());
        println!(
            "old arm {:.2} tok/s ({} device tokens, {} fallbacks); device arm {:.2} tok/s ({} device tokens, {} fallbacks)",
            ids.len() as f64 / old.secs,
            old.dd_tokens,
            old.dd_fallbacks,
            ids.len() as f64 / dev.secs,
            dev.dd_tokens,
            dev.dd_fallbacks
        );
        println!("{:>4} {:>8} {:>8}  {:<4} {:>10}", "pos", "old", "device", "ok", "max|dlogit|");
        let mut rows = Vec::new();
        for i in 0..ids.len() {
            let d = max_abs_diff(&old.logits[i], &dev.logits[i]);
            worst = worst.max(d);
            let ok = old.top1[i] == dev.top1[i];
            agree += ok as usize;
            positions += 1;
            println!("{:>4} {:>8} {:>8}  {:<4} {:>10.3e}", i, old.top1[i], dev.top1[i], if ok { "yes" } else { "NO" }, d);
            rows.push(json!({"pos": i, "old": old.top1[i], "device": dev.top1[i], "max_abs_logit_diff": d}));
        }
        let rd = old.routes.iter().zip(&dev.routes).filter(|(a, b)| a != b).count() + old.routes.len().abs_diff(dev.routes.len());
        route_diff += rd;
        println!("routes: old {} activations, device {}, differing {rd}", old.routes.len(), dev.routes.len());
        println!("alternating device/old every token: top-1 == old {alt_agree}/{}, max |logit diff| {alt_worst:.3e} ({} device tokens)", ids.len(), alt.dd_tokens);
        report.push(json!({"prompt": prompt, "tokens": ids, "positions": rows, "route_activations": old.routes.len(), "route_diff": rd,
            "old_tok_s": ids.len() as f64 / old.secs, "device_tok_s": ids.len() as f64 / dev.secs, "device_tokens": dev.dd_tokens, "device_fallbacks": dev.dd_fallbacks}));
    }
    println!("\nteacher-forced top-1 old==device: {agree}/{positions}; max |logit diff| {worst:.3e}; route activations differing: {route_diff}");
    println!("alternating: top-1 == old {}/{positions}; max |logit diff| {:.3e}", alt_total.0, alt_total.1);

    let mut ids = tokenize(&args.greedy_prompt)?;
    if args.bos {
        ids.insert(0, cfg.bos);
    }
    let (go, gro, gso, _) = greedy_arm(model, &ids, args.greedy_n, false, cfg.eos);
    let (gd, grd, gsd, gdt) = greedy_arm(model, &ids, args.greedy_n, true, cfg.eos);
    let text = |v: &[u32]| codec.and_then(|c| c.decode(v, true).ok()).unwrap_or_else(|| format!("{v:?}"));
    let same = go == gd;
    let first_diff = go.iter().zip(&gd).position(|(a, b)| a != b);
    let step_worst: Vec<f32> = gro.iter().zip(&grd).map(|(a, b)| max_abs_diff(a, b)).collect();
    println!("\ngreedy {} tokens from {:?}", args.greedy_n, args.greedy_prompt);
    println!("old    {:.2} tok/s sha {} {:?}", go.len() as f64 / gso, sha16(&text(&go)), text(&go));
    println!("device {:.2} tok/s sha {} {:?} ({gdt} device tokens)", gd.len() as f64 / gsd, sha16(&text(&gd)), text(&gd));
    println!(
        "greedy token-for-token: {}{}; max |logit diff| per step (while the texts agree): {:.3e}",
        if same { "IDENTICAL" } else { "DIFFERENT" },
        first_diff.map(|i| format!(" (first difference at step {i})")).unwrap_or_default(),
        step_worst.iter().take(first_diff.unwrap_or(usize::MAX)).cloned().fold(0f32, f32::max)
    );
    model.knobs().set_device_decode(false);
    if let Some(p) = &args.report {
        std::fs::write(
            p,
            serde_json::to_string_pretty(&json!({"model": args.model, "agree": agree, "positions": positions, "max_abs_logit_diff": worst, "route_diff": route_diff,
                "prompts": report, "greedy": {"old": go, "device": gd, "identical": same, "first_diff": first_diff, "step_max_abs_logit_diff": step_worst}}))?,
        )?;
    }
    if let Some(rt) = model.store.remote.as_ref() {
        println!("w2w: {}", rt.status());
    }
    Ok(())
}

/// `--ab-prefill`: device prefill against the host prompt pass (see the flag's doc).
#[cfg(feature = "gpu")]
fn ab_prefill(args: &Args, model: &Lfm2, tokenize: &dyn Fn(&str) -> Result<Vec<u32>>, codec: Option<&tokenizers::Tokenizer>) -> Result<()> {
    let cfg = model.cfg();
    let k = model.knobs();
    let vocab = cfg.n_vocab;
    println!("ab-prefill: ubatch {} (LB_PREFILL_UBATCH); host arm = device_prefill off, device arm = on; decode device-resident in both", lb_lfm2::device_prefill::ubatch());
    let prefill = |ids: &[u32], device: bool, all: bool| -> (Vec<f32>, f64, lb_lfm2::model::Cache) {
        k.set_device_prefill(device);
        let mut cache = model.new_cache();
        let t = Instant::now();
        let lg = model.forward(ids, &mut cache, all);
        let v = lg.into_data().try_to_vec::<f32>().expect("f32 logits");
        (v, t.elapsed().as_secs_f64(), cache)
    };
    let argmax = |row: &[f32]| row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;
    let (mut positions, mut agree, mut worst) = (0usize, 0usize, 0f32);
    let mut report = Vec::new();
    for prompt in &args.prompts {
        let mut ids = tokenize(prompt)?;
        if args.bos {
            ids.insert(0, cfg.bos);
        }
        let dp0 = model.timings.dp_tokens.load(std::sync::atomic::Ordering::Relaxed);
        let (host, hs, _) = prefill(&ids, false, true);
        let (dev, ds, _) = prefill(&ids, true, true);
        let ran = model.timings.dp_tokens.load(std::sync::atomic::Ordering::Relaxed) - dp0;
        println!("\n== {:?} ({} positions; host {:.1} tok/s, device {:.1} tok/s, {} tokens took the device path)", prompt, ids.len(), ids.len() as f64 / hs, ids.len() as f64 / ds, ran);
        let mut rows = Vec::new();
        for i in 0..ids.len() {
            let (h, d) = (&host[i * vocab..(i + 1) * vocab], &dev[i * vocab..(i + 1) * vocab]);
            let diff = max_abs_diff(h, d);
            let (th, td) = (argmax(h), argmax(d));
            worst = worst.max(diff);
            agree += (th == td) as usize;
            positions += 1;
            if th != td || i + 1 == ids.len() || i < 3 {
                println!("{:>4} host {:>7} device {:>7} {:<3} max|dlogit| {:.3e}", i, th, td, if th == td { "ok" } else { "NO" }, diff);
            }
            rows.push(json!({"pos": i, "host": th, "device": td, "max_abs_logit_diff": diff}));
        }
        report.push(json!({"prompt": prompt, "n": ids.len(), "host_tok_s": ids.len() as f64 / hs, "device_tok_s": ids.len() as f64 / ds, "device_tokens": ran, "positions": rows}));
    }
    println!("\nteacher-forced top-1 host==device over the prompts: {agree}/{positions}; max |logit diff| {worst:.3e}");

    // the continuation from each arm's prefilled state: decode is device-resident in both arms
    let mut ids = tokenize(&args.greedy_prompt)?;
    if args.bos {
        ids.insert(0, cfg.bos);
    }
    k.set_device_decode(true);
    let greedy = |device: bool| -> (Vec<u32>, f64, f64) {
        let (last, ps, mut cache) = prefill(&ids, device, false);
        let mut tok = argmax(&last);
        let mut out = vec![tok];
        let t = Instant::now();
        while out.len() < args.greedy_n && tok != cfg.eos {
            let lg = model.forward(&[tok], &mut cache, false).into_data().try_to_vec::<f32>().expect("f32");
            tok = argmax(&lg);
            out.push(tok);
        }
        (out, ps, t.elapsed().as_secs_f64())
    };
    let (go, pso, dso) = greedy(false);
    let (gd, psd, dsd) = greedy(true);
    let text = |v: &[u32]| codec.and_then(|c| c.decode(v, true).ok()).unwrap_or_else(|| format!("{v:?}"));
    let same = go == gd;
    println!("\ngreedy {} tokens after a {}-token prefill of {:?}", args.greedy_n, ids.len(), args.greedy_prompt);
    println!("host prefill   {:>8.1} prompt tok/s, then {:.2} tok/s decode, sha {} {:?}", ids.len() as f64 / pso, go.len() as f64 / dso, sha16(&text(&go)), text(&go));
    println!("device prefill {:>8.1} prompt tok/s, then {:.2} tok/s decode, sha {} {:?}", ids.len() as f64 / psd, gd.len() as f64 / dsd, sha16(&text(&gd)), text(&gd));
    if let Some(p) = model.pcard.as_ref() {
        let st = &p.stats;
        println!("prefill card: {} chunks; its layers {:.3} s, hops {:.3} s, state copies {:.3} s", st.chunks.load(std::sync::atomic::Ordering::Relaxed),
            st.card_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, st.hop_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, st.state_copy_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    }
    println!("continuation token-for-token: {}", if same { "IDENTICAL".to_string() } else { format!("DIFFERENT (first at step {})", go.iter().zip(&gd).position(|(a, b)| a != b).unwrap_or(go.len().min(gd.len()))) });

    // last-row prefill timings at fixed lengths (the greedy prompt repeated), device arm, and the
    // host arm only up to 512 tokens (it is the slow path being replaced)
    let mut timing = Vec::new();
    for &n in &args.prefill_lens {
        let mut p: Vec<u32> = Vec::with_capacity(n);
        while p.len() < n {
            p.extend_from_slice(&ids[1..]);
        }
        p.truncate(n);
        p[0] = cfg.bos;
        let _ = prefill(&p, true, false); // warm
        let (dl, ds, _) = prefill(&p, true, false);
        let (hrate, top_ok) = if n <= 512 {
            let (hl, hs, _) = prefill(&p, false, false);
            (Some(n as f64 / hs), Some(argmax(&hl) == argmax(&dl)))
        } else {
            (None, None)
        };
        println!("prefill {n:>6} tokens: device {:>8.1} tok/s ({:.3} s){}", n as f64 / ds, ds, hrate.map(|r| format!("; host {r:.1} tok/s; last-row top-1 equal: {}", top_ok.unwrap())).unwrap_or_default());
        timing.push(json!({"n": n, "device_tok_s": n as f64 / ds, "device_s": ds, "host_tok_s": hrate, "top1_equal": top_ok}));
    }
    k.set_device_prefill(false);
    if let Some(p) = &args.report {
        std::fs::write(p, serde_json::to_string_pretty(&json!({"model": args.model, "ubatch": lb_lfm2::device_prefill::ubatch(), "agree": agree, "positions": positions,
            "max_abs_logit_diff": worst, "prompts": report, "greedy": {"host": go, "device": gd, "identical": same}, "timing": timing}))?)?;
    }
    Ok(())
}

/// KL(p || q) of the softmaxes of two logit rows (natural log), computed in f64.
fn kld(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p_logits), lse(q_logits));
    p_logits.iter().zip(q_logits).map(|(&a, &b)| {
        let la = a as f64 - lp;
        la.exp() * (la - (b as f64 - lq))
    }).sum::<f64>().max(0.0)
}

/// `--ab-math`: X7a model gates (docs/WORK_TO_WEIGHTS.md "## X7"). The f32 device decode is the
/// reference arm; each listed math runs on the same upload with `Gpu::set_q4_math`.
#[cfg(feature = "gpu")]
fn ab_math(args: &Args, model: &Lfm2, tokenize: &dyn Fn(&str) -> Result<Vec<u32>>, codec: Option<&tokenizers::Tokenizer>) -> Result<()> {
    use lb_q4_gpu::Q4Math;
    let cfg = model.cfg();
    let gpu = model.store.gpu.as_ref().context("--ab-math needs --gpu (a device)")?;
    let maths: Vec<Q4Math> = args.ab_math.iter().map(|s| Q4Math::parse(s).with_context(|| format!("unknown math {s:?}"))).collect::<Result<_>>()?;
    println!("ab-math: device {} cc {:?}, auto would pick {:?}; q4 rows {}; arms f32 + {:?}", gpu.name, gpu.compute_capability, Q4Math::for_card(gpu.compute_capability), gpu.q4_rows(), maths);
    let mut report = Vec::new();
    // teacher-forced: positions, top-1, KLD
    let mut tf: Vec<(usize, usize, f64, f64, f32)> = vec![(0, 0, 0.0, 0.0, 0.0); maths.len()];
    for prompt in &args.prompts {
        let mut ids = tokenize(prompt)?;
        if args.bos {
            ids.insert(0, cfg.bos);
        }
        gpu.set_q4_math(Q4Math::F32);
        let base = run_arm(model, &ids, true, false);
        for (mi, &m) in maths.iter().enumerate() {
            gpu.set_q4_math(m);
            let arm = run_arm(model, &ids, true, false);
            let mut kl_max = 0f64;
            for i in 0..ids.len() {
                let k = kld(&base.logits[i], &arm.logits[i]);
                tf[mi].0 += 1;
                tf[mi].1 += (base.top1[i] == arm.top1[i]) as usize;
                tf[mi].2 += k;
                kl_max = kl_max.max(k);
                tf[mi].4 = tf[mi].4.max(max_abs_diff(&base.logits[i], &arm.logits[i]));
            }
            tf[mi].3 = tf[mi].3.max(kl_max);
            println!("{:>4} {:?}: {} positions, top-1 so far {}/{}, prompt max KLD {kl_max:.3e}, fallbacks {}", m.name(), &prompt[..prompt.len().min(40)], ids.len(), tf[mi].1, tf[mi].0, arm.dd_fallbacks);
        }
    }
    gpu.set_q4_math(Q4Math::F32);
    for (mi, &m) in maths.iter().enumerate() {
        let (n, ok, ksum, kmax, dmax) = tf[mi];
        let mean = if n > 0 { ksum / n as f64 } else { 0.0 };
        println!("teacher-forced {:>4} vs f32: top-1 {ok}/{n}, mean KLD {mean:.3e}, max KLD {kmax:.3e}, max |dlogit| {dmax:.3e}", m.name());
        report.push(json!({"kind": "teacher_forced", "math": m.name(), "exact": m.exact(), "positions": n, "top1_agree": ok, "kld_mean": mean, "kld_max": kmax, "max_abs_logit_diff": dmax}));
    }
    // greedy, interleaved rounds: f32 then each math
    let mut ids = tokenize(&args.greedy_prompt)?;
    if args.bos {
        ids.insert(0, cfg.bos);
    }
    let text = |v: &[u32]| codec.and_then(|c| c.decode(v, true).ok()).unwrap_or_else(|| format!("{v:?}"));
    let mut arms = vec![Q4Math::F32];
    arms.extend(maths.iter().copied().filter(|&m| m != Q4Math::F32));
    for round in 0..args.ab_rounds {
        for &m in &arms {
            gpu.set_q4_math(m);
            let (g, _, secs, dd) = greedy_arm(model, &ids, args.greedy_n, true, cfg.eos);
            let sha = sha16(&text(&g));
            let tps = g.len() as f64 / secs;
            println!("greedy r{round} {:>4}: {tps:.2} tok/s sha {sha} ({dd} device tokens) {:?}", m.name(), text(&g));
            report.push(json!({"kind": "greedy", "round": round, "math": m.name(), "exact": m.exact(), "tokens": g.len(), "tok_s": tps, "sha": sha, "device_tokens": dd}));
        }
    }
    gpu.set_q4_math(Q4Math::F32);
    model.knobs().set_device_decode(false);
    if let Some(p) = &args.report {
        std::fs::write(p, serde_json::to_string_pretty(&json!({"model": args.model, "device": gpu.name, "cc": gpu.compute_capability, "rows": report}))?)?;
    }
    Ok(())
}
