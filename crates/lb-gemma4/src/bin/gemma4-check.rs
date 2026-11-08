//! Greedy parity of lb-gemma4 against a llama.cpp server on the same GGUF (the oracle).
//!
//! Per prompt: tokenize through the oracle (`/tokenize`, BOS added; `--chat` prompts are first
//! rendered by the oracle's own chat template, `/apply-template`, thinking off), ask the oracle
//! for `--n` greedy tokens with the top-5 probabilities of every step (`/completion`,
//! `n_probs 5`, `post_sampling_probs false` = softmax of the raw logits). Then run our engine
//! teacher-forced on the same tokens: prefill the prompt, then one decode pass per oracle token.
//! At each step compare our argmax with the oracle's token, our top-5 set with its top-5 set,
//! and `|p_ours - p_oracle|` over the oracle's top-5. Our argmax matching at every step means
//! our free-running greedy text is the oracle's.
//!
//! `--trace-dir`: also write our named last-row intermediates of the prompt's prefill
//! (`trace-<i>.json`) for comparison with llama.cpp's eval callback.

use anyhow::{Context, Result};
use clap::Parser;
use lb_gemma4::Gemma4;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "http://127.0.0.1:18193")]
    oracle: String,
    /// Raw prompt (BOS + text); repeatable.
    #[arg(long = "prompt")]
    prompts: Vec<String>,
    /// User message rendered through the chat template (system line from `--system`); repeatable.
    #[arg(long = "chat")]
    chats: Vec<String>,
    #[arg(
        long,
        default_value = "Answer directly. Do not write out your reasoning or analysis; give only the final answer in the requested format."
    )]
    system: String,
    #[arg(long, default_value_t = 16)]
    n: usize,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long)]
    trace_dir: Option<PathBuf>,
    /// Prefill one token per pass (decode path only) instead of one batched pass.
    #[arg(long, default_value_t = false)]
    serial_prefill: bool,
}

fn post(base: &str, path: &str, body: Value) -> Result<Value> {
    Ok(ureq::post(&format!("{base}{path}"))
        .timeout(std::time::Duration::from_secs(3600))
        .send_json(body)?
        .into_json()?)
}

fn tokenize(base: &str, text: &str) -> Result<Vec<u32>> {
    let r = post(
        base,
        "/tokenize",
        json!({"content": text, "add_special": true}),
    )?;
    r["tokens"]
        .as_array()
        .context("tokens")?
        .iter()
        .map(|v| v.as_u64().map(|x| x as u32).context("id"))
        .collect()
}

/// (token, top-5 [(id, p)]) per step.
fn oracle_greedy(base: &str, ids: &[u32], n: usize) -> Result<Vec<(u32, Vec<(u32, f32)>)>> {
    let r = post(
        base,
        "/completion",
        json!({"prompt": ids, "n_predict": n, "temperature": 0.0, "n_probs": 5, "post_sampling_probs": false, "cache_prompt": false}),
    )?;
    let steps = r["completion_probabilities"]
        .as_array()
        .context("completion_probabilities")?;
    steps
        .iter()
        .map(|s| {
            let id = s["id"].as_u64().context("id")? as u32;
            let top = s["top_logprobs"]
                .as_array()
                .context("top_logprobs")?
                .iter()
                .map(|t| {
                    Ok((
                        t["id"].as_u64().context("id")? as u32,
                        (t["logprob"].as_f64().context("logprob")?).exp() as f32,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((id, top))
        })
        .collect()
}

fn softmax(x: &[f32]) -> Vec<f32> {
    lb_gemma4::moe::softmax(x)
}

fn top5(p: &[f32]) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.select_nth_unstable_by(5, |&a, &b| p[b].partial_cmp(&p[a]).unwrap());
    let mut t: Vec<(u32, f32)> = idx[..5].iter().map(|&i| (i as u32, p[i])).collect();
    t.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    t
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t = Instant::now();
    let model = Gemma4::load(&args.model)?;
    let c = model.cfg();
    eprintln!(
        "loaded {} in {:.1}s: {} layers, {} experts top-{}, swa layers {:?}",
        args.model.display(),
        t.elapsed().as_secs_f64(),
        c.n_layer,
        c.n_expert,
        c.n_expert_used,
        (0..c.n_layer)
            .filter(|&l| !c.swa[l])
            .map(|l| format!("{l}=global"))
            .collect::<Vec<_>>()
    );
    let mut cases: Vec<(String, String)> = args
        .prompts
        .iter()
        .map(|p| ("raw".to_string(), p.clone()))
        .collect();
    for u in &args.chats {
        let r = post(
            &args.oracle,
            "/apply-template",
            json!({"messages": [{"role": "system", "content": args.system}, {"role": "user", "content": u}], "chat_template_kwargs": {"enable_thinking": false}}),
        )?;
        cases.push((
            "chat".to_string(),
            r["prompt"].as_str().context("prompt")?.to_string(),
        ));
    }
    let mut reports = Vec::new();
    let mut all_ok = true;
    for (ci, (kind, text)) in cases.iter().enumerate() {
        let ids = tokenize(&args.oracle, text)?;
        let t0 = Instant::now();
        let oracle = oracle_greedy(&args.oracle, &ids, args.n)?;
        let oracle_s = t0.elapsed().as_secs_f64();
        let mut cache = model.new_cache();
        if args.trace_dir.is_some() {
            model.set_trace(true);
        }
        let t1 = Instant::now();
        let mut logits = if args.serial_prefill {
            let mut l = Vec::new();
            for &t in &ids {
                l = model.forward(&[t], &mut cache, false)?;
            }
            l
        } else {
            model.forward(&ids, &mut cache, false)?
        };
        let prefill_s = t1.elapsed().as_secs_f64();
        if let Some(dir) = &args.trace_dir {
            std::fs::create_dir_all(dir)?;
            let tr: Vec<Value> = model
                .take_trace()
                .into_iter()
                .map(|t| json!({"name": t.name, "layer": t.layer, "data": t.data}))
                .collect();
            std::fs::write(
                dir.join(format!("trace-{ci}.json")),
                serde_json::to_string(&json!({"ids": ids, "trace": tr}))?,
            )?;
            model.set_trace(false);
        }
        println!(
            "\n== case {ci} ({kind}, {} prompt tokens): {:?}",
            ids.len(),
            text.chars().take(80).collect::<String>()
        );
        println!(
            "{:>4} {:>8} {:>8} {:>5} {:>6} {:>9}  oracle top-5 / ours top-5",
            "step", "oracle", "ours", "same", "top5=", "max|dp|"
        );
        let (mut same_run, mut first_diff, mut worst_dp, mut top5_all) = (0usize, None, 0f32, true);
        let mut steps = Vec::new();
        let mut decode_s = 0.0;
        for (s, (tok, otop)) in oracle.iter().enumerate() {
            let p = softmax(&logits);
            let ours = Gemma4::argmax(&logits);
            let mine = top5(&p);
            let oset: std::collections::BTreeSet<u32> = otop.iter().map(|x| x.0).collect();
            let mset: std::collections::BTreeSet<u32> = mine.iter().map(|x| x.0).collect();
            let dp = otop
                .iter()
                .map(|(id, po)| (p[*id as usize] - po).abs())
                .fold(0f32, f32::max);
            let same = ours == *tok;
            if same && first_diff.is_none() {
                same_run += 1;
            } else if first_diff.is_none() {
                first_diff = Some(s);
            }
            worst_dp = worst_dp.max(dp);
            top5_all &= oset == mset;
            println!(
                "{:>4} {:>8} {:>8} {:>5} {:>6} {:>9.4}  {:?} / {:?}",
                s,
                tok,
                ours,
                if same { "yes" } else { "NO" },
                oset == mset,
                dp,
                otop.iter()
                    .map(|(i, p)| format!("{i}:{p:.3}"))
                    .collect::<Vec<_>>(),
                mine.iter()
                    .map(|(i, p)| format!("{i}:{p:.3}"))
                    .collect::<Vec<_>>()
            );
            steps.push(json!({"step": s, "oracle": tok, "ours": ours, "top5_match": oset == mset, "max_dp": dp,
                "oracle_top5": otop, "ours_top5": mine}));
            if s + 1 < oracle.len() {
                let td = Instant::now();
                logits = model.forward(&[*tok], &mut cache, false)?;
                decode_s += td.elapsed().as_secs_f64();
            }
        }
        let ok = same_run >= 8.min(oracle.len()) && top5_all && worst_dp <= 0.03;
        all_ok &= ok;
        println!(
            "case {ci}: identical greedy steps {same_run}/{} (first diff {:?}), top-5 sets all match {top5_all}, max|dp| {worst_dp:.4} -> {}; ours prefill {prefill_s:.1}s, decode {:.2} tok/s [measured, t440 CPU, shared host]; oracle {oracle_s:.1}s",
            oracle.len(),
            first_diff,
            if ok { "PASS" } else { "FAIL" },
            (oracle.len().saturating_sub(1)) as f64 / decode_s.max(1e-9)
        );
        reports.push(json!({"case": ci, "kind": kind, "prompt": text, "ids": ids, "identical_steps": same_run, "first_diff": first_diff,
            "top5_all_match": top5_all, "max_dp": worst_dp, "pass": ok, "prefill_s": prefill_s, "decode_s": decode_s, "steps": steps}));
    }
    println!("\nOVERALL {}", if all_ok { "PASS" } else { "FAIL" });
    if let Some(p) = &args.report {
        std::fs::write(
            p,
            serde_json::to_string_pretty(
                &json!({"model": args.model, "oracle": args.oracle, "n": args.n, "pass": all_ok, "cases": reports}),
            )?,
        )?;
    }
    Ok(())
}
