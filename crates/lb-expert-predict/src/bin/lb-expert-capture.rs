//! Capture runner for the expert-prediction experiment (`docs/EXPERT_PREDICT.md`).
//!
//! For each prompt of a JSONL file (`{"id", "kind", "text"}`), one *chunk*: ChatML-wrap the
//! prompt, prefill it in one pass, greedily decode up to `--gen` tokens, and save every
//! layer's residual stream (`x_mid`, `x_out`) and the engine's routes for every token (see
//! `lb_expert_predict::capture`). Chunks already complete on disk are skipped, so the same
//! command resumes a run that died, on any node that sees the run directory.
//!
//! The capture only copies tensors out of the host forward pass; it does not change what the
//! model computes (`--check-sha` re-runs one prompt without capture and compares the tokens).

use anyhow::{Context, Result};
use burn::tensor::Device;
use clap::Parser;
use lb_expert_predict::capture::{ChunkMeta, RunDir};
use lb_lfm2::model::Lfm2;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    /// JSONL of `{"id", "kind", "text"}`.
    #[arg(long)]
    prompts: PathBuf,
    /// Run directory (states/, chunks/, progress.jsonl).
    #[arg(long)]
    out: PathBuf,
    /// Greedy decode tokens per prompt (fewer if the model stops).
    #[arg(long, default_value_t = 128)]
    gen: usize,
    /// Prompt text longer than this many tokens is cut.
    #[arg(long, default_value_t = 512)]
    max_prompt: usize,
    /// Stop after this many new chunks (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Upload this many MB of experts to the GPU (needs the `gpu` feature; 0 = CPU only).
    #[arg(long, default_value_t = 0)]
    gpu_expert_mb: usize,
    /// With `--gpu-expert-mb`: also make up to this many MB of dense weights resident.
    #[arg(long, default_value_t = 0)]
    gpu_dense_mb: usize,
    /// Before the run: decode the first prompt with and without capture and require the same
    /// tokens (proves the capture does not touch the forward pass).
    #[arg(long, default_value_t = false)]
    check_sha: bool,
}

#[derive(Deserialize)]
struct Prompt {
    id: String,
    kind: String,
    text: String,
}

struct Codec {
    tok: tokenizers::Tokenizer,
}

impl Codec {
    fn enc(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(s, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().to_vec())
    }
    /// `<bos><|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`, text cut to `max` tokens.
    fn chat(&self, bos: u32, text: &str, max: usize) -> Result<Vec<u32>> {
        let mut body = self.enc(text.trim())?;
        body.truncate(max);
        let mut ids = vec![bos];
        ids.extend(self.enc("<|im_start|>user\n")?);
        ids.extend(body);
        ids.extend(self.enc("<|im_end|>\n<|im_start|>assistant\n")?);
        Ok(ids)
    }
}

/// Prefill + greedy decode. Returns (fed tokens, n_prefill, token after the last fed one).
fn generate(model: &Lfm2, prompt: &[u32], gen: usize, stops: &[u32]) -> (Vec<u32>, usize, u32) {
    let mut cache = model.new_cache();
    let logits = model.forward(prompt, &mut cache, false);
    let mut next = Lfm2::argmax_rows(&logits)[0];
    let mut fed = prompt.to_vec();
    for _ in 0..gen {
        if stops.contains(&next) {
            break;
        }
        fed.push(next);
        let logits = model.forward(&[next], &mut cache, false);
        next = Lfm2::argmax_rows(&logits)[0];
    }
    (fed, prompt.len(), next)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::default();
    let t0 = Instant::now();
    #[allow(unused_mut)]
    let mut model = Lfm2::load(&args.model, &device)?;
    let cfg = model.cfg().clone();
    eprintln!(
        "loaded {} in {:.1}s: {} layers ({} dense lead), n_embd {}, {} experts top-{}, n_ff_exp {}",
        args.model.display(),
        t0.elapsed().as_secs_f32(),
        cfg.n_layer,
        cfg.n_dense_lead,
        cfg.n_embd,
        cfg.n_expert,
        cfg.n_expert_used,
        cfg.n_ff_exp
    );
    if args.gpu_expert_mb > 0 {
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("--gpu-expert-mb needs a build with the `gpu` feature");
        #[cfg(feature = "gpu")]
        {
            let bytes: Vec<usize> = (0..cfg.n_layer).map(|l| model.w.expert_bytes(l)).collect();
            model.placement = lb_lfm2::moe::Placement::whole_layers(cfg.n_expert, &bytes, args.gpu_expert_mb << 20);
            let n = model.load_residents().map_err(|e| anyhow::anyhow!(e))?;
            let d = if args.gpu_dense_mb > 0 { model.load_dense(args.gpu_dense_mb << 20).map_err(|e| anyhow::anyhow!(e))? } else { 0 };
            eprintln!("resident on the device: {n} experts, {d} dense matrices");
        }
    }
    let codec = Codec { tok: tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))? };
    let mut stops = vec![cfg.eos];
    if let Some(id) = codec.tok.token_to_id("<|im_end|>") {
        stops.push(id);
    }
    let prompts: Vec<Prompt> = std::fs::read_to_string(&args.prompts)
        .with_context(|| format!("read {}", args.prompts.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).with_context(|| format!("prompt line: {l}")))
        .collect::<Result<_>>()?;
    let run = RunDir::new(&args.out)?;
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string();
    let model_id = args.model.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();

    if args.check_sha {
        let p = &prompts[0];
        let ids = codec.chat(cfg.bos, &p.text, args.max_prompt)?;
        let n = args.gen.min(16);
        model.set_capture(false);
        let (a, _, an) = generate(&model, &ids, n, &stops);
        model.set_capture(true);
        let (b, _, bn) = generate(&model, &ids, n, &stops);
        model.set_capture(false);
        anyhow::ensure!(a == b && an == bn, "capture changed the greedy tokens: {a:?} vs {b:?}");
        eprintln!("check-sha: {} greedy tokens identical with and without capture", a.len() - ids.len() + 1);
    }

    let mut new_chunks = 0usize;
    for p in &prompts {
        if run.done(&p.id).is_some() {
            eprintln!("{}: done, skipping", p.id);
            continue;
        }
        if args.limit > 0 && new_chunks >= args.limit {
            break;
        }
        let t = Instant::now();
        let ids = codec.chat(cfg.bos, &p.text, args.max_prompt)?;
        model.set_capture(true);
        let (fed, n_prefill, last) = generate(&model, &ids, args.gen, &stops);
        let cap = model.take_capture();
        model.set_capture(false);

        // passes come in order: one prefill pass (m = n_prefill) then m = 1 decode passes,
        // each contributing one CapturedLayer per layer in layer order
        let (nl, e) = (cfg.n_layer, cfg.n_embd);
        let n_tok = fed.len();
        let mut states = vec![0f32; n_tok * nl * 2 * e];
        let mut routes = vec![vec![Vec::<u16>::new(); nl]; n_tok];
        let mut weights = vec![vec![Vec::<f32>::new(); nl]; n_tok];
        let mut seen = 0usize;
        for c in &cap {
            for r in 0..c.m {
                let t = c.pos0 + r;
                anyhow::ensure!(t < n_tok, "{}: captured position {t} beyond {n_tok} fed tokens", p.id);
                let base = (t * nl + c.layer) * 2 * e;
                states[base..base + e].copy_from_slice(&c.x_mid[r * e..(r + 1) * e]);
                states[base + e..base + 2 * e].copy_from_slice(&c.x_out[r * e..(r + 1) * e]);
                if let Some(rt) = c.routes.get(r) {
                    routes[t][c.layer] = rt.experts.iter().map(|&x| x as u16).collect();
                    weights[t][c.layer] = rt.weights.clone();
                }
                seen += 1;
            }
        }
        anyhow::ensure!(seen == n_tok * nl, "{}: captured {seen} token-layers, expected {}", p.id, n_tok * nl);
        let text = codec.tok.decode(&fed[n_prefill..], true).unwrap_or_default();
        let meta = ChunkMeta {
            id: p.id.clone(),
            kind: p.kind.clone(),
            model: model_id.clone(),
            n_layer: nl,
            n_embd: e,
            n_expert: cfg.n_expert,
            k: cfg.n_expert_used,
            tokens: fed,
            n_prefill,
            routes,
            weights,
            last_token: last,
            text_head: text.chars().take(240).collect(),
            host: host.clone(),
            secs: t.elapsed().as_secs_f64(),
        };
        run.write_chunk(&meta, &states)?;
        new_chunks += 1;
        eprintln!(
            "{}: {} prefill + {} decode tokens in {:.1}s ({:.2} decode tok/s incl. capture) | {:?}",
            p.id,
            n_prefill,
            n_tok - n_prefill,
            meta.secs,
            (n_tok - n_prefill) as f64 / meta.secs.max(1e-9),
            meta.text_head.chars().take(80).collect::<String>()
        );
    }
    let done = run.list_done()?;
    let toks: usize = done.iter().map(|m| m.n_tokens()).sum();
    let dec: usize = done.iter().map(|m| m.n_tokens() - m.n_prefill).sum();
    eprintln!("run {}: {} chunks done, {toks} tokens ({dec} decode)", args.out.display(), done.len());
    Ok(())
}
