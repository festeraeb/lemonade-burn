//! Minimal token-level server for lb-gemma4, for evaluation drivers that render the chat
//! template and (de)tokenize through a llama.cpp server on the same GGUF, so both engines see
//! byte-identical token ids.
//!
//! * `GET /health` -> `{"status":"ok"}`
//! * `POST /completion` `{"prompt": [ids], "n_predict": N, "stop_ids": [ids]}` -> greedy
//!   `{"tokens": [...], "stop": "eos"|"limit", "timings": {prompt_n, prompt_ms, predicted_n,
//!   predicted_ms, predicted_per_second}, "stats": {...}}`. The stop token is not included.
//!   Default `stop_ids` are gemma4's EOG set in llama.cpp e14a779: `<eos>` 1, `<|tool_response>`
//!   50, `<turn|>` 106. One request at a time; scrapes are flushed per request when
//!   `NAUTIBURN_SCRAPE_DIR` is set.

use anyhow::Result;
use clap::Parser;
use lb_gemma4::Gemma4;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1:18197")]
    listen: String,
}

fn generate(model: &Gemma4, body: &Value) -> Result<Value> {
    let ids: Vec<u32> = body["prompt"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("prompt must be an array of token ids"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    anyhow::ensure!(!ids.is_empty(), "empty prompt");
    let n_predict = body["n_predict"].as_u64().unwrap_or(128) as usize;
    let stops: Vec<u32> = match body["stop_ids"].as_array() {
        Some(a) => a
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        None => vec![1, 50, 106],
    };
    let mut cache = model.new_cache();
    let t0 = Instant::now();
    let mut logits = model.forward(&ids, &mut cache, false)?;
    let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let mut out = Vec::new();
    let mut stop = "limit";
    for i in 0..n_predict {
        let tok = Gemma4::argmax(&logits);
        if stops.contains(&tok) {
            stop = "eos";
            break;
        }
        out.push(tok);
        if i + 1 < n_predict {
            logits = model.forward(&[tok], &mut cache, false)?;
        }
    }
    let predicted_ms = t1.elapsed().as_secs_f64() * 1e3;
    let mut all = ids.clone();
    all.extend(&out);
    if let Err(e) = model.scraper.flush("decode", &all) {
        eprintln!("scrape flush failed: {e}");
    }
    let s = model.stats.snapshot();
    Ok(json!({
        "tokens": out, "stop": stop,
        "timings": {"prompt_n": ids.len(), "prompt_ms": prompt_ms, "predicted_n": out.len(), "predicted_ms": predicted_ms,
                     "predicted_per_second": if predicted_ms > 0.0 { out.len() as f64 / predicted_ms * 1e3 } else { 0.0 }},
        "stats": {"gpu_hits": s.gpu_hits, "cpu_calls": s.cpu_calls, "cpu_s": s.cpu_s},
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t = Instant::now();
    let model = Gemma4::load(&args.model)?;
    eprintln!(
        "gemma4-serve: loaded {} in {:.1}s, numerics {:?}, listening on {}",
        args.model.display(),
        t.elapsed().as_secs_f64(),
        lb_gemma4::numerics::mode(),
        args.listen
    );
    let server = tiny_http::Server::http(&args.listen)
        .map_err(|e| anyhow::anyhow!("bind {}: {e}", args.listen))?;
    let hdr =
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    for mut req in server.incoming_requests() {
        let (code, body) = match (req.method().as_str(), req.url()) {
            ("GET", "/health") => (200, json!({"status": "ok"})),
            ("POST", "/completion") => {
                let mut s = String::new();
                let parsed = req
                    .as_reader()
                    .read_to_string(&mut s)
                    .map_err(anyhow::Error::from)
                    .and_then(|_| Ok(serde_json::from_str::<Value>(&s)?));
                match parsed.and_then(|b| generate(&model, &b)) {
                    Ok(v) => (200, v),
                    Err(e) => (400, json!({"error": e.to_string()})),
                }
            }
            _ => (404, json!({"error": "not found"})),
        };
        let resp = tiny_http::Response::from_string(body.to_string())
            .with_status_code(code)
            .with_header(hdr.clone());
        if let Err(e) = req.respond(resp) {
            eprintln!("respond failed: {e}");
        }
    }
    Ok(())
}
