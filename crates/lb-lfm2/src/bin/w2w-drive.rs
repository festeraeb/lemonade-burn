//! Client side of one measured server session (W3): warm up, run every prompt of a JSONL file
//! (`{"id","text","max_tokens"}`) greedily `--passes` times, and append one row per completion:
//! decode tok/s, text sha, and the work-to-weights hop counters for that completion (the
//! difference of `GET /v1/placement` -> `w2w.decode` before and after). Measurement harness only:
//! the engine under test is the server.

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    port: u16,
    #[arg(long)]
    prompts: PathBuf,
    #[arg(long, default_value_t = 1)]
    passes: usize,
    #[arg(long)]
    out: PathBuf,
    /// Copied into every row: `key=value`, repeatable (arm, round, label, cards, ...).
    #[arg(long = "tag")]
    tags: Vec<String>,
    /// `id=sha`: the sha a prompt must give; repeatable. A mismatch is recorded and fails the run.
    #[arg(long = "want")]
    want: Vec<String>,
}

fn sha16(text: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(text.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let base = format!("http://127.0.0.1:{}", args.port);
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(1800)).build();
    let prompts: Vec<Value> = std::fs::read_to_string(&args.prompts)
        .with_context(|| format!("read {}", args.prompts.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow!("prompt line: {e}")))
        .collect::<Result<_>>()?;
    let complete = |p: &Value| -> Result<Value> {
        Ok(agent.post(&format!("{base}/v1/completions")).send_json(json!({"prompt": p["text"], "max_tokens": p["max_tokens"], "temperature": 0}))?.into_json()?)
    };
    let placement = || -> Value { agent.get(&format!("{base}/v1/placement")).call().ok().and_then(|r| r.into_json().ok()).unwrap_or(Value::Null) };
    let dec = |v: &Value, k: &str| v["w2w"]["decode"][k].as_f64().unwrap_or(0.0);
    let load1 = || std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| s.split_whitespace().next().and_then(|v| v.parse::<f64>().ok())).unwrap_or(-1.0);
    complete(&prompts[0])?; // warm-up, not recorded
    let mut bad = 0;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&args.out).with_context(|| format!("open {}", args.out.display()))?;
    for pass in 0..args.passes {
        for p in &prompts {
            let before = placement();
            let r = complete(p)?;
            let after = placement();
            let text = r["choices"][0]["text"].as_str().unwrap_or("");
            let sha = sha16(text);
            let id = p["id"].as_str().unwrap_or("");
            let want = args.want.iter().find_map(|w| w.split_once('=').filter(|(i, _)| *i == id).map(|(_, s)| s.to_string()));
            let sha_ok = want.as_ref().map(|w| *w == sha);
            if sha_ok == Some(false) {
                bad += 1;
            }
            let d = |k: &str| dec(&after, k) - dec(&before, k);
            let toks = d("tokens");
            let per = |x: f64| if toks > 0.0 { (x / toks * 1000.0).round() / 1000.0 } else { 0.0 };
            let mut row = json!({
                "ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
                "step": "W3", "prompt": id, "pass": pass,
                "tokens": r["usage"]["completion_tokens"], "decode_tok_s": r["timings"]["predicted_per_second"], "prefill_s": r["timings"]["prompt_s"],
                "text_sha": sha, "want_sha": want, "sha_ok": sha_ok,
                "w2w": !after["w2w"].is_null(),
                "hops_per_token": per(d("hops")), "hop_wall_ms_per_token": per(d("hop_wall_ms")), "hop_net_ms_per_token": per(d("hop_net_ms")),
                "host_ms_per_token": per(d("host_ms")), "experts_remote_per_token": per(d("experts_remote")), "experts_fallback": d("experts_fallback"),
                "hosts_up": after["w2w"]["hosts"].as_array().map(|a| a.iter().filter(|h| h["up"] == true).count()),
                "hosts": after["w2w"]["hosts"].as_array().map(|a| a.len()),
                "load1": load1(),
            });
            for t in &args.tags {
                if let Some((k, v)) = t.split_once('=') {
                    row[k] = v.parse::<f64>().map(|x| json!(x)).unwrap_or_else(|_| json!(v));
                }
            }
            writeln!(f, "{row}")?;
            println!("{}", json!({"prompt": id, "pass": pass, "tok_s": row["decode_tok_s"], "sha": row["text_sha"], "sha_ok": sha_ok, "hops": row["hops_per_token"], "hop_ms": row["hop_wall_ms_per_token"], "fallback": row["experts_fallback"]}));
        }
    }
    if bad > 0 {
        return Err(anyhow!("{bad} completion(s) gave a sha other than the wanted one"));
    }
    Ok(())
}
