//! Work-to-weights parity harness: the trunk with experts on hosts against the one-process
//! path, in one process on one loaded model, logits compared **bit for bit**.
//!
//! The reference arm runs with no remote tier. Each `--arm name=placement.json` then attaches
//! that placement (its hosts must be running) and repeats:
//! * **prefill**: every prompt as one pass (the batched wire shape), all positions' logits;
//! * **decode**: every prompt token by token, teacher-forced (the one-row wire shape);
//! * **greedy**: `--greedy-n` tokens from `--greedy-prompt`, and the sha of the text.
//!
//! `--kill arm=phase@index:pid[:STOP]` kills a host process (SIGKILL; with `:STOP` it is frozen
//! instead, so only the deadline can find it) just before that decode
//! position / greedy step of that arm, to prove the fallback mid-decode.
//!
//! One JSON line per arm is appended to `--out`; exit status is non-zero if any arm differs.

use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::Device;
use clap::Parser;
use lb_lfm2::model::Lfm2;
use lb_lfm2::remote::PassRecord;
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// The three prompts of the lfm2-check harness (29 + 44 + 13 = 86 positions with BOS).
const HARNESS: [&str; 3] = [
    "A search and rescue team on Lake Erie found the missing kayaker after three hours. The coordinator said the side-scan sonar image showed",
    "Great Lakes shipwreck survey notes: the schooner lies in 45 m of water, bow to the north, with the mainmast collapsed across the deck. Visibility was 3 m and the",
    "The capital of France is Paris. The capital of Germany is",
];

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, env = "LB_MODEL")]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    /// `name=placement.json`; repeatable, run in order.
    #[arg(long = "arm")]
    arms: Vec<String>,
    /// `arm=decode@<position>:<pid>` or `arm=greedy@<step>:<pid>`, optionally `:STOP`; repeatable.
    #[arg(long = "kill")]
    kills: Vec<String>,
    /// Phases to run: any of prefill, decode, greedy.
    #[arg(long, default_value = "prefill,decode,greedy")]
    phases: String,
    /// Phases for one arm only (`arm=greedy`), overriding `--phases`; repeatable.
    #[arg(long = "arm-phases")]
    arm_phases: Vec<String>,
    /// Prompts for prefill/decode; default: the three lfm2-check harness prompts.
    #[arg(long = "prompt")]
    prompts: Vec<String>,
    #[arg(long, default_value = "Explain in two sentences why a sonar return from a hard sand bottom differs from one over silt.")]
    greedy_prompt: String,
    #[arg(long, default_value_t = 32)]
    greedy_n: usize,
    /// The one-card greedy sha of the sonar prompt, to compare against.
    #[arg(long, default_value = "062392b8d2b0361a")]
    want_sha: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "")]
    label: String,
}

struct Kill {
    arm: String,
    phase: String,
    at: usize,
    pid: u32,
    /// `KILL` (the process dies: the connection resets) or `STOP` (it hangs: only the deadline
    /// finds it).
    signal: String,
}

fn parse_kill(s: &str) -> Result<Kill> {
    let bad = || anyhow!("--kill {s:?}: expected arm=decode@<n>:<pid> or arm=greedy@<n>:<pid>");
    let (arm, rest) = s.split_once('=').ok_or_else(bad)?;
    let (phase, rest) = rest.split_once('@').ok_or_else(bad)?;
    let (at, pid) = rest.split_once(':').ok_or_else(bad)?;
    let (pid, signal) = pid.split_once(':').unwrap_or((pid, "KILL"));
    if signal != "KILL" && signal != "STOP" {
        return Err(bad());
    }
    if phase != "decode" && phase != "greedy" {
        return Err(bad());
    }
    Ok(Kill { arm: arm.into(), phase: phase.into(), at: at.parse().map_err(|_| bad())?, pid: pid.parse().map_err(|_| bad())?, signal: signal.into() })
}

fn kill(pid: u32, signal: &str) -> bool {
    std::process::Command::new("kill").args([&format!("-{signal}"), &pid.to_string()]).status().map(|s| s.success()).unwrap_or(false)
}

fn row(t: burn::tensor::Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
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

fn sha16(text: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(text.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// What one arm computed.
#[derive(Default)]
struct Out {
    /// Per prompt: all positions' logits from one pass, `[n * vocab]`.
    prefill: Vec<Vec<f32>>,
    /// Per position over all prompts: the logits of that token's own decode pass.
    decode: Vec<Vec<f32>>,
    greedy_tokens: Vec<u32>,
    greedy_logits: Vec<Vec<f32>>,
    secs: [f64; 3],
}

struct Cmp {
    n: usize,
    identical: usize,
    top1: usize,
    max_abs: f32,
}

/// Compare logits rows of `vocab`: how many are the same bits, how many agree on top-1.
fn compare(a: &[f32], b: &[f32], vocab: usize) -> Cmp {
    let mut c = Cmp { n: 0, identical: 0, top1: 0, max_abs: 0.0 };
    for (ra, rb) in a.chunks(vocab).zip(b.chunks(vocab)) {
        c.n += 1;
        c.identical += (ra.len() == rb.len() && ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits())) as usize;
        c.top1 += (argmax(ra) == argmax(rb)) as usize;
        c.max_abs = ra.iter().zip(rb).map(|(x, y)| (x - y).abs()).fold(c.max_abs, f32::max);
    }
    if a.len() != b.len() {
        c.n = c.n.max(a.len().max(b.len()).div_ceil(vocab));
    }
    c
}

fn run_arm(model: &Lfm2, prompts: &[Vec<u32>], greedy: &[u32], n_greedy: usize, phases: &[&str], kills: &[&Kill]) -> Out {
    let mut o = Out::default();
    let eos = model.cfg().eos;
    if phases.contains(&"prefill") {
        let t = Instant::now();
        for ids in prompts {
            let mut cache = model.new_cache();
            o.prefill.push(row(model.forward(ids, &mut cache, true)));
        }
        o.secs[0] = t.elapsed().as_secs_f64();
    }
    if phases.contains(&"decode") {
        let t = Instant::now();
        let mut pos = 0usize;
        for ids in prompts {
            let mut cache = model.new_cache();
            for &id in ids {
                for k in kills.iter().filter(|k| k.phase == "decode" && k.at == pos) {
                    eprintln!("w2w-check: SIG{} to pid {} before decode position {pos}: {}", k.signal, k.pid, if kill(k.pid, &k.signal) { "sent" } else { "FAILED" });
                }
                o.decode.push(row(model.forward(&[id], &mut cache, false)));
                pos += 1;
            }
        }
        o.secs[1] = t.elapsed().as_secs_f64();
    }
    if phases.contains(&"greedy") {
        let t = Instant::now();
        let mut cache = model.new_cache();
        let mut next = argmax(&row(model.forward(greedy, &mut cache, false)));
        for step in 0..n_greedy {
            o.greedy_tokens.push(next);
            if next == eos {
                break;
            }
            for k in kills.iter().filter(|k| k.phase == "greedy" && k.at == step) {
                eprintln!("w2w-check: SIG{} to pid {} before greedy step {step}: {}", k.signal, k.pid, if kill(k.pid, &k.signal) { "sent" } else { "FAILED" });
            }
            let r = row(model.forward(&[next], &mut cache, false));
            next = argmax(&r);
            o.greedy_logits.push(r);
        }
        o.secs[2] = t.elapsed().as_secs_f64();
    }
    o
}

/// Hop statistics over the decode passes (one token each) of a record list.
fn hop_stats(recs: &[PassRecord]) -> serde_json::Value {
    let dec: Vec<&PassRecord> = recs.iter().filter(|r| r.m == 1).collect();
    let n = dec.len().max(1) as f64;
    let sum = |f: &dyn Fn(&PassRecord) -> f64| dec.iter().map(|r| f(r)).sum::<f64>();
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    json!({
        "decode_tokens": dec.len(),
        "hops_per_token_mean": r3(sum(&|r| r.hops as f64) / n),
        "hops_per_token_min": dec.iter().map(|r| r.hops).min().unwrap_or(0),
        "hops_per_token_max": dec.iter().map(|r| r.hops).max().unwrap_or(0),
        "experts_remote": dec.iter().map(|r| r.experts_remote).sum::<u64>(),
        "experts_fallback": dec.iter().map(|r| r.experts_fallback).sum::<u64>(),
        "hop_wall_ms_per_token": r3(sum(&|r| r.hop_wall_ms) / n),
        "hop_net_ms_per_token": r3(sum(&|r| r.hop_net_ms) / n),
        "host_compute_ms_per_token": r3(sum(&|r| r.host_ms) / n),
        "prefill_passes": recs.iter().filter(|r| r.m > 1).count(),
        "prefill_hops": recs.iter().filter(|r| r.m > 1).map(|r| r.hops).sum::<u64>(),
        "tokens_with_a_host_down": dec.iter().filter(|r| !r.down.is_empty()).count(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kills: Vec<Kill> = args.kills.iter().map(|s| parse_kill(s)).collect::<Result<_>>()?;
    let codec = tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow!("tokenizer {}: {e}", args.tokenizer.display()))?;
    let t0 = Instant::now();
    let mut model = Lfm2::load(&args.model, &Device::default())?;
    let cfg = model.cfg().clone();
    let tokenize = |text: &str| -> Result<Vec<u32>> {
        let mut ids = codec.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec();
        ids.insert(0, cfg.bos);
        Ok(ids)
    };
    let texts: Vec<String> = if args.prompts.is_empty() { HARNESS.iter().map(|s| s.to_string()).collect() } else { args.prompts.clone() };
    let prompts: Vec<Vec<u32>> = texts.iter().map(|p| tokenize(p)).collect::<Result<_>>()?;
    let greedy = tokenize(&args.greedy_prompt)?;
    let positions: usize = prompts.iter().map(Vec::len).sum();
    let all_phases: Vec<&str> = args.phases.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    for p in &all_phases {
        if !["prefill", "decode", "greedy"].contains(p) {
            bail!("unknown phase {p:?}");
        }
    }
    eprintln!(
        "w2w-check: loaded {} in {:.1}s; {} prompts, {positions} positions ({:?}); greedy prompt {} tokens; phases {:?}",
        args.model.display(),
        t0.elapsed().as_secs_f32(),
        prompts.len(),
        prompts.iter().map(Vec::len).collect::<Vec<_>>(),
        greedy.len(),
        all_phases
    );
    let text = |v: &[u32]| codec.decode(v, true).unwrap_or_else(|_| format!("{v:?}"));
    let load1 = || std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| s.split_whitespace().next().and_then(|v| v.parse::<f64>().ok())).unwrap_or(-1.0);

    // the reference: this process, no hosts
    model.store.remote = None;
    let reference = run_arm(&model, &prompts, &greedy, args.greedy_n, &all_phases, &[]);
    let ref_text = text(&reference.greedy_tokens);
    let ref_sha = sha16(&ref_text);
    let ref_local_calls = model.stats.cpu_calls.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!(
        "w2w-check: reference (one process): prefill {:.1}s, decode {:.1}s ({} positions), greedy {:.1}s ({} tokens) sha {ref_sha}{} {ref_text:?}",
        reference.secs[0],
        reference.secs[1],
        reference.decode.len(),
        reference.secs[2],
        reference.greedy_tokens.len(),
        if !all_phases.contains(&"greedy") { " (not run)" } else if ref_sha == args.want_sha { " == the one-card sha" } else { " (NOT the one-card sha)" },
    );
    let mut rows = vec![json!({
        "ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "step": "W1/W2", "label": args.label, "arm": "reference (one process, no hosts)", "phases": all_phases,
        "positions": positions, "prefill_s": reference.secs[0], "decode_s": reference.secs[1], "greedy_s": reference.secs[2],
        "decode_tok_s": if reference.secs[1] > 0.0 { reference.decode.len() as f64 / reference.secs[1] } else { 0.0 },
        "greedy_tokens": reference.greedy_tokens, "greedy_text": ref_text, "greedy_sha": ref_sha, "want_sha": args.want_sha,
        "greedy_sha_is_one_card_sha": ref_sha == args.want_sha, "local_expert_calls": ref_local_calls, "load1": load1(),
    })];
    let mut failed = false;
    for spec in &args.arms {
        let (name, path) = spec.split_once('=').with_context(|| format!("--arm {spec:?}: expected name=placement.json"))?;
        let phases: Vec<&str> = match args.arm_phases.iter().find_map(|s| s.split_once('=').filter(|(a, _)| *a == name).map(|(_, p)| p)) {
            Some(p) => p.split(',').map(str::trim).filter(|s| all_phases.contains(s)).collect(),
            None => all_phases.clone(),
        };
        let rt = model.attach_remote(std::path::Path::new(path)).map_err(|e| anyhow!("arm {name}: {e}"))?;
        let hosts_before = rt.hosts();
        let my_kills: Vec<&Kill> = kills.iter().filter(|k| k.arm == name).collect();
        let c0 = model.stats.cpu_calls.load(std::sync::atomic::Ordering::Relaxed);
        let got = run_arm(&model, &prompts, &greedy, args.greedy_n, &phases, &my_kills);
        let local_calls = model.stats.cpu_calls.load(std::sync::atomic::Ordering::Relaxed) - c0;
        let recs = rt.take_records();
        let hosts_after = rt.hosts();
        model.store.remote = None;

        let mut ok = true;
        let mut phase_rows = serde_json::Map::new();
        if phases.contains(&"prefill") {
            let (mut n, mut same, mut top1, mut worst) = (0, 0, 0, 0f32);
            for (a, b) in reference.prefill.iter().zip(&got.prefill) {
                let c = compare(a, b, cfg.n_vocab);
                n += c.n;
                same += c.identical;
                top1 += c.top1;
                worst = worst.max(c.max_abs);
            }
            ok &= n == positions && same == n;
            phase_rows.insert("prefill".into(), json!({"positions": n, "bit_identical": same, "top1_agree": top1, "max_abs_logit_diff": worst, "secs": got.secs[0]}));
        }
        if phases.contains(&"decode") {
            let (mut same, mut top1, mut worst) = (0, 0, 0f32);
            for (a, b) in reference.decode.iter().zip(&got.decode) {
                let c = compare(a, b, cfg.n_vocab);
                same += c.identical;
                top1 += c.top1;
                worst = worst.max(c.max_abs);
            }
            let n = got.decode.len();
            ok &= n == positions && same == n;
            phase_rows.insert("decode".into(), json!({"positions": n, "bit_identical": same, "top1_agree": top1, "max_abs_logit_diff": worst, "secs": got.secs[1], "tok_s": n as f64 / got.secs[1].max(1e-9)}));
        }
        let mut sha = String::new();
        if phases.contains(&"greedy") {
            let same_tokens = got.greedy_tokens == reference.greedy_tokens;
            let (mut same, mut worst) = (0, 0f32);
            for (a, b) in reference.greedy_logits.iter().zip(&got.greedy_logits) {
                let c = compare(a, b, cfg.n_vocab);
                same += c.identical;
                worst = worst.max(c.max_abs);
            }
            let t = text(&got.greedy_tokens);
            sha = sha16(&t);
            ok &= same_tokens && same == reference.greedy_logits.len() && got.greedy_logits.len() == reference.greedy_logits.len() && sha == ref_sha;
            phase_rows.insert(
                "greedy".into(),
                json!({"tokens": got.greedy_tokens.len(), "tokens_identical": same_tokens, "steps_bit_identical": same, "steps": got.greedy_logits.len(), "max_abs_logit_diff": worst,
                    "sha": sha, "sha_equals_reference": sha == ref_sha, "sha_is_one_card_sha": sha == args.want_sha, "text": t, "secs": got.secs[2], "tok_s": got.greedy_tokens.len() as f64 / got.secs[2].max(1e-9)}),
            );
        }
        // an arm that never reached a host proves nothing: it must have started with every host
        // up and had experts answered remotely; without a kill, nothing may have fallen back
        let s = &rt.stats;
        let o = std::sync::atomic::Ordering::Relaxed;
        let used = hosts_before.iter().all(|h| h.up) && !hosts_before.is_empty() && s.experts.load(o) > 0;
        let clean = !my_kills.is_empty() || (s.fallback_experts.load(o) == 0 && hosts_after.iter().all(|h| h.up));
        let killed_seen = my_kills.is_empty() || (s.mark_downs.load(o) > 0 && s.fallback_experts.load(o) > 0);
        if !used {
            eprintln!("w2w-check: arm {name}: INVALID, the hosts were not all up or answered nothing");
        }
        if !clean {
            eprintln!("w2w-check: arm {name}: INVALID, experts fell back to the trunk without a kill");
        }
        if !killed_seen {
            eprintln!("w2w-check: arm {name}: INVALID, a kill was asked for but no host was marked down");
        }
        ok &= used && clean && killed_seen;
        failed |= !ok;
        let row = json!({
            "ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            "step": "W1/W2", "label": args.label, "arm": name, "placement": path, "phases": phases, "pass": ok, "hosts_used": used, "no_unexpected_fallback": clean,
            "results": phase_rows,
            "kills": my_kills.iter().map(|k| json!({"phase": k.phase, "at": k.at, "pid": k.pid, "signal": k.signal})).collect::<Vec<_>>(),
            "hops": hop_stats(&recs),
            "totals": {"hops": s.hops.load(o), "experts_remote": s.experts.load(o), "experts_fallback": s.fallback_experts.load(o), "refusals": s.refusals.load(o),
                "timeouts": s.timeouts.load(o), "mark_downs": s.mark_downs.load(o), "trunk_local_expert_calls": local_calls},
            "hosts_before": hosts_before, "hosts_after": hosts_after, "load1": load1(),
        });
        eprintln!(
            "w2w-check: arm {name}: {} | {} | hops {}",
            if ok { "PASS (bit-identical to the one-process path)" } else { "FAIL" },
            serde_json::to_string(&row["results"]).unwrap_or_default().chars().take(700).collect::<String>(),
            row["hops"]
        );
        let _ = sha;
        rows.push(row);
    }
    if let Some(p) = &args.out {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).with_context(|| format!("open {}", p.display()))?;
        for r in &rows {
            writeln!(f, "{r}")?;
        }
    }
    if failed {
        bail!("at least one arm differs from the one-process path");
    }
    Ok(())
}
