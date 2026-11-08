//! Hit rate of next-layer expert prediction from hidden states (experiment #1,
//! `docs/EXPERT_PREDICT.md`), from a capture run directory.
//!
//! For every captured token and every MoE layer T, the engine's chosen set is compared with:
//! * `mid{d}`: layer T's router (with T's FFN pre-norm) applied to `x_mid` of layer T-d, the
//!   residual stream after T-d's operator, i.e. available *before* T-d's experts run.
//!   `mid0` is T's own router input: a sanity row that must be ~1.0.
//! * `out{d}`: the same on `x_out` of layer T-d (after T-d's FFN; `out1` is T's input).
//! * `mid{d}+N`: the `mid` guess widened to k+N experts (a staging ring that fetches more
//!   than k): recall = share of the actual experts inside the wider guess.
//! * baselines on the same tokens: `prev` (the previous token's set at T, NautiBurn's
//!   "previous-token copy") and `freq` (T's k most used experts in all *other* chunks).
//!
//! Writes `<out>/report.json`, `<out>/report.md` and per-token rows with the gate margins
//! (`<out>/rows/<chunk>.jsonl`) so confident guesses can be selected later.

use anyhow::Result;
use clap::Parser;
use lb_expert_predict::capture::{ChunkMeta, RunDir, Which};
use lb_expert_predict::metrics::{rms_norm, router_scores, score_sets, top_by_count, top_share, topk_margin, Acc};
use lb_gguf::Gguf;
use lb_lfm2::weights::{Ffn, Weights};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Capture run directory.
    #[arg(long)]
    run: PathBuf,
    /// Output directory (default `<run>/analysis`).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Skip writing the per-token rows.
    #[arg(long, default_value_t = false)]
    no_rows: bool,
}

/// (name, which state, distance, guess size over k)
const GUESSES: &[(&str, Which, usize, usize)] = &[
    ("mid0", Which::Mid, 0, 0),
    ("mid1", Which::Mid, 1, 0),
    ("out1", Which::Out, 1, 0),
    ("mid2", Which::Mid, 2, 0),
    ("out2", Which::Out, 2, 0),
    ("mid1+2", Which::Mid, 1, 2),
    ("mid1+4", Which::Mid, 1, 4),
    ("mid2+4", Which::Mid, 2, 4),
];
const BASELINES: &[&str] = &["prev", "freq"];

/// Accumulators: predictor -> phase -> layer.
type Table = BTreeMap<String, BTreeMap<String, BTreeMap<usize, Acc>>>;

fn add(t: &mut Table, pred: &str, phase: &str, layer: usize, s: lb_expert_predict::metrics::SetScore) {
    t.entry(pred.into()).or_default().entry(phase.into()).or_default().entry(layer).or_default().add(s);
}

fn merge(a: &mut Table, b: &Table) {
    for (p, phases) in b {
        for (ph, layers) in phases {
            for (l, acc) in layers {
                a.entry(p.clone()).or_default().entry(ph.clone()).or_default().entry(*l).or_default().merge(acc);
            }
        }
    }
}

/// One guess with its margin, for the confidence curve.
#[derive(Clone, Copy)]
struct Conf {
    margin: f32,
    hits: u8,
    k: u8,
    decode: bool,
}

struct ChunkOut {
    table: Table,
    conf: BTreeMap<&'static str, Vec<Conf>>,
    mismatch: u64,
    checked: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let run = RunDir::new(&args.run)?;
    let out = args.out.clone().unwrap_or_else(|| args.run.join("analysis"));
    std::fs::create_dir_all(out.join("rows"))?;
    let w = Weights::load(Arc::new(Gguf::open(&args.model)?))?;
    let cfg = &w.cfg;
    let chunks = run.list_done()?;
    anyhow::ensure!(!chunks.is_empty(), "no finished chunks in {}", args.run.display());
    let k = cfg.n_expert_used;
    let moe_layers: Vec<usize> = (0..cfg.n_layer).filter(|&l| cfg.is_moe(l)).collect();

    // per-layer usage counts per chunk (concentration + leave-one-chunk-out frequency baseline)
    let counts_of = |m: &ChunkMeta| -> Vec<Vec<u64>> {
        let mut c = vec![vec![0u64; cfg.n_expert]; cfg.n_layer];
        for t in 0..m.n_tokens() {
            for &l in &moe_layers {
                for &e in &m.routes[t][l] {
                    c[l][e as usize] += 1;
                }
            }
        }
        c
    };
    let per_chunk: Vec<Vec<Vec<u64>>> = chunks.iter().map(counts_of).collect();
    let mut total = vec![vec![0u64; cfg.n_expert]; cfg.n_layer];
    for c in &per_chunk {
        for l in 0..cfg.n_layer {
            for e in 0..cfg.n_expert {
                total[l][e] += c[l][e];
            }
        }
    }

    let results: Vec<ChunkOut> = chunks
        .par_iter()
        .enumerate()
        .map(|(ci, m)| -> Result<ChunkOut> {
            let st = run.open_states(m)?;
            let mut table = Table::new();
            let mut conf: BTreeMap<&'static str, Vec<Conf>> = BTreeMap::new();
            let (mut mismatch, mut checked) = (0u64, 0u64);
            // held-out frequency sets: all chunks but this one
            let freq: Vec<Vec<usize>> = (0..cfg.n_layer)
                .map(|l| {
                    let c: Vec<u64> = (0..cfg.n_expert).map(|e| total[l][e] - per_chunk[ci][l][e]).collect();
                    top_by_count(&c, k)
                })
                .collect();
            let mut rows = String::new();
            for t in 0..m.n_tokens() {
                let phase = if m.is_decode(t) { "decode" } else { "prefill" };
                // cache the state rows this token needs
                let mut mid: Vec<Option<Vec<f32>>> = vec![None; cfg.n_layer];
                let mut outs: Vec<Option<Vec<f32>>> = vec![None; cfg.n_layer];
                for &tl in &moe_layers {
                    let Ffn::Moe { router, bias, .. } = &w.layers[tl].ffn else { unreachable!() };
                    let norm = &w.layers[tl].ffn_norm;
                    let actual: Vec<usize> = m.routes[t][tl].iter().map(|&e| e as usize).collect();
                    let mut row = serde_json::Map::new();
                    row.insert("t".into(), t.into());
                    row.insert("dec".into(), m.is_decode(t).into());
                    row.insert("L".into(), tl.into());
                    row.insert("act".into(), serde_json::json!(actual));
                    for &(name, which, d, extra) in GUESSES {
                        if tl < d {
                            continue;
                        }
                        let sl = tl - d;
                        let slot = match which {
                            Which::Mid => &mut mid[sl],
                            Which::Out => &mut outs[sl],
                        };
                        let x = slot.get_or_insert_with(|| st.row(t, sl, which));
                        let h = rms_norm(x, norm, cfg.rms_eps);
                        let (_, sel) = router_scores(router, bias, &h);
                        let (guess, margin) = topk_margin(&sel, k + extra);
                        let s = score_sets(&guess, &actual);
                        add(&mut table, name, phase, tl, s);
                        if name == "mid0" {
                            checked += 1;
                            mismatch += (!s.all_hit) as u64;
                            // the actual gate's own margin (how close it came to choosing differently)
                            row.insert("am".into(), serde_json::json!(margin));
                            row.insert("mid0ok".into(), s.all_hit.into());
                        } else if extra == 0 {
                            conf.entry(name).or_default().push(Conf { margin, hits: s.hits as u8, k: k as u8, decode: m.is_decode(t) });
                            row.insert(name.into(), serde_json::json!(guess));
                            row.insert(format!("{name}m"), serde_json::json!(margin));
                        }
                    }
                    if t > 0 {
                        let prev: Vec<usize> = m.routes[t - 1][tl].iter().map(|&e| e as usize).collect();
                        add(&mut table, "prev", phase, tl, score_sets(&prev, &actual));
                    }
                    add(&mut table, "freq", phase, tl, score_sets(&freq[tl], &actual));
                    if !args.no_rows {
                        rows.push_str(&serde_json::Value::Object(row).to_string());
                        rows.push('\n');
                    }
                }
            }
            if !args.no_rows {
                std::fs::write(out.join("rows").join(format!("{}.jsonl", m.id)), rows)?;
            }
            Ok(ChunkOut { table, conf, mismatch, checked })
        })
        .collect::<Result<_>>()?;

    let mut table = Table::new();
    let mut conf: BTreeMap<&'static str, Vec<Conf>> = BTreeMap::new();
    let (mut mismatch, mut checked) = (0u64, 0u64);
    for r in &results {
        merge(&mut table, &r.table);
        for (n, v) in &r.conf {
            conf.entry(n).or_default().extend_from_slice(v);
        }
        mismatch += r.mismatch;
        checked += r.checked;
    }

    // pooled over layers
    let pooled = |pred: &str, phase: Option<&str>| -> Acc {
        let mut a = Acc::default();
        if let Some(phases) = table.get(pred) {
            for (ph, layers) in phases {
                if phase.is_none_or(|p| p == ph) {
                    for acc in layers.values() {
                        a.merge(acc);
                    }
                }
            }
        }
        a
    };
    let preds: Vec<&str> = GUESSES.iter().map(|g| g.0).chain(BASELINES.iter().copied()).collect();

    // confidence curve: keep the most confident X% of guesses (decode tokens)
    #[derive(Serialize)]
    struct ConfRow {
        keep: f64,
        n: usize,
        min_margin: f32,
        recall: f64,
        all_k: f64,
    }
    let mut conf_out: BTreeMap<String, Vec<ConfRow>> = BTreeMap::new();
    for (name, v) in &conf {
        for (tag, want_dec) in [("decode", Some(true)), ("all", None)] {
            let mut v: Vec<Conf> = v.iter().copied().filter(|c| want_dec.is_none_or(|d| d == c.decode)).collect();
            v.sort_by(|a, b| b.margin.total_cmp(&a.margin));
            let rows = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                .iter()
                .filter_map(|&keep| {
                    let n = ((v.len() as f64) * keep).round() as usize;
                    if n == 0 {
                        return None;
                    }
                    let s = &v[..n];
                    let hits: u64 = s.iter().map(|c| c.hits as u64).sum();
                    let kt: u64 = s.iter().map(|c| c.k as u64).sum();
                    let all = s.iter().filter(|c| c.hits == c.k).count();
                    Some(ConfRow { keep, n, min_margin: s[n - 1].margin, recall: hits as f64 / kt as f64, all_k: all as f64 / n as f64 })
                })
                .collect();
            conf_out.insert(format!("{name}/{tag}"), rows);
        }
    }

    let concentration: BTreeMap<usize, BTreeMap<String, f64>> = moe_layers
        .iter()
        .map(|&l| (l, [8usize, 16, 24, 32, 48, 64].iter().map(|&n| (format!("top{n}"), top_share(&total[l], n))).collect()))
        .collect();

    let n_tok: usize = chunks.iter().map(|m| m.n_tokens()).sum();
    let n_dec: usize = chunks.iter().map(|m| m.n_tokens() - m.n_prefill).sum();
    let summary: BTreeMap<String, BTreeMap<String, serde_json::Value>> = preds
        .iter()
        .map(|&p| {
            let row = ["all", "prefill", "decode"]
                .iter()
                .map(|&ph| {
                    let a = pooled(p, if ph == "all" { None } else { Some(ph) });
                    (ph.to_string(), serde_json::json!({"n": a.n, "recall": a.recall(), "jaccard": a.jaccard(), "all_k": a.all_k()}))
                })
                .collect();
            (p.to_string(), row)
        })
        .collect();
    let report = serde_json::json!({
        "model": args.model.file_name().and_then(|s| s.to_str()),
        "run": args.run,
        "chunks": chunks.len(),
        "tokens": n_tok,
        "decode_tokens": n_dec,
        "k": k,
        "n_expert": cfg.n_expert,
        "moe_layers": moe_layers.len(),
        "sanity_mid0_mismatch": {"checked": checked, "mismatch": mismatch},
        "summary": summary,
        "per_layer": table,
        "confidence": conf_out,
        "concentration": concentration,
    });
    std::fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;

    // markdown
    let mut md = String::new();
    let f = |x: f64| format!("{x:.3}");
    md.push_str(&format!(
        "# Expert prediction hit rate: {}\n\n{} chunks, {n_tok} tokens ({n_dec} decode, {} prefill), {} MoE layers, {} experts top-{k}. \
         Token-layer pairs scored per predictor = tokens x MoE layers (fewer for d=2 at the first MoE layer).\n\n\
         Sanity: the engine's own router input (`mid0`) reproduces the engine's set on {}/{} token-layers.\n\n",
        report["model"].as_str().unwrap_or("?"),
        chunks.len(),
        n_tok - n_dec,
        moe_layers.len(),
        cfg.n_expert,
        checked - mismatch,
        checked
    ));
    md.push_str("## Pooled over layers\n\n| predictor | phase | n | recall@k | Jaccard | all-k hit |\n|---|---|---|---|---|---|\n");
    for &p in &preds {
        for ph in ["decode", "prefill", "all"] {
            let a = pooled(p, if ph == "all" { None } else { Some(ph) });
            md.push_str(&format!("| {p} | {ph} | {} | {} | {} | {} |\n", a.n, f(a.recall()), f(a.jaccard()), f(a.all_k())));
        }
    }
    md.push_str("\n## Per layer, decode tokens (recall@k / all-k)\n\n| layer | mid1 | out1 | mid2 | out2 | mid1+4 | prev | freq |\n|---|---|---|---|---|---|---|---|\n");
    for &l in &moe_layers {
        let cell = |p: &str| -> String {
            table
                .get(p)
                .and_then(|x| x.get("decode"))
                .and_then(|x| x.get(&l))
                .map(|a| format!("{} / {}", f(a.recall()), f(a.all_k())))
                .unwrap_or_else(|| "-".into())
        };
        md.push_str(&format!(
            "| {l} | {} | {} | {} | {} | {} | {} | {} |\n",
            cell("mid1"),
            cell("out1"),
            cell("mid2"),
            cell("out2"),
            cell("mid1+4"),
            cell("prev"),
            cell("freq")
        ));
    }
    md.push_str("\n## Confidence: keep the guesses with the largest gate margin (decode tokens)\n\n| predictor | keep | n | min margin | recall@k | all-k hit |\n|---|---|---|---|---|---|\n");
    for (name, rows) in &conf_out {
        if !name.ends_with("/decode") {
            continue;
        }
        for r in rows {
            md.push_str(&format!("| {} | {:.0}% | {} | {:.4} | {} | {} |\n", name.trim_end_matches("/decode"), r.keep * 100.0, r.n, r.min_margin, f(r.recall), f(r.all_k)));
        }
    }
    md.push_str("\n## Routing concentration (share of routings taken by the top-N experts of the layer, all tokens)\n\n| layer | top8 | top16 | top24 | top32 | top48 |\n|---|---|---|---|---|---|\n");
    for (l, c) in &concentration {
        md.push_str(&format!("| {l} | {} | {} | {} | {} | {} |\n", f(c["top8"]), f(c["top16"]), f(c["top24"]), f(c["top32"]), f(c["top48"])));
    }
    std::fs::File::create(out.join("report.md"))?.write_all(md.as_bytes())?;
    print!("{md}");
    Ok(())
}
