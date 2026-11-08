//! Expert stand-in finder (`docs/EXPERT_PREDICT.md`, goal 2): for one MoE layer at a time,
//! which other expert (or which router-weighted pair of experts) can fill in for expert `i`
//! if `i` is dropped or moved to slow storage.
//!
//! Inputs are the router inputs (`rms(x_mid, ffn_norm)`) that the engine actually routed to
//! each expert during a capture run (`lb-expert-capture`). Up to `--per-expert` of them per
//! expert, spread evenly over the run; every expert of the layer runs on all of them; then
//! for each `i` every `j` is scored on `i`'s samples by cosine and relative L2 error
//! (`lb_expert_predict::standin`). Also scored per `i`:
//! * `top2`: the two best single stand-ins blended by the router's own probabilities for them
//!   on each token (the "almost expert");
//! * `router_next`: the expert the router itself would pick next if `i` were masked (the best
//!   unchosen expert by selection score), i.e. what dropping `i` does today with no planning.
//!
//! Per layer it writes `layer-LL.jsonl` (all (i, j) pairs and the per-expert rows) and
//! `layer-LL.summary.json`, then `standin-summary.md` over every finished layer. A layer whose
//! summary exists is skipped, so the same command resumes.

use anyhow::Result;
use clap::Parser;
use lb_expert_predict::capture::{RunDir, Which};
use lb_expert_predict::metrics::{rms_norm, router_scores, top_share};
use lb_expert_predict::standin::{best_two, blend2, score_table, DenseExpert, PairAcc};
use lb_gguf::Gguf;
use lb_lfm2::weights::{ExpertMats, Ffn, Weights};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Capture run directory.
    #[arg(long)]
    run: PathBuf,
    /// Output directory (default `<run>/standin`).
    #[arg(long)]
    out: Option<PathBuf>,
    /// `all` or a comma list of layer indices.
    #[arg(long, default_value = "all")]
    layers: String,
    /// Samples per expert (evenly spread over the run's routings of that expert).
    #[arg(long, default_value_t = 48)]
    per_expert: usize,
    /// Only rebuild standin-summary.md from the finished layers.
    #[arg(long, default_value_t = false)]
    summary_only: bool,
}

/// NaN (an expert with no samples) is written as JSON null; read null back as NaN.
fn nan_or<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

#[derive(Serialize, Deserialize, Clone)]
struct ExpertRow {
    i: usize,
    routed: u64,
    #[serde(deserialize_with = "nan_or")]
    share: f64,
    n: u64,
    best: usize,
    #[serde(deserialize_with = "nan_or")]
    best_cos: f64,
    #[serde(deserialize_with = "nan_or")]
    best_rel: f64,
    second: usize,
    #[serde(deserialize_with = "nan_or")]
    top2_cos: f64,
    #[serde(deserialize_with = "nan_or")]
    top2_rel: f64,
    #[serde(deserialize_with = "nan_or")]
    next_cos: f64,
    #[serde(deserialize_with = "nan_or")]
    next_rel: f64,
    /// The whole MoE layer output on i's tokens, compared with the full top-k output when i is
    /// dropped and the other weights renormalized (`drop`), when the router's next pick takes
    /// its place (`lnext`), and when the best single stand-in takes its place at i's weight (`lbest`).
    #[serde(deserialize_with = "nan_or")]
    drop_cos: f64,
    #[serde(deserialize_with = "nan_or")]
    drop_rel: f64,
    #[serde(deserialize_with = "nan_or")]
    lnext_cos: f64,
    #[serde(deserialize_with = "nan_or")]
    lnext_rel: f64,
    #[serde(deserialize_with = "nan_or")]
    lbest_cos: f64,
    #[serde(deserialize_with = "nan_or")]
    lbest_rel: f64,
}

#[derive(Serialize, Deserialize)]
struct LayerSummary {
    layer: usize,
    tokens: u64,
    per_expert: usize,
    experts: Vec<ExpertRow>,
    concentration: HashMap<String, f64>,
    /// Max |engine Q4 kernel - f32 copy| / |engine| over the check rows (dequant sanity).
    #[serde(deserialize_with = "nan_or")]
    kernel_check_rel: f64,
    secs: f64,
}

fn dequant(w: &Weights, m: &ExpertMats, e: usize) -> Vec<f32> {
    let q = w.mat(&m.expert(e));
    let mut out = vec![0f32; m.rows * m.cols];
    out.par_chunks_mut(m.cols).enumerate().for_each(|(r, dst)| q.dequant_row(r, dst));
    out
}

fn layer_run(args: &Args, w: &Weights, run: &RunDir, layer: usize, out: &std::path::Path) -> Result<()> {
    let t0 = Instant::now();
    let cfg = &w.cfg;
    let (e, ne) = (cfg.n_embd, cfg.n_expert);
    let lw = &w.layers[layer];
    let Ffn::Moe { router, bias, gate, up, down } = &lw.ffn else { anyhow::bail!("layer {layer} is not MoE") };
    let chunks = run.list_done()?;
    // every routing of this layer: expert -> [(chunk, token)]
    let mut routed: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
    let mut tokens = 0u64;
    for (ci, m) in chunks.iter().enumerate() {
        for t in 0..m.n_tokens() {
            tokens += 1;
            for &x in &m.routes[t][layer] {
                routed[x as usize].push((ci, t));
            }
        }
    }
    let counts: Vec<u64> = routed.iter().map(|v| v.len() as u64).collect();
    let total: u64 = counts.iter().sum();
    // evenly spaced samples per expert, deduplicated into one row set U
    let mut row_of: HashMap<(usize, usize), usize> = HashMap::new();
    let mut rows_ref: Vec<(usize, usize)> = Vec::new();
    let samples: Vec<Vec<usize>> = routed
        .iter()
        .map(|v| {
            let n = v.len().min(args.per_expert);
            (0..n)
                .map(|s| {
                    let key = v[s * v.len() / n];
                    *row_of.entry(key).or_insert_with(|| {
                        rows_ref.push(key);
                        rows_ref.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    let nu = rows_ref.len();
    // router inputs and router probabilities for every row
    let states: Vec<_> = chunks.iter().map(|m| run.open_states(m)).collect::<Result<_>>()?;
    let mut u = vec![0f32; nu * e];
    u.par_chunks_mut(e).zip(&rows_ref).for_each(|(dst, &(ci, t))| {
        dst.copy_from_slice(&rms_norm(&states[ci].row(t, layer, Which::Mid), &lw.ffn_norm, cfg.rms_eps));
    });
    let scores: Vec<(Vec<f32>, Vec<f32>)> = u.par_chunks(e).map(|h| router_scores(router, bias, h)).collect();
    eprintln!("layer {layer}: {tokens} tokens, {nu} sample rows; running {ne} experts");

    let outputs: Vec<Vec<f32>> = (0..ne)
        .map(|j| {
            let ex = DenseExpert { n_embd: e, n_ff: gate.rows, gate: dequant(w, gate, j), up: dequant(w, up, j), down: dequant(w, down, j) };
            ex.forward(&u, nu)
        })
        .collect();

    // dequant sanity: the engine's Q4 kernel on 4 rows of expert 0's samples vs the f32 copy
    let mut kernel_check_rel = 0f64;
    if let Some(i) = (0..ne).find(|&i| !samples[i].is_empty()) {
        for &r in samples[i].iter().take(4) {
            let y = lb_lfm2::moe::expert_ffn(w, gate, up, down, i, &u[r * e..(r + 1) * e], 1);
            let rel = lb_expert_predict::standin::rel_l2(&outputs[i][r * e..(r + 1) * e], &y);
            kernel_check_rel = kernel_check_rel.max(rel);
        }
    }

    let table = score_table(&outputs, &samples, e);
    let mut jl = String::new();
    let mut experts = Vec::with_capacity(ne);
    for i in 0..ne {
        for j in 0..ne {
            let a = table[i][j];
            if i != j && a.n > 0 {
                jl.push_str(&format!(
                    "{{\"layer\":{layer},\"i\":{i},\"j\":{j},\"cos\":{:.5},\"rel_err\":{:.5},\"n_samples\":{}}}\n",
                    a.cos(),
                    a.rel_err(),
                    a.n
                ));
            }
        }
        let (j1, j2) = best_two(&table[i], i);
        let (mut top2, mut next) = (PairAcc::default(), PairAcc::default());
        let (mut ldrop, mut lnext, mut lbest) = (PairAcc::default(), PairAcc::default(), PairAcc::default());
        for &r in &samples[i] {
            let yi = &outputs[i][r * e..(r + 1) * e];
            let (probs, sel) = &scores[r];
            let b = blend2(&outputs[j1][r * e..(r + 1) * e], probs[j1], &outputs[j2][r * e..(r + 1) * e], probs[j2]);
            top2.add(&b, yi);
            // the expert the router picks if i is masked: best selection score outside the token's set
            let (ci, t) = rows_ref[r];
            let chosen: Vec<usize> = chunks[ci].routes[t][layer].iter().map(|&x| x as usize).collect();
            let nx = (0..ne).filter(|x| !chosen.contains(x)).max_by(|&a, &b| sel[a].total_cmp(&sel[b]).then(b.cmp(&a)));
            if let Some(nx) = nx {
                next.add(&outputs[nx][r * e..(r + 1) * e], yi);
            }
            // layer-level: weighted sums over the token's set, weights = router probs normalized
            // over the experts used (as route_sigmoid does)
            let mix = |set: &[(usize, f32)]| -> Vec<f32> {
                let s: f32 = set.iter().map(|x| x.1).sum::<f32>().max(6.103515625e-5);
                let mut o = vec![0f32; e];
                for &(x, wt) in set {
                    for (d, v) in o.iter_mut().zip(&outputs[x][r * e..(r + 1) * e]) {
                        *d += wt / s * v;
                    }
                }
                o
            };
            let full_set: Vec<(usize, f32)> = chosen.iter().map(|&x| (x, probs[x])).collect();
            let full = mix(&full_set);
            let rest: Vec<(usize, f32)> = full_set.iter().copied().filter(|x| x.0 != i).collect();
            ldrop.add(&mix(&rest), &full);
            if let Some(nx) = nx {
                let mut s2 = rest.clone();
                s2.push((nx, probs[nx]));
                lnext.add(&mix(&s2), &full);
            }
            // best stand-in at i's own weight (output scaled as i's share of the normalized sum)
            let s: f32 = full_set.iter().map(|x| x.1).sum::<f32>().max(6.103515625e-5);
            let wi = probs[i] / s;
            let mut ob = full.clone();
            for ((d, a), b) in ob.iter_mut().zip(&outputs[i][r * e..(r + 1) * e]).zip(&outputs[j1][r * e..(r + 1) * e]) {
                *d += wi * (b - a);
            }
            lbest.add(&ob, &full);
        }
        let row = ExpertRow {
            i,
            routed: counts[i],
            share: if total > 0 { counts[i] as f64 / total as f64 } else { 0.0 },
            n: samples[i].len() as u64,
            best: j1,
            best_cos: table[i][j1].cos(),
            best_rel: table[i][j1].rel_err(),
            second: j2,
            top2_cos: top2.cos(),
            top2_rel: top2.rel_err(),
            next_cos: next.cos(),
            next_rel: next.rel_err(),
            drop_cos: ldrop.cos(),
            drop_rel: ldrop.rel_err(),
            lnext_cos: lnext.cos(),
            lnext_rel: lnext.rel_err(),
            lbest_cos: lbest.cos(),
            lbest_rel: lbest.rel_err(),
        };
        jl.push_str(&format!("{}\n", serde_json::json!({"layer": layer, "kind": "expert", "row": row})));
        experts.push(row);
    }
    let concentration: HashMap<String, f64> = [8usize, 16, 24, 32, 48, 64].iter().map(|&n| (format!("top{n}"), top_share(&counts, n))).collect();
    let summary = LayerSummary { layer, tokens, per_expert: args.per_expert, experts, concentration, kernel_check_rel, secs: t0.elapsed().as_secs_f64() };
    let jp = out.join(format!("layer-{layer:02}.jsonl"));
    std::fs::write(jp.with_extension("jsonl.tmp"), jl)?;
    std::fs::rename(jp.with_extension("jsonl.tmp"), &jp)?;
    let sp = out.join(format!("layer-{layer:02}.summary.json"));
    std::fs::write(sp.with_extension("json.tmp"), serde_json::to_vec(&summary)?)?;
    std::fs::rename(sp.with_extension("json.tmp"), &sp)?;
    eprintln!("layer {layer}: done in {:.0}s (kernel check rel {:.2e})", summary.secs, kernel_check_rel);
    Ok(())
}

fn write_summary(out: &std::path::Path, n_layer: usize) -> Result<()> {
    let mut layers: Vec<LayerSummary> = Vec::new();
    for l in 0..n_layer {
        if let Ok(b) = std::fs::read(out.join(format!("layer-{l:02}.summary.json"))) {
            layers.push(serde_json::from_slice(&b)?);
        }
    }
    let f = |x: f64| format!("{x:.3}");
    let mut md = String::from("# Expert stand-in finder: summary\n\n");
    if let Some(l0) = layers.first() {
        md.push_str(&format!("{} layers analysed, {} tokens per layer, up to {} samples per expert. Scores are means over expert i's samples of cos(y_j, y_i) and |y_j - y_i| / |y_i|.\n\n", layers.len(), l0.tokens, l0.per_expert));
    }
    md.push_str("## Per layer: how many experts have a stand-in\n\n| layer | experts | best single >=0.90 | >=0.95 | top-2 blend >=0.90 | >=0.95 | router-next mean cos | median best cos | layer out, i dropped: median cos / rel_err | router-next fills in | best stand-in fills in | top-16 share | top-32 share | top-48 share | kernel check |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    let (mut tot, mut s90, mut s95, mut b90, mut b95) = (0, 0, 0, 0, 0);
    for s in &layers {
        let c = |th: f64, top2: bool| s.experts.iter().filter(|x| if top2 { x.top2_cos >= th } else { x.best_cos >= th }).count();
        let mut best: Vec<f64> = s.experts.iter().map(|x| x.best_cos).filter(|x| x.is_finite()).collect();
        best.sort_by(|a, b| a.total_cmp(b));
        let med = best.get(best.len() / 2).copied().unwrap_or(f64::NAN);
        let next: Vec<f64> = s.experts.iter().map(|x| x.next_cos).filter(|x| x.is_finite()).collect();
        let next_mean = next.iter().sum::<f64>() / next.len().max(1) as f64;
        let med_of = |f: &dyn Fn(&ExpertRow) -> f64| -> f64 {
            let mut v: Vec<f64> = s.experts.iter().map(f).filter(|x| x.is_finite()).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let pair = |c: &dyn Fn(&ExpertRow) -> f64, r: &dyn Fn(&ExpertRow) -> f64| format!("{:.3} / {:.3}", med_of(c), med_of(r));
        let (dp, nxp, bp) = (pair(&|x| x.drop_cos, &|x| x.drop_rel), pair(&|x| x.lnext_cos, &|x| x.lnext_rel), pair(&|x| x.lbest_cos, &|x| x.lbest_rel));
        tot += s.experts.len();
        s90 += c(0.9, false);
        s95 += c(0.95, false);
        b90 += c(0.9, true);
        b95 += c(0.95, true);
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1e} |\n",
            s.layer,
            s.experts.len(),
            c(0.9, false),
            c(0.95, false),
            c(0.9, true),
            c(0.95, true),
            f(next_mean),
            f(med),
            dp,
            nxp,
            bp,
            f(s.concentration["top16"]),
            f(s.concentration["top32"]),
            f(s.concentration["top48"]),
            s.kernel_check_rel
        ));
    }
    md.push_str(&format!("\nAll layers: {tot} experts; best single stand-in >=0.90: {s90}, >=0.95: {s95}; top-2 blend >=0.90: {b90}, >=0.95: {b95}.\n"));
    md.push_str("\n## Best stand-in per expert (ranked by best single cosine, per layer)\n\n");
    for s in &layers {
        md.push_str(&format!("### Layer {}\n\n| expert | routed share | n | best j | cos | rel_err | 2nd j | top-2 cos | top-2 rel_err | router-next cos | layer rel_err: drop / next / best |\n|---|---|---|---|---|---|---|---|---|---|---|\n", s.layer));
        let mut ex = s.experts.clone();
        let key = |x: f64| if x.is_nan() { f64::NEG_INFINITY } else { x };
        ex.sort_by(|a, b| key(b.best_cos).total_cmp(&key(a.best_cos)));
        for x in &ex {
            md.push_str(&format!(
                "| {} | {:.4} | {} | {} | {} | {} | {} | {} | {} | {} | {:.3} / {:.3} / {:.3} |\n",
                x.i,
                x.share,
                x.n,
                x.best,
                f(x.best_cos),
                f(x.best_rel),
                x.second,
                f(x.top2_cos),
                f(x.top2_rel),
                f(x.next_cos),
                x.drop_rel,
                x.lnext_rel,
                x.lbest_rel
            ));
        }
        let none: Vec<String> = s.experts.iter().filter(|x| !(x.best_cos >= 0.9)).map(|x| x.i.to_string()).collect();
        md.push_str(&format!("\nNo single stand-in at cos >= 0.9 ({} experts): {}\n\n", none.len(), none.join(", ")));
    }
    std::fs::File::create(out.join("standin-summary.md"))?.write_all(md.as_bytes())?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let run = RunDir::new(&args.run)?;
    let out = args.out.clone().unwrap_or_else(|| args.run.join("standin"));
    std::fs::create_dir_all(&out)?;
    let w = Weights::load(Arc::new(Gguf::open(&args.model)?))?;
    let n_layer = w.cfg.n_layer;
    if !args.summary_only {
        let layers: Vec<usize> = if args.layers == "all" {
            (0..n_layer).filter(|&l| w.cfg.is_moe(l)).collect()
        } else {
            args.layers.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?
        };
        for l in layers {
            if out.join(format!("layer-{l:02}.summary.json")).exists() {
                eprintln!("layer {l}: done, skipping");
                continue;
            }
            layer_run(&args, &w, &run, l, &out)?;
            write_summary(&out, n_layer)?;
        }
    }
    write_summary(&out, n_layer)?;
    eprintln!("summary: {}", out.join("standin-summary.md").display());
    Ok(())
}
