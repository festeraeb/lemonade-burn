//! GGUF -> LittleBit sidecar, init-only (no QAT): every block linear (`blk.N.*` 2-D weights except norms,
//! conv taps and routers) becomes a factorised layer at `--bpw`. Embeddings / tied head stay in the GGUF.
//!
//! `lb-littlebit-convert --model m.gguf --bpw 0.55 --out m.lb055.safetensors [--itq-iters 50] [--no-residual]`

use anyhow::{Context, Result};
use clap::Parser;
use lb_gguf::{Gguf, QuantMat};
use lb_littlebit::{convert, rank_for_bpw, sidecar};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Bits per weight budget for each converted matrix (paper accounting).
    #[arg(long)]
    bpw: f64,
    /// Joint-ITQ iterations (LittleBit-2 initialisation); 0 = LittleBit v1 initialisation.
    #[arg(long, default_value_t = 50)]
    itq_iters: usize,
    /// One path only (no residual).
    #[arg(long, default_value_t = false)]
    no_residual: bool,
    /// Only tensors whose name contains this.
    #[arg(long)]
    only: Option<String>,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Hybrid mode: routing counts `layer<TAB>expert<TAB>count` (from the scrape DB's histogram_counts).
    /// Only MoE experts are converted: per layer, the hottest experts covering `--keep-mass` of the
    /// routing stay as they are; every other expert (including never-seen ones) becomes LittleBit.
    #[arg(long)]
    experts_cold: Option<PathBuf>,
    #[arg(long, default_value_t = 0.9)]
    keep_mass: f64,
}

/// (layer, expert) pairs to convert: outside the hottest set covering `keep` of each layer's routing.
fn cold_experts(tsv: &std::path::Path, n_layer_experts: &BTreeMap<usize, usize>, keep: f64) -> Result<Vec<(usize, usize)>> {
    let mut counts: BTreeMap<usize, Vec<u64>> = n_layer_experts.iter().map(|(&l, &n)| (l, vec![0; n])).collect();
    for line in std::fs::read_to_string(tsv)?.lines() {
        let f: Vec<&str> = line.split(['\t', '|']).collect();
        if f.len() < 3 {
            continue;
        }
        let (Ok(l), Ok(e), Ok(c)) = (f[0].trim().parse::<usize>(), f[1].trim().parse::<usize>(), f[2].trim().parse::<u64>()) else { continue };
        if let Some(v) = counts.get_mut(&l) {
            if e < v.len() {
                v[e] += c;
            }
        }
    }
    let mut cold = Vec::new();
    for (l, v) in counts {
        let total: u64 = v.iter().sum();
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by_key(|&e| std::cmp::Reverse(v[e]));
        let mut acc = 0u64;
        let mut hot = 0;
        for &e in &order {
            if total > 0 && acc as f64 >= keep * total as f64 {
                break;
            }
            acc += v[e];
            hot += 1;
        }
        let hot_set: std::collections::HashSet<usize> = order[..hot].iter().copied().collect();
        eprintln!("layer {l:>2}: {hot}/{} experts carry {:.1}% of {} routings; {} converted", v.len(), if total > 0 { 100.0 * acc as f64 / total as f64 } else { 0.0 }, total, v.len() - hot);
        cold.extend((0..v.len()).filter(|e| !hot_set.contains(e)).map(|e| (l, e)));
    }
    Ok(cold)
}

const LINEARS: [&str; 9] = [
    "shortconv.in_proj.weight",
    "shortconv.out_proj.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

fn main() -> Result<()> {
    let a = Args::parse();
    let g = Gguf::open(&a.model).with_context(|| format!("open {}", a.model.display()))?;
    let paths = if a.no_residual { 1 } else { 2 };
    if let Some(tsv) = a.experts_cold.as_ref() {
        return convert_experts(&a, &g, tsv, paths);
    }
    let picked: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.") && t.dims.len() == 2 && LINEARS.iter().any(|s| t.name.ends_with(s)))
        .filter(|t| a.only.as_ref().map_or(true, |o| t.name.contains(o.as_str())))
        .collect();
    anyhow::ensure!(!picked.is_empty(), "no block linears found");
    eprintln!("{} matrices, bpw {}, {} path(s), ITQ {} iterations", picked.len(), a.bpw, paths, a.itq_iters);
    let t0 = Instant::now();
    let results: Vec<Result<(String, lb_littlebit::Linear, String)>> = picked
        .par_iter()
        .map(|t| {
            let (cols, rows) = (t.dims[0], t.dims[1]);
            let q = QuantMat::new(t.ty, rows, cols, g.tensor_bytes(t));
            let mut w = vec![0f32; rows * cols];
            for r in 0..rows {
                q.dequant_row(r, &mut w[r * cols..(r + 1) * cols]);
            }
            let r = rank_for_bpw(rows, cols, a.bpw, paths);
            let opts = convert::Opts { r, residual: paths == 2, itq_iters: a.itq_iters, seed: a.seed };
            let (lin, rel, st) = convert::convert(&w, rows, cols, &opts);
            let s0 = st[0];
            let line = format!(
                "{:<34} [{:>5} x {:>5}] r {:>4}  bpw {:.3}  rel.err {:.4}  flip-risk {:.2}% -> {:.2}%  corner {:.3} -> {:.3}",
                t.name,
                rows,
                cols,
                r,
                lin.bits_per_weight(),
                rel,
                100.0 * s0.risk_before,
                100.0 * s0.risk_after,
                s0.corner_before,
                s0.corner_after
            );
            eprintln!("{line}");
            Ok((t.name.clone(), lin, line))
        })
        .collect();
    let mut layers = BTreeMap::new();
    let mut report = Vec::new();
    for r in results {
        let (n, l, line) = r?;
        layers.insert(n, l);
        report.push(line);
    }
    report.sort();
    let mut meta = BTreeMap::new();
    meta.insert("source".into(), a.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    meta.insert("bpw_target".into(), a.bpw.to_string());
    meta.insert("itq_iters".into(), a.itq_iters.to_string());
    meta.insert("paths".into(), paths.to_string());
    meta.insert("method".into(), "init-only (Dual-SVID + Joint-ITQ from the papers' equations), no QAT".into());
    sidecar::write(&a.out, &layers, &meta)?;
    let bits: f64 = layers.values().map(|l| l.bits_per_weight() * (l.out * l.inp) as f64).sum();
    let n: usize = layers.values().map(|l| l.out * l.inp).sum();
    eprintln!(
        "wrote {} ({} layers, {:.1} MiB for {:.1} M weights = {:.3} bpw) in {:.1} s",
        a.out.display(),
        layers.len(),
        bits / 8.0 / 1048576.0,
        n as f64 / 1e6,
        bits / n as f64,
        t0.elapsed().as_secs_f64()
    );
    std::fs::write(a.out.with_extension("report.txt"), report.join("\n") + "\n")?;
    Ok(())
}

/// Hybrid mode: convert only the cold experts' gate/up/down, named `<stacked tensor>#e<id>`.
fn convert_experts(a: &Args, g: &Gguf, tsv: &std::path::Path, paths: usize) -> Result<()> {
    let exps: Vec<_> = g.tensors.iter().filter(|t| t.dims.len() == 3 && t.name.starts_with("blk.") && (t.name.ends_with("ffn_gate_exps.weight") || t.name.ends_with("ffn_up_exps.weight") || t.name.ends_with("ffn_down_exps.weight"))).collect();
    anyhow::ensure!(!exps.is_empty(), "no stacked expert tensors (blk.N.ffn_*_exps.weight) in this GGUF");
    let layer_of = |n: &str| n.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|v| v.parse::<usize>().ok()).unwrap();
    let n_layer_experts: BTreeMap<usize, usize> = exps.iter().map(|t| (layer_of(&t.name), t.dims[2])).collect();
    let cold = cold_experts(tsv, &n_layer_experts, a.keep_mass)?;
    let total: usize = n_layer_experts.values().sum();
    eprintln!("{} of {} experts converted (keep-mass {}), bpw {}, ITQ {}", cold.len(), total, a.keep_mass, a.bpw, a.itq_iters);
    let jobs: Vec<(&lb_gguf::TensorInfo, usize)> = cold
        .iter()
        .flat_map(|&(l, e)| exps.iter().filter(move |t| layer_of(&t.name) == l).map(move |t| (*t, e)))
        .filter(|(t, _)| a.only.as_ref().map_or(true, |o| t.name.contains(o.as_str())))
        .collect();
    eprintln!("{} matrices to convert", jobs.len());
    let t0 = Instant::now();
    let results: Vec<(String, lb_littlebit::Linear, f32)> = jobs
        .par_iter()
        .map(|(t, e)| {
            let (cols, rows, n) = (t.dims[0], t.dims[1], t.dims[2]);
            let q = QuantMat::new(t.ty, n * rows, cols, g.tensor_bytes(t));
            let mut w = vec![0f32; rows * cols];
            for r in 0..rows {
                q.dequant_row(e * rows + r, &mut w[r * cols..(r + 1) * cols]);
            }
            let r = rank_for_bpw(rows, cols, a.bpw, paths);
            let opts = convert::Opts { r, residual: paths == 2, itq_iters: a.itq_iters, seed: a.seed.wrapping_add(*e as u64) };
            let t1 = Instant::now();
            let (lin, rel, st) = convert::convert(&w, rows, cols, &opts);
            eprintln!("{}#e{} [{} x {}] r {} rel.err {:.4} flip-risk {:.2}% -> {:.2}% in {:.1} s", t.name, e, rows, cols, r, rel, 100.0 * st[0].risk_before, 100.0 * st[0].risk_after, t1.elapsed().as_secs_f64());
            (format!("{}#e{}", t.name, e), lin, rel)
        })
        .collect();
    let mean_rel = results.iter().map(|r| r.2 as f64).sum::<f64>() / results.len().max(1) as f64;
    let mut layers = BTreeMap::new();
    for (n, l, _) in results {
        layers.insert(n, l);
    }
    let mut meta = BTreeMap::new();
    meta.insert("source".into(), a.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    meta.insert("bpw_target".into(), a.bpw.to_string());
    meta.insert("itq_iters".into(), a.itq_iters.to_string());
    meta.insert("keep_mass".into(), a.keep_mass.to_string());
    meta.insert("cold_experts".into(), cold.len().to_string());
    meta.insert("method".into(), "hybrid: cold experts only, init-only (Dual-SVID + Joint-ITQ), no QAT".into());
    sidecar::write(&a.out, &layers, &meta)?;
    eprintln!("wrote {} ({} matrices, mean rel.err {:.3}) in {:.1} s", a.out.display(), layers.len(), mean_rel, t0.elapsed().as_secs_f64());
    Ok(())
}
