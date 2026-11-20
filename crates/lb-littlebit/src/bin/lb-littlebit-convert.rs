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
