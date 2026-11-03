//! Cross-quant check: dequantize the same tensor from two GGUFs of the same model and compare.
//!
//! This is how a new dequantizer is verified without a reference implementation to link
//! against. Two quantizations of the same weights must agree closely; a wrong bit layout
//! shows up immediately as a large error or a dead correlation, because quantization noise
//! is small next to any structural mistake.

use anyhow::{bail, Context, Result};
use lb_gguf::Gguf;
use std::path::Path;

pub struct Cmp {
    pub ty_a: String,
    pub ty_b: String,
    pub n: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    /// RMS of the difference over the RMS of `a`.
    pub rel_rms: f32,
    pub cosine: f32,
}

fn rows(g: &Gguf, name: &str, n_rows: usize) -> Result<(String, Vec<f32>)> {
    let t = g.tensor(name).with_context(|| format!("tensor {name} not in file"))?;
    let cols = t.dims[0];
    let rb = t.ty.row_bytes(cols).with_context(|| format!("{} rows of {cols} are not whole blocks", t.ty.name()))?;
    let bytes = g.tensor_bytes(t);
    let have = bytes.len() / rb;
    let n = n_rows.min(have);
    let mut out = vec![0f32; n * cols];
    for r in 0..n {
        t.ty.dequant_row(&bytes[r * rb..(r + 1) * rb], &mut out[r * cols..(r + 1) * cols]);
    }
    Ok((t.ty.name(), out))
}

pub fn compare(a: &Path, b: &Path, tensor: &str, n_rows: usize) -> Result<Cmp> {
    let (ga, gb) = (Gguf::open(a)?, Gguf::open(b)?);
    let (ty_a, va) = rows(&ga, tensor, n_rows)?;
    let (ty_b, vb) = rows(&gb, tensor, n_rows)?;
    if va.len() != vb.len() {
        bail!("different element counts: {} vs {}", va.len(), vb.len());
    }
    let n = va.len();
    let (mut max_abs, mut sum_abs, mut sd2, mut sa2, mut dot, mut na2, mut nb2) =
        (0f32, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (x, y) = (va[i], vb[i]);
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sd2 += (d as f64) * (d as f64);
        sa2 += (x as f64) * (x as f64);
        dot += (x as f64) * (y as f64);
        na2 += (x as f64) * (x as f64);
        nb2 += (y as f64) * (y as f64);
    }
    Ok(Cmp {
        ty_a,
        ty_b,
        n,
        max_abs,
        mean_abs: (sum_abs / n as f64) as f32,
        rel_rms: ((sd2 / sa2.max(1e-30)).sqrt()) as f32,
        cosine: (dot / (na2.sqrt() * nb2.sqrt()).max(1e-30)) as f32,
    })
}
