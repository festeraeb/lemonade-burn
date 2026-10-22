//! `lb-convert --src <hf model dir> --dst <out dir> [--check <quant_summary.json>]`
//!
//! Reads the bf16 master weights of BitNet b1.58 2B4T, ternarizes every BitLinear weight
//! exactly as `transformers`' `WeightQuant` does, packs it 4 weights/byte and writes the
//! lemonade-burn v1 safetensors file. With `--check`, each tensor's ternary values (sha256
//! of the int8 array) and scale are compared with the torch-computed summary written by
//! `scripts/hf_reference.py`.

use anyhow::{bail, Context, Result};
use clap::Parser;
use lb_convert::*;
use lb_ternary::pack::{absmean_f32, ternarize_with_scale, PackedTernary};
use rayon::prelude::*;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    src: PathBuf,
    #[arg(long)]
    dst: PathBuf,
    /// quant_summary.json from scripts/hf_reference.py
    #[arg(long)]
    check: Option<PathBuf>,
}

struct Out {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

struct QInfo {
    scale: f32,
    sha: String,
    counts: [usize; 3],
}

fn main() -> Result<()> {
    let a = Args::parse();
    let src = a.src.join("model.safetensors");
    let file = std::fs::File::open(&src).with_context(|| format!("open {}", src.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    std::fs::create_dir_all(&a.dst)?;
    let t0 = std::time::Instant::now();

    let mut names: Vec<String> = st.names().into_iter().map(String::from).collect();
    names.sort();
    let results: Vec<Result<(Vec<Out>, Option<(String, QInfo)>)>> = names
        .par_iter()
        .map(|name| -> Result<(Vec<Out>, Option<(String, QInfo)>)> {
            let tv = st.tensor(name)?;
            if tv.dtype() != Dtype::BF16 {
                bail!("{name}: expected BF16, got {:?}", tv.dtype());
            }
            let shape = tv.shape().to_vec();
            if is_bitlinear_weight(name) {
                let (rows, cols) = (shape[0], shape[1]);
                let w = bf16_bytes_to_f32(tv.data());
                let mean = absmean_f32(&w);
                let scale = 1.0f32 / mean.max(1e-5);
                let t = ternarize_with_scale(&w, scale);
                let mut counts = [0usize; 3];
                for &v in &t {
                    counts[(v + 1) as usize] += 1;
                }
                let sha = hex(&Sha256::digest(bytemuck_i8(&t)));
                let p = PackedTernary::from_ternary(rows, cols, &t, scale);
                let linear = name.trim_end_matches(".weight").to_string();
                let rb = p.data.len() / rows;
                Ok((
                    vec![
                        Out { name: qweight_name(&linear), dtype: Dtype::U8, shape: vec![rows, rb], data: p.data },
                        Out {
                            name: wscale_name(&linear),
                            dtype: Dtype::F32,
                            shape: vec![1],
                            data: scale.to_le_bytes().to_vec(),
                        },
                    ],
                    Some((name.clone(), QInfo { scale, sha, counts })),
                ))
            } else if name.ends_with("norm.weight") {
                let w = bf16_bytes_to_f32(tv.data());
                let data = w.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok((vec![Out { name: name.clone(), dtype: Dtype::F32, shape, data }], None))
            } else if name == "model.embed_tokens.weight" {
                Ok((vec![Out { name: name.clone(), dtype: Dtype::BF16, shape, data: tv.data().to_vec() }], None))
            } else {
                bail!("unexpected tensor {name}")
            }
        })
        .collect();

    let mut outs = Vec::new();
    let mut qinfo = BTreeMap::new();
    for r in results {
        let (o, q) = r?;
        outs.extend(o);
        if let Some((n, q)) = q {
            qinfo.insert(n, q);
        }
    }
    eprintln!("ternarized {} BitLinear tensors in {:.1}s", qinfo.len(), t0.elapsed().as_secs_f64());

    let mut meta = HashMap::new();
    meta.insert("format".to_string(), FORMAT.to_string());
    meta.insert("version".to_string(), VERSION.to_string());
    meta.insert("quant".to_string(), "weights: absmean per-tensor ternary (WeightQuant); acts: per-token absmax int8 at runtime".to_string());
    meta.insert("layout".to_string(), "row-major, 128-weight blocks of 32 bytes, bits 2k..2k+1 of byte j = code(w+1) of weight k*32+j".to_string());
    let views: Vec<(String, TensorView)> = outs
        .iter()
        .map(|o| Ok((o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data)?)))
        .collect::<Result<_>>()?;
    let dst_file = a.dst.join(FILE_NAME);
    safetensors::serialize_to_file(views, Some(meta), &dst_file)?;
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "LICENSE"] {
        let s = a.src.join(f);
        if s.exists() {
            std::fs::copy(&s, a.dst.join(f))?;
        }
    }
    let size = std::fs::metadata(&dst_file)?.len();
    eprintln!("wrote {} ({:.1} MB) in {:.1}s", dst_file.display(), size as f64 / 1e6, t0.elapsed().as_secs_f64());

    let total: [usize; 3] = qinfo.values().fold([0; 3], |acc, q| [acc[0] + q.counts[0], acc[1] + q.counts[1], acc[2] + q.counts[2]]);
    let n: usize = total.iter().sum();
    eprintln!(
        "ternary distribution: -1 {:.2}%  0 {:.2}%  +1 {:.2}%  ({} weights)",
        100.0 * total[0] as f64 / n as f64,
        100.0 * total[1] as f64 / n as f64,
        100.0 * total[2] as f64 / n as f64,
        n
    );

    if let Some(check) = a.check {
        let summary: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&check)?)?;
        let (mut ok, mut bad_sha, mut bad_scale) = (0, 0, 0);
        for (name, q) in &qinfo {
            let s = &summary[name];
            if s.is_null() {
                bail!("{name} missing from {}", check.display());
            }
            let ref_scale = s["scale"].as_f64().unwrap() as f32;
            let ref_sha = s["sha256"].as_str().unwrap();
            let scale_ok = ref_scale.to_bits() == q.scale.to_bits();
            let sha_ok = ref_sha == q.sha;
            if !scale_ok {
                bad_scale += 1;
                eprintln!("  scale mismatch {name}: ours {:e} torch {:e}", q.scale, ref_scale);
            }
            if !sha_ok {
                bad_sha += 1;
                eprintln!("  ternary mismatch {name}");
            }
            if scale_ok && sha_ok {
                ok += 1;
            }
        }
        eprintln!("check vs torch: {ok}/{} identical, {bad_scale} scale mismatches, {bad_sha} ternary mismatches", qinfo.len());
        if bad_sha > 0 {
            std::process::exit(2);
        }
    }
    Ok(())
}

fn bytemuck_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
