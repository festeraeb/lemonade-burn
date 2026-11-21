//! mimo2-check: per-layer parity of the CPU reference against llama.cpp activations, then greedy decoding.
//!
//! Reference activations: a directory of `<name>-<layer>.bin` raw f32 files as written by an eval callback over llama.cpp
//! (`/mnt/fast/lb-models-ref/run_mimo_ref.sh`). For every tapped name present in both, prints per layer the max over tokens
//! of the relative L2 error `|a - r| / |r|` and of `max|a - r| / max|r|`; compares the chosen experts per token; and
//! compares the final logits (top-5). Then decodes `--gen` tokens greedily, appending each one to `--out` (JSON lines), so
//! a restarted run resumes (`--resume`) by re-reading the prompt plus what was generated.
use anyhow::{Context, Result};
use clap::Parser;
use lb_gguf::GgufSet;
use lb_mimo2::{argmax, tokenizer, Cache, Model, Taps};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: Option<String>,
    /// comma-separated token ids (instead of --prompt)
    #[arg(long)]
    tokens: Option<String>,
    #[arg(long)]
    ref_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    gen: usize,
    /// expected greedy ids (comma-separated) to compare against
    #[arg(long)]
    expect: Option<String>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// comma-separated layers: check each sublayer on llama.cpp's own inputs (needs --ref-dir), then stop
    #[arg(long)]
    isolate: Option<String>,
    #[arg(long, env = "LB_THREADS", default_value_t = 16)]
    threads: usize,
}

fn read_f32(p: &std::path::Path) -> Option<Vec<f32>> {
    let b = std::fs::read(p).ok()?;
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn errs(a: &[f32], r: &[f32], d: usize) -> (f64, f64) {
    let (mut rel, mut mx) = (0f64, 0f64);
    for (ar, rr) in a.chunks_exact(d).zip(r.chunks_exact(d)) {
        let num: f64 = ar.iter().zip(rr).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
        let den: f64 = rr.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt().max(1e-30);
        let ma = ar.iter().zip(rr).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max);
        let mr = rr.iter().map(|y| y.abs() as f64).fold(0.0, f64::max).max(1e-30);
        rel = rel.max(num / den);
        mx = mx.max(ma / mr);
    }
    (rel, mx)
}

fn ids(s: &str) -> Vec<u32> {
    s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect()
}

fn main() -> Result<()> {
    let a = Args::parse();
    rayon::ThreadPoolBuilder::new().num_threads(a.threads).build_global().ok();
    let g = GgufSet::open(&a.model).context("open model")?;
    let model = Model::load(&g)?;
    let tk = tokenizer::from_gguf(g.first())?;
    let c = &model.cfg;
    eprintln!("mimo2: {} layers, embd {}, heads {} (kv {:?}), experts {} top-{}, kv_f16 {}, threads {}", c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.n_expert, c.n_used, model.kv_f16, a.threads);
    let prompt: Vec<u32> = match (&a.tokens, &a.prompt) {
        (Some(t), _) => ids(t),
        (None, Some(p)) => tokenizer::encode(&tk, p)?,
        _ => anyhow::bail!("--prompt or --tokens"),
    };
    println!("prompt ids {prompt:?}");
    let d = c.n_embd;
    let mut generated: Vec<u32> = Vec::new();
    if a.resume {
        if let Some(o) = &a.out {
            if let Ok(s) = std::fs::read_to_string(o) {
                for line in s.lines() {
                    if let Some(v) = line.split("\"id\":").nth(1) {
                        generated.push(v.trim_end_matches('}').trim().parse()?);
                    }
                }
                eprintln!("resume: {} tokens already generated", generated.len());
            }
        }
    }
    if let (Some(dir), Some(layers)) = (&a.ref_dir, &a.isolate) {
        // each sublayer fed llama.cpp's own input: attn_norm-L -> attention + attn_output vs attn_out-L,
        // ffn_norm-L -> router logits / experts vs ffn_moe_logits-L / ffn_moe_out-L
        let m = prompt.len();
        for il in ids(layers).into_iter().map(|v| v as usize) {
            let l = &model.layers[il];
            if let Some(x) = read_f32(&dir.join(format!("attn_norm-{il}.bin"))) {
                if x.len() == m * d {
                    let mut cache = Cache::new(c.n_layer);
                    let att = model.attention(il, &x, m, 0, &mut cache);
                    let o = lb_mimo2::mm(l.wo, &att, m);
                    if let Some(r) = read_f32(&dir.join(format!("attn_out-{il}.bin"))) {
                        let o: &[f32] = if r.len() == d { &o[(m - 1) * d..] } else { &o };
                        let (rel, mx) = errs(o, &r, d);
                        println!("isolated {il} | attn_out | {rel:.3e} | {mx:.3e}");
                    }
                }
            }
            if let (Some(x), Some(moe)) = (read_f32(&dir.join(format!("ffn_norm-{il}.bin"))), l.moe.as_ref()) {
                let mx_ = x.len() / d;
                let mut t = Taps::new();
                let o = model.moe(moe, &x, mx_, il, Some(&mut t));
                if let Some(r) = read_f32(&dir.join(format!("ffn_moe_logits-{il}.bin"))) {
                    let (rel, mx) = errs(&t[&format!("ffn_moe_logits-{il}")], &r, c.n_expert);
                    println!("isolated {il} | router logits | {rel:.3e} | {mx:.3e}");
                }
                if let Some(r) = read_f32(&dir.join(format!("ffn_moe_weights-{il}.bin"))) {
                    println!("isolated {il} | weights mine {:?}", &t[&format!("ffn_moe_weights-{il}")][..c.n_used]);
                    println!("isolated {il} | weights ref  {:?}", &r[..c.n_used.min(r.len())]);
                }
                if let Some(r) = read_f32(&dir.join(format!("ffn_moe_out-{il}.bin"))) {
                    for tt in 0..mx_ {
                        let (rel, mx) = errs(&o[tt * d..(tt + 1) * d], &r[tt * d..(tt + 1) * d], d);
                        println!("isolated {il} | ffn_moe_out token {tt} | {rel:.3e} | {mx:.3e}");
                    }
                }
            }
        }
        return Ok(());
    }
    let mut cache = Cache::new(c.n_layer);
    let mut taps = Taps::new();
    let t0 = Instant::now();
    let first: Vec<u32> = prompt.iter().chain(&generated).cloned().collect();
    let logits = model.forward(&first, &mut cache, if a.ref_dir.is_some() && generated.is_empty() { Some(&mut taps) } else { None });
    eprintln!("prefill {} tokens: {:.1} s", first.len(), t0.elapsed().as_secs_f64());

    if let (Some(dir), true) = (&a.ref_dir, generated.is_empty()) {
        let m = prompt.len();
        println!("layer | name | max rel L2 | max abs/max|ref| (over tokens)");
        let mut worst_lout = 0f64;
        for il in 0..c.n_layer {
            for name in ["attn_norm", "attn_out", "ffn_norm", "ffn_out", "ffn_moe_out", "l_out"] {
                let key = format!("{name}-{il}");
                let (Some(mine), Some(r)) = (taps.get(&key), read_f32(&dir.join(format!("{key}.bin")))) else { continue };
                // llama.cpp crops the last layer to the output token after attention
                let mine: &[f32] = if r.len() == d && mine.len() == m * d { &mine[(m - 1) * d..] } else { mine };
                if r.len() != mine.len() {
                    println!("{il} | {name} | size {} vs ref {}", mine.len(), r.len());
                    continue;
                }
                let (rel, mx) = errs(mine, &r, d);
                if name == "l_out" {
                    worst_lout = worst_lout.max(rel);
                }
                println!("{il} | {name} | {rel:.3e} | {mx:.3e}");
            }
            let key = format!("ffn_moe_topk-{il}");
            if let (Some(mine), Ok(raw)) = (taps.get(&key), std::fs::read(dir.join(format!("{key}.bin")))) {
                let r: Vec<i32> = raw.chunks_exact(4).map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
                let k = c.n_used;
                let mt = mine.len() / k;
                // the reference is a strided view: token t's ids at t * n_expert
                let stride = if mt > 1 { (r.len() - k) / (mt - 1) } else { k };
                let mut same = 0;
                for t in 0..mt {
                    let mut a: Vec<i32> = mine[t * k..(t + 1) * k].iter().map(|&v| v as i32).collect();
                    let mut b: Vec<i32> = r[t * stride..t * stride + k].to_vec();
                    a.sort();
                    b.sort();
                    if a == b {
                        same += 1;
                    } else {
                        println!("{il} | topk token {t} differs: {a:?} vs ref {b:?}");
                    }
                }
                println!("{il} | topk | {same}/{mt} tokens same expert set");
            }
        }
        if let Some(r) = read_f32(&dir.join("result_norm--1.bin")) {
            let (rel, mx) = errs(&taps["result_norm--1"], &r[r.len() - d..], d);
            println!("head | result_norm | {rel:.3e} | {mx:.3e}");
        }
        if let Some(r) = read_f32(&dir.join("result_output--1.bin")) {
            let v = c.vocab;
            let r = &r[r.len() - v..];
            let (rel, mx) = errs(&logits, r, v);
            let top = |x: &[f32]| {
                let mut i: Vec<usize> = (0..x.len()).collect();
                i.sort_by(|&p, &q| x[q].partial_cmp(&x[p]).unwrap());
                i.truncate(5);
                i.iter().map(|&j| (j, x[j])).collect::<Vec<_>>()
            };
            println!("head | logits | {rel:.3e} | {mx:.3e}");
            println!("top5 mine {:?}", top(&logits));
            println!("top5 ref  {:?}", top(r));
        }
        println!("worst l_out rel L2 over layers: {worst_lout:.3e}");
    }

    if a.gen > 0 {
        let mut out = a.out.as_ref().map(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap());
        let mut next = argmax(&logits) as u32;
        while generated.len() < a.gen {
            generated.push(next);
            if let Some(f) = out.as_mut() {
                writeln!(f, "{{\"i\":{},\"id\":{next}}}", generated.len() - 1)?;
                f.flush()?;
            }
            eprintln!("gen {} id {next} {:?} ({:.1} s)", generated.len(), tokenizer::decode(&tk, &[next])?, t0.elapsed().as_secs_f64());
            if generated.len() == a.gen {
                break;
            }
            let lg = model.forward(&[next], &mut cache, None);
            next = argmax(&lg) as u32;
        }
        println!("generated ids {generated:?}");
        println!("text {:?}", tokenizer::decode(&tk, &generated)?);
        if let Some(e) = &a.expect {
            let e = ids(e);
            let n = e.len().min(generated.len());
            let first_diff = (0..n).find(|&i| e[i] != generated[i]);
            println!("expect: {} compared, first difference at {:?}", n, first_diff);
        }
    }
    Ok(())
}
