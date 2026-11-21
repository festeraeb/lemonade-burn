//! ds4-check: per-layer parity of the deepseek41 CPU reference against llama.cpp activations, then greedy decoding.
//!
//! Reference activations: a directory of `<name>-<layer>.bin` raw dumps written by an eval callback over llama.cpp
//! (`/mnt/fast/deepseek41-port/evalcb.cpp`: f32, or i32 for `ffn_moe_topk`; when several graph nodes share a name the
//! last one wins, e.g. `engram_kv` holds the 4 keys only and `comp_latent` of a ratio-2 layer carries one padding block).
//! For every tapped name present in both, prints per layer the max over tokens of the relative L2 error `|a - r| / |r|`
//! and of `max|a - r| / max|r|`; compares the chosen experts per token; and compares the final logits (top-5). Then
//! decodes `--gen` tokens greedily, appending each one to `--out` (JSON lines), so a restarted run resumes (`--resume`)
//! by re-reading the prompt plus what was generated. `--load-only` stops after loading (metadata and small F32 tensors).
use anyhow::{Context, Result};
use clap::Parser;
use lb_deepseek4::{argmax, tokenizer, types_used, Cache, Model, Taps, MAX_CTX};
use lb_gguf::GgufSet;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// shard 1 of the split GGUF (the other shards are found next to it)
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
    #[arg(long, env = "LB_THREADS", default_value_t = 16)]
    threads: usize,
    /// load the model and tokenizer, print the config and the prompt ids, and stop (no tensor data beyond small F32 ones)
    #[arg(long, default_value_t = false)]
    load_only: bool,
    /// comma-separated layers: run the output projection on llama.cpp's attn_derope-L and the FFN on its ffn_norm-L,
    /// compare with attn_out-L / ffn_moe_out-L / ffn_shexp-L, then stop (needs --ref-dir)
    #[arg(long)]
    isolate: Option<String>,
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

/// Tap names compared per layer, in graph order, with the per-row width used for the error (None = per token).
const LAYER_TAPS: [(&str, Option<usize>); 20] = [
    ("engram_kv", None),
    ("engram_gate", None),
    ("engram_out", None),
    ("hc_attn_pre", None),
    ("attn_norm", None),
    ("qr_norm", None),
    ("q", None),
    ("kv", None),
    ("comp_latent", Some(512)),
    ("attn_raw", None),
    ("attn_comp", None),
    ("attn_derope", None),
    ("attn_out", None),
    ("hc_ffn_pre", None),
    ("ffn_norm", None),
    ("ffn_moe_logits", None),
    ("ffn_moe_weights", None),
    ("ffn_moe_out", None),
    ("ffn_shexp", None),
    ("ffn_out", None),
];

fn main() -> Result<()> {
    let a = Args::parse();
    rayon::ThreadPoolBuilder::new().num_threads(a.threads).build_global().ok();
    let g = GgufSet::open(&a.model).context("open model")?;
    let model = Model::load(&g)?;
    let tk = tokenizer::from_gguf(g.first())?;
    let c = &model.cfg;
    eprintln!(
        "deepseek41: {} layers, embd {}, heads {} x {}, q_lora {}, experts {} top-{} (ff {}), hc {}, ratios {:?}, kv_src {:?}, engram layers {:?}, kv_f16 {}, act_emu {}, threads {}",
        c.n_layer,
        c.n_embd,
        c.n_head,
        c.head_dim,
        c.q_lora,
        c.n_expert,
        c.n_used,
        c.ff_exp,
        c.hc,
        c.ratio,
        c.kv_src,
        c.engram.iter().enumerate().filter_map(|(i, e)| e.map(|_| i)).collect::<Vec<_>>(),
        model.kv_f16,
        model.act_emu,
        a.threads
    );
    let prompt: Vec<u32> = match (&a.tokens, &a.prompt) {
        (Some(t), _) => ids(t),
        (None, Some(p)) => tokenizer::encode(&tk, p)?,
        _ => anyhow::bail!("--prompt or --tokens"),
    };
    println!("prompt ids {prompt:?}");
    if a.load_only {
        println!("types {:?}", types_used(&g).iter().map(|(t, n)| format!("{}x{}", t.name(), n)).collect::<Vec<_>>());
        println!("rope layer 0: {:?}", c.rope(0, c.head_dim - c.n_rot));
        println!("rope layer 2: {:?}", c.rope(2, c.head_dim - c.n_rot));
        println!("engram pad {} token_map len {} vocab {} ple rows {:?}", c.engram_pad, c.token_map.len(), c.vocab, model.ple.as_ref().map(|p| p.rows));
        println!("decode {:?}", tokenizer::decode(&tk, &prompt)?);
        return Ok(());
    }
    if prompt.len() + a.gen > MAX_CTX {
        anyhow::bail!("prompt {} + gen {} exceeds {MAX_CTX} tokens, the length this reference is valid for", prompt.len(), a.gen);
    }
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
        let d = c.n_embd;
        for il in ids(layers).into_iter().map(|v| v as usize) {
            if let (Some(x), Some(r)) = (read_f32(&dir.join(format!("attn_derope-{il}.bin"))), read_f32(&dir.join(format!("attn_out-{il}.bin")))) {
                let m = x.len() / (c.n_head * c.head_dim);
                let o = model.out_proj(il, &x, m);
                let o: &[f32] = if r.len() == d { &o[(m - 1) * d..] } else { &o };
                let (rel, mx) = errs(o, &r, d);
                println!("isolated {il} | attn_out from ref attn_derope | {rel:.3e} | {mx:.3e}");
            }
            if let Some(x) = read_f32(&dir.join(format!("ffn_norm-{il}.bin"))) {
                let m = x.len() / d;
                let mut t = Taps::new();
                let _ = model.ffn(il, &x, m, &mut Some(&mut t));
                for name in ["ffn_moe_logits", "ffn_moe_out", "ffn_shexp"] {
                    let key = format!("{name}-{il}");
                    if let (Some(mine), Some(r)) = (t.get(&key), read_f32(&dir.join(format!("{key}.bin")))) {
                        if mine.len() == r.len() {
                            let w = if name == "ffn_moe_logits" { c.n_expert } else { d };
                            let (rel, mx) = errs(mine, &r, w);
                            println!("isolated {il} | {name} from ref ffn_norm | {rel:.3e} | {mx:.3e}");
                        } else {
                            println!("isolated {il} | {name} size {} vs ref {}", mine.len(), r.len());
                        }
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
    let logits = model.forward(&first, &mut cache, if a.ref_dir.is_some() && generated.is_empty() { Some(&mut taps) } else { None })?;
    eprintln!("prefill {} tokens: {:.1} s", first.len(), t0.elapsed().as_secs_f64());

    if let (Some(dir), true) = (&a.ref_dir, generated.is_empty()) {
        let m = prompt.len();
        println!("layer | name | max rel L2 | max abs/max|ref| (over tokens)");
        let mut worst_lout = 0f64;
        let mut topk_same = (0usize, 0usize);
        for il in 0..c.n_layer {
            for (name, width) in LAYER_TAPS.iter().cloned().chain(std::iter::once(("l_out", None))) {
                let key = format!("{name}-{il}");
                let (Some(mine), Some(r)) = (taps.get(&key), read_f32(&dir.join(format!("{key}.bin")))) else { continue };
                let w = width.unwrap_or(mine.len() / m);
                let (mine, r): (&[f32], &[f32]) = if r.len() == mine.len() {
                    (mine, &r)
                } else if name == "comp_latent" && r.len() > mine.len() {
                    // llama.cpp pads the block count to ceil(n_tokens / ratio); the extra row is a dummy write
                    (mine, &r[..mine.len()])
                } else if r.len() * m == mine.len() {
                    // cropped to the output token
                    (&mine[mine.len() - r.len()..], &r)
                } else {
                    println!("{il} | {name} | size {} vs ref {}", mine.len(), r.len());
                    continue;
                };
                let (rel, mx) = errs(mine, r, w);
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
                // the reference is a strided view of the argsort: token t's ids at t * n_expert
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
                topk_same.0 += same;
                topk_same.1 += mt;
                println!("{il} | topk | {same}/{mt} tokens same expert set");
            }
        }
        for (name, n) in [("hc_out", d), ("result_norm", d)] {
            let key = format!("{name}--1");
            if let (Some(mine), Some(r)) = (taps.get(&key), read_f32(&dir.join(format!("{key}.bin")))) {
                let (rel, mx) = errs(mine, &r[r.len() - n..], n);
                println!("head | {name} | {rel:.3e} | {mx:.3e}");
            }
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
        println!("expert sets: {}/{} (token, layer) pairs agree", topk_same.0, topk_same.1);
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
            let lg = model.forward(&[next], &mut cache, None)?;
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
