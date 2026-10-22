//! `lb-run generate --model DIR --prompt TEXT [--chat] [--max-tokens N]`
//! `lb-run compare  --model DIR --ref REFDIR`   (REFDIR from scripts/hf_reference.py)
//! `lb-run bench    --model DIR [--prompt-len N] [--gen N]`

use anyhow::{bail, Context, Result};
use burn::tensor::Device;
use clap::{Parser, Subcommand};
use lb_model::sample::{argmax, Sampler, SamplingParams};
use lb_model::text::{ChatMessage, StreamDecoder, TextCodec};
use lb_model::BitNet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        chat: bool,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
    },
    Compare {
        #[arg(long)]
        model: PathBuf,
        #[arg(long = "ref")]
        reference: PathBuf,
    },
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 128)]
        prompt_len: usize,
        #[arg(long, default_value_t = 64)]
        gen: usize,
        #[arg(long, default_value_t = 1)]
        reps: usize,
    },
}

fn to_vec(t: burn::tensor::Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().expect("f32 logits")
}

fn load(model: &Path) -> Result<(BitNet, TextCodec)> {
    let t0 = Instant::now();
    let dev = Device::flex();
    let m = BitNet::load(model, &dev)?;
    let codec = TextCodec::load(model, m.cfg.bos_token_id)?;
    eprintln!(
        "loaded in {:.2}s; kernel={} threads={} rss={:.0} MB",
        t0.elapsed().as_secs_f64(),
        lb_ternary::Kernel::detect().name(),
        rayon::current_num_threads(),
        rss_mb()
    );
    Ok((m, codec))
}

pub fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).map(|l| l.split_whitespace().nth(1).unwrap_or("0").parse::<f64>().unwrap_or(0.0)))
        .unwrap_or(0.0)
        / 1024.0
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Generate { model, prompt, chat, max_tokens, temperature } => {
            let (m, codec) = load(&model)?;
            let ids = if chat {
                codec.encode_chat(&[ChatMessage { role: "user".into(), content: prompt }])?
            } else {
                codec.encode(&prompt)?
            };
            let mut sampler = Sampler::new(SamplingParams { temperature, ..Default::default() });
            let mut cache = m.new_cache();
            let t0 = Instant::now();
            let logits = to_vec(m.forward(&ids, &mut cache, false));
            let tp = t0.elapsed().as_secs_f64();
            let mut next = sampler.sample(&logits);
            let mut dec = StreamDecoder::new();
            let t1 = Instant::now();
            let mut n = 0;
            while n < max_tokens && !TextCodec::is_stop(next) {
                print!("{}", dec.push(&codec, next)?);
                std::io::stdout().flush()?;
                n += 1;
                let l = to_vec(m.forward(&[next], &mut cache, false));
                next = sampler.sample(&l);
            }
            let td = t1.elapsed().as_secs_f64();
            println!();
            eprintln!(
                "prompt {} tok in {:.2}s ({:.1} tok/s); generated {} tok in {:.2}s ({:.2} tok/s); rss {:.0} MB",
                ids.len(),
                tp,
                ids.len() as f64 / tp,
                n,
                td,
                n as f64 / td,
                rss_mb()
            );
        }
        Cmd::Compare { model, reference } => compare(&model, &reference)?,
        Cmd::Bench { model, prompt_len, gen, reps } => {
            let (m, _codec) = load(&model)?;
            // deterministic synthetic prompt of ordinary tokens
            let ids: Vec<u32> = std::iter::once(m.cfg.bos_token_id).chain((0..prompt_len as u32 - 1).map(|i| 1000 + (i * 7919) % 30000)).collect();
            for r in 0..reps {
                lb_model::ops::prof_reset();
                let mut cache = m.new_cache();
                let t0 = Instant::now();
                let logits = to_vec(m.forward(&ids, &mut cache, false));
                let tp = t0.elapsed().as_secs_f64();
                let mut next = argmax(&logits);
                let t1 = Instant::now();
                for _ in 0..gen {
                    let l = to_vec(m.forward(&[next], &mut cache, false));
                    next = argmax(&l);
                }
                let td = t1.elapsed().as_secs_f64();
                let tern = lb_model::ops::PROF_TERNARY_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
                let bf = lb_model::ops::PROF_BF16_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
                eprintln!("  profile (prefill+decode): ternary ops {tern:.2}s, lm_head {bf:.2}s, other {:.2}s", tp + td - tern - bf);
                lb_model::ops::prof_reset();
                println!(
                    "rep {r}: prefill {} tok {:.2}s = {:.2} tok/s | decode {} tok {:.2}s = {:.2} tok/s | rss {:.0} MB | kernel {} threads {}",
                    ids.len(),
                    tp,
                    ids.len() as f64 / tp,
                    gen,
                    td,
                    gen as f64 / td,
                    rss_mb(),
                    lb_ternary::Kernel::detect().name(),
                    rayon::current_num_threads()
                );
            }
        }
    }
    Ok(())
}

fn read_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    if &b[..6] != b"\x93NUMPY" {
        bail!("not npy");
    }
    let (hlen, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes(b[8..12].try_into()?) as usize, 12) };
    let header = std::str::from_utf8(&b[off..off + hlen])?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        bail!("unsupported npy header {header}");
    }
    let s = header.split("'shape': (").nth(1).context("shape")?;
    let s = &s[..s.find(')').context("shape")?];
    let shape: Vec<usize> = s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect();
    let data: Vec<f32> = b[off + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((shape, data))
}

fn compare(model: &Path, refdir: &Path) -> Result<()> {
    let (m, codec) = load(model)?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refdir.join("prompts.json"))?)?;
    println!("reference dtype: {}", meta["dtype"]);
    let (mut agree_all, mut n_all, mut maxdiff_all) = (0usize, 0usize, 0f32);
    for (i, rec) in meta["records"].as_array().context("records")?.iter().enumerate() {
        let ids = |k: &str| -> Vec<u32> { rec[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
        let prompt_ids = ids("prompt_ids");
        let full = ids("full_ids");
        // tokenizer check
        let p = &rec["prompt"];
        let ours = if p["kind"] == "chat" {
            let msgs: Vec<ChatMessage> = p["messages"].as_array().unwrap().iter().map(|mm| ChatMessage { role: mm["role"].as_str().unwrap().into(), content: mm["content"].as_str().unwrap().into() }).collect();
            codec.encode_chat(&msgs)?
        } else {
            codec.encode(p["text"].as_str().unwrap())?
        };
        let tok_ok = ours == prompt_ids;

        let (shape, refl) = read_npy_f32(&refdir.join(format!("logits_{i}.npy")))?;
        let vocab = shape[1];
        assert_eq!(shape[0], full.len());
        let mut cache = m.new_cache();
        let t0 = Instant::now();
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let pl = to_vec(m.forward(&prompt_ids, &mut cache, true));
        rows.extend(pl.chunks_exact(vocab).map(|c| c.to_vec()));
        for &t in &full[prompt_ids.len()..] {
            rows.push(to_vec(m.forward(&[t], &mut cache, false)));
        }
        let el = t0.elapsed().as_secs_f64();
        let (mut agree, mut maxd, mut sumd, mut agree_margin, mut n_margin) = (0, 0f32, 0f64, 0, 0);
        let mut first_disagree = None;
        for (pos, row) in rows.iter().enumerate() {
            let r = &refl[pos * vocab..(pos + 1) * vocab];
            let (a, b) = (argmax(row), argmax(r));
            if a == b {
                agree += 1;
            } else if first_disagree.is_none() {
                first_disagree = Some(pos);
            }
            // margin of the reference's top-1 over its runner-up
            let mut top2 = f32::NEG_INFINITY;
            for (j, &v) in r.iter().enumerate() {
                if j as u32 != b && v > top2 {
                    top2 = v;
                }
            }
            if r[b as usize] - top2 > 0.5 {
                n_margin += 1;
                if a == b {
                    agree_margin += 1;
                }
            }
            for (x, y) in row.iter().zip(r) {
                let d = (x - y).abs();
                maxd = maxd.max(d);
                sumd += d as f64;
            }
        }
        let n = rows.len();
        println!(
            "[{i}] tokenizer {} | positions {n} | top1 agree {agree}/{n} (ref margin>0.5: {agree_margin}/{n_margin}) | max|dlogit| {maxd:.4} | mean|dlogit| {:.5} | first disagreement {:?} | {:.1}s",
            if tok_ok { "match" } else { "MISMATCH" },
            sumd / (n * vocab) as f64,
            first_disagree,
            el
        );
        agree_all += agree;
        n_all += n;
        maxdiff_all = maxdiff_all.max(maxd);
    }
    println!("TOTAL top1 agree {agree_all}/{n_all}, max|dlogit| {maxdiff_all:.4}");
    Ok(())
}
