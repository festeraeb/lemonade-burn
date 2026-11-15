//! qwen35-check <model.gguf> <ids.txt> <ref.f32> <out.jsonl>  teacher-forced agreement against a reference logits file
//! (`ref_logits ... logits`, llama.cpp), one JSONL line per position (resume point: positions already present are skipped).
//! qwen35-check <model.gguf> <ids.txt> --greedy <n> <out.txt>   greedy continuation of every prompt line.
use anyhow::Result;
use lb_qwen35::Model;
use std::io::Write;

fn read_ids(p: &str) -> Vec<Vec<u32>> {
    std::fs::read_to_string(p).unwrap().lines().map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect()).filter(|v: &Vec<u32>| !v.is_empty()).collect()
}

fn log_softmax(l: &[f32]) -> Vec<f64> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = l.iter().map(|&v| (v as f64 - m).exp()).sum();
    l.iter().map(|&v| v as f64 - m - s.ln()).collect()
}

fn argmax(l: &[f32]) -> usize {
    l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[1]))?;
    let m = Model::load(&g)?;
    let prompts = read_ids(&a[2]);
    if a[3] == "--greedy" {
        let n: usize = a[4].parse()?;
        let mut out = std::fs::File::create(&a[5])?;
        for p in &prompts {
            let mut st = m.new_state();
            let mut logits = None;
            for (i, &t) in p.iter().enumerate() {
                logits = m.forward(&mut st, t, i + 1 == p.len());
            }
            let mut gen = Vec::new();
            let mut l = logits.unwrap();
            for s in 0..n {
                let t = argmax(&l) as u32;
                gen.push(t);
                if s + 1 < n {
                    l = m.forward(&mut st, t, true).unwrap();
                }
            }
            writeln!(out, "{}|{}", p.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "), gen.iter().map(|t| format!(" {t}")).collect::<String>())?;
            out.flush()?;
            eprintln!("greedy prompt done ({} tokens)", p.len());
        }
        return Ok(());
    }
    let vocab = m.cfg.vocab;
    let mut rf = std::fs::File::open(&a[3])?;
    let done: usize = std::fs::read_to_string(&a[4]).map(|s| s.lines().count()).unwrap_or(0);
    let mut jl = std::fs::OpenOptions::new().create(true).append(true).open(&a[4])?;
    let mut refl = vec![0f32; vocab];
    let mut pos_global = 0usize;
    for (pi, p) in prompts.iter().enumerate() {
        // resume: if every position of this prompt is already done, skip it; recurrent state cannot be restored mid-prompt
        let only: Option<Vec<usize>> = std::env::var("QWEN_ONLY").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
        if only.as_ref().is_some_and(|o| !o.contains(&pi)) || pos_global + p.len() <= done {
            pos_global += p.len();
            continue;
        }
        let mut st = m.new_state();
        for (i, &t) in p.iter().enumerate() {
            let l = m.forward(&mut st, t, true).unwrap();
            let off = (pos_global + i) as u64 * vocab as u64 * 4;
            use std::io::{Read, Seek, SeekFrom};
            rf.seek(SeekFrom::Start(off))?;
            let mut buf = vec![0u8; vocab * 4];
            rf.read_exact(&mut buf)?;
            for (r, c) in refl.iter_mut().zip(buf.chunks_exact(4)) {
                *r = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            if pos_global + i < done {
                continue;
            }
            let (lo, lr) = (log_softmax(&l), log_softmax(&refl));
            // KL(ref || ours)
            let kl: f64 = lr.iter().zip(&lo).map(|(r, o)| r.exp() * (r - o)).sum();
            let maxd = l.iter().zip(&refl).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            let (to, tr) = (argmax(&l), argmax(&refl));
            // reference top-1 margin over its runner-up, to separate real disagreements from near ties
            let mut s2 = refl.clone();
            s2[tr] = f32::NEG_INFINITY;
            let margin = refl[tr] - s2[argmax(&s2)];
            writeln!(jl, "{{\"prompt\":{pi},\"i\":{i},\"top1_ours\":{to},\"top1_ref\":{tr},\"agree\":{},\"max_logit_diff\":{maxd},\"kl\":{kl},\"ref_margin\":{margin}}}", to == tr)?;
            jl.flush()?;
        }
        pos_global += p.len();
        eprintln!("prompt {pi} done");
    }
    Ok(())
}
