//! Whole-model device checks (feature `cuda`). Cards: `CUDA_VISIBLE_DEVICES=uuid0,uuid1,...` in stage order; `QWEN_SPLIT=20,20`.
//!   tf    <model> <ids> <ref_batch.f32> <ref_single.f32> <out.jsonl> [--no-cpu]   teacher-forced: device vs CPU decoder, device vs
//!         llama.cpp (batch path), llama.cpp batch vs single (self-agreement), per position
//!   greedy <model> <ids> <n> <out.txt>                                             greedy continuation per prompt line
//!   bench <model> <ids> <n_new>                                                    decode tokens/s after the prompt (first line)
use anyhow::Result;
use lb_qwen35::{pipeline::Pipeline, Model};
use std::io::{Read, Seek, SeekFrom, Write};

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
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}
fn maxd(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}
fn load_row(f: &mut std::fs::File, idx: usize, vocab: usize) -> Result<Vec<f32>> {
    f.seek(SeekFrom::Start(idx as u64 * vocab as u64 * 4))?;
    let mut buf = vec![0u8; vocab * 4];
    f.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[2]))?;
    let m = Model::load(&g)?;
    let split: Vec<usize> = std::env::var("QWEN_SPLIT").unwrap_or_else(|_| format!("{}", m.cfg.n_layer)).split(',').map(|x| x.parse().unwrap()).collect();
    let cap: usize = std::env::var("QWEN_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let pipe = Pipeline::new(&m, &split, cap)?;
    let prompts = read_ids(&a[3]);
    match a[1].as_str() {
        "tf" => {
            let vocab = m.cfg.vocab;
            let mut rb = std::fs::File::open(&a[4])?;
            let mut rs = std::fs::File::open(&a[5])?;
            let mut out = std::fs::File::create(&a[6])?;
            let cpu = !a.iter().any(|x| x == "--no-cpu");
            let mut gp = 0usize;
            for (pi, p) in prompts.iter().enumerate() {
                pipe.reset();
                let mut st = m.new_state();
                for (i, &t) in p.iter().enumerate() {
                    let d = pipe.step(t, i, true)?.unwrap();
                    let b = load_row(&mut rb, gp + i, vocab)?;
                    let mut line = format!("{{\"prompt\":{pi},\"i\":{i},\"dev_top1\":{},\"llama_top1\":{},\"kl_llama_dev\":{},\"agree_dev_llama\":{}", argmax(&d), argmax(&b), kl(&b, &d), argmax(&d) == argmax(&b));
                    // the single-token reference may cover only the leading positions of the file
                    if rs.metadata()?.len() >= (gp + i + 1) as u64 * vocab as u64 * 4 {
                        let s = load_row(&mut rs, gp + i, vocab)?;
                        line += &format!(",\"llama_single_top1\":{},\"kl_batch_single\":{},\"agree_batch_single\":{}", argmax(&s), kl(&b, &s), argmax(&b) == argmax(&s));
                    }
                    if cpu {
                        let c = m.forward(&mut st, t, true).unwrap();
                        line += &format!(",\"cpu_top1\":{},\"agree_dev_cpu\":{},\"maxd_dev_cpu\":{},\"kl_cpu_dev\":{}", argmax(&c), argmax(&c) == argmax(&d), maxd(&c, &d), kl(&c, &d));
                    }
                    writeln!(out, "{line}}}")?;
                    out.flush()?;
                }
                gp += p.len();
                eprintln!("prompt {pi} done");
            }
        }
        "greedy" => {
            let n: usize = a[4].parse()?;
            let mut out = std::fs::File::create(&a[5])?;
            for p in &prompts {
                pipe.reset();
                let mut l = None;
                for (i, &t) in p.iter().enumerate() {
                    l = pipe.step(t, i, i + 1 == p.len())?;
                }
                let mut gen = Vec::new();
                let mut l = l.unwrap();
                for s in 0..n {
                    let t = argmax(&l) as u32;
                    gen.push(t);
                    if s + 1 < n {
                        l = pipe.step(t, p.len() + s, true)?.unwrap();
                    }
                }
                writeln!(out, "{}|{}", p.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "), gen.iter().map(|t| format!(" {t}")).collect::<String>())?;
            }
        }
        "bench" => {
            let n: usize = a[4].parse()?;
            let p = &prompts[0];
            for round in 0..2 {
                pipe.reset();
                let t0 = std::time::Instant::now();
                let mut l = None;
                for (i, &t) in p.iter().enumerate() {
                    l = pipe.step(t, i, i + 1 == p.len())?;
                }
                let pre = t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let mut l = l.unwrap();
                for s in 0..n {
                    let t = argmax(&l) as u32;
                    if s + 1 < n {
                        l = pipe.step(t, p.len() + s, true)?.unwrap();
                    }
                }
                let dt = t1.elapsed().as_secs_f64();
                eprintln!("round {round}: prompt {} tokens in {pre:.2}s ({:.1} tok/s); decode {} tokens in {dt:.2}s = {:.1} tok/s", p.len(), p.len() as f64 / pre, n - 1, (n - 1) as f64 / dt);
            }
        }
        "pcheck" => {
            // pcheck <model> <ids> <n> [chunk]: a prompt of n tokens (the ids lines concatenated, repeated) by prefill and by single steps; last-position
            // logits compared, then 24 greedy tokens continued from each state
            let n: usize = a[4].parse()?;
            let chunk: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(512);
            let mut toks: Vec<u32> = Vec::new();
            while toks.len() < n {
                for p in &prompts {
                    toks.extend_from_slice(p);
                }
            }
            toks.truncate(n);
            pipe.reset();
            let mut l1 = None;
            let t0 = std::time::Instant::now();
            for (i, &t) in toks.iter().enumerate() {
                l1 = pipe.step(t, i, i + 1 == toks.len())?;
            }
            let t_single = t0.elapsed().as_secs_f64();
            let l1 = l1.unwrap();
            let mut g1 = Vec::new();
            let mut l = l1.clone();
            for s in 0..24 {
                let t = argmax(&l) as u32;
                g1.push(t);
                l = pipe.step(t, toks.len() + s, true)?.unwrap();
            }
            pipe.reset();
            let t0 = std::time::Instant::now();
            let l2 = pipe.prefill(&toks, 0, chunk)?;
            let t_pre = t0.elapsed().as_secs_f64();
            let mut g2 = Vec::new();
            let mut l = l2.clone();
            for s in 0..24 {
                let t = argmax(&l) as u32;
                g2.push(t);
                l = pipe.step(t, toks.len() + s, true)?.unwrap();
            }
            eprintln!("n={n} chunk={chunk}: single steps {t_single:.2}s ({:.1} tok/s), prefill {t_pre:.2}s ({:.1} tok/s)", n as f64 / t_single, n as f64 / t_pre);
            eprintln!("last-position logits: top-1 {} vs {} ({}), max |diff| {:.3e}, KL {:.3e}; greedy continuation {}", argmax(&l1), argmax(&l2), if argmax(&l1) == argmax(&l2) { "same" } else { "DIFFERENT" }, maxd(&l1, &l2), kl(&l1, &l2), if g1 == g2 { "identical (24 tokens)".to_string() } else { format!("differs: {:?} vs {:?}", &g1[..8], &g2[..8]) });
        }
        "pbench" => {
            let n: usize = a[4].parse()?;
            let chunk: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(512);
            let mut toks: Vec<u32> = Vec::new();
            while toks.len() < n {
                for p in &prompts {
                    toks.extend_from_slice(p);
                }
            }
            toks.truncate(n);
            for round in 0..2 {
                pipe.reset();
                let t0 = std::time::Instant::now();
                let _ = pipe.prefill(&toks, 0, chunk)?;
                let dt = t0.elapsed().as_secs_f64();
                eprintln!("round {round}: prefill {n} tokens (chunk {chunk}) in {dt:.2}s = {:.1} tok/s", n as f64 / dt);
                for (name, s) in pipe.ptime_report() {
                    eprintln!("   {name:<16} {s:7.3} s");
                }
            }
        }
        "bench2" => {
            // bench2 <model> <ids> <n_new> [chunk] [rounds]: the server's path for every prompt line: prefill (prompts of 32+ tokens; chunk 0 =
            // auto) then `n_new` greedy tokens with the argmax on the device. One JSON line per prompt and round: prefill and decode tok/s, a sha
            // of the generated ids (token identity across builds and knobs).
            let n: usize = a[4].parse()?;
            let chunk: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(0);
            let rounds: usize = a.get(6).and_then(|v| v.parse().ok()).unwrap_or(2);
            pipe.warm_decode(prompts[0][0])?;
            for round in 0..rounds {
                for (pi, p) in prompts.iter().enumerate() {
                    pipe.reset();
                    let t0 = std::time::Instant::now();
                    let mut tok = if p.len() >= 32 {
                        argmax(&pipe.prefill(p, 0, chunk)?) as u32
                    } else {
                        let mut l = None;
                        for (i, &t) in p.iter().enumerate() {
                            l = pipe.step(t, i, i + 1 == p.len())?;
                        }
                        argmax(&l.unwrap()) as u32
                    };
                    let pre = t0.elapsed().as_secs_f64();
                    let mut gen = vec![tok];
                    let t1 = std::time::Instant::now();
                    for s in 1..n {
                        tok = pipe.step_argmax(tok, p.len() + s - 1)?;
                        gen.push(tok);
                    }
                    let dt = t1.elapsed().as_secs_f64();
                    let mut h: u64 = 0xcbf29ce484222325;
                    for &t in &gen {
                        for b in t.to_le_bytes() {
                            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                        }
                    }
                    println!(
                        "{{\"round\":{round},\"prompt\":{pi},\"len\":{},\"chunk\":{chunk},\"prefill_s\":{pre:.4},\"prefill_tps\":{:.1},\"n\":{n},\"decode_tps\":{:.2},\"sha\":\"{h:016x}\",\"head\":{:?}}}",
                        p.len(),
                        p.len() as f64 / pre,
                        (n - 1) as f64 / dt,
                        &gen[..gen.len().min(12)]
                    );
                }
            }
            // the stage graphs replayed alone (state is scratch by now: timing only)
            let g = pipe.bench_stage_graphs(50);
            println!("{{\"stage_graph_ms\":{:?}}}", g);
        }
        "sweep" => {
            // sweep <model> <ids>: LB_SWEEP="A=1,B=2;A=2,B=2;..." -- per configuration set those env vars in this process (the launch-time knobs of the
            // K-quant kernels) and time the parts of the LB_PARTS_LAYERS layers; one model load for the whole grid
            let layers: Vec<usize> = std::env::var("LB_PARTS_LAYERS").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![0, 3]);
            let grid = std::env::var("LB_SWEEP").unwrap_or_default();
            for cfg in grid.split(';').filter(|c| !c.is_empty()) {
                for kv in cfg.split(',') {
                    let (k, v) = kv.split_once('=').expect("A=B");
                    unsafe { std::env::set_var(k, v) };
                }
                for &il in &layers {
                    for (name, us) in pipe.bench_parts(il, 300) {
                        println!("{{\"cfg\":\"{cfg}\",\"layer\":{il},\"part\":\"{name}\",\"us\":{us:.2}}}");
                    }
                }
            }
        }
        "parts" => {
            // time the parts of one delta-net layer (0) and one attention layer (3) on stage 0 (LB_PARTS_LAYERS=a,b,.. for others), then the head
            let layers: Vec<usize> = std::env::var("LB_PARTS_LAYERS").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![0, 3]);
            for il in layers {
                let mut tot = 0.0;
                for (name, us) in pipe.bench_parts(il, 300) {
                    eprintln!("layer {il}: {name:<26} {us:8.1} us");
                    tot += us;
                }
                eprintln!("layer {il}: {:<26} {tot:8.1} us", "SUM");
            }
            for (name, us) in pipe.bench_head(100) {
                eprintln!("head: {name:<26} {us:8.1} us");
            }
        }
        _ => eprintln!("unknown mode"),
    }
    Ok(())
}
