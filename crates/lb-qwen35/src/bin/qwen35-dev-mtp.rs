//! qwen35-dev-mtp <model.gguf> <n_gen> <depths e.g. 0,1,2,3> [prompt.txt ...] : device greedy decode, plain (depth 0) vs MTP speculation, per prompt:
//! token-for-token equality with the plain run, decode tokens/s, rounds and acceptance. One JSON line per (prompt, depth).
//! Cards: `CUDA_VISIBLE_DEVICES=uuid0,...` in stage order, `QWEN_SPLIT`, `QWEN_CAP` (KV positions, default 8192), `LB_QW_PREFILL_CHUNK`.
use anyhow::Result;
use lb_qwen35::{chat, pipeline::Pipeline, spec::argmax, spec_dev::SpecDev, tokenizer, Model};
use std::sync::Mutex;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[1]))?;
    let m = Model::load(&g)?;
    let tk = tokenizer::from_gguf(&g)?;
    let n: usize = a[2].parse()?;
    let depths: Vec<usize> = a[3].split(',').map(|x| x.parse().unwrap()).collect();
    let split: Vec<usize> = std::env::var("QWEN_SPLIT").unwrap_or_else(|_| format!("{}", m.cfg.n_layer)).split(',').map(|x| x.parse().unwrap()).collect();
    let cap: usize = std::env::var("QWEN_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    let chunk: usize = std::env::var("LB_QW_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let maxd = depths.iter().cloned().max().unwrap_or(0);
    let mut pipe = Pipeline::new_spec(&m, &split, cap, if maxd > 0 { maxd + 1 } else { 0 })?;
    if maxd > 0 {
        let mtp = m.load_mtp(&g)?;
        let t = std::time::Instant::now();
        pipe.attach_mtp(&mtp)?;
        eprintln!("nextn block uploaded in {:.1}s", t.elapsed().as_secs_f64());
    }
    pipe.warm_decode(g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(248044) as u32)?;
    pipe.warm_spec(g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(248044) as u32)?;
    let prompts: Vec<(String, String)> = if a.len() > 4 {
        a[4..].iter().map(|p| (p.clone(), std::fs::read_to_string(p).unwrap())).collect()
    } else {
        vec![("code".into(), "Write a Rust function that merges two sorted vectors and explain its complexity.".into()),
             ("prose".into(), "Describe the main causes of the French Revolution in a few paragraphs.".into()),
             ("primes".into(), "List the first twenty prime numbers and then write a Python function is_prime(n).".into())]
    };
    for (name, p) in &prompts {
        let text = chat::render(&[chat::Msg { role: "user", content: p }], true, false);
        let ids = tokenizer::encode(&tk, &text)?;
        let mut plain: Vec<u32> = Vec::new();
        for &d in &depths {
            if d > 0 {
                // warm-up (compiles kernels, records the graphs) so the timed run measures steady state
                pipe.reset();
                let rows = Mutex::new(Vec::new());
                let lg = pipe.prefill_h(&ids, 0, chunk, Some(&rows))?;
                let mut sd = SpecDev::new(d);
                sd.after_prefill(&pipe, &ids, 0, &rows.into_inner().unwrap())?;
                let (mut next, mut len) = (argmax(&lg), ids.len());
                for _ in 0..12 {
                    next = sd.next(&pipe, next, len)?;
                    len += 1;
                }
            }
            pipe.reset();
            let rows = Mutex::new(Vec::new());
            let t0 = std::time::Instant::now();
            let lg = pipe.prefill_h(&ids, 0, chunk, Some(&rows))?;
            let rows = rows.into_inner().unwrap();
            let mut sd = SpecDev::new(d.max(1));
            if d > 0 {
                sd.after_prefill(&pipe, &ids, 0, &rows)?;
            }
            let prefill_s = t0.elapsed().as_secs_f64();
            let mut next = argmax(&lg);
            let mut out = Vec::new();
            let t1 = std::time::Instant::now();
            let mut len = ids.len();
            for _ in 0..n {
                out.push(next);
                next = if d == 0 { pipe.step_argmax(next, len)? } else { sd.next(&pipe, next, len)? };
                len += 1;
            }
            let dec_s = t1.elapsed().as_secs_f64();
            if lb_qwen35::device::ptime_on() {
                eprintln!("ptime depth {d}: {:?}", pipe.ptime_report());
            }
            if d > 0 {
                sd.trim(&pipe, len);
            }
            if d == 0 {
                plain = out.clone();
            }
            let div = if d == 0 { None } else { plain.iter().zip(&out).position(|(x, y)| x != y) };
            let s = &sd.stats;
            let acc: Vec<String> = (0..d).map(|i| format!("{:.3}", s.prefix_ok[i] as f64 / s.rounds.max(1) as f64)).collect();
            println!("{{\"prompt\":\"{name}\",\"prompt_tokens\":{},\"n\":{n},\"depth\":{d},\"prefill_s\":{prefill_s:.2},\"decode_s\":{dec_s:.3},\"tok_s\":{:.2},\"equal_to_plain\":{},\"first_divergence\":{},\"rounds\":{},\"tokens_per_round\":{:.3},\"prefix_accept\":[{}],\"draft_ms_per_round\":{:.2},\"verify_ms_per_round\":{:.2},\"fixup_ms_per_round\":{:.2}}}",
                ids.len(), n as f64 / dec_s, d == 0 || div.is_none(), div.map(|x| x as i64).unwrap_or(-1), s.rounds, s.emitted as f64 / s.rounds.max(1) as f64, acc.join(","),
                s.draft_s * 1e3 / s.rounds.max(1) as f64, s.verify_s * 1e3 / s.rounds.max(1) as f64, s.fixup_s * 1e3 / s.rounds.max(1) as f64);
        }
    }
    Ok(())
}
