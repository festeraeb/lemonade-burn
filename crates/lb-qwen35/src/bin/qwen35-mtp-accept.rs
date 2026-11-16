//! qwen35-mtp-accept <model.gguf> <prompt.txt> <n_gen> : acceptance of the file's own nextn (MTP) head against the trunk's greedy
//! continuation (teacher-forced: the head is fed the trunk's real tokens), for draft depths 1..3, in four wiring variants
//! (embedding first or hidden first in the eh_proj input; RoPE position of the embedded token or one before). CPU reference decoder.
use anyhow::Result;
use lb_qwen35::{chat, tokenizer, MtpState};
use std::io::Write;

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[1]))?;
    let m = lb_qwen35::Model::load(&g)?;
    let mtp = m.load_mtp(&g)?;
    let tk = tokenizer::from_gguf(&g)?;
    let prompt = std::fs::read_to_string(&a[2])?;
    let n_gen: usize = a[3].parse()?;
    let text = chat::render(&[chat::Msg { role: "user", content: &prompt }], true, false);
    let ids = tokenizer::encode(&tk, &text)?;
    let mut st = m.new_state();
    // trunk: feed the prompt, then greedy-generate; keep each position's hidden and each position's argmax (the token after it)
    let mut hidden: Vec<Vec<f32>> = Vec::new();
    let mut toks: Vec<u32> = ids.clone();
    let mut next: Vec<u32> = Vec::new(); // next[t] = trunk argmax after consuming toks[t]
    let t0 = std::time::Instant::now();
    for (i, &t) in ids.iter().enumerate() {
        let l = m.forward(&mut st, t, true).unwrap();
        hidden.push(st.last_hidden.clone());
        next.push(argmax(&l));
        let _ = i;
    }
    eprintln!("prompt {} tokens in {:.1}s", ids.len(), t0.elapsed().as_secs_f64());
    let np = ids.len();
    for _ in 0..n_gen {
        let t = *next.last().unwrap();
        toks.push(t);
        let l = m.forward(&mut st, t, true).unwrap();
        hidden.push(st.last_hidden.clone());
        next.push(argmax(&l));
    }
    eprintln!("generated {} tokens, total {:.1}s", n_gen, t0.elapsed().as_secs_f64());
    let gen_text = tokenizer::decode(&tk, &toks[np..])?;
    eprintln!("--- generated text (first 400 chars)\n{}", gen_text.chars().take(400).collect::<String>());
    // MTP over the sequence. Draft for position t+1's successor: head(hidden[t], toks[t+1]) should equal next[t+1].
    let total = toks.len();
    let mut out = std::io::stdout();
    for &embed_first in &[true, false] {
        for &pos_off in &[1usize, 0usize] {
            let mut ms = MtpState::default();
            let mut hit = [0usize; 3];
            let mut cnt = [0usize; 3];
            let mut chain_ok = [0usize; 3]; // prefix of length d all right
            let mut chain_n = 0usize;
            for t in 0..total - 1 {
                // real step (teacher-forced), advances the head's KV
                let (lg, hid) = m.mtp_step(&mtp, &mut ms, &hidden[t], toks[t + 1], t + pos_off, embed_first);
                let d1 = argmax(&lg);
                if t >= np - 1 && t + 2 < total {
                    // generated region only: next[t+1] is the trunk's token at t+2
                    let want = [next[t + 1], if t + 2 < total { next[t + 2] } else { u32::MAX }, if t + 3 < total { next[t + 3] } else { u32::MAX }];
                    cnt[0] += 1;
                    if d1 == want[0] { hit[0] += 1; }
                    // chain: depth 2 and 3 on a copy of the KV
                    let mut ok_prefix = d1 == want[0];
                    chain_n += 1;
                    if ok_prefix { chain_ok[0] += 1; }
                    let (mut cm, mut ch, mut ct) = (ms.clone(), hid, d1);
                    for d in 1..3 {
                        if t + 2 + d >= total + 0 && want[d] == u32::MAX { break; }
                        let (lg2, hid2) = m.mtp_step(&mtp, &mut cm, &ch, ct, t + pos_off + d, embed_first);
                        let dd = argmax(&lg2);
                        cnt[d] += 1;
                        // per-depth accuracy with the chain fed its own drafts
                        if dd == want[d] { hit[d] += 1; }
                        ok_prefix = ok_prefix && dd == want[d];
                        if ok_prefix { chain_ok[d] += 1; }
                        ch = hid2; ct = dd;
                    }
                }
            }
            let r = |i: usize| hit[i] as f64 / cnt[i].max(1) as f64;
            let c = |i: usize| chain_ok[i] as f64 / chain_n.max(1) as f64;
            writeln!(out, "{{\"embed_first\":{embed_first},\"pos_off\":{pos_off},\"n\":{},\"acc_d1\":{:.3},\"acc_d2_chain\":{:.3},\"acc_d3_chain\":{:.3},\"prefix_ok_d1\":{:.3},\"prefix_ok_d2\":{:.3},\"prefix_ok_d3\":{:.3}}}", cnt[0], r(0), r(1), r(2), c(0), c(1), c(2))?;
        }
    }
    Ok(())
}
