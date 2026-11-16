//! qwen35-mtp-parity <model.gguf> <n_gen> <depth> [prompt files...] : greedy output with MTP speculation vs plain decode, token for token, on
//! the CPU reference; prints one JSON line per prompt (equal, first divergence, acceptance per depth).
use anyhow::Result;
use lb_qwen35::{chat, spec, tokenizer};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[1]))?;
    let m = lb_qwen35::Model::load(&g)?;
    let mtp = m.load_mtp(&g)?;
    let tk = tokenizer::from_gguf(&g)?;
    let n: usize = a[2].parse()?;
    let k: usize = a[3].parse()?;
    let prompts: Vec<String> = if a.len() > 4 { a[4..].iter().map(|p| std::fs::read_to_string(p).unwrap()).collect() } else {
        vec!["Write a Rust function that merges two sorted vectors and explain its complexity.".into(),
             "Describe the main causes of the French Revolution in a few paragraphs.".into(),
             "List the first twenty prime numbers and then write a Python function is_prime(n).".into()]
    };
    for (i, p) in prompts.iter().enumerate() {
        let text = chat::render(&[chat::Msg { role: "user", content: p }], true, false);
        let ids = tokenizer::encode(&tk, &text)?;
        let t0 = std::time::Instant::now();
        let plain = spec::plain_greedy(&m, &ids, n);
        let t1 = t0.elapsed().as_secs_f64();
        let (sp, st) = spec::mtp_greedy(&m, &mtp, &ids, n, k);
        let t2 = t0.elapsed().as_secs_f64() - t1;
        let div = plain.iter().zip(&sp).position(|(x, y)| x != y);
        let acc: Vec<f64> = st.accepted_at.iter().map(|&c| c as f64 / st.rounds.max(1) as f64).collect();
        println!("{{\"prompt\":{i},\"n\":{n},\"depth\":{k},\"equal\":{},\"first_divergence\":{},\"rounds\":{},\"tokens_per_round\":{:.3},\"prefix_accept_by_depth\":{:?},\"plain_s\":{t1:.1},\"spec_s\":{t2:.1}}}", div.is_none() && plain.len() == sp.len(), div.map(|d| d as i64).unwrap_or(-1), st.rounds, n as f64 / st.rounds as f64, acc);
    }
    Ok(())
}
