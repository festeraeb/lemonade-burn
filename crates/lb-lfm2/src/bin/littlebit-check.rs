//! Teacher-forced comparison of an LFM2 GGUF against the same model with a LittleBit sidecar attached
//! (`lb_littlebit`): perplexity of each, mean KL(original ‖ littlebit) per position, top-1 agreement,
//! forward times; then a short greedy continuation from each.
//!
//! `littlebit-check --model m.gguf --sidecar m.lb.safetensors --tokenizer tokenizer.json --text-file t.txt`

use anyhow::{Context, Result};
use burn::tensor::{Device, Tensor};
use clap::Parser;
use lb_lfm2::model::Lfm2;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    sidecar: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    text_file: PathBuf,
    /// Use at most this many tokens of the text.
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    #[arg(long, default_value = "The search team launched the boat at dawn and")]
    greedy_prompt: String,
    #[arg(long, default_value_t = 32)]
    greedy_n: usize,
    /// Run the LittleBit layers on the GPU (build with --features gpu / cuda).
    #[arg(long, default_value_t = false)]
    gpu: bool,
}

fn rows(t: Tensor<2>) -> (usize, Vec<f32>) {
    let [_, v] = t.dims();
    (v, t.into_data().to_vec::<f32>().expect("f32 logits"))
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = x.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln() + mx;
    x.iter().map(|&v| v as f64 - lse).collect()
}

fn greedy(m: &Lfm2, tok: &tokenizers::Tokenizer, prompt: &str, n: usize) -> Result<String> {
    let ids = tok.encode(prompt, true).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let mut cache = m.new_cache();
    let (v, l) = rows(m.forward(&ids, &mut cache, false));
    let mut next = argmax(&l[l.len() - v..]);
    let mut out = vec![next];
    for _ in 1..n {
        let (_, l) = rows(m.forward(&[next], &mut cache, false));
        next = argmax(&l);
        out.push(next);
    }
    tok.decode(&out, true).map_err(|e| anyhow::anyhow!("{e}"))
}

fn argmax(x: &[f32]) -> u32 {
    x.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0)
}

fn main() -> Result<()> {
    let a = Args::parse();
    let tok = tokenizers::Tokenizer::from_file(&a.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let text = std::fs::read_to_string(&a.text_file).with_context(|| format!("read {}", a.text_file.display()))?;
    let mut ids = tok.encode(text.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    ids.truncate(a.max_tokens);
    let device = Device::default();

    let run = |m: &Lfm2| -> (usize, Vec<f32>, f64) {
        let t = Instant::now();
        let mut cache = m.new_cache();
        let (v, l) = rows(m.forward(&ids, &mut cache, true));
        (v, l, t.elapsed().as_secs_f64())
    };
    let base = Lfm2::load(&a.model, &device)?;
    let (v, lb_base, t_base) = run(&base);
    let g_base = greedy(&base, &tok, &a.greedy_prompt, a.greedy_n)?;
    drop(base);

    let mut lbm = Lfm2::load(&a.model, &device)?;
    let n = lbm.attach_littlebit(&a.sidecar)?;
    let mut where_ = "CPU reference kernels";
    if a.gpu {
        #[cfg(feature = "gpu")]
        {
            let (k, b) = lbm.littlebit_to_device().map_err(|e| anyhow::anyhow!(e))?;
            eprintln!("littlebit: {k} layers on the device ({:.1} MiB)", b as f64 / 1048576.0);
            where_ = "littlebit layers on the GPU, the rest on the CPU";
        }
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("--gpu needs the gpu feature");
    }
    let (_, lb_lb, t_lb) = run(&lbm);
    let g_lb = greedy(&lbm, &tok, &a.greedy_prompt, a.greedy_n)?;

    let (mut nll_b, mut nll_l, mut kld, mut agree) = (0f64, 0f64, 0f64, 0usize);
    let pos = ids.len() - 1;
    for i in 0..pos {
        let pb = log_softmax(&lb_base[i * v..(i + 1) * v]);
        let pl = log_softmax(&lb_lb[i * v..(i + 1) * v]);
        let next = ids[i + 1] as usize;
        nll_b -= pb[next];
        nll_l -= pl[next];
        kld += pb.iter().zip(&pl).map(|(b, l)| b.exp() * (b - l)).sum::<f64>();
        let ab = pb.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        let al = pl.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        agree += (ab == al) as usize;
    }
    let p = pos as f64;
    println!("sidecar {} ({} matrices replaced), {} tokens teacher-forced", a.sidecar.display(), n, ids.len());
    println!("perplexity: original {:.2}  littlebit {:.2}", (nll_b / p).exp(), (nll_l / p).exp());
    println!("mean KL(original || littlebit) {:.4} nats; top-1 agreement {:.1}%", kld / p, 100.0 * agree as f64 / p);
    println!("prompt pass: original {:.2} s (CPU), littlebit {:.2} s ({where_})", t_base, t_lb);
    println!("greedy original : {}", g_base.replace('\n', " "));
    println!("greedy littlebit: {}", g_lb.replace('\n', " "));
    Ok(())
}
