//! Prompt-prefix reuse check: rewinding to a checkpoint must give the same logits as never
//! having left it.
//!
//!   prefix_check <model.gguf> [n_a] [n_x] [n_b]
//!
//! Reference: a fresh cache runs A then B. Test: a cache runs A, checkpoints, runs X one token
//! at a time (as decode would), rewinds, runs B. Prints the max abs logit difference of B's last
//! row and exits 1 unless it is 0 (the host path is deterministic).
use lb_lfm2::model::Lfm2;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(a.get(1).expect("model.gguf"));
    let n = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (na, nx, nb) = (n(2, 12), n(3, 5), n(4, 6));
    let model = Lfm2::load(&path, &Default::default())?;
    let vocab = model.cfg().n_vocab as u64;
    let mut s = 12345u64;
    let mut tok = |k: usize| -> Vec<u32> {
        (0..k)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % (vocab - 100) + 50) as u32
            })
            .collect()
    };
    let (ta, tx, tb) = (tok(na), tok(nx), tok(nb));
    let last = |t: burn::tensor::Tensor<2>| -> Vec<f32> { t.into_data().try_to_vec::<f32>().expect("logits") };

    let mut c0 = model.new_cache();
    model.forward(&ta, &mut c0, false);
    let want = last(model.forward(&tb, &mut c0, false));

    let mut c1 = model.new_cache();
    model.forward(&ta, &mut c1, false);
    let cp = model.checkpoint(&mut c1).map_err(anyhow::Error::msg)?;
    for t in &tx {
        model.forward(&[*t], &mut c1, false);
    }
    let wandered = last(model.forward(&tb[..1], &mut c1, false));
    model.rewind(&mut c1, &cp).map_err(anyhow::Error::msg)?;
    let got = last(model.forward(&tb, &mut c1, false));

    let diff = want.iter().zip(&got).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
    let off = want.iter().zip(&wandered).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
    println!("checkpoint at {} ({} bytes); A {na} X {nx} B {nb} tokens", cp.pos, cp.bytes());
    println!("max |logit diff| after rewind: {diff:e}   (without rewind the state differs by {off:e})");
    if diff != 0.0 {
        println!("FAIL");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
