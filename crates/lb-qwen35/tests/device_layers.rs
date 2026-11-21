//! Device layers against the CPU decoder, on real activations (feature `cuda`; run on a GPU host with `LB_CUDA_DEVICE=0`).
//! Env: QWEN_GGUF (model), QWEN_IDS (a line of token ids), QWEN_NL (layers, default 12), QWEN_NTOK (tokens, default 24).
//!
//! Phase 1, isolated: layer L's device copy is fed the CPU decoder's own input of layer L for every token (its recurrent state and KV
//! cache evolve on the device from those inputs); outputs (after the mixer, after the MoE block) are compared with the CPU's.
//! Phase 2, chain: the device runs layers 0..NL one after another from layer 0's input with its own states, compared with the CPU's
//! layer outputs at every layer, so error growth is visible.
#![cfg(feature = "cuda")]
use lb_qwen35::{device::DevModel, Model};
use std::collections::HashMap;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

struct Stat {
    max_abs: f64,
    max_rel: f64,
}

fn cmp(got: &[f32], want: &[f32]) -> (f64, f64) {
    let maxabs = got.iter().zip(want).fold(0f64, |m, (a, b)| m.max((*a as f64 - *b as f64).abs()));
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (maxabs, (num / den.max(1e-30)).sqrt())
}

#[test]
fn device_layers_match_cpu_decoder() {
    let path = env("QWEN_GGUF", "/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf");
    if !std::path::Path::new(&path).exists() {
        eprintln!("skip: no model");
        return;
    }
    let ntok: usize = env("QWEN_NTOK", "24").parse().unwrap();
    let nl: usize = env("QWEN_NL", "12").parse().unwrap();
    let from: usize = env("QWEN_FROM", "0").parse().unwrap();
    let ids: Vec<u32> = std::fs::read_to_string(env("QWEN_IDS", "tf_ids.txt")).unwrap().lines().next().unwrap().split_whitespace().take(ntok).map(|t| t.parse().unwrap()).collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let m = Model::load(&g).unwrap();
    // CPU pass, recording per token and layer
    let mut st = m.new_state();
    let mut rec: Vec<HashMap<(String, usize), Vec<f32>>> = Vec::new();
    for &t in &ids {
        let mut h = HashMap::new();
        m.forward_ex(&mut st, t, false, &mut |name, il, v| {
            if il >= from && il < from + nl && matches!(name, "layer_in" | "attn_residual" | "post_moe") {
                h.insert((name.to_string(), il), v.to_vec());
            }
        });
        rec.push(h);
    }
    eprintln!("cpu pass done: {} tokens", ids.len());
    let gpu = lb_q4_gpu::Gpu::open().unwrap();
    let dev = DevModel::upload(&gpu, &m, from..from + nl, 64, 0).unwrap();
    eprintln!("uploaded {nl} layers");
    let kind = |l: usize| if m.cfg.is_recurrent(from + l) { "delta-net" } else { "attention" };
    let n = m.cfg.n_embd;
    let mut worst: HashMap<&str, (f64, f64)> = HashMap::new();
    eprintln!("--- isolated (CPU input to each device layer)");
    for l in 0..nl {
        dev.reset(&gpu);
        let (mut mid, mut out) = (Stat { max_abs: 0.0, max_rel: 0.0 }, Stat { max_abs: 0.0, max_rel: 0.0 });
        for (pos, h) in rec.iter().enumerate() {
            let x = gpu.upload_f32(&h[&("layer_in".to_string(), from + l)]);
            dev.set_pos(&gpu, pos);
            let (x1, y) = dev.layer_forward(&gpu, l, &x, pos);
            let r = gpu.read_all(vec![(x1, n), (y, n)]).unwrap();
            let (a, b) = cmp(&r[0], &h[&("attn_residual".to_string(), from + l)]);
            mid.max_abs = mid.max_abs.max(a);
            mid.max_rel = mid.max_rel.max(b);
            let (a, b) = cmp(&r[1], &h[&("post_moe".to_string(), from + l)]);
            out.max_abs = out.max_abs.max(a);
            out.max_rel = out.max_rel.max(b);
        }
        eprintln!("layer {:2} {:<9} after mixer: max|d| {:.2e} rel L2 {:.2e} | layer out: max|d| {:.2e} rel L2 {:.2e}", from + l, kind(l), mid.max_abs, mid.max_rel, out.max_abs, out.max_rel);
        let w = worst.entry(kind(l)).or_insert((0.0, 0.0));
        w.0 = w.0.max(out.max_abs);
        w.1 = w.1.max(out.max_rel);
    }
    for (k, (a, r)) in &worst {
        eprintln!("worst isolated layer output, {k}: max|d| {a:.2e} rel L2 {r:.2e}");
    }
    eprintln!("--- chain of {nl} layers on the device from layer 0's input");
    dev.reset(&gpu);
    let mut chain = vec![(0f64, 0f64); nl];
    for (pos, h) in rec.iter().enumerate() {
        let mut x = gpu.upload_f32(&h[&("layer_in".to_string(), from)]);
        for l in 0..nl {
            dev.set_pos(&gpu, pos);
            let (_x1, y) = dev.layer_forward(&gpu, l, &x, pos);
            let r = gpu.read_all(vec![(y.clone(), n)]).unwrap();
            let (a, b) = cmp(&r[0], &h[&("post_moe".to_string(), from + l)]);
            chain[l].0 = chain[l].0.max(a);
            chain[l].1 = chain[l].1.max(b);
            x = gpu.upload_f32(&r[0]);
        }
    }
    for (l, (a, r)) in chain.iter().enumerate() {
        eprintln!("after layer {:2}: max|d| {a:.2e} rel L2 {r:.2e}", from + l);
    }
    assert!(worst.values().all(|(_, r)| *r < 1e-3), "isolated layer error too large");
    assert!(chain.iter().all(|(_, r)| *r < 2e-2), "chain error too large");
}
