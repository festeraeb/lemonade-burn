//! A tiny synthetic `lfm2moe` GGUF for tests: every tensor the trunk (`lb-lfm2`) and a host
//! need, with deterministic pseudo-random weights, written in a few milliseconds. It lets the
//! whole work-to-weights path (forward pass, hosts, transports, mark-down) run under
//! `cargo test` with no 13 GB model file. The format is the GGUF v3 layout `lb-gguf` reads.

use lb_gguf::types::quantize_q4_0;
use std::io::Write;
use std::path::Path;

/// Shape of the synthetic model.
#[derive(Clone, Debug)]
pub struct Tiny {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    /// Per layer; 0 = short-conv layer.
    pub kv_heads: Vec<usize>,
    pub n_ff: usize,
    pub n_dense_lead: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_vocab: usize,
    pub l_cache: usize,
}

impl Default for Tiny {
    /// Five layers: one dense lead, four MoE (one of them attention), 8 experts top-3, d = 64.
    fn default() -> Self {
        Self { n_layer: 5, n_embd: 64, n_head: 2, kv_heads: vec![0, 0, 1, 0, 0], n_ff: 96, n_dense_lead: 1, n_expert: 8, n_expert_used: 3, n_ff_exp: 32, n_vocab: 96, l_cache: 3 }
    }
}

struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * scale).collect()
    }
}

enum Val {
    U32(u32),
    F32(f32),
    Str(String),
    U32s(Vec<u32>),
}

struct W {
    kv: Vec<(String, Val)>,
    /// name, dims (GGUF order), type id, data
    tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)>,
}

fn s(out: &mut Vec<u8>, x: &str) {
    out.extend_from_slice(&(x.len() as u64).to_le_bytes());
    out.extend_from_slice(x.as_bytes());
}

impl W {
    fn f32(&mut self, name: String, dims: Vec<usize>, v: &[f32]) {
        assert_eq!(dims.iter().product::<usize>(), v.len(), "{name}");
        self.tensors.push((name, dims, 0, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }

    /// `rows` rows of `cols` values, Q4_0; `dims` is the GGUF shape (cols first).
    fn q4(&mut self, name: String, dims: Vec<usize>, cols: usize, v: &[f32]) {
        assert_eq!(dims.iter().product::<usize>(), v.len(), "{name}");
        assert_eq!(dims[0], cols);
        self.tensors.push((name, dims, 2, v.chunks_exact(cols).flat_map(quantize_q4_0).collect()));
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            s(&mut out, k);
            match v {
                Val::U32(x) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&x.to_le_bytes());
                }
                Val::F32(x) => {
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&x.to_le_bytes());
                }
                Val::Str(x) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    s(&mut out, x);
                }
                Val::U32s(xs) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        let mut offset = 0usize;
        for (name, dims, ty, data) in &self.tensors {
            s(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset += data.len().div_ceil(32) * 32;
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
        for (_, _, _, data) in &self.tensors {
            out.extend_from_slice(data);
            out.resize(out.len().div_ceil(32) * 32, 0);
        }
        out
    }
}

/// Write the synthetic model to `path`. The same `(tiny, seed)` gives the same bytes.
pub fn write(path: &Path, t: &Tiny, seed: u64) -> std::io::Result<()> {
    let mut r = Rng(seed.wrapping_mul(2654435761).wrapping_add(12345));
    let mut w = W { kv: Vec::new(), tensors: Vec::new() };
    let a = "lfm2moe";
    let (e, hd) = (t.n_embd, t.n_embd / t.n_head);
    let u = |k: &str, v: usize| (format!("{a}.{k}"), Val::U32(v as u32));
    w.kv.push(("general.architecture".into(), Val::Str(a.into())));
    w.kv.extend([
        u("block_count", t.n_layer),
        u("embedding_length", e),
        u("feed_forward_length", t.n_ff),
        u("attention.head_count", t.n_head),
        u("expert_count", t.n_expert),
        u("expert_used_count", t.n_expert_used),
        u("expert_feed_forward_length", t.n_ff_exp),
        u("leading_dense_block_count", t.n_dense_lead),
        u("expert_gating_func", 2),
        u("vocab_size", t.n_vocab),
        u("context_length", 256),
        u("shortconv.l_cache", t.l_cache),
    ]);
    w.kv.push((format!("{a}.attention.head_count_kv"), Val::U32s(t.kv_heads.iter().map(|&x| x as u32).collect())));
    w.kv.push((format!("{a}.attention.layer_norm_rms_epsilon"), Val::F32(1e-5)));
    w.kv.push((format!("{a}.rope.freq_base"), Val::F32(10000.0)));
    w.kv.push(("tokenizer.ggml.bos_token_id".into(), Val::U32(1)));
    w.kv.push(("tokenizer.ggml.eos_token_id".into(), Val::U32(2)));

    let norm = |r: &mut Rng, n: usize| -> Vec<f32> { (0..n).map(|_| 1.0 + 0.2 * r.f()).collect() };
    for i in 0..t.n_layer {
        let p = format!("blk.{i}");
        let v = norm(&mut r, e);
        w.f32(format!("{p}.attn_norm.weight"), vec![e], &v);
        let v = norm(&mut r, e);
        w.f32(format!("{p}.ffn_norm.weight"), vec![e], &v);
        if t.kv_heads[i] > 0 {
            let kvd = t.kv_heads[i] * hd;
            for (name, cols, rows) in [("attn_q", e, t.n_head * hd), ("attn_k", e, kvd), ("attn_v", e, kvd), ("attn_output", t.n_head * hd, e)] {
                let v = r.vec(rows * cols, 0.15);
                w.q4(format!("{p}.{name}.weight"), vec![cols, rows], cols, &v);
            }
            let v = norm(&mut r, hd);
            w.f32(format!("{p}.attn_q_norm.weight"), vec![hd], &v);
            let v = norm(&mut r, hd);
            w.f32(format!("{p}.attn_k_norm.weight"), vec![hd], &v);
        } else {
            let v = r.vec(3 * e * e, 0.15);
            w.q4(format!("{p}.shortconv.in_proj.weight"), vec![e, 3 * e], e, &v);
            let v = r.vec(t.l_cache * e, 0.5);
            w.f32(format!("{p}.shortconv.conv.weight"), vec![t.l_cache, e], &v);
            let v = r.vec(e * e, 0.15);
            w.q4(format!("{p}.shortconv.out_proj.weight"), vec![e, e], e, &v);
        }
        if i < t.n_dense_lead {
            for (name, cols, rows) in [("ffn_gate", e, t.n_ff), ("ffn_up", e, t.n_ff), ("ffn_down", t.n_ff, e)] {
                let v = r.vec(rows * cols, 0.15);
                w.q4(format!("{p}.{name}.weight"), vec![cols, rows], cols, &v);
            }
        } else {
            let v = r.vec(t.n_expert * e, 0.4);
            w.f32(format!("{p}.ffn_gate_inp.weight"), vec![e, t.n_expert], &v);
            let v = r.vec(t.n_expert, 0.05);
            w.f32(format!("{p}.exp_probs_b.bias"), vec![t.n_expert], &v);
            for (name, cols, rows) in [("ffn_gate_exps", e, t.n_ff_exp), ("ffn_up_exps", e, t.n_ff_exp), ("ffn_down_exps", t.n_ff_exp, e)] {
                let v = r.vec(t.n_expert * rows * cols, 0.2);
                w.q4(format!("{p}.{name}.weight"), vec![cols, rows, t.n_expert], cols, &v);
            }
        }
    }
    let v = r.vec(t.n_vocab * e, 0.5);
    w.q4("token_embd.weight".into(), vec![e, t.n_vocab], e, &v);
    let v = norm(&mut r, e);
    w.f32("token_embd_norm.weight".into(), vec![e], &v);

    let mut f = std::fs::File::create(path)?;
    f.write_all(&w.bytes())
}

/// Write the default tiny model into the system temp directory under a name unique to this
/// process and `tag`; the caller removes it (or leaves it to the temp cleaner).
pub fn temp_model(tag: &str, seed: u64) -> std::io::Result<std::path::PathBuf> {
    let p = std::env::temp_dir().join(format!("w2w-tiny-{}-{tag}-{seed}.gguf", std::process::id()));
    write(&p, &Tiny::default(), seed)?;
    Ok(p)
}
