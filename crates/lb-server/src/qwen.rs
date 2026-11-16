//! The `qwen35moe` family behind the server: a [`Decoder`] over `lb_qwen35::pipeline::Pipeline` (N CUDA cards, one layer block each).
//! Cards are chosen by UUID the way every engine here does it: `CUDA_VISIBLE_DEVICES=<uuid0>,<uuid1>,...` in stage order;
//! `QWEN_SPLIT=20,20` sets layers per card (default: even). One sequence at a time lives on the device (recurrent state and KV cache are
//! the pipeline's): a `Cache` is just the position counter, a new one (position 0) resets the device state, and rewinding to a checkpoint
//! is not supported, so a request reuses the kept conversation only when it extends it exactly.

use crate::Decoder;
use anyhow::Result;
use burn::tensor::{Device, Tensor, TensorData};
use lb_model::text::{ChatStyle, TextCodec};
use lb_qwen35::pipeline::Pipeline;
use std::path::Path;
use std::sync::Arc;

pub struct QwenDev {
    pipe: Pipeline<'static, 'static>,
    cap: usize,
    bos: u32,
    vocab: usize,
}

// The pipeline is created on the main thread and then only ever used by the single model worker thread.
unsafe impl Send for QwenDev {}

pub struct QCache {
    len: usize,
}

pub fn is_qwen(p: &Path) -> bool {
    lb_gguf::Gguf::open(p).ok().and_then(|g| g.meta_str("general.architecture").map(|a| a == "qwen35moe")) == Some(true)
}

pub fn load(path: &Path, ctx: usize) -> Result<(QwenDev, Arc<TextCodec>, usize)> {
    let g: &'static lb_gguf::Gguf = Box::leak(Box::new(lb_gguf::Gguf::open(path)?));
    let model: &'static lb_qwen35::Model<'static> = Box::leak(Box::new(lb_qwen35::Model::load(g)?));
    let n_layer = model.cfg.n_layer;
    let split: Vec<usize> = match std::env::var("QWEN_SPLIT") {
        Ok(s) => s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>()?,
        Err(_) => {
            let cards = std::env::var("CUDA_VISIBLE_DEVICES").map(|v| v.split(',').filter(|x| !x.trim().is_empty()).count()).unwrap_or(1).max(1);
            (0..cards).map(|i| n_layer / cards + usize::from(i < n_layer % cards)).collect()
        }
    };
    let cap = ctx.min(model.cfg_context(g));
    let pipe = Pipeline::new(model, &split, cap)?;
    let tok = lb_qwen35::tokenizer::from_gguf(g)?;
    let bos = g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(248044) as u32;
    let eos = g.meta_u64("tokenizer.ggml.eos_token_id").unwrap_or(248046) as u32;
    let codec = Arc::new(TextCodec { tok, bos, style: ChatStyle::Qwen35, stops: vec![eos, bos] });
    let vocab = model.cfg.vocab;
    Ok((QwenDev { pipe, cap, bos, vocab }, codec, cap))
}

impl Decoder for QwenDev {
    type Cache = QCache;
    type Checkpoint = ();
    fn new_cache(&self) -> QCache {
        QCache { len: 0 }
    }
    fn forward(&self, tokens: &[u32], cache: &mut QCache, all_logits: bool) -> Tensor<2> {
        let v = self.forward_vec(tokens, cache, all_logits);
        Tensor::from_data(TensorData::new(v, [1, self.vocab]), &Device::flex())
    }
    fn forward_vec(&self, tokens: &[u32], cache: &mut QCache, all_logits: bool) -> Vec<f32> {
        assert!(!all_logits, "qwen35moe decoder returns the last position's logits only");
        if cache.len == 0 {
            self.pipe.reset();
        }
        assert!(cache.len + tokens.len() <= self.cap, "sequence exceeds the KV capacity {}", self.cap);
        let mut last = None;
        for (i, &t) in tokens.iter().enumerate() {
            last = self.pipe.step(t, cache.len, i + 1 == tokens.len()).expect("device step");
            cache.len += 1;
        }
        last.expect("logits")
    }
    fn forward_argmax(&self, tokens: &[u32], cache: &mut QCache) -> Option<u32> {
        if cache.len == 0 {
            self.pipe.reset();
        }
        assert!(cache.len + tokens.len() <= self.cap, "sequence exceeds the KV capacity {}", self.cap);
        let (last, head) = tokens.split_last().expect("tokens");
        for &t in head {
            self.pipe.step(t, cache.len, false).expect("device step");
            cache.len += 1;
        }
        let tok = self.pipe.step_argmax(*last, cache.len).expect("device step");
        cache.len += 1;
        Some(tok)
    }
    fn max_context(&self) -> usize {
        self.cap
    }
    fn bos(&self) -> u32 {
        self.bos
    }
    fn kernel_name(&self) -> &'static str {
        "qwen35moe q5_k/q6_k/q8_0 file-density kernels, CUDA layer pipeline"
    }
}
