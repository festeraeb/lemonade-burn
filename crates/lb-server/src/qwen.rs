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
use lb_qwen35::spec_dev::SpecDev;
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::Arc;

pub struct QwenDev {
    pipe: Pipeline<'static, 'static>,
    cap: usize,
    bos: u32,
    vocab: usize,
    /// MTP speculative decoding (greedy requests only); `None` when off or the file has no nextn layer.
    spec: Option<RefCell<SpecDev>>,
    /// The `cache.len` of the last forward served (what the caller believes the sequence length is).
    served: Cell<usize>,
    spec_cap: usize,
}

// The pipeline is created on the main thread and then only ever used by the single model worker thread.
unsafe impl Send for QwenDev {}

pub struct QCache {
    len: usize,
}

/// A rewind point: the sequence state at `len` tokens, on the host.
pub struct QSnap {
    len: usize,
    state: Vec<Vec<lb_qwen35::device::LayerSnap>>,
}

/// Prompts of at least this many new tokens go through the batched prefill (`LB_QW_PREFILL_MIN`, default 32; 0 = never).
fn trace() -> bool {
    std::env::var("LB_QW_TRACE").is_ok()
}

fn prefill_min() -> usize {
    std::env::var("LB_QW_PREFILL_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(32)
}

/// Rows per prefill chunk (`LB_QW_PREFILL_CHUNK`; 0 = automatic 64..1024). A smaller chunk lowers the device temporaries, which matters when the weights nearly fill the cards.
fn prefill_chunk() -> usize {
    std::env::var("LB_QW_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
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
    let nextn = g.meta_u64("qwen35moe.nextn_predict_layers").unwrap_or(0) > 0;
    // LB_MTP=1 turns the draft on (default off); LB_MTP_DEPTH (1..7, default 3) is the most tokens the nextn head drafts per verify;
    // LB_MTP_ADAPT=0 drafts exactly LB_MTP_DEPTH every round instead of picking 0..=LB_MTP_DEPTH by measured acceptance and round time
    let mtp_on = nextn && std::env::var("LB_MTP").map(|v| v == "1" || v == "on").unwrap_or(false);
    let depth: usize = std::env::var("LB_MTP_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(3).clamp(1, 7);
    let adapt = std::env::var("LB_MTP_ADAPT").map(|v| v != "0").unwrap_or(true);
    let mut pipe = Pipeline::new_spec(model, &split, cap, if mtp_on { depth + 1 } else { 0 })?;
    if mtp_on {
        pipe.attach_mtp(&model.load_mtp(g)?)?;
        eprintln!("lb-qwen35: MTP speculative decoding on, depth {} {depth} (greedy requests)", if adapt { "adaptive 0..=" } else { "fixed" });
    }
    pipe.warm_decode(g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(248044) as u32)?;
    pipe.warm_spec(g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(248044) as u32)?;
    let tok = lb_qwen35::tokenizer::from_gguf(g)?;
    let bos = g.meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(248044) as u32;
    let eos = g.meta_u64("tokenizer.ggml.eos_token_id").unwrap_or(248046) as u32;
    let codec = Arc::new(TextCodec { tok, bos, style: ChatStyle::Qwen35, stops: vec![eos, bos] });
    let vocab = model.cfg.vocab;
    Ok((QwenDev { pipe, cap, bos, vocab, spec: mtp_on.then(|| RefCell::new(if adapt { SpecDev::adaptive(depth) } else { SpecDev::new(depth) })), served: Cell::new(0), spec_cap: depth + 1 }, codec, cap))
}

impl QwenDev {
    /// Greedy next token for a multi-token (or non-continuation) input with the draft on: the trunk consumes `tokens` through a path that also
    /// returns the hidden rows, and the nextn head's cache is built from them.
    fn argmax_spec(&self, sp: &RefCell<SpecDev>, tokens: &[u32], cache: &mut QCache) -> u32 {
        self.settle();
        if cache.len == 0 {
            self.pipe.reset();
            sp.borrow_mut().invalidate();
            sp.borrow_mut().trunk_len = 0;
        }
        assert!(cache.len + tokens.len() <= self.cap, "sequence exceeds the KV capacity {}", self.cap);
        let (tok, rows) = if tokens.len() <= self.spec_cap && cache.len + tokens.len() <= self.cap {
            let (g, hs) = self.pipe.run_rows(tokens, cache.len).expect("device verify");
            (*g.last().unwrap(), hs)
        } else {
            let rows = std::sync::Mutex::new(Vec::new());
            let l = self.pipe.prefill_h(tokens, cache.len, prefill_chunk(), Some(&rows)).expect("device prefill");
            let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
            (am, rows.into_inner().unwrap())
        };
        sp.borrow_mut().after_prefill(&self.pipe, tokens, cache.len, &rows).expect("nextn cache");
        cache.len += tokens.len();
        self.served.set(cache.len);
        tok
    }
    /// Put the trunk back to exactly the length the caller believes (verified-ahead tokens undone).
    fn settle(&self) {
        if let Some(sp) = &self.spec {
            sp.borrow_mut().trim(&self.pipe, self.served.get());
        }
    }
    /// A plain (non-speculative) excursion moved the trunk: the draft's pending hidden is stale.
    fn spec_forget(&self, len: usize) {
        if let Some(sp) = &self.spec {
            let mut sp = sp.borrow_mut();
            sp.invalidate();
            sp.trunk_len = len;
        }
    }
}

impl Decoder for QwenDev {
    type Cache = QCache;
    type Checkpoint = QSnap;
    fn checkpoint(&self, cache: &mut QCache) -> Option<QSnap> {
        self.settle();
        let t0 = std::time::Instant::now();
        let r = self.pipe.snapshot(cache.len);
        if trace() {
            eprintln!("qwen: checkpoint at {} tokens took {:.0} ms", cache.len, t0.elapsed().as_secs_f64() * 1e3);
        }
        match r {
            Ok(state) => Some(QSnap { len: cache.len, state }),
            Err(e) => {
                eprintln!("lb-server: qwen checkpoint failed ({e}); this turn will not be reusable");
                None
            }
        }
    }
    fn rewind(&self, cache: &mut QCache, cp: &QSnap) -> bool {
        let t0 = std::time::Instant::now();
        self.pipe.restore(&cp.state);
        cache.len = cp.len;
        self.served.set(cp.len);
        self.spec_forget(cp.len);
        if trace() {
            eprintln!("qwen: rewind to {} tokens took {:.0} ms", cp.len, t0.elapsed().as_secs_f64() * 1e3);
        }
        true
    }
    fn new_cache(&self) -> QCache {
        QCache { len: 0 }
    }
    fn forward(&self, tokens: &[u32], cache: &mut QCache, all_logits: bool) -> Tensor<2> {
        let v = self.forward_vec(tokens, cache, all_logits);
        Tensor::from_data(TensorData::new(v, [1, self.vocab]), &Device::flex())
    }
    fn forward_vec(&self, tokens: &[u32], cache: &mut QCache, all_logits: bool) -> Vec<f32> {
        assert!(!all_logits, "qwen35moe decoder returns the last position's logits only");
        self.settle();
        if cache.len == 0 {
            self.pipe.reset();
        }
        self.spec_forget(cache.len + tokens.len());
        self.served.set(cache.len + tokens.len());
        assert!(cache.len + tokens.len() <= self.cap, "sequence exceeds the KV capacity {}", self.cap);
        let pm = prefill_min();
        if pm > 0 && tokens.len() >= pm && cache.len + tokens.len() <= self.cap {
            let t0 = std::time::Instant::now();
            let l = self.pipe.prefill(tokens, cache.len, prefill_chunk()).expect("device prefill");
            if trace() {
                eprintln!("qwen: prefill of {} tokens at {} took {:.0} ms", tokens.len(), cache.len, t0.elapsed().as_secs_f64() * 1e3);
            }
            cache.len += tokens.len();
            return l;
        }
        let mut last = None;
        for (i, &t) in tokens.iter().enumerate() {
            last = self.pipe.step(t, cache.len, i + 1 == tokens.len()).expect("device step");
            cache.len += 1;
        }
        last.expect("logits")
    }
    fn forward_argmax(&self, tokens: &[u32], cache: &mut QCache) -> Option<u32> {
        if let Some(sp) = &self.spec {
            // decode continuation: one token, the draft's hidden is current
            if tokens.len() == 1 && cache.len > 0 && cache.len + self.spec_cap < self.cap && sp.borrow().ready_for(cache.len) {
                let t = sp.borrow_mut().next(&self.pipe, tokens[0], cache.len).expect("speculative step");
                cache.len += 1;
                self.served.set(cache.len);
                return Some(t);
            }
            return Some(self.argmax_spec(sp, tokens, cache));
        }
        if cache.len == 0 {
            self.pipe.reset();
        }
        assert!(cache.len + tokens.len() <= self.cap, "sequence exceeds the KV capacity {}", self.cap);
        let pm = prefill_min();
        if pm > 0 && tokens.len() >= pm && cache.len + tokens.len() <= self.cap {
            let t0 = std::time::Instant::now();
            let l = self.pipe.prefill(tokens, cache.len, prefill_chunk()).expect("device prefill");
            if trace() {
                eprintln!("qwen: prefill of {} tokens at {} took {:.0} ms", tokens.len(), cache.len, t0.elapsed().as_secs_f64() * 1e3);
            }
            cache.len += tokens.len();
            return Some(l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32);
        }
        let (last, head) = tokens.split_last().expect("tokens");
        for &t in head {
            self.pipe.step(t, cache.len, false).expect("device step");
            cache.len += 1;
        }
        let tok = self.pipe.step_argmax(*last, cache.len).expect("device step");
        cache.len += 1;
        Some(tok)
    }
    fn after_request(&self, _prompt: &[u32], _generated: &[u32]) {
        if let (Some(sp), true) = (&self.spec, trace() || std::env::var("LB_MTP_LOG").is_ok()) {
            let sp = sp.borrow();
            let s = &sp.stats;
            eprintln!("qwen mtp: {} rounds, {:.2} tokens/round, prefix accept {:?}, draft {:.1} verify {:.1} fixup {:.1} ms/round, rounds at depth {:?}", s.rounds, s.emitted as f64 / s.rounds.max(1) as f64, &s.prefix_ok[..sp.depth], s.draft_s * 1e3 / s.rounds.max(1) as f64, s.verify_s * 1e3 / s.rounds.max(1) as f64, s.fixup_s * 1e3 / s.rounds.max(1) as f64, &s.at_depth[..=sp.depth]);
        }
    }
    fn discard_request(&self) {
        self.settle();
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
