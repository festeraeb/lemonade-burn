//! N-card layer pipeline for the `qwen35moe` decoder. Each stage owns one CUDA device (opened by its position in
//! `CUDA_VISIBLE_DEVICES`, which is how a card is chosen by UUID: list the UUIDs there, in stage order) and a contiguous block of
//! layers; the last stage also holds the output norm and the LM head. A token's activation vector (`n_embd` f32) is read from one card
//! and written to the next at a stage boundary; nothing else crosses cards and nothing is read back inside a layer. Launches are
//! flushed to the device thread after every layer (the lesson of `ENGINE_SYNC_COST.md`: CubeCL delivers a batch only when it is full or
//! flushed, so without it the card idles while the host builds the next launches).

use crate::device::DevModel;
use crate::Model;
use anyhow::{anyhow, Result};
use lb_gguf::QuantMat;
use lb_q4_gpu::{CubeGraph, Gpu, GpuQ6KNative, GpuQ8Mat};
use std::cell::RefCell;

type Handle = lb_q4_gpu::cubecl_server::Handle;

enum HeadMat {
    Q6(GpuQ6KNative),
    Q8(GpuQ8Mat),
}

struct Head {
    norm: Handle,
    out: HeadMat,
}

enum GraphState {
    None,
    Prepared,
    Ready { graph: CubeGraph, out: Handle, aux: Option<Handle>, _held: Vec<Handle> },
    Failed,
}

struct Stage {
    graph: RefCell<GraphState>,
    x_in: Handle,
    gpu: Gpu,
    dev: DevModel,
    first: usize,
    head: Option<Head>,
}

enum Out {
    Logits,
    Argmax,
    Nothing,
}

enum Got {
    Logits(Vec<f32>),
    Token(u32),
    Nothing,
}

pub struct Pipeline<'m, 'a> {
    model: &'m Model<'a>,
    stages: Vec<Stage>,
    flush: u32,
    graph_on: bool,
    timing: bool,
}

impl<'m, 'a> Pipeline<'m, 'a> {
    /// `split[i]` = number of layers on stage `i` (must sum to the layer count); stage `i` opens CUDA device `i` of
    /// `CUDA_VISIBLE_DEVICES`. `cap` = positions of KV cache per attention layer.
    pub fn new(model: &'m Model<'a>, split: &[usize], cap: usize) -> Result<Self> {
        let n = model.cfg.n_layer;
        if split.iter().sum::<usize>() != n {
            return Err(anyhow!("split {split:?} does not cover {n} layers"));
        }
        let mut stages = Vec::new();
        let mut first = 0;
        for (i, &cnt) in split.iter().enumerate() {
            let gpu = Gpu::open_cuda(i).map_err(|e| anyhow!(e))?;
            let dev = DevModel::upload(&gpu, model, first..first + cnt, cap)?;
            let head = if i + 1 == split.len() {
                let out: QuantMat<'_> = model.output;
                let out = match out.ty {
                    lb_gguf::GgmlType::Q8_0 => HeadMat::Q8(GpuQ8Mat::upload(&gpu, out).map_err(|e| anyhow!(e))?),
                    _ => HeadMat::Q6(GpuQ6KNative::upload(&gpu, out).map_err(|e| anyhow!(e))?),
                };
                Some(Head { norm: gpu.upload_f32(&model.out_norm), out })
            } else {
                None
            };
            eprintln!("stage {i}: layers {first}..{} uploaded", first + cnt);
            let x_in = gpu.upload_f32(&vec![0.0; model.cfg.n_embd]);
            stages.push(Stage { graph: RefCell::new(GraphState::None), x_in, gpu, dev, first, head });
            first += cnt;
        }
        let flush = std::env::var("LB_QW_FLUSH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let graph_on = std::env::var("LB_QW_GRAPH").map(|v| v != "0").unwrap_or(true);
        Ok(Pipeline { model, stages, flush, graph_on, timing: std::env::var("LB_QW_TIMING").is_ok() })
    }

    /// Run three single-token steps (plain, capture, replay) and reset: the CUDA graphs of every stage exist before any prefill runs (a capture started after
    /// stage threads have written the shared buffers is invalidated by CUDA; a replay is not).
    pub fn warm_decode(&self, tok: u32) -> Result<()> {
        for i in 0..3 {
            self.step(tok, i, true)?;
        }
        self.reset();
        Ok(())
    }

    pub fn reset(&self) {
        for s in &self.stages {
            s.dev.reset(&s.gpu);
        }
    }

    /// The launches of stage `si` for one token: its layers from `x_in`, then (last stage) the final norm and head.
    fn build(&self, si: usize, flush: bool) -> (Handle, Option<Handle>) {
        let c = &self.model.cfg;
        let s = &self.stages[si];
        let mut cur = s.x_in.clone();
        for l in 0..s.dev.layers.len() {
            let (_x1, y) = s.dev.layer_forward(&s.gpu, l, &cur, 0);
            cur = y;
            if flush {
                s.gpu.flush_queue();
            }
        }
        if let Some(h) = &s.head {
            let nrm = s.gpu.qw_rms(&cur, &h.norm, 1, c.n_embd, c.eps).0;
            let logits = match &h.out {
                HeadMat::Q6(m) => m.launch(&s.gpu, &nrm, 1),
                HeadMat::Q8(m) => m.launch(&s.gpu, &nrm, 1),
            };
            let am = s.gpu.qw_argmax(&logits, c.vocab);
            return (logits, Some(am));
        }
        (cur, None)
    }

    /// Run stage `si` on `xh` at `pos`; returns the handle holding its output (activation, or logits for the last stage).
    /// With `LB_QW_GRAPH` (default) the stage is recorded as one CUDA graph on its second token and replayed afterwards.
    fn run_stage(&self, si: usize, xh: &[f32], pos: usize) -> Result<(Handle, Option<Handle>)> {
        let s = &self.stages[si];
        s.gpu.write_f32(&s.x_in, xh);
        s.dev.set_pos(&s.gpu, pos);
        let mut gs = s.graph.borrow_mut();
        if self.graph_on {
            if matches!(&*gs, GraphState::None) {
                *gs = match s.gpu.client.graph_prepare() {
                    Ok(()) => GraphState::Prepared,
                    Err(e) => {
                        eprintln!("lb-qwen35: graph_prepare refused ({e:?}); running without graphs");
                        GraphState::Failed
                    }
                };
                return Ok(self.build(si, self.flush > 0)); // the warm-up token after graph_prepare runs normally
            }
            if matches!(&*gs, GraphState::Prepared) {
                if let Err(e) = s.gpu.client.start_capture() {
                    eprintln!("lb-qwen35: start_capture refused ({e:?})");
                    *gs = GraphState::Failed;
                } else {
                    let (out, aux) = self.build(si, false);
                    match s.gpu.client.stop_capture() {
                        Ok(graph) => {
                            let held = s.gpu.take_in_flight();
                            eprintln!("lb-qwen35: stage {si} recorded as a CUDA graph");
                            *gs = GraphState::Ready { graph, out, aux, _held: held };
                        }
                        Err(e) => {
                            eprintln!("lb-qwen35: stop_capture refused ({e:?})");
                            // the recording runs nothing: this token is simply run again, un-graphed, from here on
                            *gs = GraphState::Failed;
                            drop(gs);
                            return Ok(self.build(si, self.flush > 0));
                        }
                    }
                }
            }
            if let GraphState::Ready { graph, out, aux, .. } = &*gs {
                // SAFETY: every handle the recording used is held by the stage or by `held`; replayed only on the stream it was
                // recorded on (this thread's), and this token's inputs (x_in, position) were written on that stream above.
                unsafe { graph.replay() }.map_err(|e| anyhow!("graph replay: {e:?}"))?;
                return Ok((out.clone(), aux.clone()));
            }
        }
        drop(gs);
        Ok(self.build(si, self.flush > 0))
    }

    /// One token at `pos`. Returns the logits when `want_logits`.
    pub fn step(&self, tok: u32, pos: usize, want_logits: bool) -> Result<Option<Vec<f32>>> {
        self.step_ex(tok, pos, if want_logits { Out::Logits } else { Out::Nothing }).map(|o| match o {
            Got::Logits(v) => Some(v),
            _ => None,
        })
    }

    /// One token at `pos`, returning the greedy next token (argmax on the device: 4 bytes come back instead of the logits).
    pub fn step_argmax(&self, tok: u32, pos: usize) -> Result<u32> {
        match self.step_ex(tok, pos, Out::Argmax)? {
            Got::Token(t) => Ok(t),
            _ => unreachable!(),
        }
    }

    fn step_ex(&self, tok: u32, pos: usize, want: Out) -> Result<Got> {
        let c = &self.model.cfg;
        let mut xh = vec![0f32; c.n_embd];
        self.model.tok_embd.dequant_row(tok as usize, &mut xh);
        let nst = self.stages.len();
        for si in 0..nst {
            let t0 = std::time::Instant::now();
            let (out, aux) = self.run_stage(si, &xh, pos)?;
            let s = &self.stages[si];
            if self.timing {
                s.gpu.flush_queue();
                eprintln!("stage {si} issue {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            if si + 1 == nst {
                if let (Out::Argmax, Some(am)) = (&want, aux) {
                    let r = s.gpu.read_all(vec![(am, 1)]).map_err(|e| anyhow!(e))?.remove(0);
                    return Ok(Got::Token(r[0].to_bits()));
                }
                let n = if s.head.is_some() { c.vocab } else { c.n_embd };
                let r = s.gpu.read_all(vec![(out, n)]).map_err(|e| anyhow!(e))?.remove(0);
                return Ok(if matches!(want, Out::Logits) { Got::Logits(r) } else { Got::Nothing });
            }
            xh = s.gpu.read_all(vec![(out, c.n_embd)]).map_err(|e| anyhow!(e))?.remove(0);
        }
        unreachable!()
    }

    /// Parts timing of layer `il` (on whichever stage holds it).
    pub fn bench_parts(&self, il: usize, reps: usize) -> Vec<(&'static str, f64)> {
        let s = self.stages.iter().find(|s| il >= s.first && il < s.first + s.dev.layers.len()).expect("layer on a stage");
        s.dev.bench_parts(&s.gpu, il - s.first, reps)
    }

    /// Final norm + head on one hidden row (outside the stage graphs): the logits.
    fn head_logits(&self, xrow: &[f32]) -> Result<Vec<f32>> {
        let c = &self.model.cfg;
        let s = self.stages.last().unwrap();
        let h = s.head.as_ref().ok_or_else(|| anyhow!("no head"))?;
        let x = s.gpu.upload_f32(xrow);
        let nrm = s.gpu.qw_rms(&x, &h.norm, 1, c.n_embd, c.eps).0;
        let logits = match &h.out {
            HeadMat::Q6(m) => m.launch(&s.gpu, &nrm, 1),
            HeadMat::Q8(m) => m.launch(&s.gpu, &nrm, 1),
        };
        Ok(s.gpu.read_all(vec![(logits, c.vocab)]).map_err(|e| anyhow!(e))?.remove(0))
    }

    pub fn ptime_report(&self) -> Vec<(&'static str, f64)> {
        crate::device::ptime_report()
    }

    /// Batched prompt processing: `tokens` at positions `pos0 ..` in chunks (tiled GEMMs, batched kernels), the stages as a software pipeline: one thread a
    /// stage, chunk `c + 1` on stage 0 while chunk `c` is on stage 1, a chunk's hidden rows handed over through the host. Returns the logits of the last
    /// token. State and KV cache afterwards are those of token-by-token steps. Needs `pos0 + tokens.len() <= 4096` (the attention kernel keeps its scores in
    /// shared memory). `chunk` 0 picks a size giving each stage about four chunks (64..1024 tokens).
    pub fn prefill(&self, tokens: &[u32], pos0: usize, chunk: usize) -> Result<Vec<f32>> {
        use std::sync::mpsc;
        let c = &self.model.cfg;
        if pos0 + tokens.len() > 4096 {
            return Err(anyhow!("prefill supports positions below 4096"));
        }
        let nst = self.stages.len();
        let chunk = if chunk > 0 { chunk } else { (tokens.len().div_ceil(2 * nst).div_ceil(64) * 64).clamp(64, 1024) };
        let mut spans = Vec::new();
        let mut p = 0usize;
        while p < tokens.len() {
            let m = chunk.min(tokens.len() - p);
            spans.push((p, m));
            p += m;
        }
        struct Raw<T>(*const T);
        unsafe impl<T> Send for Raw<T> {}
        // channels: stage i sends its output rows to stage i + 1
        let mut txs = Vec::new();
        let mut rxs = Vec::new();
        for _ in 0..nst {
            let (t, r) = mpsc::channel::<Result<(usize, Vec<f32>)>>();
            txs.push(t);
            rxs.push(Some(r));
        }
        let (last_tx, last_rx) = mpsc::channel::<Result<Vec<f32>>>();
        std::thread::scope(|sc| {
            for si in 0..nst {
                let s = Raw(&self.stages[si] as *const Stage);
                let model = Raw(self.model as *const Model<'a>);
                let rx = rxs[si].take().unwrap();
                let next = if si + 1 < nst { Some(txs[si + 1].clone()) } else { None };
                let last_tx = last_tx.clone();
                let spans = spans.clone();
                let flush = self.flush;
                let ncfg = (c.n_embd, spans.len());
                let tok_ptr = Raw(tokens.as_ptr());
                let tok_len = tokens.len();
                sc.spawn(move || {
                    let (s, model, tok_ptr) = (s, model, tok_ptr);
                    // SAFETY: the stage, the model and the tokens outlive the scope; each stage is touched by its own thread only
                    let (s, model, tokens) = unsafe { (&*s.0, &*model.0, std::slice::from_raw_parts(tok_ptr.0, tok_len)) };
                    let n_embd = ncfg.0;
                    for (ci, &(p, m)) in spans.iter().enumerate() {
                        let run = || -> Result<Vec<f32>> {
                            let xh = if si == 0 {
                                let mut xh = vec![0f32; m * n_embd];
                                for (i, &t) in tokens[p..p + m].iter().enumerate() {
                                    model.tok_embd.dequant_row(t as usize, &mut xh[i * n_embd..(i + 1) * n_embd]);
                                }
                                xh
                            } else {
                                match rx.recv() {
                                    Ok(Ok((_, v))) => v,
                                    Ok(Err(e)) => return Err(e),
                                    Err(e) => return Err(anyhow!("stage hand-over: {e}")),
                                }
                            };
                            let mut cur = s.gpu.upload_f32(&xh);
                            s.dev.set_chunk(&s.gpu, pos0 + p, m);
                            for l in 0..s.dev.layers.len() {
                                cur = s.dev.prefill_layer(&s.gpu, l, &cur, m)?;
                                if flush > 0 {
                                    s.gpu.flush_queue();
                                }
                            }
                            s.gpu.read_all(vec![(cur, m * n_embd)]).map_err(|e| anyhow!(e)).map(|mut v| v.remove(0))
                        };
                        match run() {
                            Ok(v) => {
                                if let Some(nx) = &next {
                                    let _ = nx.send(Ok((ci, v)));
                                } else if ci + 1 == spans.len() {
                                    let _ = last_tx.send(Ok(v[(m - 1) * n_embd..m * n_embd].to_vec()));
                                }
                            }
                            Err(e) => {
                                let msg = e.to_string();
                                if let Some(nx) = &next {
                                    let _ = nx.send(Err(anyhow!(msg.clone())));
                                }
                                let _ = last_tx.send(Err(anyhow!(msg)));
                                return;
                            }
                        }
                    }
                });
            }
            drop(last_tx);
        });
        // The stage threads wrote the states and KV caches on their own streams: touch every persistent buffer once from this thread so the
        // cross-stream ordering is settled here, outside any CUDA-graph capture window (a capture that meets a pending foreign write is invalidated).
        for s in &self.stages {
            let hs: Vec<(Handle, usize)> = s.dev.persistent().into_iter().map(|h| (h, 1usize)).collect();
            s.gpu.read_all(hs).map_err(|e| anyhow!(e))?;
        }
        let row = last_rx.recv().map_err(|e| anyhow!("prefill produced no result: {e}"))??;
        self.head_logits(&row)
    }

    /// Host copy of the whole sequence state at `len` tokens.
    pub fn snapshot(&self, len: usize) -> Result<Vec<Vec<crate::device::LayerSnap>>> {
        self.stages.iter().map(|s| s.dev.snapshot(&s.gpu, len)).collect()
    }

    pub fn restore(&self, snap: &[Vec<crate::device::LayerSnap>]) {
        for (s, sn) in self.stages.iter().zip(snap) {
            s.dev.restore(&s.gpu, sn);
        }
    }

    pub fn first_layer_of_stage(&self, i: usize) -> usize {
        self.stages[i].first
    }
}
