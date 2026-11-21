//! N-card layer pipeline for the `qwen35moe` decoder. Each stage owns one CUDA device (opened by its position in
//! `CUDA_VISIBLE_DEVICES`, which is how a card is chosen by UUID: list the UUIDs there, in stage order) and a contiguous block of
//! layers; the last stage also holds the output norm and the LM head. A token's activation vector (`n_embd` f32) is read from one card
//! and written to the next at a stage boundary; nothing else crosses cards and nothing is read back inside a layer. Launches are
//! flushed to the device thread after every layer (the lesson of `ENGINE_SYNC_COST.md`: CubeCL delivers a batch only when it is full or
//! flushed, so without it the card idles while the host builds the next launches).

use crate::device::{DMat, DevModel};
use crate::Mtp;
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
    /// per verify chunk size `m` in `1..=spec_m` (index `m - 1`): the input rows (`m * n_embd`) and the CUDA graph of that size (recorded on its
    /// second pass). One graph per size so an adaptive draft depth (and the depth-0 round) replays instead of running eagerly.
    vx_in: Vec<Handle>,
    vgraph: Vec<RefCell<VState>>,
}

enum VState {
    None,
    Prepared,
    Ready { graph: CubeGraph, out: (Handle, Option<Handle>), _held: Vec<Handle> },
    Failed,
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

/// The nextn block on the last stage's card: `eh_proj`, one attention+MoE layer with its own KV cache, the shared head norm.
struct MtpDev {
    dev: DevModel,
    eh: DMat,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    shead: Handle,
    x_in: Handle,
    /// the draft step as a CUDA graph (inputs: `x_in`, the position buffer): outputs (hidden, argmax)
    graph: RefCell<VState>,
}

pub struct Pipeline<'m, 'a> {
    model: &'m Model<'a>,
    mtp: Option<MtpDev>,
    stages: Vec<Stage>,
    flush: u32,
    graph_on: bool,
    timing: bool,
    vm: usize,
    vwarm: std::cell::Cell<bool>,
}

impl<'m, 'a> Pipeline<'m, 'a> {
    /// `split[i]` = number of layers on stage `i` (must sum to the layer count); stage `i` opens CUDA device `i` of
    /// `CUDA_VISIBLE_DEVICES`. `cap` = positions of KV cache per attention layer.
    pub fn new(model: &'m Model<'a>, split: &[usize], cap: usize) -> Result<Self> {
        Self::new_spec(model, split, cap, 0)
    }

    /// As [`new`](Self::new) with the speculative-verify scratch for chunks of up to `spec_m` tokens (0 = none; needed by [`run_rows`](Self::run_rows)).
    pub fn new_spec(model: &'m Model<'a>, split: &[usize], cap: usize, spec_m: usize) -> Result<Self> {
        let n = model.cfg.n_layer;
        if split.iter().sum::<usize>() != n {
            return Err(anyhow!("split {split:?} does not cover {n} layers"));
        }
        let mut stages = Vec::new();
        let mut first = 0;
        for (i, &cnt) in split.iter().enumerate() {
            let gpu = Gpu::open_cuda(i).map_err(|e| anyhow!(e))?;
            let dev = DevModel::upload(&gpu, model, first..first + cnt, cap, spec_m)?;
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
            let vx_in = (1..=spec_m.max(1)).map(|m| gpu.upload_f32(&vec![0.0; m * model.cfg.n_embd])).collect();
            let vgraph = (1..=spec_m.max(1)).map(|_| RefCell::new(VState::None)).collect();
            stages.push(Stage { graph: RefCell::new(GraphState::None), x_in, gpu, dev, first, head, vx_in, vgraph });
            first += cnt;
        }
        // env > card profiles (the most constrained stage card decides) > default
        let cards: Vec<_> = stages.iter().map(|s| s.gpu.profile.clone()).collect();
        let flush = lb_card_profile::min_knob(&cards, "LB_QW_FLUSH").and_then(|v| v.parse().ok()).unwrap_or(1);
        let graph_on = lb_card_profile::min_knob(&cards, "LB_QW_GRAPH").map(|v| v != "0").unwrap_or(true);
        Ok(Pipeline { model, mtp: None, stages, flush, graph_on, timing: std::env::var("LB_QW_TIMING").is_ok(), vm: spec_m, vwarm: std::cell::Cell::new(false) })
    }

    /// A card-profile knob over this pipeline's stage cards: env, else the smallest value any stage card's profile gives, else `None`.
    pub fn knob_min(&self, name: &str) -> Option<String> {
        let cards: Vec<_> = self.stages.iter().map(|s| s.gpu.profile.clone()).collect();
        lb_card_profile::min_knob(&cards, name)
    }

    /// The card profiles of the first `n` CUDA cards (stage order), probed from the driver without opening them: for settings needed
    /// before the pipeline exists (the MTP draft depth sizes its buffers).
    pub fn probe_stage_profiles(n: usize) -> Vec<lb_card_profile::Resolved> {
        lb_q4_gpu::store::probe_cuda_cards().iter().take(n).map(lb_card_profile::resolve_quiet).collect()
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

    /// The speculative graphs (the verify pass of every stage, the nextn draft step) are recorded here, before any prefill has run, for the same
    /// reason as the decode graphs: a recording started after stage threads wrote shared buffers can be invalidated. Needs `new_spec` and `attach_mtp`.
    pub fn warm_spec(&self, tok: u32) -> Result<()> {
        if self.mtp.is_none() || self.vm < 2 {
            return Ok(());
        }
        let zeros = vec![0f32; self.model.cfg.n_embd];
        for i in 0..3 {
            for m in 1..=self.vm {
                self.run_rows(&vec![tok; m], 0)?;
            }
            self.mtp_draft(&zeros, tok, i)?;
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

    /// Milliseconds a replay of each stage's recorded decode graph takes on its own (`reps` replays queued, one sync): the device time of a
    /// token's stage with the in-graph launch gaps, without the host hand-over between stages. Empty for a stage with no recorded graph.
    pub fn bench_stage_graphs(&self, reps: usize) -> Vec<Option<f64>> {
        self.stages
            .iter()
            .map(|s| {
                if let GraphState::Ready { graph, out, .. } = &*s.graph.borrow() {
                    let t0 = std::time::Instant::now();
                    for _ in 0..reps {
                        unsafe { graph.replay() }.ok()?;
                    }
                    s.gpu.read_all(vec![(out.clone(), 1)]).ok()?;
                    Some(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Device time of the final norm, the output head and the argmax (last stage), microseconds a token.
    pub fn bench_head(&self, reps: usize) -> Vec<(&'static str, f64)> {
        let c = &self.model.cfg;
        let s = self.stages.last().unwrap();
        let h = s.head.as_ref().unwrap();
        let x = s.gpu.upload_f32(&vec![0.1f32; c.n_embd]);
        let nrm = s.gpu.qw_rms(&x, &h.norm, 1, c.n_embd, c.eps).0;
        let head = || match &h.out {
            HeadMat::Q6(m) => m.launch(&s.gpu, &nrm, 1),
            HeadMat::Q8(m) => m.launch(&s.gpu, &nrm, 1),
        };
        let logits = head();
        vec![
            ("output head matvec", crate::device::graph_time_us(&s.gpu, reps, &head)),
            ("argmax", crate::device::graph_time_us(&s.gpu, reps, &|| s.gpu.qw_argmax(&logits, c.vocab))),
        ]
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
    /// token. State and KV cache afterwards are those of token-by-token steps. Any length up to the KV capacity (the attention kernel tiles its scores). `chunk` 0 picks a size giving each stage about four chunks (64..1024 tokens).
    pub fn prefill(&self, tokens: &[u32], pos0: usize, chunk: usize) -> Result<Vec<f32>> {
        self.prefill_h(tokens, pos0, chunk, None)
    }

    /// [`prefill`](Self::prefill) that also appends the last stage's output row (the trunk hidden before the final norm) of every token to `rows`.
    pub fn prefill_h(&self, tokens: &[u32], pos0: usize, chunk: usize, rows: Option<&std::sync::Mutex<Vec<f32>>>) -> Result<Vec<f32>> {
        use std::sync::mpsc;
        let c = &self.model.cfg;
        let cap = self.stages[0].dev.cap;
        if pos0 + tokens.len() > cap {
            return Err(anyhow!("prefill past the KV capacity {cap}"));
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
                let coll = rows;
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
                                cur = s.dev.prefill_layer(&s.gpu, l, &cur, m, false)?;
                                if flush > 0 {
                                    s.gpu.flush_queue();
                                }
                            }
                            s.gpu.read_all(vec![(cur, m * n_embd)]).map_err(|e| anyhow!(e)).map(|mut v| v.remove(0))
                        };
                        match run() {
                            Ok(v) => {
                                if let (None, Some(co)) = (&next, coll) {
                                    co.lock().unwrap().extend_from_slice(&v);
                                }
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

    /// Upload the nextn block (call once, after `new_spec`). Its attention KV cache holds the same `cap` positions as the trunk's.
    pub fn attach_mtp(&mut self, mtp: &Mtp<'_>) -> Result<()> {
        let c = &self.model.cfg;
        let s = self.stages.last().unwrap();
        let dev = DevModel::upload_layers(&s.gpu, c, &[&mtp.layer], s.dev.cap, 0)?;
        let eh = DMat::new(&s.gpu, &mtp.eh_proj)?;
        let x_in = s.gpu.upload_f32(&vec![0.0; 2 * c.n_embd]);
        self.mtp = Some(MtpDev { dev, eh, enorm: mtp.enorm.clone(), hnorm: mtp.hnorm.clone(), shead: s.gpu.upload_f32(&mtp.shead_norm), x_in, graph: RefCell::new(VState::None) });
        Ok(())
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn mtp_cat(&self, hid: &[f32], tok: u32) -> Vec<f32> {
        let c = &self.model.cfg;
        let m = self.mtp.as_ref().unwrap();
        let mut e = vec![0f32; c.n_embd];
        self.model.tok_embd.dequant_row(tok as usize, &mut e);
        let mut cat = crate::rms_norm(&e, &m.enorm, c.eps);
        cat.extend(crate::rms_norm(hid, &m.hnorm, c.eps));
        cat
    }

    /// One nextn step: given the trunk hidden `hid` of the position before `tok` and `tok` at head-cache entry `pos`, writes the entry and returns
    /// the greedy draft (the token after `tok`) and the block's hidden output (the next step's `hid`).
    pub fn mtp_draft(&self, hid: &[f32], tok: u32, pos: usize) -> Result<(u32, Vec<f32>)> {
        let c = &self.model.cfg;
        let m = self.mtp.as_ref().ok_or_else(|| anyhow!("no nextn block"))?;
        let s = self.stages.last().unwrap();
        s.gpu.write_f32(&m.x_in, &self.mtp_cat(hid, tok));
        m.dev.set_pos(&s.gpu, pos);
        let (y, am) = self.run_draft(m, self.graph_on)?;
        let mut r = s.gpu.read_all(vec![(am, 1), (y, c.n_embd)]).map_err(|e| anyhow!(e))?;
        let hid = r.pop().unwrap();
        Ok((r[0][0].to_bits(), hid))
    }

    fn build_draft(&self, m: &MtpDev) -> (Handle, Handle) {
        let c = &self.model.cfg;
        let s = self.stages.last().unwrap();
        let h = s.head.as_ref().unwrap();
        let x = m.eh.launch(&s.gpu, &m.x_in);
        let (_, y) = m.dev.layer_forward(&s.gpu, 0, &x, 0);
        let nrm = s.gpu.qw_rms(&y, &m.shead, 1, c.n_embd, c.eps).0;
        let logits = match &h.out {
            HeadMat::Q6(w) => w.launch(&s.gpu, &nrm, 1),
            HeadMat::Q8(w) => w.launch(&s.gpu, &nrm, 1),
        };
        let am = s.gpu.qw_argmax(&logits, c.vocab);
        (y, am)
    }

    /// One draft step on the device: plain the first time (capture mode on), recorded as a graph the second time, replayed after.
    fn run_draft(&self, m: &MtpDev, graph_on: bool) -> Result<(Handle, Handle)> {
        let s = self.stages.last().unwrap();
        if !graph_on {
            return Ok(self.build_draft(m));
        }
        let kind = match &*m.graph.borrow() {
            VState::Ready { .. } => 0,
            VState::Failed => 1,
            VState::None => 2,
            VState::Prepared => 3,
        };
        if kind == 0 {
            if let VState::Ready { graph, out, .. } = &*m.graph.borrow() {
                // SAFETY: as in `run_stage`
                unsafe { graph.replay() }.map_err(|e| anyhow!("draft graph replay: {e:?}"))?;
                return Ok((out.0.clone(), out.1.clone().unwrap()));
            }
        }
        if kind == 1 {
            return Ok(self.build_draft(m));
        }
        if kind == 2 {
            match s.gpu.client.graph_prepare() {
                Ok(()) => {
                    if std::env::var("LB_SCRATCH_LOG").is_ok() {
                        eprintln!("--- draft pass A (eager)");
                    }
                    let o = self.build_draft(m);
                    s.gpu.read_all(vec![(o.1, 1)]).map_err(|e| anyhow!(e))?;
                    if std::env::var("LB_SCRATCH_LOG").is_ok() {
                        eprintln!("--- draft pass A synced, pooled buffers: {}", s.gpu.pooled());
                    }
                }
                Err(e) => {
                    eprintln!("lb-qwen35: draft graph_prepare refused ({e:?})");
                    *m.graph.borrow_mut() = VState::Failed;
                    return Ok(self.build_draft(m));
                }
            }
        }
        match s.gpu.client.start_capture() {
            Err(e) => {
                eprintln!("lb-qwen35: draft start_capture refused ({e:?})");
                *m.graph.borrow_mut() = VState::Failed;
                Ok(self.build_draft(m))
            }
            Ok(()) => {
                if std::env::var("LB_SCRATCH_LOG").is_ok() {
                    eprintln!("--- draft capture begins");
                }
                let (y, am) = self.build_draft(m);
                match s.gpu.client.stop_capture() {
                    Ok(graph) => {
                        let held = s.gpu.take_in_flight();
                        eprintln!("lb-qwen35: nextn draft step recorded as a CUDA graph");
                        unsafe { graph.replay() }.map_err(|e| anyhow!("draft graph replay: {e:?}"))?;
                        *m.graph.borrow_mut() = VState::Ready { graph, out: (y.clone(), Some(am.clone())), _held: held };
                        Ok((y, am))
                    }
                    Err(e) => {
                        eprintln!("lb-qwen35: draft stop_capture refused ({e:?})");
                        *m.graph.borrow_mut() = VState::Failed;
                        Ok(self.build_draft(m))
                    }
                }
            }
        }
    }

    /// The head-cache entry `pos` only (no draft): the catch-up after a verify.
    pub fn mtp_kv(&self, hid: &[f32], tok: u32, pos: usize) {
        let m = self.mtp.as_ref().unwrap();
        let s = self.stages.last().unwrap();
        s.gpu.write_f32(&m.x_in, &self.mtp_cat(hid, tok));
        m.dev.set_pos(&s.gpu, pos);
        let x = m.eh.launch(&s.gpu, &m.x_in);
        m.dev.kv_only(&s.gpu, 0, &x);
    }

    /// Head-cache entries `j0 .. j0 + toks.len()` in bulk: entry `j0 + i` pairs the trunk hidden row `i` of `hs` (`[n, n_embd]`) with `toks[i]`.
    pub fn mtp_prefill(&self, hs: &[f32], toks: &[u32], j0: usize) -> Result<()> {
        let c = &self.model.cfg;
        let m = self.mtp.as_ref().ok_or_else(|| anyhow!("no nextn block"))?;
        let s = self.stages.last().unwrap();
        let chunk = 128usize;
        let mut off = 0;
        while off < toks.len() {
            let mm = chunk.min(toks.len() - off);
            let mut cat = Vec::with_capacity(mm * 2 * c.n_embd);
            for i in off..off + mm {
                cat.extend(self.mtp_cat(&hs[i * c.n_embd..(i + 1) * c.n_embd], toks[i]));
            }
            let xin = s.gpu.upload_f32(&cat);
            let x = m.eh.gemm(&s.gpu, &xin, mm);
            m.dev.set_chunk(&s.gpu, j0 + off, mm);
            let y = m.dev.prefill_layer(&s.gpu, 0, &x, mm, false)?;
            s.gpu.read_all(vec![(y, 1)]).map_err(|e| anyhow!(e))?;
            off += mm;
        }
        Ok(())
    }

    /// Verify pass: `tokens` at `pos0 ..` through every stage as one batched chunk (the prefill kernels, on this thread so the stream order matches
    /// the decode graphs'). Checkpoints the recurrent state first and records the delta-net inputs so [`spec_rollback`](Self::spec_rollback) can
    /// shorten the chunk. Returns the greedy token after each row and the trunk hidden of each row (`[m, n_embd]`).
    pub fn run_rows(&self, tokens: &[u32], pos0: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let c = &self.model.cfg;
        let m = tokens.len();
        if pos0 + m > self.stages[0].dev.cap {
            return Err(anyhow!("verify past the KV capacity"));
        }
        let mut xh = vec![0f32; m * c.n_embd];
        for (i, &t) in tokens.iter().enumerate() {
            self.model.tok_embd.dequant_row(t as usize, &mut xh[i * c.n_embd..(i + 1) * c.n_embd]);
        }
        let tm = std::env::var("LB_MTP_TIMING").is_ok();
        let nst = self.stages.len();
        let mut g: Vec<u32> = Vec::new();
        for si in 0..nst {
            let t0 = std::time::Instant::now();
            let s = &self.stages[si];
            s.dev.set_chunk(&s.gpu, pos0, m);
            let full = m >= 1 && m <= self.vm && self.graph_on;
            if full {
                s.gpu.write_f32(&s.vx_in[m - 1], &xh);
            }
            let (cur, am) = self.run_verify_stage(si, &xh, m, full)?;
            let mut want = vec![(cur, m * c.n_embd)];
            if let Some(am) = am {
                want.push((am, m));
            }
            let mut r = s.gpu.read_all(want).map_err(|e| anyhow!(e))?;
            if r.len() == 2 {
                g = r.pop().unwrap().into_iter().map(|v| v.to_bits()).collect();
            }
            xh = r.remove(0);
            if tm {
                eprintln!("verify stage {si}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        Ok((g, xh))
    }

    /// The nextn block's graph-free helpers live above; the verify pass is recorded per stage:
    fn build_verify(&self, si: usize, x: &Handle, m: usize, flush: bool) -> (Handle, Option<Handle>) {
        let c = &self.model.cfg;
        let s = &self.stages[si];
        s.dev.spec_checkpoint(&s.gpu);
        let mut cur = x.clone();
        for l in 0..s.dev.layers.len() {
            cur = s.dev.verify_layer(&s.gpu, l, &cur, m);
            if flush {
                s.gpu.flush_queue();
            }
        }
        if let Some(h) = &s.head {
            // final norm, the head over every row, and the argmax of each: part of the last stage's recording
            let nrm = s.gpu.qw_rms(&cur, &h.norm, m, c.n_embd, c.eps).0;
            let logits = match &h.out {
                HeadMat::Q6(w) => w.launch(&s.gpu, &nrm, m),
                HeadMat::Q8(w) => w.launch(&s.gpu, &nrm, m),
            };
            return (cur, Some(s.gpu.qw_argmax_rows(&logits, m, c.vocab)));
        }
        (cur, None)
    }

    /// One stage of a verify pass. The full chunk size runs eagerly the first time, is recorded as a CUDA graph the second time and replayed after.
    fn run_verify_stage(&self, si: usize, xh: &[f32], m: usize, full: bool) -> Result<(Handle, Option<Handle>)> {
        let s = &self.stages[si];
        if !full {
            let x = s.gpu.upload_f32(xh);
            return Ok(self.build_verify(si, &x, m, true));
        }
        let kind = match &*s.vgraph[m - 1].borrow() {
            VState::Ready { .. } => 0,
            VState::Failed => 1,
            VState::None => 2,
            VState::Prepared => 3,
        };
        if kind == 0 {
            if let VState::Ready { graph, out, .. } = &*s.vgraph[m - 1].borrow() {
                // SAFETY: as in `run_stage`: the recording's buffers are held, replayed on the stream it was recorded on, inputs written on it above
                unsafe { graph.replay() }.map_err(|e| anyhow!("verify graph replay: {e:?}"))?;
                return Ok((out.0.clone(), out.1.clone()));
            }
        }
        if kind == 1 {
            return Ok(self.build_verify(si, &s.vx_in[m - 1], m, true));
        }
        if kind == 2 {
            // capture mode on, one plain pass (it fills the scratch pool and settles the cached constants), the chunk undone, and the
            // recording follows at once: nothing else touches this stream between the prepare and the capture
            match s.gpu.client.graph_prepare() {
                Ok(()) => {
                    let o = self.build_verify(si, &s.vx_in[m - 1], m, true);
                    s.dev.spec_restore(&s.gpu);
                    s.gpu.read_all(vec![(o.0, 1)]).map_err(|e| anyhow!(e))?;
                }
                Err(e) => {
                    eprintln!("lb-qwen35: verify graph_prepare refused ({e:?})");
                    *s.vgraph[m - 1].borrow_mut() = VState::Failed;
                    return Ok(self.build_verify(si, &s.vx_in[m - 1], m, true));
                }
            }
        }
        let mut vs = s.vgraph[m - 1].borrow_mut();
        match s.gpu.client.start_capture() {
            Err(e) => {
                eprintln!("lb-qwen35: verify start_capture refused ({e:?})");
                *vs = VState::Failed;
                drop(vs);
                Ok(self.build_verify(si, &s.vx_in[m - 1], m, true))
            }
            Ok(()) => {
                if std::env::var("LB_SCRATCH_LOG").is_ok() {
                    eprintln!("--- verify capture begins, stage {si}");
                }
                let out = self.build_verify(si, &s.vx_in[m - 1], m, false);
                match s.gpu.client.stop_capture() {
                    Ok(graph) => {
                        let held = s.gpu.take_in_flight();
                        eprintln!("lb-qwen35: stage {si} verify pass (m = {m}) recorded as a CUDA graph");
                        unsafe { graph.replay() }.map_err(|e| anyhow!("verify graph replay: {e:?}"))?;
                        *vs = VState::Ready { graph, out: out.clone(), _held: held };
                        Ok(out)
                    }
                    Err(e) => {
                        eprintln!("lb-qwen35: verify stop_capture refused ({e:?})");
                        *vs = VState::Failed;
                        drop(vs);
                        Ok(self.build_verify(si, &s.vx_in[m - 1], m, true))
                    }
                }
            }
        }
    }

    /// After [`run_rows`](Self::run_rows) at `pos0`: leave the recurrent state as after only the first `keep` of its tokens.
    pub fn spec_rollback(&self, pos0: usize, keep: usize) {
        for s in &self.stages {
            s.dev.spec_replay(&s.gpu, pos0, keep);
        }
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
