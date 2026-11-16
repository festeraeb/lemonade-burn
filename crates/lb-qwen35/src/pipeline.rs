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
    Ready { graph: CubeGraph, out: Handle, _held: Vec<Handle> },
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

    pub fn reset(&self) {
        for s in &self.stages {
            s.dev.reset(&s.gpu);
        }
    }

    /// The launches of stage `si` for one token: its layers from `x_in`, then (last stage) the final norm and head.
    fn build(&self, si: usize, flush: bool) -> Handle {
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
            return match &h.out {
                HeadMat::Q6(m) => m.launch(&s.gpu, &nrm, 1),
                HeadMat::Q8(m) => m.launch(&s.gpu, &nrm, 1),
            };
        }
        cur
    }

    /// Run stage `si` on `xh` at `pos`; returns the handle holding its output (activation, or logits for the last stage).
    /// With `LB_QW_GRAPH` (default) the stage is recorded as one CUDA graph on its second token and replayed afterwards.
    fn run_stage(&self, si: usize, xh: &[f32], pos: usize) -> Result<Handle> {
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
                    let out = self.build(si, false);
                    match s.gpu.client.stop_capture() {
                        Ok(graph) => {
                            let held = s.gpu.take_in_flight();
                            eprintln!("lb-qwen35: stage {si} recorded as a CUDA graph");
                            *gs = GraphState::Ready { graph, out, _held: held };
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
            if let GraphState::Ready { graph, out, .. } = &*gs {
                // SAFETY: every handle the recording used is held by the stage or by `held`; replayed only on the stream it was
                // recorded on (this thread's), and this token's inputs (x_in, position) were written on that stream above.
                unsafe { graph.replay() }.map_err(|e| anyhow!("graph replay: {e:?}"))?;
                return Ok(out.clone());
            }
        }
        drop(gs);
        Ok(self.build(si, self.flush > 0))
    }

    /// One token at `pos`. Returns the logits when `want_logits`.
    pub fn step(&self, tok: u32, pos: usize, want_logits: bool) -> Result<Option<Vec<f32>>> {
        let c = &self.model.cfg;
        let mut xh = vec![0f32; c.n_embd];
        self.model.tok_embd.dequant_row(tok as usize, &mut xh);
        let nst = self.stages.len();
        for si in 0..nst {
            let t0 = std::time::Instant::now();
            let out = self.run_stage(si, &xh, pos)?;
            let s = &self.stages[si];
            if self.timing {
                s.gpu.flush_queue();
                eprintln!("stage {si} issue {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            if si + 1 == nst {
                let n = if s.head.is_some() { c.vocab } else { c.n_embd };
                let r = s.gpu.read_all(vec![(out, n)]).map_err(|e| anyhow!(e))?.remove(0);
                return Ok(want_logits.then_some(r));
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

    pub fn first_layer_of_stage(&self, i: usize) -> usize {
        self.stages[i].first
    }
}
