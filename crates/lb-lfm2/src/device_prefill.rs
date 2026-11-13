//! Device-resident prefill: a prompt of `n` tokens in chunks of `LB_PREFILL_UBATCH` rows (default
//! 2048), each chunk from the embedding upload to the last row's logits on the device, with the KV
//! cache and the conv states written where device decode reads them. `LB_DEVICE_PREFILL=1` (or the
//! live `device_prefill` knob) turns it on.
//!
//! Why: the host pass (`Lfm2::forward_host`) ran attention, norms, RoPE, the conv, routing and the
//! expert sum on the host through Burn, and attention materialised `[heads, m, total]` scores plus a
//! host-built mask. On a P100 that was 63-69 prompt tok/s and out of memory at a 2k prompt (X6,
//! `docs/WORK_TO_WEIGHTS.md`), against llama.cpp's 600-1000 on the same card.
//!
//! ```text
//! per chunk (rows c0 .. c0+mc at positions p0 = pos0 + c0):
//!   host: dequantize mc embedding rows, upload as the chunk's residual x            (1 upload)
//!   per layer:  h  = rms_rows(x, attn_norm)
//!               conv: in_proj -> conv_prefill (device state advances mc steps) -> out_proj
//!               attn: q, k, v -> qk_norm_rope_rows (+ KV rows p0..) -> attn_prefill -> o
//!               h2 = rms_rows(x += op_out, ffn_norm)
//!               dense FFN: gate_up_silu -> down       MoE: router_rows -> experts -> sum
//!               x += ffn_out; sync + recycle the layer's scratch (the card is nearly full)
//!   last chunk: rms_rows(last row, final norm) -> head -> read logits                   (1 read)
//! ```
//!
//! The matmuls: chunks of at least `LB_PREFILL_GEMM_MIN` rows (16) take the tiled GEMM
//! (`lb_q4_gpu::gemm`: each weight tile read once per 64 rows, on a tensor-core card the f16
//! tensor-core form; MoE slots grouped by expert on the host after one read of the routes);
//! smaller chunks the rows kernels. Every glue kernel is the batched form of its decode kernel and
//! bit-identical to running it token by token (`tests/prefill_kernels_match_decode.rs`).
//!
//! **Prefill card (the split).** With a [`PrefillCard`] attached (`Lfm2::attach_prefill_card`,
//! `LB_PREFILL_CARD=<cuda index>`), a second card of this process holds whole layers `0 .. K` and
//! their own KV / conv state: each chunk runs layers `0 .. K` there, the residual hops to the
//! decode card (`Gpu::copy_to`, device to device when NCCL loads, else through the host), and the
//! decode card runs `K ..` and the head. After the prompt the prefill card's KV and conv states
//! for its layers are copied into the decode card's cache, so decode continues exactly as if one
//! card had done it. The point: a Turing card's tensor cores take most of the prompt's FLOPs while
//! the P100's bandwidth does the decode (X6: 3.7-4.1x the P100's prefill rate in llama.cpp).
//!
//! **When it runs.** Everything device decode needs (no blocker, every MoE layer a stacked resident
//! layer), no remote tier and no in-box expert cards. Otherwise it declines before touching any
//! state and the host path runs.
//!
//! **State.** Transactional: if the device is ahead of the host it is pulled first, so a failure
//! part-way only drops the device copy (`cache.dev = None`) and the host path redoes the prompt.
//! The prefill card's state is marked invalid while a prompt runs and valid only on success. On
//! success the device is ahead (`cache.host_len` stays behind `cache.len`), exactly as after device
//! decode tokens, so decode continues on the device and a later host pass pulls once.

use crate::device_decode::{head, pull_if_stale, push, Head, LayerPrep, OpPrep, Prep};
use crate::model::{Cache, Lfm2};
use crate::moe::Route;
use crate::weights::{Ffn, MatRef, Operator};
use burn::tensor::{Tensor, TensorData};
use lb_q4_gpu::cubecl_server::Handle;
use lb_q4_gpu::decode::rope_table;
use lb_q4_gpu::{Gpu, GpuExpertLayer, GpuQ4Mat};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Smallest chunk that takes the tiled GEMM (`lb_q4_gpu::gemm`) instead of the rows kernels:
/// `LB_PREFILL_GEMM_MIN`, default 16 (`0` = always the rows kernels).
pub fn gemm_min() -> usize {
    std::env::var("LB_PREFILL_GEMM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
}

/// Rows per chunk: `LB_PREFILL_UBATCH`, default 2048 (fills the expert tiles: 780 vs 594 tok/s on a
/// P100 against 512, docs/PREFILL_SPLIT.md).
pub fn ubatch() -> usize {
    std::env::var("LB_PREFILL_UBATCH").ok().and_then(|v| v.parse().ok()).filter(|v: &usize| *v > 0).unwrap_or(2048)
}

/// `LB_PREFILL_TIMING=1`: sync after every kind of step and total the wall time per kind
/// ([`TIMING`]). Slower (hundreds of extra syncs a chunk); for finding where a chunk's time goes.
pub fn timing_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LB_PREFILL_TIMING").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Step kinds of [`TIMING`].
pub const TIMING_NAMES: [&str; 8] = ["embed+upload", "norms+adds", "dense gemm", "conv", "attention", "router+grouping", "expert gemm", "expert sum"];
/// Nanoseconds per step kind, summed over every chunk and card since start (timing mode only).
pub static TIMING: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// In timing mode: wait for the card and charge the time since `t` to kind `k`; restart `t`.
fn mark(gpu: &Gpu, t: &mut Instant, k: usize) {
    if timing_on() {
        let _ = gpu.sync();
        TIMING[k].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        *t = Instant::now();
    }
}

fn decline(why: &str) -> Option<Tensor<2>> {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| eprintln!("lb-lfm2: device prefill declined, the host path runs: {why}"));
    None
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// What one card needs to run a range of layers on a chunk: its device, the matrices and expert
/// stacks it holds (keyed as the model keys them: `MatRef::start`, layer index), the per-layer
/// constants (`LayerPrep`), its KV / conv buffers, its RoPE table and its routes buffers.
struct CardView<'a> {
    gpu: &'a Gpu,
    mats: &'a HashMap<usize, GpuQ4Mat>,
    stacks: &'a HashMap<usize, GpuExpertLayer>,
    layers: &'a [LayerPrep],
    k: &'a [Option<Handle>],
    v: &'a [Option<Handle>],
    conv: &'a [Option<Handle>],
    rope: &'a Handle,
    rope_rows: usize,
    routes: &'a Handle,
    weights: &'a Handle,
}

impl CardView<'_> {
    fn mat(&self, r: &MatRef) -> &GpuQ4Mat {
        self.mats.get(&r.start).expect("matrix resident on this card (checked when the card was built)")
    }
}

/// Settings of one chunk, the same for every card.
struct Pass {
    mc: usize,
    p0: usize,
    cap: usize,
    gemm: bool,
    seq: bool,
    scrape: bool,
}

/// Run layers `range` of the chunk whose residual is `x` (on `cv`'s card). Each layer ends with a
/// sync that recycles its scratch, so only `x`, the caches and the routes stay allocated.
fn run_layers(m: &Lfm2, cv: &CardView, range: Range<usize>, x: &Handle, ps: &Pass, hits: &mut u64) -> Result<(), String> {
    let cfg = m.cfg();
    let (e, l, hd, eps) = (cfg.n_embd, cfg.l_cache, cfg.head_dim(), cfg.rms_eps);
    let (nh, k_used, n_exp) = (cfg.n_head, cfg.n_expert_used, cfg.n_expert);
    let (gpu, mc, p0, cap) = (cv.gpu, ps.mc, ps.p0, ps.cap);
    let lin = |r: &MatRef, h: &Handle| if ps.gemm { cv.mat(r).launch_gemm(gpu, h, mc) } else { cv.mat(r).launch(gpu, h, mc) };
    let mut tm = Instant::now();
    for li in range {
        let layer = &m.w.layers[li];
        let lp = &cv.layers[li];
        let h = gpu.rms_norm_rows(x, None, &lp.attn_norm, mc, e, eps);
        mark(gpu, &mut tm, 1);
        let op_out = match (&layer.op, &lp.op) {
            (Operator::Conv { in_proj, out_proj, .. }, OpPrep::Conv { taps }) => {
                let bcx = lin(in_proj, &h);
                mark(gpu, &mut tm, 2);
                let state = cv.conv[li].as_ref().expect("conv layer state");
                let yy = gpu.conv_prefill(&bcx, taps, state, mc, e, l);
                mark(gpu, &mut tm, 3);
                let o = lin(out_proj, &yy);
                mark(gpu, &mut tm, 2);
                o
            }
            (Operator::Attn { q, k, v, o, .. }, OpPrep::Attn { q_norm, k_norm }) => {
                let kvh = cfg.kv_heads[li];
                let (qh, kh, vh) = (lin(q, &h), lin(k, &h), lin(v, &h));
                mark(gpu, &mut tm, 2);
                let (kc, vc) = (cv.k[li].as_ref().expect("k cache"), cv.v[li].as_ref().expect("v cache"));
                let qo = gpu.qk_norm_rope_rows(&qh, &kh, &vh, q_norm, k_norm, cv.rope, cv.rope_rows, kc, vc, (mc, p0, cap), (nh, kvh, hd), eps);
                let att = gpu.attn_prefill(&qo, kc, vc, (mc, p0, cap), (nh, kvh, hd));
                mark(gpu, &mut tm, 4);
                let ao = lin(o, &att);
                mark(gpu, &mut tm, 2);
                ao
            }
            _ => unreachable!("layer {li}: operator and its device copy disagree"),
        };
        let h2 = gpu.rms_norm_rows(x, Some(&op_out), &lp.ffn_norm, mc, e, eps);
        mark(gpu, &mut tm, 1);
        let mut is_moe = false;
        let mut scraped = false;
        let ffn_out = match &layer.ffn {
            Ffn::Dense { gate, up, down } => {
                let hh = if ps.gemm { GpuQ4Mat::launch_gemm_gate_up(gpu, cv.mat(gate), cv.mat(up), &h2, mc) } else { GpuQ4Mat::launch_gate_up_silu(gpu, cv.mat(gate), cv.mat(up), &h2, mc) };
                let o = lin(down, &hh);
                mark(gpu, &mut tm, 2);
                o
            }
            Ffn::Moe { .. } => {
                let stack = cv.stacks.get(&li).expect("stacked layer resident on this card");
                let (rh, bh) = lp.router.as_ref().expect("MoE layer router");
                gpu.router_topk_rows(&h2, rh, bh, cv.routes, cv.weights, (mc, n_exp, k_used, e), ps.seq);
                is_moe = true;
                *hits += (mc * k_used) as u64;
                let slots = mc * k_used;
                let outs = if ps.gemm {
                    // the host groups the chunk's slots by expert: one read of the routes (and the
                    // weights, for the scraper) in the middle of the layer
                    let want = if ps.scrape { vec![(cv.routes.clone(), 8 * slots), (cv.weights.clone(), 4 * slots)] } else { vec![(cv.routes.clone(), 8 * slots)] };
                    let b = gpu.read_raw_keep(want).map_err(|err| format!("layer {li} routes: {err}"))?;
                    let r = u32s(&b[0]);
                    if ps.scrape {
                        record_routes(m, li, p0, mc, k_used, &r, &f32s(&b[1]));
                        scraped = true;
                    }
                    let (perm, tiles) = lb_q4_gpu::gemm::group_slots(&r, n_exp);
                    let n_tiles = tiles.len() / 3;
                    let (ph, th) = (gpu.upload_u32(perm), gpu.upload_u32(tiles));
                    mark(gpu, &mut tm, 5);
                    stack.launch_grouped(gpu, &h2, mc, cv.routes, slots, &ph, &th, n_tiles).0
                } else {
                    stack.launch_device(gpu, &h2, mc, cv.routes, slots, 0, slots).0
                };
                mark(gpu, &mut tm, 6);
                let y = gpu.weighted_sum_rows(&outs, cv.weights, cv.routes, (mc, k_used, e));
                mark(gpu, &mut tm, 7);
                y
            }
        };
        gpu.add_rows(x, &ffn_out, mc * e);
        mark(gpu, &mut tm, 1);
        if is_moe && ps.scrape && !scraped {
            let b = gpu.read_raw(vec![(cv.routes.clone(), 8 * mc * k_used), (cv.weights.clone(), 4 * mc * k_used)]).map_err(|err| format!("layer {li} routes: {err}"))?;
            record_routes(m, li, p0, mc, k_used, &u32s(&b[0]), &f32s(&b[1]));
        } else {
            gpu.sync_recycle().map_err(|err| format!("layer {li} fence: {err}"))?;
        }
    }
    Ok(())
}

/// One layer's routes for a chunk, to the scraper (the host path's record call, same shape).
fn record_routes(m: &Lfm2, li: usize, p0: usize, mc: usize, k: usize, r: &[u32], w: &[f32]) {
    let rs: Vec<Route> = (0..mc).map(|t| Route { experts: (0..k).map(|i| r[2 * (t * k + i)] as usize).collect(), weights: w[t * k..(t + 1) * k].to_vec() }).collect();
    m.scraper.record(li, p0, &rs);
}

/// A second card of this process that runs the first `n_layers` layers of every prefill chunk
/// (see the module doc). Built by [`Lfm2::attach_prefill_card`].
pub struct PrefillCard {
    pub index: usize,
    pub gpu: Gpu,
    /// Layers `0 .. n_layers` run here.
    pub n_layers: usize,
    pub bytes: usize,
    mats: HashMap<usize, GpuQ4Mat>,
    stacks: HashMap<usize, GpuExpertLayer>,
    layers: Vec<LayerPrep>,
    state: Mutex<CardState>,
    pub stats: PrefillCardStats,
}

#[derive(Default, Debug)]
pub struct PrefillCardStats {
    pub chunks: AtomicU64,
    /// Wall time of this card's layers, of the residual hops, and of the state copies (decode card
    /// to prefill card before a prompt, back after it).
    pub card_ns: AtomicU64,
    pub hop_ns: AtomicU64,
    pub state_copy_ns: AtomicU64,
}

/// The prefill card's own KV / conv buffers for its layers, valid for positions `0 .. len`
/// (`usize::MAX` = invalid: copy from the decode card before use).
struct CardState {
    k: Vec<Option<Handle>>,
    v: Vec<Option<Handle>>,
    conv: Vec<Option<Handle>>,
    cap: usize,
    len: usize,
    rope: Option<(Handle, usize)>,
    routes: Option<(Handle, Handle, usize)>,
}

/// Device bytes a card needs for layer `li`'s quantized weights (Q4_0: 18 bytes a 32-block).
fn layer_bytes(m: &Lfm2, li: usize) -> usize {
    let q4 = |r: &MatRef| r.rows * r.cols / 32 * 18;
    let layer = &m.w.layers[li];
    let op = match &layer.op {
        Operator::Conv { in_proj, out_proj, .. } => q4(in_proj) + q4(out_proj),
        Operator::Attn { q, k, v, o, .. } => q4(q) + q4(k) + q4(v) + q4(o),
    };
    let ffn = match &layer.ffn {
        Ffn::Dense { gate, up, down } => q4(gate) + q4(up) + q4(down),
        Ffn::Moe { gate, up, down, .. } => q4(&gate.all()) + q4(&up.all()) + q4(&down.all()),
    };
    op + ffn
}

impl Lfm2 {
    /// Open CUDA device `index` as the prefill card and upload whole layers `0 .. K` to it, `K` =
    /// `n_layers` if given, else as many leading layers as fit `budget_mb`. Needs the cuda build.
    pub fn attach_prefill_card(&mut self, index: usize, n_layers: Option<usize>, budget_mb: usize) -> Result<String, String> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (index, n_layers, budget_mb);
            Err("a prefill card needs the cuda build".into())
        }
        #[cfg(feature = "cuda")]
        {
            let n = self.cfg().n_layer;
            let k = match n_layers {
                Some(k) => k.min(n),
                None => {
                    let (mut k, mut used) = (0usize, 0usize);
                    while k < n && used + layer_bytes(self, k) <= budget_mb * 1_000_000 {
                        used += layer_bytes(self, k);
                        k += 1;
                    }
                    k
                }
            };
            if k == 0 {
                return Err(format!("no whole layer fits the prefill card's budget of {budget_mb} MB (layer 0 needs {:.0} MB)", layer_bytes(self, 0) as f64 / 1e6));
            }
            let t = Instant::now();
            let gpu = Gpu::open_cuda(index)?;
            let (mut mats, mut stacks, mut bytes) = (HashMap::new(), HashMap::new(), 0usize);
            for li in 0..k {
                let layer = &self.w.layers[li];
                let mut put = |r: &MatRef| -> Result<(), String> {
                    let g = GpuQ4Mat::upload(&gpu, self.w.mat(r))?;
                    bytes += g.bytes_on_device();
                    mats.insert(r.start, g);
                    Ok(())
                };
                match &layer.op {
                    Operator::Conv { in_proj, out_proj, .. } => {
                        put(in_proj)?;
                        put(out_proj)?;
                    }
                    Operator::Attn { q, k, v, o, .. } => {
                        for r in [q, k, v, o] {
                            put(r)?;
                        }
                    }
                }
                match &layer.ffn {
                    Ffn::Dense { gate, up, down } => {
                        for r in [gate, up, down] {
                            put(r)?;
                        }
                    }
                    Ffn::Moe { gate, up, down, .. } => {
                        let s = GpuExpertLayer::upload_stacked(&gpu, self.w.mat(&gate.all()), self.w.mat(&up.all()), self.w.mat(&down.all()), gate.n)?;
                        bytes += s.bytes_on_device();
                        stacks.insert(li, s);
                    }
                }
            }
            // the small per-layer constants (norms, taps, routers) for every layer: cheap, and the
            // layer index stays the model's
            let layers = Prep::new(self, &gpu).layers;
            gpu.sync()?;
            let msg = format!(
                "prefill card: CUDA device {index} ({}), layers 0..{k} of {n}, {:.0} MiB in {:.1}s; tensor cores {}; hops {}",
                gpu.name,
                bytes as f64 / 1048576.0,
                t.elapsed().as_secs_f64(),
                if gpu.tensor_cores { "yes" } else { "NO (f32 tiles)" },
                if gpu.device_transport() { "device to device" } else { "through the host" }
            );
            let st = CardState { k: vec![None; n], v: vec![None; n], conv: vec![None; n], cap: 0, len: usize::MAX, rope: None, routes: None };
            self.pcard = Some(PrefillCard { index, gpu, n_layers: k, bytes, mats, stacks, layers, state: Mutex::new(st), stats: PrefillCardStats::default() });
            Ok(msg)
        }
    }

    /// `LB_PREFILL_CARD=<cuda index>` (with `LB_PREFILL_LAYERS=K` or `LB_PREFILL_CARD_MB`, default
    /// 10500) attaches a prefill card; nothing set = none. For hosts (lfm2-check, lb-server).
    pub fn attach_prefill_card_from_env(&mut self) -> Result<Option<String>, String> {
        let Some(index) = std::env::var("LB_PREFILL_CARD").ok().filter(|v| !v.is_empty()) else { return Ok(None) };
        let index: usize = index.parse().map_err(|_| format!("LB_PREFILL_CARD={index}: not a CUDA device index"))?;
        let layers = std::env::var("LB_PREFILL_LAYERS").ok().and_then(|v| v.parse().ok());
        let mb = std::env::var("LB_PREFILL_CARD_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(10_500);
        self.attach_prefill_card(index, layers, mb).map(Some)
    }
}

/// Run `tokens` at positions `cache.len ..` on the device. Returns the logits (`[n, vocab]` with
/// `all_logits`, else `[1, vocab]` for the last position), or `None` with nothing changed.
pub fn prefill(m: &Lfm2, tokens: &[u32], cache: &mut Cache, all_logits: bool) -> Option<Tensor<2>> {
    let gpu = m.store.gpu.as_ref()?;
    if m.store.remote.is_some() {
        return decline("a remote expert tier is attached");
    }
    if m.store.inbox.is_some() {
        return decline("in-box expert cards are attached (not wired into prefill yet)");
    }
    let n = tokens.len();
    if n == 0 {
        return None;
    }
    let knobs = m.store.knobs.clone();
    let mut guard = m.dd.lock().expect("device decode state");
    let prep: &mut Prep = guard.get_or_insert_with(|| Prep::new(m, gpu));
    if let Some(b) = prep.blocker.as_ref() {
        return decline(&b.clone());
    }
    if !prep.host_moe_layers.is_empty() {
        return decline(&format!("{} MoE layer(s) not resident as stacked layers", prep.host_moe_layers.len()));
    }
    let t0 = Instant::now();
    let cfg = m.cfg();
    let (e, hd, eps, k_used, vocab, n_layer) = (cfg.n_embd, cfg.head_dim(), cfg.rms_eps, cfg.n_expert_used, cfg.n_vocab, cfg.n_layer);
    let pos0 = cache.len;
    let total = pos0 + n;

    // transactional: the host holds everything before pos0, so a failure can drop the device copy
    pull_if_stale(m, cache);
    m.ensure_cap(cache, total);
    let fresh = cache.dev.as_ref().is_none_or(|d| d.cap != cache.cap || d.len != pos0);
    if fresh {
        push(m, gpu, cache);
    }
    let cap = cache.cap;
    if prep.rope.as_ref().is_none_or(|(_, rows)| *rows < cap) {
        prep.rope = Some((gpu.upload_f32(&rope_table(cfg.rope_theta, hd, cap)), cap));
    }
    let (rope, rope_rows) = prep.rope.as_ref().map(|(h, r)| (h.clone(), *r)).expect("just built");
    let (dev_k, dev_v, dev_conv) = {
        let d = cache.dev.as_ref().expect("just pushed");
        (d.k.clone(), d.v.clone(), d.conv.clone())
    };

    let ub = ubatch().min(n);
    // one routes/weights pair for the chunk, reused by every MoE layer
    let routes = gpu.upload_u32(vec![0u32; 2 * ub * k_used]);
    let weights = gpu.upload_f32(&vec![0f32; ub * k_used]);
    let mut hits = 0u64;
    let gmin = gemm_min();
    let (seq, scrape) = (knobs.dd_router_seq(), m.scraper.enabled());
    let main = CardView { gpu, mats: &m.dense.resident, stacks: &m.store.layers, layers: &prep.layers, k: &dev_k, v: &dev_v, conv: &dev_conv, rope: &rope, rope_rows, routes: &routes, weights: &weights };

    // the prefill card: its state brought to pos0 from the decode card's (the source of truth),
    // then marked invalid until the prompt has finished
    let pc = m.pcard.as_ref().filter(|p| p.n_layers > 0);
    let split = pc.map(|p| p.n_layers.min(n_layer)).unwrap_or(0);
    let mut pstate = pc.map(|p| p.state.lock().expect("prefill card state"));
    if let (Some(p), Some(st)) = (pc, pstate.as_mut()) {
        let t = Instant::now();
        let pg = &p.gpu;
        if st.cap != cap || st.len != pos0 {
            for li in 0..split {
                if let (Some(k), Some(v)) = (&dev_k[li], &dev_v[li]) {
                    st.k[li] = Some(gpu.copy_to(k, pg));
                    st.v[li] = Some(gpu.copy_to(v, pg));
                }
                if let Some(c) = &dev_conv[li] {
                    st.conv[li] = Some(gpu.copy_to(c, pg));
                }
            }
            st.cap = cap;
        }
        st.len = usize::MAX;
        if st.rope.as_ref().is_none_or(|(_, rows)| *rows < cap) {
            st.rope = Some((pg.upload_f32(&rope_table(cfg.rope_theta, hd, cap)), cap));
        }
        if st.routes.as_ref().is_none_or(|(_, _, u)| *u < ub) {
            st.routes = Some((pg.upload_u32(vec![0u32; 2 * ub * k_used]), pg.upload_f32(&vec![0f32; ub * k_used]), ub));
        }
        p.stats.state_copy_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let mut logits_out: Vec<f32> = Vec::with_capacity(if all_logits { n * vocab } else { vocab });
    let mut c0 = 0usize;
    while c0 < n {
        let mc = ub.min(n - c0);
        let p0 = pos0 + c0;
        let ps = Pass { mc, p0, cap, gemm: gmin > 0 && mc >= gmin, seq, scrape };
        let t_emb = Instant::now();
        let mut emb = vec![0f32; mc * e];
        for (i, &t) in tokens[c0..c0 + mc].iter().enumerate() {
            m.w.embed_row(t, &mut emb[i * e..(i + 1) * e]);
        }
        if timing_on() {
            TIMING[0].fetch_add(t_emb.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        // the chunk's residual stream: a real buffer (not scratch), so a layer-end recycle keeps it
        let x: Handle = match (pc, pstate.as_ref()) {
            (Some(p), Some(st)) => {
                let t = Instant::now();
                let (rt, wt, _) = st.routes.as_ref().expect("built above");
                let (prope, prows) = st.rope.as_ref().map(|(h, r)| (h, *r)).expect("built above");
                let cv = CardView { gpu: &p.gpu, mats: &p.mats, stacks: &p.stacks, layers: &p.layers, k: &st.k, v: &st.v, conv: &st.conv, rope: prope, rope_rows: prows, routes: rt, weights: wt };
                let xa = p.gpu.upload_f32(&emb);
                if let Err(err) = run_layers(m, &cv, 0..split, &xa, &ps, &mut hits) {
                    return fail(m, cache, &format!("prefill card, chunk at {p0}: {err}"));
                }
                p.stats.card_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let t = Instant::now();
                let xb = p.gpu.copy_to(&xa, gpu);
                p.stats.hop_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                p.stats.chunks.fetch_add(1, Ordering::Relaxed);
                xb
            }
            _ => gpu.upload_f32(&emb),
        };
        if let Err(err) = run_layers(m, &main, split..n_layer, &x, &ps, &mut hits) {
            return fail(m, cache, &format!("chunk at {p0}: {err}"));
        }
        let last = c0 + mc == n;
        if all_logits || last {
            let (from, rows) = if all_logits { (0, mc) } else { (mc - 1, 1) };
            let xs = if from == 0 { x.clone() } else { x.clone().offset_start((from * e * 4) as u64) };
            let hf = gpu.rms_norm_rows(&xs, None, &prep.final_norm, rows, e, eps);
            let lg = match head(m).expect("checked in Prep::new") {
                Head::Q6(h6) => h6.launch(gpu, &hf, rows),
                Head::Q4(h4) => h4.launch(gpu, &hf, rows),
            };
            match gpu.read_raw(vec![(lg, rows * vocab * 4)]) {
                Ok(b) => logits_out.extend(f32s(&b[0])),
                Err(err) => return fail(m, cache, &format!("logits of the chunk at {p0}: {err}")),
            }
        }
        c0 += mc;
    }

    // the prefill card's layers: its KV and conv states become the decode card's
    if let (Some(p), Some(st)) = (pc, pstate.as_mut()) {
        let t = Instant::now();
        let d = cache.dev.as_mut().expect("pushed above");
        for li in 0..split {
            if let (Some(k), Some(v)) = (&st.k[li], &st.v[li]) {
                d.k[li] = Some(p.gpu.copy_to(k, gpu));
                d.v[li] = Some(p.gpu.copy_to(v, gpu));
            }
            if let Some(c) = &st.conv[li] {
                d.conv[li] = Some(p.gpu.copy_to(c, gpu));
            }
        }
        if let Err(err) = gpu.sync() {
            return fail(m, cache, &format!("prefill card state copy: {err}"));
        }
        st.len = total;
        p.stats.state_copy_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    m.stats.gpu_hits.fetch_add(hits, Ordering::Relaxed);
    if let Some(d) = cache.dev.as_mut() {
        d.len = total;
    }
    cache.len = total;
    let t = &m.timings;
    t.dp_tokens.fetch_add(n as u64, Ordering::Relaxed);
    t.dp_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let rows = if all_logits { n } else { 1 };
    Some(Tensor::from_data(TensorData::new(logits_out, [rows, vocab]), &m.device))
}

/// A failure part-way: the device copy is half-advanced, so drop it. The host state is current
/// (pulled at the start), so the host path can redo the prompt from `cache.len`. A prefill card's
/// state was marked invalid when the prompt started, so it is re-copied next time.
fn fail(m: &Lfm2, cache: &mut Cache, why: &str) -> Option<Tensor<2>> {
    assert!(cache.host_len >= cache.len, "device prefill: {why}, with the host behind: cannot continue");
    eprintln!("lb-lfm2: device prefill abandoned ({why}); the host path runs the prompt");
    cache.dev = None;
    m.timings.dd_fallbacks.fetch_add(1, Ordering::Relaxed);
    None
}
