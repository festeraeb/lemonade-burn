//! Device-resident decode: one token, one upload, one read-back (Phase D1 of
//! `docs/DEVICE_DECODE_PLAN.md`; on by default since 2026-11-07, `LB_DEVICE_DECODE=0` or the
//! live `device_decode` knob switches it off).
//!
//! The old decode token left the card about 145 times: every quantized matvec read its result
//! back, the norms, RoPE, attention and the residual adds ran as Burn ops on the host, and every
//! MoE layer routed on the host from a read-back activation and summed its experts there. Here
//! the whole token stays on the device and only `lb_q4_gpu` handles appear between the embedding
//! upload and the logits:
//!
//! ```text
//! host: dequantize the token's embedding row, write it into the residual buffer x  (1 upload)
//! per layer:  h  = add_rms_norm(x, y_prev, attn_norm)           y_prev: last layer's FFN out
//!             conv: in_proj -> conv_decode (device state) -> out_proj
//!             attn: q, k, v -> qk_norm_rope (+ KV append) -> attn_decode -> o
//!             h2 = add_rms_norm(x, op_out, ffn_norm)
//!             dense FFN: gate_up_silu -> down
//!             MoE: router_topk -> stacked experts on the device routes -> weighted_sum
//! final:      add_rms_norm(x, y_last, token_embd_norm) -> head
//! one read:   logits + every MoE layer's routes and weights                       (1 sync)
//! ```
//! The scraper is fed from the read-back routes afterwards, in layer order, so a scrape file is
//! the same as the old path's for the same routes.
//!
//! **When it runs.** A single-token pass with the knob on, the device open, `LB_CONV_BLOCK` off,
//! every dense matrix and the head resident, and every MoE layer resident as a stacked layer
//! (`LB_EXPERT_LAYERS`, `Placement::whole_layers` or a full placement). Anything missing and the
//! token goes to the old path unchanged (counted in `Timings::dd_fallbacks`, reason logged once).
//! With `dd_host_moe` (`LB_DD_HOST_MOE=1`) a MoE layer that is not resident runs on the old host
//! path inside the device token instead, at one extra sync for that layer: that is how a card too
//! small for the model (the 1070 test card) runs the device path at all.
//!
//! **State.** The KV cache and the conv states have a device copy (`DevCache`) beside the host's
//! Burn tensors. After a device token the device is ahead (`Cache::host_len < Cache::len`); the
//! next host pass (a prefill, a fallback token, the knob switched off) first reads the device
//! state back, once. A device token whose device copy is behind (after a prefill, or a host
//! token) first uploads the host's, once. So the two paths can alternate freely, token by token.
//!
//! **Numbers.** RoPE uses the host's f64 table (uploaded once per cache size, not per token: a
//! deviation from the plan's in-kernel cos/sin, chosen so the rotation is the host's to the bit).
//! The norms, attention softmax and router dots sum in a different order from the Burn ops and
//! the host router, so logits agree to rounding, not bit for bit; the parity gates in the plan
//! (top-1 at every position, greedy text) are how that is checked, see `lfm2-check --ab-device`.

use crate::dense::conv_blocks_enabled;
use crate::model::{Cache, Lfm2};
use crate::moe::{moe_ffn_with, Route};
use crate::weights::{Ffn, MatRef, Operator};
use burn::tensor::{Tensor, TensorData};
use lb_q4_gpu::cubecl_server::Handle;
use lb_q4_gpu::decode::rope_table;
use lb_q4_gpu::{Gpu, GpuQ4Mat, TokenParams};
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Device copies of the small per-layer weights and the per-token buffers, built on the first
/// device token and kept for the model's life.
pub struct Prep {
    pub(crate) layers: Vec<LayerPrep>,
    pub(crate) final_norm: Handle,
    /// The residual stream, `[n_embd]`. Persistent (never scratch), because a host MoE layer's
    /// read recycles the scratch pool in the middle of a token.
    x: Handle,
    /// Every MoE layer's routes `[n_slots, 2]` u32 and weights `[n_slots]`, slot `moe_index * k`.
    routes: Handle,
    weights: Handle,
    n_slots: usize,
    /// The same shape as `routes`, holding cache-slot numbers instead of expert ids: what the expert
    /// kernels index the slot bank with (engine step E2, `routes` stays expert ids for the sum).
    slot_routes: Handle,
    /// The on-device next-layer guess, when `LB_DEVICE_GUESS=1`: routes and weights at slot
    /// `moe_index(target) * k`.
    guess: Option<(Handle, Handle)>,
    pub(crate) rope: Option<(Handle, usize)>,
    /// Bumped whenever `rope` is rebuilt (a graph recorded against the old table is stale).
    rope_gen: u64,
    /// Router scratch for the split router (`LB_DD_FAST`): probs and score, `n_expert` f32 each.
    rscratch: (Handle, Handle),
    /// The captured token (`LB_DD_GRAPH`), for the sequence whose `DevCache::gen` it names.
    graph: Option<GraphSlot>,
    tp: TokenParams,
    /// Why no token can take this path (a matrix not resident, ...); `None` = it can.
    pub(crate) blocker: Option<String>,
    /// MoE layers without a stacked resident layer: the host runs these (`dd_host_moe`) or the
    /// token falls back.
    pub(crate) host_moe_layers: Vec<usize>,
    warned: bool,
}

/// A token recorded as a CUDA graph for one sequence's device state.
struct GraphSlot {
    gen: u64,
    rope_gen: u64,
    stream: u64,
    state: GraphState,
}

enum GraphState {
    /// `graph_prepare` was called and the token that followed ran normally (the warm-up the
    /// capture window needs: kernels compiled, scratch buffers allocated).
    Prepared,
    /// Recorded: replaying it is the token.
    Ready(ReadyGraph),
    /// The capture was refused (logged once); this sequence runs un-graphed.
    Failed,
}

struct ReadyGraph {
    graph: lb_q4_gpu::CubeGraph,
    logits: Handle,
    /// The scratch buffers the recording was given: the graph writes these exact buffers on every
    /// replay, so they are held here and never go back to the scratch pool.
    _held: Vec<Handle>,
}

pub(crate) struct LayerPrep {
    pub(crate) attn_norm: Handle,
    pub(crate) ffn_norm: Handle,
    pub(crate) op: OpPrep,
    /// Index of this layer's routes in the per-token buffer (MoE layers only).
    pub(crate) moe_index: Option<usize>,
    pub(crate) router: Option<(Handle, Handle)>,
}

pub(crate) enum OpPrep {
    Conv { taps: Handle },
    Attn { q_norm: Handle, k_norm: Handle },
}

/// The device copy of a sequence's state: per attention layer K and V `[kvh, cap, hd]`, per
/// conv layer the rolling state `[l_cache - 1, n_embd]`, valid for positions `0..len`.
pub struct DevCache {
    pub(crate) k: Vec<Option<Handle>>,
    pub(crate) v: Vec<Option<Handle>>,
    pub(crate) conv: Vec<Option<Handle>>,
    pub(crate) cap: usize,
    pub(crate) len: usize,
    /// Which upload this is (`push` makes a new one): the handles are only the same buffers while
    /// the generation is.
    pub(crate) gen: u64,
}

static DEV_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// What the static check says about a pass, before any state is touched.
#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    /// Run the token on the device; the listed MoE layers on the host inside it.
    Device { host_moe_layers: usize },
    /// Give the token to the old path.
    Fallback(String),
}

/// The rule, separate from the device so it can be tested anywhere: a blocker always falls
/// back; missing MoE layers fall back unless the host may run them inside the token.
pub fn decide(blocker: Option<&str>, host_moe_layers: usize, host_moe_allowed: bool) -> Decision {
    match blocker {
        Some(b) => Decision::Fallback(b.to_string()),
        None if host_moe_layers > 0 && !host_moe_allowed => {
            Decision::Fallback(format!("{host_moe_layers} MoE layer(s) not resident as stacked layers (LB_DD_HOST_MOE=1 runs them on the host inside the token)"))
        }
        None => Decision::Device { host_moe_layers },
    }
}

pub(crate) enum Head<'a> {
    Q6(&'a lb_q4_gpu::GpuQ6KMat),
    Q4(&'a GpuQ4Mat),
}

impl Prep {
    pub(crate) fn new(m: &Lfm2, gpu: &Gpu) -> Self {
        let cfg = m.cfg();
        let (e, k) = (cfg.n_embd, cfg.n_expert_used.max(1));
        let mut blocker: Vec<String> = Vec::new();
        if conv_blocks_enabled() {
            blocker.push("LB_CONV_BLOCK is on (its device conv state is separate from this path's)".into());
        }
        if cfg.n_expert > lb_q4_gpu::decode::MAX_EXPERTS {
            blocker.push(format!("{} experts > {}", cfg.n_expert, lb_q4_gpu::decode::MAX_EXPERTS));
        }
        let hd = cfg.head_dim();
        if !hd.is_power_of_two() || e % 4 != 0 {
            blocker.push(format!("head_dim {hd} / n_embd {e}: the kernels need a power-of-two head and n_embd % 4 == 0"));
        }
        if head(m).is_none() {
            blocker.push("the head (token_embd) is not resident".into());
        }
        let need = |r: &MatRef, what: &str, blocker: &mut Vec<String>, li: usize| {
            if !m.dense.resident.contains_key(&r.start) {
                blocker.push(format!("layer {li} {what} not resident"));
            }
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut host_moe_layers = Vec::new();
        let mut n_moe = 0usize;
        for (li, layer) in m.w.layers.iter().enumerate() {
            let op = match &layer.op {
                Operator::Conv { in_proj, conv, out_proj } => {
                    need(in_proj, "in_proj", &mut blocker, li);
                    need(out_proj, "out_proj", &mut blocker, li);
                    OpPrep::Conv { taps: gpu.upload_f32(conv) }
                }
                Operator::Attn { q, k, v, o, q_norm, k_norm } => {
                    for (r, w) in [(q, "q"), (k, "k"), (v, "v"), (o, "o")] {
                        need(r, w, &mut blocker, li);
                    }
                    OpPrep::Attn { q_norm: gpu.upload_f32(q_norm), k_norm: gpu.upload_f32(k_norm) }
                }
            };
            let (moe_index, router) = match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    for (r, w) in [(gate, "ffn gate"), (up, "ffn up"), (down, "ffn down")] {
                        need(r, w, &mut blocker, li);
                    }
                    (None, None)
                }
                Ffn::Moe { router, bias, .. } => {
                    let in_box = m.store.inbox.as_ref().is_some_and(|ib| ib.card_of(li).is_some());
                    if (!m.store.layers.contains_key(&li) && !in_box) || m.store.remote.is_some() {
                        host_moe_layers.push(li);
                    }
                    n_moe += 1;
                    (Some(n_moe - 1), Some((gpu.upload_f32(router), gpu.upload_f32(bias))))
                }
            };
            layers.push(LayerPrep { attn_norm: gpu.upload_f32(&layer.attn_norm), ffn_norm: gpu.upload_f32(&layer.ffn_norm), op, moe_index, router });
        }
        let n_slots = (n_moe * k).max(1);
        let blocker = if blocker.is_empty() {
            None
        } else {
            let n = blocker.len();
            blocker.truncate(4);
            Some(format!("{}{}", blocker.join("; "), if n > 4 { format!(" (+{} more)", n - 4) } else { String::new() }))
        };
        Self {
            layers,
            final_norm: gpu.upload_f32(&m.w.final_norm),
            x: gpu.upload_f32(&vec![0f32; e]),
            routes: gpu.upload_u32(vec![0u32; 2 * n_slots]),
            weights: gpu.upload_f32(&vec![0f32; n_slots]),
            n_slots,
            slot_routes: gpu.upload_u32(vec![0u32; 2 * n_slots]),
            guess: std::env::var("LB_DEVICE_GUESS").is_ok_and(|v| !v.is_empty() && v != "0").then(|| (gpu.upload_u32(vec![0u32; 2 * n_slots]), gpu.upload_f32(&vec![0f32; n_slots]))),
            rope: None,
            rope_gen: 0,
            rscratch: (gpu.upload_f32(&vec![0f32; cfg.n_expert.max(1)]), gpu.upload_f32(&vec![0f32; cfg.n_expert.max(1)])),
            graph: None,
            tp: TokenParams::new(gpu, 0, 1, 1),
            blocker,
            host_moe_layers,
            warned: false,
        }
    }
}

pub(crate) fn head(m: &Lfm2) -> Option<Head<'_>> {
    let s = m.w.embd.start;
    m.dense.q6.get(&s).map(Head::Q6).or_else(|| m.dense.resident.get(&s).map(Head::Q4))
}

pub(crate) fn mat<'a>(m: &'a Lfm2, r: &MatRef) -> &'a GpuQ4Mat {
    m.dense.resident.get(&r.start).expect("checked resident in Prep::new")
}

fn to_vec<const D: usize>(t: &Tensor<D>) -> Vec<f32> {
    t.clone().into_data().try_to_vec::<f32>().expect("f32 tensor")
}

/// Bring the host's KV cache and conv state up to the device's, if the device is ahead. One
/// read for all of it. Called at the top of every host pass; a no-op when nothing is stale.
pub fn pull_if_stale(m: &Lfm2, cache: &mut Cache) {
    if cache.host_len >= cache.len {
        return;
    }
    let gpu = m.store.gpu.as_ref().expect("device state without a device");
    let dev = cache.dev.as_ref().expect("host behind with no device copy");
    assert_eq!(dev.len, cache.len, "device copy at {} for a sequence at {}", dev.len, cache.len);
    let cfg = m.cfg();
    let (e, l, hd) = (cfg.n_embd, cfg.l_cache, cfg.head_dim());
    let mut want: Vec<(usize, u8, Handle, usize)> = Vec::new();
    for li in 0..cfg.n_layer {
        let kvh = cfg.kv_heads[li];
        if let (Some(k), Some(v)) = (&dev.k[li], &dev.v[li]) {
            want.push((li, 0, k.clone(), kvh * dev.cap * hd));
            want.push((li, 1, v.clone(), kvh * dev.cap * hd));
        }
        if let Some(c) = &dev.conv[li] {
            want.push((li, 2, c.clone(), (l - 1) * e));
        }
    }
    let bytes = gpu.read_raw(want.iter().map(|(_, _, h, n)| (h.clone(), n * 4)).collect()).expect("device state read-back (the host copy would be behind: cannot continue)");
    for ((li, what, _, _), b) in want.iter().zip(bytes) {
        let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        match what {
            0 | 1 => {
                let t = Tensor::<3>::from_data(TensorData::new(v, [cfg.kv_heads[*li], dev.cap, hd]), &m.device);
                if *what == 0 { cache.k[*li] = Some(t) } else { cache.v[*li] = Some(t) }
            }
            _ => cache.conv[*li] = Some(Tensor::<2>::from_data(TensorData::new(v, [l - 1, e]), &m.device)),
        }
    }
    cache.host_len = cache.len;
}

/// Upload the host's state for positions `0..cache.len` as the device copy (`cap` = the host's).
pub(crate) fn push(m: &Lfm2, gpu: &Gpu, cache: &mut Cache) {
    assert_eq!(cache.host_len, cache.len, "pushing a host state that is behind");
    let cfg = m.cfg();
    let (e, l, hd, cap) = (cfg.n_embd, cfg.l_cache, cfg.head_dim(), cache.cap);
    let mut dev = DevCache { k: vec![None; cfg.n_layer], v: vec![None; cfg.n_layer], conv: vec![None; cfg.n_layer], cap, len: cache.len, gen: DEV_GEN.fetch_add(1, Ordering::Relaxed) };
    for li in 0..cfg.n_layer {
        let kvh = cfg.kv_heads[li];
        if kvh > 0 {
            let up = |t: &Option<Tensor<3>>| match t {
                Some(t) => {
                    assert_eq!(t.dims(), [kvh, cap, hd], "host cache layer {li}");
                    gpu.upload_f32(&to_vec(t))
                }
                None => gpu.upload_f32(&vec![0f32; kvh * cap * hd]),
            };
            dev.k[li] = Some(up(&cache.k[li]));
            dev.v[li] = Some(up(&cache.v[li]));
        } else {
            dev.conv[li] = Some(match &cache.conv[li] {
                Some(t) => gpu.upload_f32(&to_vec(t)),
                None => gpu.upload_f32(&vec![0f32; (l - 1) * e]),
            });
        }
    }
    cache.dev = Some(dev);
}

/// Run one decode token device-resident. `None` hands it to the old path (nothing changed).
pub fn token(m: &Lfm2, tok: u32, cache: &mut Cache) -> Option<Tensor<2>> {
    let gpu = m.store.gpu.as_ref()?;
    let knobs = m.store.knobs.clone();
    let mut guard = m.dd.lock().expect("device decode state");
    let prep = guard.get_or_insert_with(|| Prep::new(m, gpu));
    let decision = decide(prep.blocker.as_deref(), prep.host_moe_layers.len(), knobs.dd_host_moe() || m.store.remote.is_some());
    if let Decision::Fallback(why) = decision {
        if !prep.warned {
            eprintln!("lb-lfm2: device decode declined, the old path runs: {why}");
            prep.warned = true;
        }
        m.timings.dd_fallbacks.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let t0 = Instant::now();
    let cfg = m.cfg();
    let (e, l, hd, eps) = (cfg.n_embd, cfg.l_cache, cfg.head_dim(), cfg.rms_eps);
    let (nh, k_used, n_exp) = (cfg.n_head, cfg.n_expert_used, cfg.n_expert);
    let pos0 = cache.len;
    let total = pos0 + 1;

    // state: grow (pulling first if the device is ahead), then make the device copy current
    if total > cache.cap && cache.host_len < cache.len {
        pull_if_stale(m, cache);
    }
    m.ensure_cap(cache, total);
    let fresh = cache.dev.as_ref().is_none_or(|d| d.cap != cache.cap || d.len != pos0);
    if fresh {
        push(m, gpu, cache);
    }
    if prep.rope.as_ref().is_none_or(|(_, rows)| *rows < cache.cap) {
        prep.rope = Some((gpu.upload_f32(&rope_table(cfg.rope_theta, hd, cache.cap)), cache.cap));
        prep.rope_gen += 1;
    }
    let (rope, rope_rows) = prep.rope.as_ref().map(|(h, r)| (h.clone(), *r)).expect("just built");
    prep.tp.set(gpu, pos0, total, cache.cap);
    // the device buffers by handle (cheap clones), so `cache` stays free for `abandon`
    let (dev_k, dev_v, dev_conv) = {
        let d = cache.dev.as_ref().expect("just pushed");
        (d.k.clone(), d.v.clone(), d.conv.clone())
    };

    // LB_DD_GRAPH: this token as one CUDA graph. Eligible only when it is nothing but launches: every
    // MoE layer a resident stack on this card, no remote tier, no staging ring, no in-box cards,
    // no host-run layers. Sequence per device state: the token after `graph_prepare` runs normally
    // (warm-up), the next one is recorded (the window records launches and runs none of them) and
    // replayed, every later one only writes its inputs and replays.
    let graph_ok = knobs.dd_graph()
        && prep.host_moe_layers.is_empty()
        && m.store.remote.is_none()
        && m.store.inbox.is_none()
        && m.store.ring.lock().expect("stage ring").is_none()
        && m.w.layers.iter().enumerate().all(|(li, l)| !matches!(l.ffn, Ffn::Moe { .. }) || m.store.layers.contains_key(&li));
    let gen = cache.dev.as_ref().map(|d| d.gen).unwrap_or(0);
    let stream = gpu.stream_key();
    let mut capturing = false;
    let mut replaying = false;
    if graph_ok {
        if prep.graph.as_ref().is_some_and(|g| g.gen != gen || g.rope_gen != prep.rope_gen || g.stream != stream) {
            // the old graph (and the scratch it holds) goes first: destroying it syncs the stream
            prep.graph = None;
        }
        match prep.graph.as_ref().map(|g| &g.state) {
            None => match gpu.client.graph_prepare() {
                Ok(()) => prep.graph = Some(GraphSlot { gen, rope_gen: prep.rope_gen, stream, state: GraphState::Prepared }),
                Err(e) => {
                    eprintln!("lb-lfm2: LB_DD_GRAPH: graph_prepare refused ({e:?}); tokens run un-graphed");
                    prep.graph = Some(GraphSlot { gen, rope_gen: prep.rope_gen, stream, state: GraphState::Failed });
                }
            },
            Some(GraphState::Prepared) => capturing = true,
            Some(GraphState::Ready(_)) => replaying = true,
            Some(GraphState::Failed) => {}
        }
    }

    // the staging ring (`stage.rs`): a new token makes everything it holds stale
    let mut ring_guard = m.store.ring.lock().expect("stage ring");
    if let Some(r) = ring_guard.as_mut() {
        r.begin_token();
    }
    m.stats.stage.tokens.fetch_add(1, Ordering::Relaxed);
    // predictor gate: guess only while the cache is missing (a moving average of demand misses a token)
    let pred_active = if m.store.la.pred.is_some() {
        let st = &m.stats.stage;
        let od = m.stats.on_demand.load(Ordering::Relaxed);
        let miss = od.saturating_sub(st.pred_prev_demand.swap(od, Ordering::Relaxed)) as i64 * 1000;
        let ema = st.pred_ema_milli.load(Ordering::Relaxed) as i64;
        let ema = (ema + (miss - ema) / 8).max(0);
        st.pred_ema_milli.store(ema as u64, Ordering::Relaxed);
        ema >= knobs.pred_min_miss() as i64 * 1000
    } else {
        true
    };
    let mut emb = vec![0f32; e];
    m.w.embed_row(tok, &mut emb);
    gpu.write_f32(&prep.x, &emb);
    // handles uploaded mid-token (host MoE outputs) stay alive until the token's read
    let mut keep: Vec<Handle> = Vec::new();
    let mut host_routes: Vec<(usize, Route)> = Vec::new();
    let mut y: Option<Handle> = None;
    // fused glue launches (`LB_DD_FUSE=0` keeps one kernel per step): norm+router, sum+next norm
    let fuse = m.store.remote.is_none() && std::env::var("LB_DD_FUSE").map(|v| !v.is_empty() && v != "0").unwrap_or(true);
    // LB_DD_FAST: the router over one cube per expert, the expert sum's slot order once per cube
    let fast = knobs.dd_fast();
    let fast_router = fast && gpu.router_split_ok(knobs.dd_router_seq());
    let fast_sum = fast && k_used <= 16;
    let mut pend: Option<(Handle, usize)> = None;
    let mut dev_moe = 0u64;
    let mut inbox_layers = 0u64;
    // engine step E2: experts routed on the device and served from the cache's slot bank
    // with a remote tier every MoE layer goes through the host side (`moe_ffn_with`), where the
    // tier is asked first; the device router and the stacked layers are not used
    let remote_on = m.store.remote.is_some();
    let e2 = knobs.e2_device_router() && !remote_on && m.store.cache.lock().expect("expert cache").is_some();
    // LB_E2_SPEC: launch a layer's experts from the device slot table before the routes are read (needs the
    // look-ahead off: its landings rename slots under the table)
    let spec = e2 && knobs.e2_spec() && !m.store.la.on && m.store.cache.lock().expect("expert cache").as_ref().is_some_and(|c| c.landings.is_none());
    let mut slot_words = vec![0u32; 2 * prep.n_slots];
    let mut guessed: Option<Vec<u32>> = None;
    // a guess job on the worker, to be picked up and issued before the next layer's read
    let mut deferred = false;
    let stage = &m.stats.stage;
    let vocab = cfg.n_vocab;
    // LB_E2_FLUSH=N: queued launches go to the device thread after every Nth launch group (see `Gpu::flush_queue`)
    let e2_fuse = e2 && knobs.e2_fuse();
    let flush_n = if e2 { knobs.e2_flush() as usize } else { 0 };
    let mut flush_i = 0usize;
    macro_rules! fl {
        () => {
            if flush_n > 0 {
                flush_i += 1;
                if flush_i % flush_n == 0 {
                    gpu.flush_queue();
                }
            }
        };
    }
    let t_issue = Instant::now();
    // LB_E2_PROFILE=<file>: one JSON line a token: per MoE layer [layer, misses, sync wait us, host busy us since the last sync, bytes read, read_begin us, read_finish us, host us from the read's return to the end of the layer's launches]
    let mut prof: Vec<[u64; 8]> = Vec::new();
    let mut last_ret = t_issue;
    let logits: Handle = if replaying {
        match prep.graph.as_ref().map(|g| &g.state) {
            Some(GraphState::Ready(r)) => r.logits.clone(),
            _ => unreachable!("replaying without a ready graph"),
        }
    } else {
        if capturing {
            if let Err(err) = gpu.client.start_capture() {
                return graph_refused(m, cache, tok, guard, ring_guard, &format!("start_capture: {err:?}"));
            }
        }
        if spec {
            m.store.cache.lock().expect("expert cache").as_mut().expect("checked").sync_table(gpu);
        }
        for (li, layer) in m.w.layers.iter().enumerate() {
            let lp = &prep.layers[li];
            if gpu.trace.is_on() {
                let op = if matches!(layer.op, Operator::Conv { .. }) { "conv" } else { "attn" };
                gpu.trace.label(&format!("L{li}:{op}/{}", if matches!(layer.ffn, Ffn::Moe { .. }) { "moe" } else { "dense" }));
            }
            let h = match pend.take() {
                // the previous layer's expert sum, folded into this pre-norm's launch
            Some((outs, base)) if fast_sum => gpu.sum_add_rms_norm_ord(&outs, &prep.weights, &prep.routes, &prep.x, &lp.attn_norm, (k_used, e, base, prep.n_slots), eps),
            Some((outs, base)) => gpu.sum_add_rms_norm(&outs, &prep.weights, &prep.routes, &prep.x, &lp.attn_norm, (k_used, e, base, prep.n_slots), eps),
                None => gpu.add_rms_norm(&prep.x, y.as_ref(), &lp.attn_norm, e, eps),
            };
            fl!();
            let op_out = match (&layer.op, &lp.op) {
                (Operator::Conv { in_proj, out_proj, .. }, OpPrep::Conv { taps }) => {
                    let bcx = mat(m, in_proj).launch(gpu, &h, 1);
                    fl!();
                    let state = dev_conv[li].as_ref().expect("conv layer state");
                    let yy = gpu.conv_decode(&bcx, taps, state, e, l);
                    fl!();
                    let o = mat(m, out_proj).launch(gpu, &yy, 1);
                    fl!();
                    o
                }
                (Operator::Attn { q, k, v, o, .. }, OpPrep::Attn { q_norm, k_norm }) => {
                    let kvh = cfg.kv_heads[li];
                    let (qh, kh, vh) = (mat(m, q).launch(gpu, &h, 1), mat(m, k).launch(gpu, &h, 1), mat(m, v).launch(gpu, &h, 1));
                    fl!();
                    let (kc, vc) = (dev_k[li].as_ref().expect("k cache"), dev_v[li].as_ref().expect("v cache"));
                    let qo = gpu.qk_norm_rope(&qh, &kh, &vh, q_norm, k_norm, &rope, rope_rows, kc, vc, &prep.tp, (nh, kvh, hd), eps);
                    fl!();
                    let att = gpu.attn_decode(&qo, kc, vc, &prep.tp, (nh, kvh, hd));
                    fl!();
                    let oo = mat(m, o).launch(gpu, &att, 1);
                    fl!();
                    oo
                }
                _ => unreachable!("layer {li}: operator and its device copy disagree"),
            };
            // a stacked MoE layer on this card: its router runs in the pre-norm's launch (`fuse`)
            let fused = fuse
                && matches!(&layer.ffn, Ffn::Moe { .. })
                && lp.moe_index.is_some()
                && lp.router.is_some()
                && m.store.layers.contains_key(&li)
                && !m.store.inbox.as_ref().is_some_and(|ib| ib.card_of(li).is_some());
            let h2 = if fused {
                let (rh, bh) = lp.router.as_ref().expect("checked");
                let base = lp.moe_index.expect("checked") * k_used;
                if fast_router {
                    let h2 = gpu.add_rms_norm(&prep.x, Some(&op_out), &lp.ffn_norm, e, eps);
                    gpu.router_topk_split(&h2, rh, bh, &prep.rscratch, &prep.routes, &prep.weights, (n_exp, k_used, e, base, prep.n_slots));
                    h2
                } else {
                    gpu.add_rms_norm_route(&prep.x, Some(&op_out), &lp.ffn_norm, e, eps, rh, bh, &prep.routes, &prep.weights, (n_exp, k_used, base, prep.n_slots), knobs.dd_router_seq())
                }
            } else {
                gpu.add_rms_norm(&prep.x, Some(&op_out), &lp.ffn_norm, e, eps)
            };
            fl!();
            let ffn_out = match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let hh = GpuQ4Mat::launch_gate_up_silu(gpu, mat(m, gate), mat(m, up), &h2, 1);
                    mat(m, down).launch(gpu, &hh, 1)
                }
                // work-to-weights inside the box (`inbox.rs`): the layer lives on another card of this
                // process; the activation goes card to card and the routes, experts and sum run there
                // with the one-card kernels; nothing is read on the host
                Ffn::Moe { .. } if !remote_on && lp.moe_index.is_some() && m.store.inbox.as_ref().is_some_and(|ib| ib.card_of(li).is_some()) => {
                    let ib = m.store.inbox.as_ref().expect("checked");
                    let card = ib.card_of(li).expect("checked");
                    let base = lp.moe_index.expect("checked") * k_used;
                    let h2b = gpu.copy_to(&h2, &card.gpu);
                    let (rh, bh) = card.routers.get(&li).expect("router uploaded with the layer");
                    card.gpu.router_topk(&h2b, rh, bh, &card.routes, &card.weights, (n_exp, k_used, e, base, prep.n_slots), knobs.dd_router_seq());
                    let (outs, _) = card.layers[&li].launch_device(&card.gpu, &h2b, 1, &card.routes, prep.n_slots, base, k_used);
                    let yb = card.gpu.weighted_sum(&outs, &card.weights, &card.routes, (k_used, e, base, prep.n_slots));
                    let y = card.gpu.copy_to(&yb, gpu);
                    keep.push(y.clone());
                    inbox_layers += 1;
                    y
                }
                Ffn::Moe { router, bias, gate, up, down } => match (if remote_on { None } else { m.store.layers.get(&li) }, lp.moe_index, &lp.router) {
                    (Some(stack), Some(mi), Some((rh, bh))) => {
                        let base = mi * k_used;
                        if !fused {
                            gpu.router_topk(&h2, rh, bh, &prep.routes, &prep.weights, (n_exp, k_used, e, base, prep.n_slots), knobs.dd_router_seq());
                        }
                        let (outs, _) = stack.launch_device(gpu, &h2, 1, &prep.routes, prep.n_slots, base, k_used);
                        dev_moe += 1;
                        if fused {
                            // the sum runs in the next pre-norm's launch
                            pend = Some((outs.clone(), base));
                            outs
                        } else {
                            gpu.weighted_sum(&outs, &prep.weights, &prep.routes, (k_used, e, base, prep.n_slots))
                        }
                    }
                    (None, Some(mi), Some((rh, bh))) if e2 => {
                        let base = mi * k_used;
                        let seq = knobs.dd_router_seq();
                        gpu.router_topk(&h2, rh, bh, &prep.routes, &prep.weights, (n_exp, k_used, e, base, prep.n_slots), seq);
                        fl!();
                        // the guess of the next MoE layer, from the same residual stream (`x` is this
                        // layer's x_mid here): its router on its own FFN pre-norm
                        let mut want = vec![(prep.routes.clone(), prep.n_slots * 8)];
                        if let (Some((gr, gw)), Some(next)) = (prep.guess.as_ref(), prep.layers.get(li + 1)) {
                            if let (Some(nmi), Some((nr, nb))) = (next.moe_index, next.router.as_ref()) {
                                let hn = gpu.add_rms_norm(&prep.x, None, &next.ffn_norm, e, eps);
                                gpu.router_topk(&hn, nr, nb, gr, gw, (n_exp, k_used, e, nmi * k_used, prep.n_slots), seq);
                                want.push((gr.clone(), prep.n_slots * 8));
                            }
                        }
                        // E3: the residual stream comes back with the routes (same sync) for the host's
                        // guess of the next layer (`x` is this layer's x_mid here)
                        let la = &m.store.la;
                        let x_at = (la.on && la.oracle.is_none() && pred_active).then(|| { want.push((prep.x.clone(), e * 4)); want.len() - 1 });
                        // LB_CPU_MISS: the router input comes back too (same sync), for the CPU-served misses
                        let cpu_miss = knobs.cpu_miss();
                        let h2_at = cpu_miss.then(|| { want.push((h2.clone(), e * 4)); want.len() - 1 });
                        if deferred {
                            deferred = false;
                            let t = Instant::now();
                            let mut g2 = m.store.cache.lock().expect("expert cache");
                            let c2 = g2.as_mut().expect("checked above");
                            let gs = c2.worker.as_ref().map(|w| w.recv()).unwrap_or_default();
                            stage.la_guess_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            la_issue(m, gpu, c2, &gs);
                        }
                        // the host learns this layer's routes: the one sync a layer that a miss served
                        // from host memory cannot avoid
                        let t_read = Instant::now();
                        let busy_us = t_read.duration_since(last_ret).as_micros() as u64;
                        let (mut prof_begin, mut prof_finish) = (0u64, 0u64);
                        // LB_E2_SPEC: the routes read is enqueued right behind the router, and only then are the experts
                        // launched from the device table, so the host gets the routes while the card runs the experts
                        // per layer: speculate only where the layer has mostly hit lately (an EMA of "this pass had a
                        // miss", `LB_E2_SPEC_MISS` threshold, 0.3): a speculated layer that misses launches its
                        // experts twice, which costs more than the bubble it hides
                        let spec_here = spec && m.store.cache.lock().expect("expert cache").as_ref().is_some_and(|c| c.layer_miss.get(li).copied().unwrap_or(0.0) < spec_miss_threshold());
                        let (spec_outs, read) = if spec_here {
                            let pr = gpu.read_begin(want);
                            let so: Option<Handle> = {
                                let g = m.store.cache.lock().expect("expert cache");
                                let c = g.as_ref().expect("checked above");
                                let tab = c.tab_dev.as_ref().expect("table synced at token start");
                                gpu.slot_translate(&prep.routes, tab, c.tab_host.len(), &prep.slot_routes, (li * cfg.n_expert, k_used, base, 0, prep.n_slots));
                                Some(c.slots.bank().launch_device(gpu, &h2, 1, &prep.slot_routes, prep.n_slots, base, k_used).0)
                            };
                            (so, gpu.read_finish(pr))
                        } else if prof_on() {
                            // same calls as `read_raw_keep` (read_async then read_sync), timed apart
                            let t0 = Instant::now();
                            let pr = gpu.read_begin(want);
                            prof_begin = t0.elapsed().as_micros() as u64;
                            let t1 = Instant::now();
                            let r = gpu.read_finish(pr);
                            prof_finish = t1.elapsed().as_micros() as u64;
                            (None, r)
                        } else {
                            (None, gpu.read_raw_keep(want))
                        };
                        let mut bufs = match read {
                            Ok(b) => b,
                            Err(err) => return abandon(m, cache, &format!("MoE layer {li} routes read: {err}")),
                        };
                        stage.h2_read_ns.fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        stage.host_syncs.fetch_add(1, Ordering::Relaxed);
                        last_ret = Instant::now();
                        let prof_wait = t_read.elapsed().as_micros() as u64;
                        let prof_bytes: u64 = bufs.iter().map(|b| b.len() as u64).sum();
                        let u32s = |b: &[u8]| b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<u32>>();
                        let routes_now = u32s(&bufs[0]);
                        let ids: Vec<u32> = (0..k_used).map(|i| routes_now[2 * (base + i)]).collect();
                        if let Some(g) = guessed.take() {
                            stage.guess_named.fetch_add(g.iter().filter(|e| ids.contains(e)).count() as u64, Ordering::Relaxed);
                            stage.guess_total.fetch_add(k_used as u64, Ordering::Relaxed);
                        }
                        if let (Some(b), Some(nmi)) = (bufs.get(1), prep.layers.get(li + 1).and_then(|n| n.moe_index)) {
                            let g = u32s(b);
                            guessed = Some((0..k_used).map(|i| g[2 * (nmi * k_used + i)]).collect());
                        }
                        let mut needed: Vec<usize> = ids.iter().map(|&x| x as usize).collect();
                        needed.sort_unstable();
                        // E3: the host guess of the next MoE layers, from the state before this layer's
                        // experts run (`docs/ENGINE_E3_LOOKAHEAD.md`)
                        // with the worker the guess is picked up after the launches (`pending`)
                        let mut pending = false;
                        let mut guesses: Vec<crate::stage::Guess> = Vec::new();
                        if let Some(i) = x_at {
                            let t = Instant::now();
                            let worker = m.store.cache.lock().expect("expert cache").as_ref().and_then(|c| c.worker.as_ref().map(|w| { w.send(li, std::mem::take(&mut bufs[i])); }));
                            if worker.is_some() {
                                pending = true;
                            } else {
                                let x_mid: Vec<f32> = bufs[i].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                guesses.extend(la.leads.iter().filter_map(|&d| crate::stage::predict(&m.w, li + d, &x_mid, la.n_guess).map(|mut g| { g.lead = d; g })));
                            }
                            stage.la_guess_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        if let Some(i) = x_at {
                            if std::env::var_os("LB_PRED_DUMP").is_some() {
                                pred_dump(m, tok, pos0, li, &ids, &bufs[i]);
                            }
                        }
                        let t_fill = Instant::now();
                        let mut guard = m.store.cache.lock().expect("expert cache");
                        let c = guard.as_mut().expect("checked above");
                        // LB_LA_ORACLE: the recorded next pass replaces the guess (diagnostic, no host cost)
                        if let Some((passes, i)) = c.oracle.as_mut() {
                            let cur = *i;
                            *i += 1;
                            guesses = (1..=la.pf_depth)
                                .filter_map(|d| passes.get(cur + d).filter(|(l, _)| *l == li + d).map(|(l, ex)| crate::stage::Guess { layer: *l, experts: ex.clone(), margin: 0.0, scores: vec![1.0; ex.len()], ref_score: 0.0, lead: d }))
                                .collect();
                        }
                        let guess = guesses.first().cloned();
                        c.trace_pass("d", li, pos0, &needed, guess.as_ref());
                        // this sync finished every repack queued before it: their landing buffers are free
                        if let Some(l) = c.landings.as_mut() {
                            l.book.reclaim();
                        }
                        // keep guard: what the coming layers are predicted to need is not evicted now
                        let keep: Vec<(usize, usize)> = if la.keep {
                            let mut v: Vec<(usize, usize)> = guess.iter().flat_map(|g| g.experts.iter().map(|&x| (g.layer, x))).collect();
                            if la.keep_depth > 1 {
                                let x_mid: Vec<f32> = x_at.map(|i| bufs[i].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).unwrap_or_default();
                                for d in 2..=la.keep_depth {
                                    if let Some(g) = crate::stage::predict(&m.w, li + d, &x_mid, la.n_guess) {
                                        v.extend(g.experts.iter().map(|&x| (g.layer, x)));
                                    }
                                }
                            }
                            v
                        } else {
                            Vec::new()
                        };
                        // LB_CPU_MISS: a missed expert is copied in only on its Nth miss (LB_CPU_MISS_ADMIT,
                        // second touch by default); until then the CPU computes it from the host bytes and
                        // the cache is not touched for it
                        let t_cpu = Instant::now();
                        let mut cpu_set: Vec<usize> = Vec::new();
                        if cpu_miss {
                            let admit = knobs.cpu_miss_admit();
                            if c.miss_seen.is_empty() {
                                c.miss_seen = vec![0u8; cfg.n_layer * cfg.n_expert];
                            }
                            for &x in &needed {
                                if c.slru.slot(li, x).is_some() {
                                    continue;
                                }
                                let seen = &mut c.miss_seen[li * cfg.n_expert + x];
                                *seen = seen.saturating_add(1);
                                if admit != 0 && *seen as u32 >= admit {
                                    *seen = 0;
                                    stage.cpu_admitted.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    cpu_set.push(x);
                                }
                            }
                        }
                        let needed_dev: Vec<usize> = needed.iter().copied().filter(|x| !cpu_set.contains(x)).collect();
                        let plan = match c.slru.plan_la(li, &needed_dev, &keep, la.cycle_window, cfg.n_layer) {
                            Some((p, spared)) => {
                                stage.la_spared.fetch_add(spared, Ordering::Relaxed);
                                p
                            }
                            None => return abandon(m, cache, &format!("layer {li}: {} experts > {} cache slots", needed_dev.len(), c.slru.n_slots())),
                        };
                        if la.promote > 0 {
                            if let Some(g) = guess.as_ref() {
                                for &x in &g.experts {
                                    if c.slru.promote(g.layer, x, la.promote > 1) {
                                        stage.la_promoted.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                        let mut n_landed = 0;
                        let mut fence: Option<Handle> = None;
                        for &(x, s) in &plan.fills {
                            let raw = |r: &MatRef| m.w.mat(r).bytes;
                            // a miss the prefetch already copied in: repack from the landing buffer, the
                            // bus already paid
                            if let Some(l) = c.landings.as_mut() {
                                if let Some(i) = l.book.consume(li, x) {
                                    let sent = l.sentinels[i].take().expect("a landing in flight has its sentinel");
                                    match l.spare[i].take() {
                                        // the spare slot already holds the expert: it becomes this slot, and
                                        // the slot it replaces becomes a spare
                                        Some(p) => {
                                            let old = c.phys[s] as usize;
                                            c.phys[s] = p as u32;
                                            c.spares.push(old);
                                            fence = Some(sent);
                                        }
                                        None => c.slots.repack_landed(gpu, c.phys[s] as usize, &l.buffers[i], &sent),
                                    }
                                    n_landed += 1;
                                    if li < crate::stage::MAX_LAYERS {
                                        stage.useful[li].fetch_add(1, Ordering::Relaxed);
                                    }
                                    c.count_fill();
                                    continue;
                                }
                            }
                            if let Err(err) = c.slots.fill(gpu, c.phys[s] as usize, [raw(&gate.expert(x)), raw(&up.expert(x)), raw(&down.expert(x))]) {
                                return abandon(m, cache, &format!("layer {li} slot fill: {err}"));
                            }
                            c.count_fill();
                            fl!();
                        }
                        let (n_hit, n_fill) = (plan.hits.len(), plan.fills.len());
                        if prof_on() {
                            prof.push([li as u64, (n_fill + cpu_set.len()) as u64, prof_wait, busy_us, prof_bytes, prof_begin, prof_finish, 0]);
                        }
                        let n_demand = n_fill - n_landed;
                        m.stats.gpu_hits.fetch_add(n_hit as u64, Ordering::Relaxed);
                        m.stats.on_demand.fetch_add((n_fill + cpu_set.len()) as u64, Ordering::Relaxed);
                        m.stats.on_demand_ns.fetch_add(t_fill.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        stage.layer(li, n_hit, n_landed, n_demand);
                        stage.moe_layers.fetch_add(1, Ordering::Relaxed);
                        if n_demand > 0 {
                            stage.miss_layers.fetch_add(1, Ordering::Relaxed);
                        }
                        for (i, &x) in ids.iter().enumerate() {
                            // a CPU-served expert reads slot 0 (its output is overwritten below)
                            slot_words[2 * (base + i)] = match c.slru.slot(li, x as usize) {
                                Some(s) => c.phys[s],
                                None => {
                                    debug_assert!(cpu_set.contains(&(x as usize)), "planned");
                                    c.phys[0]
                                }
                            };
                        }
                        // speculative launch with no copy needed (hits, and misses the CPU serves): its output stands;
                        // otherwise the layer is repaired: real slots written, experts launched again
                        let outs = match spec_outs {
                            Some(o) if n_fill == 0 => o,
                            _ => {
                                gpu.write_u32(&prep.slot_routes, &slot_words);
                                fl!();
                                if let Some(f) = fence.take() {
                                    c.slots.fence(gpu, &f);
                                }
                                c.slots.bank().launch_device(gpu, &h2, 1, &prep.slot_routes, prep.n_slots, base, k_used).0
                            }
                        };
                        // the card starts the experts now, while the CPU computes the missed ones below
                        fl!();
                        if spec {
                            if c.layer_miss.len() <= li {
                                c.layer_miss.resize(cfg.n_layer, 0.0);
                            }
                            let miss = (n_fill > 0) as u32 as f32;
                            c.layer_miss[li] += 0.05 * (miss - c.layer_miss[li]);
                            if n_fill > 0 {
                                c.sync_table(gpu);
                            }
                        }
                        // the card runs the hits while the CPU computes the misses; each CPU output
                        // overwrites its route's row of `outs` (stream-ordered after the kernels) before
                        // the weighted sum, which keeps the ascending-expert-id order
                        if !cpu_set.is_empty() {
                            let hv: Vec<f32> = bufs[h2_at.expect("cpu_miss reads h2")].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let t_k = Instant::now();
                            let ys = crate::moe::expert_ffn_batch(&m.w, gate, up, down, &cpu_set, &hv, knobs.cpu_i8());
                            stage.cpu_kernel_ns.fetch_add(t_k.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            let res: Vec<(usize, Vec<f32>)> = cpu_set.iter().copied().zip(ys).collect();
                            for (x, y) in res {
                                let i = ids.iter().position(|&v| v as usize == x).expect("a needed expert is routed");
                                let row = outs.clone().offset_start((i * e * 4) as u64).offset_end(((k_used - 1 - i) * e * 4) as u64);
                                gpu.write_f32(&row, &y);
                            }
                            stage.cpu_misses.fetch_add(cpu_set.len() as u64, Ordering::Relaxed);
                            stage.cpu_miss_layers.fetch_add(1, Ordering::Relaxed);
                            stage.cpu_miss_ns.fetch_add(t_cpu.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        fl!();
                        // LB_E2_FUSE: the sum runs in the next pre-norm's launch, as in the all-resident path
                        let sum = if e2_fuse {
                            pend = Some((outs.clone(), base));
                            outs
                        } else {
                            gpu.weighted_sum(&outs, &prep.weights, &prep.routes, (k_used, e, base, prep.n_slots))
                        };
                        fl!();
                        // landings this layer did not use were wrong guesses: they cost bus, nothing else
                        if let Some(l) = c.landings.as_mut() {
                            let raw_b = c.slots.raw_expert_bytes() as u64;
                            let expired = l.book.expire_through(li);
                            for &i in &expired {
                                l.sentinels[i] = None;
                                if let Some(p) = l.spare[i].take() {
                                    c.spares.push(p);
                                }
                            }
                            let n = expired.len() as u64;
                            if n > 0 && li < crate::stage::MAX_LAYERS {
                                stage.wasted[li].fetch_add(n, Ordering::Relaxed);
                                stage.wasted_bytes.fetch_add(n * raw_b, Ordering::Relaxed);
                            }
                            if pending && la.defer {
                                // the job is picked up just before the next layer's read: the worker gets
                                // this layer's launches and the next operator's to finish in
                                deferred = true;
                            } else if pending {
                                let t = Instant::now();
                                if let Some(w) = c.worker.as_ref() {
                                    guesses = w.recv();
                                }
                                stage.la_guess_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            }
                        }
                        if !deferred {
                            la_issue(m, gpu, c, &guesses);
                        }
                        if let Some(p) = prof.last_mut() {
                            // host time from the read's return to the end of this layer's launches
                            p[7] = last_ret.elapsed().as_micros() as u64;
                        }
                        sum
                    }
                    _ => {
                        // not resident as a stack: the old host path for this layer, one sync. With
                        // the staging ring the same read also brings back the residual stream after
                        // this layer's operator (`x`, updated in place by `add_rms_norm`), the input
                        // of the next layer's guess.
                        let t_read = Instant::now();
                        let read = if ring_guard.is_some() { gpu.read_raw(vec![(h2, e * 4), (prep.x.clone(), e * 4)]) } else { gpu.read_raw(vec![(h2, e * 4)]) };
                        let mut bufs = match read {
                            Ok(b) => b,
                            Err(err) => {
                                drop(ring_guard);
                                return abandon(m, cache, &format!("host MoE layer {li} read: {err}"));
                            }
                        };
                        m.stats.stage.h2_read_ns.fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        let f32v = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
                        let hv = f32v(&bufs[0]);
                        let rc = match ring_guard.as_mut() {
                            Some(ring) => {
                                let x_mid = f32v(&bufs.pop().expect("x read with h2"));
                                let t = Instant::now();
                                let next = if m.store.stage.slots > 0 { crate::stage::predict(&m.w, li + 1, &x_mid, m.store.stage.slots) } else { None };
                                m.stats.stage.issue_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                                Some(crate::stage::RingCall { ring, gpu, w: &m.w, cfg: &m.store.stage, stats: &m.stats.stage, next })
                            }
                            None => None,
                        };
                        let (out, routes) = moe_ffn_with(&m.w, router, bias, gate, up, down, &hv, 1, &m.store, li, &m.stats, rc);
                        host_routes.push((li, routes.into_iter().next().expect("one token, one route")));
                        let hnd = gpu.upload_f32(&out);
                        keep.push(hnd.clone());
                        hnd
                    }
                },
            };
            y = if pend.is_some() { None } else { Some(ffn_out) };
        }
        if prof_on() {
            prof_write(&prof, t_issue.elapsed().as_micros() as u64);
        }
        // the last layer's job is for no layer: take its answer so it does not meet the next token's
        if deferred {
            if let Some(c) = m.store.cache.lock().expect("expert cache").as_ref() {
                if let Some(w) = c.worker.as_ref() {
                    let _ = w.recv();
                }
            }
        }
        let hf = match pend.take() {
        Some((outs, base)) if fast_sum => gpu.sum_add_rms_norm_ord(&outs, &prep.weights, &prep.routes, &prep.x, &prep.final_norm, (k_used, e, base, prep.n_slots), eps),
        Some((outs, base)) => gpu.sum_add_rms_norm(&outs, &prep.weights, &prep.routes, &prep.x, &prep.final_norm, (k_used, e, base, prep.n_slots), eps),
            None => gpu.add_rms_norm(&prep.x, y.as_ref(), &prep.final_norm, e, eps),
        };
        let logits_h = match head(m).expect("checked in Prep::new") {
            Head::Q6(h6) => h6.launch(gpu, &hf, 1),
            Head::Q4(h4) => h4.launch(gpu, &hf, 1),
        };
        if capturing {
            // the recording runs nothing; a refusal leaves every buffer as it was, so the token is
            // simply run again without the graph
            let graph = match gpu.client.stop_capture() {
                Ok(g) => g,
                Err(err) => return graph_refused(m, cache, tok, guard, ring_guard, &format!("stop_capture: {err:?}")),
            };
            let held = gpu.take_in_flight();
            if let Some(slot) = prep.graph.as_mut() {
                slot.state = GraphState::Ready(ReadyGraph { graph, logits: logits_h.clone(), _held: held });
            }
            m.timings.dd_graph_captures.fetch_add(1, Ordering::Relaxed);
        }
        logits_h
    };
    if capturing || replaying {
        if let Some(GraphState::Ready(r)) = prep.graph.as_ref().map(|g| &g.state) {
            // SAFETY: every handle the recording used is held by this slot or by `prep`/`cache`,
            // the graph is only replayed on the stream it was recorded on (`GraphSlot::stream`), and
            // this token's inputs were written on that same stream above.
            if let Err(err) = unsafe { r.graph.replay() } {
                prep.graph = None;
                return abandon(m, cache, &format!("graph replay: {err:?}"));
            }
            m.timings.dd_graph_replays.fetch_add(1, Ordering::Relaxed);
        }
        dev_moe = prep.layers.iter().filter(|l| l.moe_index.is_some()).count() as u64;
    }
    m.timings.dd_issue_ns.fetch_add(t_issue.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let read = gpu.read_raw(vec![(logits, vocab * 4), (prep.routes.clone(), prep.n_slots * 8), (prep.weights.clone(), prep.n_slots * 4)]);
    if e2 {
        stage.host_syncs.fetch_add(1, Ordering::Relaxed);
    }
    drop(keep);
    drop(ring_guard);
    let bytes = match read {
        Ok(b) => b,
        Err(err) => return abandon(m, cache, &format!("token read: {err}")),
    };
    let f32s = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
    let logits = f32s(&bytes[0]);
    #[allow(unused_mut)]
    let mut routes: Vec<u32> = bytes[1].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    #[allow(unused_mut)]
    let mut weights = f32s(&bytes[2]);
    // in-box cards: their work for this token is done (the read above waited for the last copy
    // back); fence each once to recycle its scratch, and bring its routes back for the scraper
    if inbox_layers > 0 {
        let ib = m.store.inbox.as_ref().expect("in-box layers ran");
        ib.stats.tokens.fetch_add(1, Ordering::Relaxed);
        ib.stats.device_layers.fetch_add(inbox_layers, Ordering::Relaxed);
        for (ci, card) in ib.cards.iter().enumerate() {
            ib.stats.card_syncs.fetch_add(1, Ordering::Relaxed);
            if m.scraper.enabled() {
                let b = match card.gpu.read_raw(vec![(card.routes.clone(), prep.n_slots * 8), (card.weights.clone(), prep.n_slots * 4)]) {
                    Ok(b) => b,
                    Err(err) => return abandon(m, cache, &format!("in-box card {} routes read: {err}", card.name)),
                };
                let (cr, cw): (Vec<u32>, Vec<f32>) = (b[0].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), f32s(&b[1]));
                for (li, lp) in prep.layers.iter().enumerate() {
                    if let (Some(mi), Some(c)) = (lp.moe_index, ib.layer_card.get(li).copied().flatten()) {
                        if c == ci {
                            let b0 = mi * k_used;
                            routes[2 * b0..2 * (b0 + k_used)].copy_from_slice(&cr[2 * b0..2 * (b0 + k_used)]);
                            weights[b0..b0 + k_used].copy_from_slice(&cw[b0..b0 + k_used]);
                        }
                    }
                }
            } else if let Err(err) = card.gpu.sync_recycle() {
                return abandon(m, cache, &format!("in-box card {} fence: {err}", card.name));
            }
        }
    }

    // the scraper, in layer order, device layers from the read-back and host layers as run
    if m.scraper.enabled() {
        let mut host = host_routes.into_iter().peekable();
        for (li, lp) in prep.layers.iter().enumerate() {
            let Some(mi) = lp.moe_index else { continue };
            let route = if host.peek().is_some_and(|(hl, _)| *hl == li) {
                host.next().expect("peeked").1
            } else {
                let b = mi * k_used;
                Route { experts: (0..k_used).map(|i| routes[2 * (b + i)] as usize).collect(), weights: weights[b..b + k_used].to_vec() }
            };
            m.scraper.record(li, pos0, std::slice::from_ref(&route));
        }
    }
    m.stats.gpu_hits.fetch_add((dev_moe + inbox_layers) * k_used as u64, Ordering::Relaxed);
    if let Some(d) = cache.dev.as_mut() {
        d.len = total;
    }
    cache.len = total;
    m.timings.dd_tokens.fetch_add(1, Ordering::Relaxed);
    m.timings.dd_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Some(Tensor::from_data(TensorData::new(logits, [1, vocab]), &m.device))
}

/// Issue the prefetch copies for `guesses` on the side stream (`stage.rs` `LookaheadCfg`): the guessed
/// experts not cached and not already in flight, best first, at most the cap per target layer.
fn la_issue(m: &Lfm2, gpu: &Gpu, c: &mut crate::cache::ExpertCache, guesses: &[crate::stage::Guess]) {
    let (la, stage) = (&m.store.la, &m.stats.stage);
    let (cap, mc) = (la.prefetch_cap, la.min_conf);
    let theta = m.store.knobs.pred_theta();
    let raw_b = c.slots.raw_expert_bytes() as u64;
    let Some(l) = c.landings.as_mut() else { return };
        for g in guesses.iter().filter(|_| cap > 0) {
            let t = Instant::now();
            let mut issued = 0usize;
            for (rank, &x) in g.experts.iter().enumerate() {
                if issued >= cap {
                    break;
                }
                // the guesses are in descending score, so nothing later passes
                if let Some(pm) = la.pred.as_ref() {
                    // the predictor's probability against the live threshold (`pred_theta`), not the raw margin
                    let gap = g.scores[rank] - g.scores.get(3).copied().unwrap_or(g.ref_score);
                    if pm.prob(g.lead, rank, g.scores[rank], gap) < theta {
                        stage.guess_low_margin.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                } else if g.scores[rank] - g.ref_score < mc {
                    stage.guess_low_margin.fetch_add((g.experts.len() - rank) as u64, Ordering::Relaxed);
                    break;
                }
                if la.ballast > 0 && g.layer % la.ballast != 0 {
                    break;
                }
                if la.ballast == 0 && c.slru.slot(g.layer, x).is_some() {
                    stage.guess_resident.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                if l.book.find(g.layer, x).is_some() {
                    stage.guess_in_ring.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let Some(i) = l.book.free_one() else {
                    stage.guess_no_room.fetch_add(1, Ordering::Relaxed);
                    break;
                };
                let Ffn::Moe { gate: ng, up: nu, down: nd, .. } = &m.w.layers[g.layer].ffn else { break };
                let raw = |r: &MatRef| m.w.mat(r).bytes;
                let raws = [raw(&ng.expert(x)), raw(&nu.expert(x)), raw(&nd.expert(x))];
                let landed = if la.spare {
                    let Some(p) = c.spares.pop() else {
                        stage.guess_no_room.fetch_add(1, Ordering::Relaxed);
                        break;
                    };
                    let si = g.layer % l.sides.len();
                let rb = l.next_raw % l.raw[si].len();
                l.next_raw += 1;
                let r = c.slots.land_spare(gpu, &l.sides[si], &l.raw[si][rb], p, raws);
                    match &r {
                        Ok(_) => l.spare[i] = Some(p),
                        Err(_) => c.spares.push(p),
                    }
                    r
                } else {
                    c.slots.land(gpu, &l.side, &l.buffers[i], raws)
                };
                match landed {
                    Ok(sent) => {
                        l.sentinels[i] = Some(sent);
                        l.book.issue(i, g.layer, x);
                        issued += 1;
                        stage.prefetched[g.layer.min(crate::stage::MAX_LAYERS - 1)].fetch_add(1, Ordering::Relaxed);
                        stage.prefetch_bytes.fetch_add(raw_b, Ordering::Relaxed);
                    }
                    Err(_) => {
                        stage.upload_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            stage.issue_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
}

/// A capture the runtime refused (or a window that could not open): mark the sequence un-graphed
/// and run this token again the normal way. The caller's state is untouched (nothing was recorded
/// to the device), so the repeat is exact.
fn graph_refused(
    m: &Lfm2,
    cache: &mut Cache,
    tok: u32,
    mut guard: std::sync::MutexGuard<'_, Option<Prep>>,
    ring_guard: std::sync::MutexGuard<'_, Option<crate::stage::Ring>>,
    why: &str,
) -> Option<Tensor<2>> {
    eprintln!("lb-lfm2: LB_DD_GRAPH: capture refused ({why}); this sequence runs un-graphed");
    if let Some(slot) = guard.as_mut().and_then(|p| p.graph.as_mut()) {
        slot.state = GraphState::Failed;
    }
    drop(ring_guard);
    drop(guard);
    token(m, tok, cache)
}

/// The device copy can no longer be trusted (a read failed mid-token, after the KV append and
/// the conv states moved): drop it, so the next device token re-uploads from the host, and give
/// this token to the old path, whose host state is still at the token's position.
fn abandon(m: &Lfm2, cache: &mut Cache, why: &str) -> Option<Tensor<2>> {
    // if the host was behind, the only copy of the earlier positions' conv state was the device's
    assert!(cache.host_len >= cache.len, "device decode: {why}, and the host state is behind the device's: the sequence cannot continue");
    eprintln!("lb-lfm2: device decode token abandoned ({why}); the old path runs it");
    cache.dev = None;
    m.timings.dd_fallbacks.fetch_add(1, Ordering::Relaxed);
    None
}

/// The routes a device token read back, for a parity check: `(layer, route)` per MoE layer.
/// Exposed so a harness can compare them with the old path's routes without a scrape file.
pub fn routes_of(acts: &[crate::moe::Activation], pos: usize) -> Vec<(u32, Vec<u32>)> {
    acts.iter().filter(|a| a.token_pos as usize == pos).map(|a| (a.layer, a.expert_ids.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blocker_or_a_missing_layer_hands_the_token_to_the_old_path() {
        assert_eq!(decide(None, 0, false), Decision::Device { host_moe_layers: 0 });
        assert_eq!(decide(None, 0, true), Decision::Device { host_moe_layers: 0 });
        // partial residency: the plan's default rule is the whole token on the old path ...
        assert!(matches!(decide(None, 22, false), Decision::Fallback(w) if w.contains("22 MoE layer")));
        // ... unless the host may run those layers inside the device token
        assert_eq!(decide(None, 22, true), Decision::Device { host_moe_layers: 22 });
        // a blocker (a dense matrix or the head missing) always falls back, whatever the knob
        assert!(matches!(decide(Some("the head is not resident"), 0, true), Decision::Fallback(w) if w.contains("head")));
    }
}

fn prof_file() -> &'static Option<std::sync::Mutex<std::fs::File>> {
    static F: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    F.get_or_init(|| std::env::var("LB_E2_PROFILE").ok().filter(|p| !p.is_empty()).and_then(|p| std::fs::File::create(p).ok()).map(std::sync::Mutex::new))
}

fn prof_on() -> bool {
    prof_file().is_some()
}

fn prof_write(layers: &[[u64; 8]], issue_us: u64) {
    use std::io::Write;
    if let Some(f) = prof_file() {
        let _ = writeln!(f.lock().expect("profile"), "{}", serde_json::json!({"layers": layers, "issue_to_last_sync_us": issue_us}));
    }
}

/// `LB_E2_SPEC_MISS`: a layer is speculated while its recent miss frequency is below this (default 0.3).
fn spec_miss_threshold() -> f32 {
    static T: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("LB_E2_SPEC_MISS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.3))
}

/// `LB_PRED_DUMP=<path>`: predictor study (docs/ENGINE_PREDICTOR.md). One binary record per decode
/// token and MoE layer: u32 token, u32 position, u32 layer, `k` u32 experts as routed, then the
/// n_embd f32 residual stream `x_mid` (the E3 read) of that layer. `<path>.weights` holds, once, the
/// FFN pre-norm, router and selection bias of every MoE layer so routing can be replayed offline.
fn pred_dump(m: &Lfm2, tok: u32, pos: usize, layer: usize, ids: &[u32], x_mid: &[u8]) {
    use std::io::Write;
    static F: std::sync::OnceLock<Option<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>> = std::sync::OnceLock::new();
    let f = F.get_or_init(|| {
        let p = std::env::var("LB_PRED_DUMP").ok().filter(|p| !p.is_empty())?;
        if let Ok(mut w) = std::fs::File::create(format!("{p}.weights")) {
            let cfg = &m.w.cfg;
            let _ = w.write_all(&[cfg.n_layer as u32, cfg.n_expert as u32, cfg.n_expert_used as u32, cfg.n_embd as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            let _ = w.write_all(&cfg.rms_eps.to_le_bytes());
            for (li, l) in m.w.layers.iter().enumerate() {
                if let Ffn::Moe { router, bias, .. } = &l.ffn {
                    let _ = w.write_all(&(li as u32).to_le_bytes());
                    for v in l.ffn_norm.iter().chain(router.iter()).chain(bias.iter()) {
                        let _ = w.write_all(&v.to_le_bytes());
                    }
                }
            }
        }
        std::fs::File::create(&p).ok().map(|f| std::sync::Mutex::new(std::io::BufWriter::with_capacity(1 << 20, f)))
    });
    if let Some(f) = f {
        let mut w = f.lock().expect("pred dump");
        let _ = w.write_all(&tok.to_le_bytes());
        let _ = w.write_all(&(pos as u32).to_le_bytes());
        let _ = w.write_all(&(layer as u32).to_le_bytes());
        for &e in ids {
            let _ = w.write_all(&e.to_le_bytes());
        }
        let _ = w.write_all(x_mid);
    }
}
