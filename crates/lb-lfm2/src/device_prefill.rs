//! Device-resident prefill: a prompt of `n` tokens in chunks of `LB_PREFILL_UBATCH` rows (default
//! 512), each chunk from the embedding upload to the last row's logits on the device, with the KV
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
//!               dense FFN: gate_up_silu -> down       MoE: router_rows -> stacked experts -> sum
//!               x += ffn_out; sync + recycle the layer's scratch (the card is nearly full)
//!   last chunk: rms_rows(last row, final norm) -> head -> read logits                   (1 read)
//! ```
//!
//! The matmuls: chunks of at least `LB_PREFILL_GEMM_MIN` rows (16) take the tiled GEMM
//! (`lb_q4_gpu::gemm`: each weight tile read once per 64 rows; MoE slots grouped by expert on the
//! host after one read of the routes); smaller chunks the rows kernels (each output row a plane). Every glue kernel is the batched form of its decode
//! kernel and bit-identical to running it token by token (`tests/prefill_kernels_match_decode.rs`).
//!
//! **When it runs.** Everything device decode needs (no blocker, every MoE layer a stacked resident
//! layer), no remote tier and no in-box cards (not wired here yet). Otherwise it declines before
//! touching any state and the host path runs.
//!
//! **State.** Transactional: if the device is ahead of the host it is pulled first, so a failure
//! part-way only drops the device copy (`cache.dev = None`) and the host path redoes the prompt.
//! On success the device is ahead (`cache.host_len` stays behind `cache.len`), exactly as after
//! device decode tokens, so decode continues on the device and a later host pass pulls once.

use crate::device_decode::{head, mat, pull_if_stale, push, Head, OpPrep, Prep};
use crate::model::{Cache, Lfm2};
use crate::moe::Route;
use crate::weights::{Ffn, Operator};
use burn::tensor::{Tensor, TensorData};
use lb_q4_gpu::cubecl_server::Handle;
use lb_q4_gpu::decode::rope_table;
use lb_q4_gpu::GpuQ4Mat;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Smallest chunk that takes the tiled GEMM (`lb_q4_gpu::gemm`) instead of the rows kernels:
/// `LB_PREFILL_GEMM_MIN`, default 16 (`0` = always the rows kernels).
pub fn gemm_min() -> usize {
    std::env::var("LB_PREFILL_GEMM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
}

/// Rows per chunk: `LB_PREFILL_UBATCH`, default 512.
pub fn ubatch() -> usize {
    std::env::var("LB_PREFILL_UBATCH").ok().and_then(|v| v.parse().ok()).filter(|v: &usize| *v > 0).unwrap_or(512)
}

fn decline(why: &str) -> Option<Tensor<2>> {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| eprintln!("lb-lfm2: device prefill declined, the host path runs: {why}"));
    None
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
    let (e, l, hd, eps) = (cfg.n_embd, cfg.l_cache, cfg.head_dim(), cfg.rms_eps);
    let (nh, k_used, n_exp, vocab) = (cfg.n_head, cfg.n_expert_used, cfg.n_expert, cfg.n_vocab);
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
    // one routes/weights pair for the chunk, reused by every MoE layer (read back per layer only
    // when the scraper wants the routes)
    let routes = gpu.upload_u32(vec![0u32; 2 * ub * k_used]);
    let weights = gpu.upload_f32(&vec![0f32; ub * k_used]);
    let scrape = m.scraper.enabled();
    let seq = knobs.dd_router_seq();
    let mut logits_out: Vec<f32> = Vec::with_capacity(if all_logits { n * vocab } else { vocab });
    let mut hits = 0u64;
    let gmin = gemm_min();

    let mut c0 = 0usize;
    while c0 < n {
        let mc = ub.min(n - c0);
        let p0 = pos0 + c0;
        let mut emb = vec![0f32; mc * e];
        for (i, &t) in tokens[c0..c0 + mc].iter().enumerate() {
            m.w.embed_row(t, &mut emb[i * e..(i + 1) * e]);
        }
        // the chunk's residual stream: a real buffer (not scratch), so a layer-end recycle keeps it
        let x: Handle = gpu.upload_f32(&emb);
        let gemm = gmin > 0 && mc >= gmin;
        let lin = |r: &crate::weights::MatRef, h: &Handle| if gemm { mat(m, r).launch_gemm(gpu, h, mc) } else { mat(m, r).launch(gpu, h, mc) };
        for (li, layer) in m.w.layers.iter().enumerate() {
            let lp = &prep.layers[li];
            let h = gpu.rms_norm_rows(&x, None, &lp.attn_norm, mc, e, eps);
            let op_out = match (&layer.op, &lp.op) {
                (Operator::Conv { in_proj, out_proj, .. }, OpPrep::Conv { taps }) => {
                    let bcx = lin(in_proj, &h);
                    let state = dev_conv[li].as_ref().expect("conv layer state");
                    let yy = gpu.conv_prefill(&bcx, taps, state, mc, e, l);
                    lin(out_proj, &yy)
                }
                (Operator::Attn { q, k, v, o, .. }, OpPrep::Attn { q_norm, k_norm }) => {
                    let kvh = cfg.kv_heads[li];
                    let (qh, kh, vh) = (lin(q, &h), lin(k, &h), lin(v, &h));
                    let (kc, vc) = (dev_k[li].as_ref().expect("k cache"), dev_v[li].as_ref().expect("v cache"));
                    let qo = gpu.qk_norm_rope_rows(&qh, &kh, &vh, q_norm, k_norm, &rope, rope_rows, kc, vc, (mc, p0, cap), (nh, kvh, hd), eps);
                    let att = gpu.attn_prefill(&qo, kc, vc, (mc, p0, cap), (nh, kvh, hd));
                    lin(o, &att)
                }
                _ => unreachable!("layer {li}: operator and its device copy disagree"),
            };
            let h2 = gpu.rms_norm_rows(&x, Some(&op_out), &lp.ffn_norm, mc, e, eps);
            let mut is_moe = false;
            let mut scraped = false;
            let ffn_out = match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let hh = if gemm { GpuQ4Mat::launch_gemm_gate_up(gpu, mat(m, gate), mat(m, up), &h2, mc) } else { GpuQ4Mat::launch_gate_up_silu(gpu, mat(m, gate), mat(m, up), &h2, mc) };
                    lin(down, &hh)
                }
                Ffn::Moe { .. } => {
                    let stack = m.store.layers.get(&li).expect("stacked layer (checked by host_moe_layers)");
                    let (rh, bh) = lp.router.as_ref().expect("MoE layer router");
                    gpu.router_topk_rows(&h2, rh, bh, &routes, &weights, (mc, n_exp, k_used, e), seq);
                    is_moe = true;
                    hits += (mc * k_used) as u64;
                    let slots = mc * k_used;
                    let outs = if gemm {
                        // the host groups the chunk's slots by expert: one read of the routes (and
                        // the weights, for the scraper) in the middle of the layer
                        let want = if scrape { vec![(routes.clone(), 8 * slots), (weights.clone(), 4 * slots)] } else { vec![(routes.clone(), 8 * slots)] };
                        let b = match gpu.read_raw_keep(want) {
                            Ok(b) => b,
                            Err(err) => return fail(m, cache, &format!("layer {li} routes of the chunk at {p0}: {err}")),
                        };
                        let r: Vec<u32> = b[0].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        if scrape {
                            let w: Vec<f32> = b[1].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            record_routes(m, li, p0, mc, k_used, &r, &w);
                            scraped = true;
                        }
                        let (perm, tiles) = lb_q4_gpu::gemm::group_slots(&r, n_exp);
                        let n_tiles = tiles.len() / 3;
                        let (ph, th) = (gpu.upload_u32(perm), gpu.upload_u32(tiles));
                        stack.launch_grouped(gpu, &h2, mc, &routes, slots, &ph, &th, n_tiles).0
                    } else {
                        stack.launch_device(gpu, &h2, mc, &routes, slots, 0, slots).0
                    };
                    gpu.weighted_sum_rows(&outs, &weights, &routes, (mc, k_used, e))
                }
            };
            gpu.add_rows(&x, &ffn_out, mc * e);
            // end of the layer: only x, the caches and the routes are live; recycle the rest
            let done = if is_moe && scrape && !scraped {
                gpu.read_raw(vec![(routes.clone(), 2 * mc * k_used * 4), (weights.clone(), mc * k_used * 4)]).map(|b| {
                    let r: Vec<u32> = b[0].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let w: Vec<f32> = b[1].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    record_routes(m, li, p0, mc, k_used, &r, &w);
                })
            } else {
                gpu.sync_recycle()
            };
            if let Err(err) = done {
                return fail(m, cache, &format!("layer {li} of the chunk at {p0}: {err}"));
            }
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
                Ok(b) => logits_out.extend(b[0].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
                Err(err) => return fail(m, cache, &format!("logits of the chunk at {p0}: {err}")),
            }
        }
        c0 += mc;
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

/// One layer's routes for a chunk, to the scraper (the host path's record call, same shape).
fn record_routes(m: &Lfm2, li: usize, p0: usize, mc: usize, k: usize, r: &[u32], w: &[f32]) {
    let rs: Vec<Route> = (0..mc).map(|t| Route { experts: (0..k).map(|i| r[2 * (t * k + i)] as usize).collect(), weights: w[t * k..(t + 1) * k].to_vec() }).collect();
    m.scraper.record(li, p0, &rs);
}

/// A failure part-way: the device copy is half-advanced, so drop it. The host state is current
/// (pulled at the start), so the host path can redo the prompt from `cache.len`.
fn fail(m: &Lfm2, cache: &mut Cache, why: &str) -> Option<Tensor<2>> {
    assert!(cache.host_len >= cache.len, "device prefill: {why}, with the host behind: cannot continue");
    eprintln!("lb-lfm2: device prefill abandoned ({why}); the host path runs the prompt");
    cache.dev = None;
    m.timings.dd_fallbacks.fetch_add(1, Ordering::Relaxed);
    None
}
