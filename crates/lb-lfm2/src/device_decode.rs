//! Device-resident decode: one token, one upload, one read-back (Phase D1 of
//! `docs/DEVICE_DECODE_PLAN.md`; `LB_DEVICE_DECODE=1` or the live `device_decode` knob).
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
use crate::moe::{moe_ffn, Route};
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
    layers: Vec<LayerPrep>,
    final_norm: Handle,
    /// The residual stream, `[n_embd]`. Persistent (never scratch), because a host MoE layer's
    /// read recycles the scratch pool in the middle of a token.
    x: Handle,
    /// Every MoE layer's routes `[n_slots, 2]` u32 and weights `[n_slots]`, slot `moe_index * k`.
    routes: Handle,
    weights: Handle,
    n_slots: usize,
    rope: Option<(Handle, usize)>,
    tp: TokenParams,
    /// Why no token can take this path (a matrix not resident, ...); `None` = it can.
    blocker: Option<String>,
    /// MoE layers without a stacked resident layer: the host runs these (`dd_host_moe`) or the
    /// token falls back.
    host_moe_layers: Vec<usize>,
    warned: bool,
}

struct LayerPrep {
    attn_norm: Handle,
    ffn_norm: Handle,
    op: OpPrep,
    /// Index of this layer's routes in the per-token buffer (MoE layers only).
    moe_index: Option<usize>,
    router: Option<(Handle, Handle)>,
}

enum OpPrep {
    Conv { taps: Handle },
    Attn { q_norm: Handle, k_norm: Handle },
}

/// The device copy of a sequence's state: per attention layer K and V `[kvh, cap, hd]`, per
/// conv layer the rolling state `[l_cache - 1, n_embd]`, valid for positions `0..len`.
pub struct DevCache {
    k: Vec<Option<Handle>>,
    v: Vec<Option<Handle>>,
    conv: Vec<Option<Handle>>,
    cap: usize,
    len: usize,
}

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

enum Head<'a> {
    Q6(&'a lb_q4_gpu::GpuQ6KMat),
    Q4(&'a GpuQ4Mat),
}

impl Prep {
    fn new(m: &Lfm2, gpu: &Gpu) -> Self {
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
                    if !m.store.layers.contains_key(&li) {
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
            rope: None,
            tp: TokenParams::new(gpu, 0, 1, 1),
            blocker,
            host_moe_layers,
            warned: false,
        }
    }
}

fn head(m: &Lfm2) -> Option<Head<'_>> {
    let s = m.w.embd.start;
    m.dense.q6.get(&s).map(Head::Q6).or_else(|| m.dense.resident.get(&s).map(Head::Q4))
}

fn mat<'a>(m: &'a Lfm2, r: &MatRef) -> &'a GpuQ4Mat {
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
fn push(m: &Lfm2, gpu: &Gpu, cache: &mut Cache) {
    assert_eq!(cache.host_len, cache.len, "pushing a host state that is behind");
    let cfg = m.cfg();
    let (e, l, hd, cap) = (cfg.n_embd, cfg.l_cache, cfg.head_dim(), cache.cap);
    let mut dev = DevCache { k: vec![None; cfg.n_layer], v: vec![None; cfg.n_layer], conv: vec![None; cfg.n_layer], cap, len: cache.len };
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
    let decision = decide(prep.blocker.as_deref(), prep.host_moe_layers.len(), knobs.dd_host_moe());
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
    }
    let (rope, rope_rows) = prep.rope.as_ref().map(|(h, r)| (h.clone(), *r)).expect("just built");
    prep.tp.set(gpu, pos0, total, cache.cap);
    // the device buffers by handle (cheap clones), so `cache` stays free for `abandon`
    let (dev_k, dev_v, dev_conv) = {
        let d = cache.dev.as_ref().expect("just pushed");
        (d.k.clone(), d.v.clone(), d.conv.clone())
    };

    let mut emb = vec![0f32; e];
    m.w.embed_row(tok, &mut emb);
    gpu.write_f32(&prep.x, &emb);
    // handles uploaded mid-token (host MoE outputs) stay alive until the token's read
    let mut keep: Vec<Handle> = Vec::new();
    let mut host_routes: Vec<(usize, Route)> = Vec::new();
    let mut y: Option<Handle> = None;
    let mut dev_moe = 0u64;
    for (li, layer) in m.w.layers.iter().enumerate() {
        let lp = &prep.layers[li];
        let h = gpu.add_rms_norm(&prep.x, y.as_ref(), &lp.attn_norm, e, eps);
        let op_out = match (&layer.op, &lp.op) {
            (Operator::Conv { in_proj, out_proj, .. }, OpPrep::Conv { taps }) => {
                let bcx = mat(m, in_proj).launch(gpu, &h, 1);
                let state = dev_conv[li].as_ref().expect("conv layer state");
                let yy = gpu.conv_decode(&bcx, taps, state, e, l);
                mat(m, out_proj).launch(gpu, &yy, 1)
            }
            (Operator::Attn { q, k, v, o, .. }, OpPrep::Attn { q_norm, k_norm }) => {
                let kvh = cfg.kv_heads[li];
                let (qh, kh, vh) = (mat(m, q).launch(gpu, &h, 1), mat(m, k).launch(gpu, &h, 1), mat(m, v).launch(gpu, &h, 1));
                let (kc, vc) = (dev_k[li].as_ref().expect("k cache"), dev_v[li].as_ref().expect("v cache"));
                let qo = gpu.qk_norm_rope(&qh, &kh, &vh, q_norm, k_norm, &rope, rope_rows, kc, vc, &prep.tp, (nh, kvh, hd), eps);
                let att = gpu.attn_decode(&qo, kc, vc, &prep.tp, (nh, kvh, hd));
                mat(m, o).launch(gpu, &att, 1)
            }
            _ => unreachable!("layer {li}: operator and its device copy disagree"),
        };
        let h2 = gpu.add_rms_norm(&prep.x, Some(&op_out), &lp.ffn_norm, e, eps);
        let ffn_out = match &layer.ffn {
            Ffn::Dense { gate, up, down } => {
                let hh = GpuQ4Mat::launch_gate_up_silu(gpu, mat(m, gate), mat(m, up), &h2, 1);
                mat(m, down).launch(gpu, &hh, 1)
            }
            Ffn::Moe { router, bias, gate, up, down } => match (m.store.layers.get(&li), lp.moe_index, &lp.router) {
                (Some(stack), Some(mi), Some((rh, bh))) => {
                    let base = mi * k_used;
                    gpu.router_topk(&h2, rh, bh, &prep.routes, &prep.weights, (n_exp, k_used, e, base, prep.n_slots), knobs.dd_router_seq());
                    let (outs, _) = stack.launch_device(gpu, &h2, 1, &prep.routes, prep.n_slots, base, k_used);
                    dev_moe += 1;
                    gpu.weighted_sum(&outs, &prep.weights, &prep.routes, (k_used, e, base, prep.n_slots))
                }
                _ => {
                    // not resident as a stack: the old host path for this layer, one sync
                    let hv = match gpu.read_all(vec![(h2, e)]) {
                        Ok(mut v) => v.remove(0),
                        Err(err) => return abandon(m, cache, &format!("host MoE layer {li} read: {err}")),
                    };
                    let (out, routes) = moe_ffn(&m.w, router, bias, gate, up, down, &hv, 1, &m.store, li, &m.stats);
                    host_routes.push((li, routes.into_iter().next().expect("one token, one route")));
                    let hnd = gpu.upload_f32(&out);
                    keep.push(hnd.clone());
                    hnd
                }
            },
        };
        y = Some(ffn_out);
    }
    let hf = gpu.add_rms_norm(&prep.x, y.as_ref(), &prep.final_norm, e, eps);
    let vocab = cfg.n_vocab;
    let logits = match head(m).expect("checked in Prep::new") {
        Head::Q6(h6) => h6.launch(gpu, &hf, 1),
        Head::Q4(h4) => h4.launch(gpu, &hf, 1),
    };
    let read = gpu.read_raw(vec![(logits, vocab * 4), (prep.routes.clone(), prep.n_slots * 8), (prep.weights.clone(), prep.n_slots * 4)]);
    drop(keep);
    let bytes = match read {
        Ok(b) => b,
        Err(err) => return abandon(m, cache, &format!("token read: {err}")),
    };
    let f32s = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
    let logits = f32s(&bytes[0]);
    let routes: Vec<u32> = bytes[1].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let weights = f32s(&bytes[2]);

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
    m.stats.gpu_hits.fetch_add(dev_moe * k_used as u64, Ordering::Relaxed);
    if let Some(d) = cache.dev.as_mut() {
        d.len = total;
    }
    cache.len = total;
    m.timings.dd_tokens.fetch_add(1, Ordering::Relaxed);
    m.timings.dd_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Some(Tensor::from_data(TensorData::new(logits, [1, vocab]), &m.device))
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
