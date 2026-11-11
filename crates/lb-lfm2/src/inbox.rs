//! Work-to-weights inside one box (W3e): expert cards opened **in the trunk's own process**.
//!
//! A socket host (`remote.rs`) costs every MoE layer a read on the trunk (the activation leaves
//! the card), a read on the host (its experts' outputs leave its card) and an upload back: two
//! host syncs a layer, 76 a token, which W3 measured as the wall (0.19x of one card all-resident
//! with the W1 host, about 0.27x with v2). When the expert card is in the same box there is no
//! reason for the activation to visit the host at all. Here the trunk keeps it on the device:
//!
//! ```text
//! trunk card A:  ... h2 = add_rms_norm(x, op_out, ffn_norm)
//!                copy h2 A -> B                         (device to device, stream ordered)
//! expert card B: router_topk(h2) -> routes, weights       (the same kernels, the same order
//!                stacked experts on the routes            as the one-card device decode)
//!                weighted_sum -> y
//!                copy y B -> A
//! trunk card A:  next layer ...           one read for the whole token, as on one card
//! ```
//!
//! Nothing on B is ever read in the middle of a token: B's routes come back after the token
//! (only when the scraper is on) and B is fenced once a token to recycle its scratch. Because
//! the router, the expert kernels and the weighted sum are exactly the one-card device decode's,
//! a token is bit-identical to the all-resident one-card token, not merely equal to rounding.
//!
//! Placement stays data: a host in the placement file whose `addr` is `gpu://N` is CUDA device
//! `N` of this process (`CUDA_VISIBLE_DEVICES` order), and it must hold whole MoE layers (its
//! experts run as one stacked buffer). Socket hosts are for other boxes; mixing both kinds in
//! one placement is not built yet and is refused.
//!
//! The host-routed paths (prefill, a token the device decode declines) run an in-box layer's
//! experts on its card through per-expert views of the same stacked buffers, as the trunk runs
//! its own residents: one upload and one read on that card per layer pass.

use crate::weights::{Ffn, Weights};
use lb_expert_host::placement::{PlacementMap, LOCAL};
use lb_q4_gpu::cubecl_server::Handle;
use lb_q4_gpu::{Gpu, GpuExpert, GpuExpertLayer};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// One expert card of this process.
pub struct Card {
    pub name: String,
    /// CUDA device index in this process.
    pub index: usize,
    pub gpu: Gpu,
    pub layers: HashMap<usize, GpuExpertLayer>,
    /// Per-expert views into `layers`, for the host-routed paths.
    pub views: HashMap<(usize, usize), GpuExpert>,
    /// Router weights and bias of the layers this card holds.
    pub routers: HashMap<usize, (Handle, Handle)>,
    /// This card's routes / weights buffers, the shape of the trunk's (`[n_slots, 2]`, `[n_slots]`).
    pub routes: Handle,
    pub weights: Handle,
    pub n_slots: usize,
    pub bytes: usize,
}

#[derive(Default, Debug)]
pub struct InBoxStats {
    /// Device tokens that used an in-box card.
    pub tokens: AtomicU64,
    /// MoE layer passes served on an in-box card in a device token (two card-to-card copies each).
    pub device_layers: AtomicU64,
    /// MoE layer passes served on an in-box card on a host-routed path (prefill, fallback).
    pub host_layers: AtomicU64,
    /// End-of-token fences / reads on in-box cards.
    pub card_syncs: AtomicU64,
}

pub struct InBox {
    pub cards: Vec<Card>,
    /// Per layer: the card that holds it whole, if any.
    pub layer_card: Vec<Option<usize>>,
    pub stats: InBoxStats,
}

/// Whether the placement names any in-box card (`gpu://N`).
pub fn names_inbox(map: &PlacementMap) -> bool {
    map.hosts.iter().any(|h| h.addr.starts_with("gpu://"))
}

impl InBox {
    /// Open the cards the placement names and upload the whole layers it puts on them.
    /// `n_slots` = MoE layers x experts used (the device decode's routes buffer).
    pub fn build(w: &Weights, map: &PlacementMap, n_slots: usize, trunk_index: Option<usize>) -> Result<Self, String> {
        if let Some(h) = map.hosts.iter().find(|h| !h.addr.starts_with("gpu://")) {
            return Err(format!("host {} ({}) is a socket host: a placement with both in-box cards (gpu://N) and socket hosts is not built yet", h.name, h.addr));
        }
        let cfg = &w.cfg;
        let mut layer_card: Vec<Option<usize>> = vec![None; cfg.n_layer];
        let mut want: Vec<Vec<usize>> = vec![Vec::new(); map.hosts.len()];
        for (l, row) in map.assign.iter().enumerate() {
            if !cfg.is_moe(l) {
                continue;
            }
            let first = row[0];
            if row.iter().all(|&a| a == LOCAL) {
                continue;
            }
            if first == LOCAL || row.iter().any(|&a| a != first) {
                return Err(format!("layer {l}: an in-box card holds whole MoE layers; this placement splits it"));
            }
            want[first as usize].push(l);
        }
        let mut cards = Vec::new();
        for (h, spec) in map.hosts.iter().enumerate() {
            let index: usize = spec.addr["gpu://".len()..].trim_end_matches('/').parse().map_err(|_| format!("host {}: {} is not gpu://<index>", spec.name, spec.addr))?;
            if Some(index) == trunk_index {
                return Err(format!("host {}: gpu://{index} is the trunk's own card", spec.name));
            }
            if want[h].is_empty() {
                continue;
            }
            let gpu = open_card(index)?;
            let t = std::time::Instant::now();
            let (mut layers, mut views, mut routers, mut bytes) = (HashMap::new(), HashMap::new(), HashMap::new(), 0usize);
            for &l in &want[h] {
                let Ffn::Moe { router, bias, gate, up, down } = &w.layers[l].ffn else { unreachable!("checked is_moe") };
                let stack = GpuExpertLayer::upload_stacked(&gpu, w.mat(&gate.all()), w.mat(&up.all()), w.mat(&down.all()), gate.n)?;
                for e in 0..gate.n {
                    views.insert((l, e), stack.expert(&gpu, e)?);
                }
                bytes += stack.bytes_on_device();
                layers.insert(l, stack);
                routers.insert(l, (gpu.upload_f32(router), gpu.upload_f32(bias)));
                layer_card[l] = Some(cards.len());
            }
            let routes = gpu.upload_u32(vec![0u32; 2 * n_slots.max(1)]);
            let weights = gpu.upload_f32(&vec![0f32; n_slots.max(1)]);
            gpu.sync()?;
            eprintln!(
                "lb-lfm2 w2w in-box: card {} = CUDA device {index} ({}): {} whole MoE layers, {:.0} MiB, in {:.1}s; card-to-card copies {}",
                spec.name,
                gpu.name,
                layers.len(),
                bytes as f64 / 1048576.0,
                t.elapsed().as_secs_f64(),
                if gpu.device_transport() { "device to device (NCCL)" } else { "THROUGH THE HOST (no device transport: NCCL not loadable)" }
            );
            cards.push(Card { name: spec.name.clone(), index, gpu, layers, views, routers, routes, weights, n_slots, bytes });
        }
        Ok(Self { cards, layer_card, stats: InBoxStats::default() })
    }

    pub fn card_of(&self, layer: usize) -> Option<&Card> {
        self.layer_card.get(layer).copied().flatten().map(|c| &self.cards[c])
    }

    /// A host-routed MoE layer pass on the card that holds `layer`: every routed expert from its
    /// view, one upload per distinct token set and one read on that card. Returns one output per
    /// call, in the order of `experts` (ascending expert id).
    pub fn run_host_routed(&self, layer: usize, experts: &[(usize, Vec<(usize, f32)>)], x: &[f32], n_embd: usize) -> Result<Vec<Vec<f32>>, String> {
        let card = self.card_of(layer).ok_or_else(|| format!("layer {layer} is on no in-box card"))?;
        let mut uploaded: HashMap<Vec<usize>, Handle> = HashMap::new();
        let mut launched = Vec::with_capacity(experts.len());
        for (e, toks) in experts {
            let key: Vec<usize> = toks.iter().map(|(t, _)| *t).collect();
            let xh = uploaded
                .entry(key)
                .or_insert_with(|| {
                    let mut xs = Vec::with_capacity(toks.len() * n_embd);
                    for (t, _) in toks {
                        xs.extend_from_slice(&x[t * n_embd..(t + 1) * n_embd]);
                    }
                    card.gpu.upload_f32(&xs)
                })
                .clone();
            let v = card.views.get(&(layer, *e)).ok_or_else(|| format!("expert ({layer}, {e}) not on card {}", card.name))?;
            launched.push(v.launch(&card.gpu, &xh, toks.len()));
        }
        let outs = card.gpu.read_all(launched)?;
        self.stats.host_layers.fetch_add(1, Ordering::Relaxed);
        Ok(outs)
    }

    pub fn status(&self) -> serde_json::Value {
        let o = Ordering::Relaxed;
        let card = |c: &Card| {
            let mut layers: Vec<usize> = c.layers.keys().copied().collect();
            layers.sort_unstable();
            serde_json::json!({"name": c.name, "cuda_index": c.index, "layers": layers, "mib": c.bytes as f64 / 1048576.0, "device_transport": c.gpu.device_transport()})
        };
        serde_json::json!({
            "cards": self.cards.iter().map(card).collect::<Vec<_>>(),
            "device_tokens": self.stats.tokens.load(o), "device_layers": self.stats.device_layers.load(o), "host_layers": self.stats.host_layers.load(o), "card_syncs": self.stats.card_syncs.load(o),
        })
    }
}

#[cfg(feature = "cuda")]
fn open_card(index: usize) -> Result<Gpu, String> {
    Gpu::open_cuda(index)
}

#[cfg(not(feature = "cuda"))]
fn open_card(index: usize) -> Result<Gpu, String> {
    Err(format!("gpu://{index}: in-box expert cards need the cuda build"))
}
