//! Read a GGUF and work out what an adapter for it would need.

use anyhow::{Context, Result};
use lb_gguf::Gguf;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

/// How a config field was settled.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum Origin {
    /// Read from this GGUF key.
    Key(String),
    /// Not in the file; this value is the engine's documented default.
    Default(String),
    /// Not in the file and there is no safe default: the adapter must not invent one.
    Missing,
}

#[derive(Clone, Debug, Serialize)]
pub struct Field {
    pub name: &'static str,
    /// Rust type the scaffold emits.
    pub ty: &'static str,
    pub value: Option<String>,
    pub origin: Origin,
    /// Why the engine needs it, for the report and the generated doc comment.
    pub why: &'static str,
}

impl Field {
    pub fn resolved(&self) -> bool {
        matches!(self.origin, Origin::Key(_))
    }
}

/// One group of tensors sharing a shape, with the per-block index collapsed to `*`.
#[derive(Clone, Debug, Serialize)]
pub struct TensorGroup {
    /// e.g. `blk.*.ffn_gate_exps.weight`
    pub pattern: String,
    pub count: usize,
    pub dims: Vec<usize>,
    pub ty: String,
}

/// What MoE weight layout the file uses.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum MoeLayout {
    /// One stacked tensor per projection: `ffn_(gate|up|down)_exps`.
    Fused,
    /// One tensor per expert: `blk.N.ffn_gate.M.weight`.
    PerExpert,
    /// Experts not found: either a dense model or a naming scheme we do not know.
    None,
}

#[derive(Clone, Debug, Serialize)]
pub struct Probe {
    pub path: String,
    pub arch: String,
    pub fields: Vec<Field>,
    pub tensor_groups: Vec<TensorGroup>,
    /// Every `{arch}.*` key in the file, including ones no field claimed.
    pub arch_keys: BTreeMap<String, String>,
    /// `{arch}.*` keys no field used: either harmless or a feature we do not model yet.
    pub unclaimed_keys: Vec<String>,
    pub moe_layout: MoeLayout,
    pub has_shared_expert: bool,
    pub has_router_bias: bool,
    pub has_qk_norm: bool,
    pub has_mtp: bool,
    pub notes: Vec<String>,
}

/// Collapse `blk.12.foo.weight` to `blk.*.foo.weight`, and `blk.3.ffn_gate.7.weight`
/// to `blk.*.ffn_gate.*.weight`, so per-block tensors group together.
fn pattern_of(name: &str) -> String {
    name.split('.').map(|p| if p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty() { "*" } else { p }).collect::<Vec<_>>().join(".")
}

fn meta_string(g: &Gguf, key: &str) -> Option<String> {
    if let Some(s) = g.meta_str(key) {
        return Some(s.to_string());
    }
    if let Some(v) = g.meta_u64(key) {
        return Some(v.to_string());
    }
    if let Some(v) = g.meta_f64(key) {
        return Some(v.to_string());
    }
    g.meta_u64_array(key).map(|a| format!("{a:?}"))
}

/// The fields every adapter in this engine needs, with the GGUF key each is read from.
/// `default`: `Some(v)` when leaving it out is legitimate and `v` is what the engine assumes.
#[allow(clippy::type_complexity)]
const SPEC: &[(&str, &str, &str, Option<&str>, &str)] = &[
    // name, rust type, key suffix under `{arch}.`, default, why
    ("n_layer", "usize", "block_count", None, "number of transformer blocks"),
    ("n_embd", "usize", "embedding_length", None, "residual width"),
    ("n_ff", "usize", "feed_forward_length", None, "dense FFN width (non-MoE layers)"),
    ("n_head", "usize", "attention.head_count", None, "attention query heads"),
    ("n_head_kv", "usize", "attention.head_count_kv", None, "KV heads (GQA); may be a per-layer array"),
    ("n_expert", "usize", "expert_count", Some("0"), "experts per MoE layer; 0 = dense model"),
    ("n_expert_used", "usize", "expert_used_count", Some("0"), "top-k experts routed per token"),
    ("n_ff_exp", "usize", "expert_feed_forward_length", Some("0"), "width of one expert's FFN"),
    ("n_ff_shexp", "usize", "expert_shared_feed_forward_length", Some("0"), "shared-expert FFN width; 0 = none"),
    ("n_dense_lead", "usize", "leading_dense_block_count", Some("0"), "leading blocks with a dense FFN instead of experts"),
    ("gating", "u64", "expert_gating_func", Some("1"), "router: 1 = softmax, 2 = sigmoid + selection bias"),
    ("expert_weights_scale", "f32", "expert_weights_scale", Some("1.0"), "scale applied to router weights"),
    ("expert_weights_norm", "bool", "expert_weights_norm", Some("false"), "renormalise the top-k router weights"),
    ("rms_eps", "f32", "attention.layer_norm_rms_epsilon", None, "RMSNorm epsilon"),
    ("rope_theta", "f64", "rope.freq_base", None, "RoPE base frequency"),
    ("rope_dim", "usize", "rope.dimension_count", Some("0"), "rotary dims; 0 = full head_dim"),
    ("key_length", "usize", "attention.key_length", Some("0"), "head dim for K when it is not n_embd/n_head"),
    ("value_length", "usize", "attention.value_length", Some("0"), "head dim for V, as above"),
    ("n_vocab", "usize", "vocab_size", None, "vocabulary size"),
    ("n_ctx", "usize", "context_length", None, "training context length"),
    ("nextn_layers", "usize", "nextn_predict_layers", Some("0"), "multi-token-prediction heads; 0 = none"),
];

pub fn probe(path: &Path) -> Result<Probe> {
    let g = Gguf::open(path).with_context(|| format!("opening {}", path.display()))?;
    let arch = g.meta_str("general.architecture").context("general.architecture missing: not a usable GGUF")?.to_string();

    let mut arch_keys = BTreeMap::new();
    for k in &g.meta_order {
        if k.starts_with(&format!("{arch}.")) {
            if let Some(v) = meta_string(&g, k) {
                arch_keys.insert(k.clone(), v);
            }
        }
    }

    let mut fields = Vec::new();
    let mut claimed = Vec::new();
    for &(name, ty, suffix, default, why) in SPEC {
        let key = format!("{arch}.{suffix}");
        let found = meta_string(&g, &key);
        claimed.push(key.clone());
        let (value, origin) = match (found, default) {
            (Some(v), _) => (Some(v), Origin::Key(key)),
            (None, Some(d)) => (Some(d.to_string()), Origin::Default(d.to_string())),
            (None, None) => (None, Origin::Missing),
        };
        fields.push(Field { name, ty, value, origin, why });
    }
    // tokenizer ids live outside the arch namespace
    for (name, key, default) in [
        ("bos", "tokenizer.ggml.bos_token_id", "1"),
        ("eos", "tokenizer.ggml.eos_token_id", "2"),
    ] {
        let (value, origin) = match meta_string(&g, key) {
            Some(v) => (Some(v), Origin::Key(key.to_string())),
            None => (Some(default.to_string()), Origin::Default(default.to_string())),
        };
        fields.push(Field { name, ty: "u32", value, origin, why: "tokenizer special token" });
    }

    let mut groups: BTreeMap<String, TensorGroup> = BTreeMap::new();
    for t in &g.tensors {
        let p = pattern_of(&t.name);
        groups
            .entry(p.clone())
            .and_modify(|gr| gr.count += 1)
            .or_insert(TensorGroup { pattern: p, count: 1, dims: t.dims.clone(), ty: t.ty.name() });
    }
    let has = |needle: &str| groups.keys().any(|k| k.contains(needle));
    let moe_layout = if has("_exps") {
        MoeLayout::Fused
    } else if groups.keys().any(|k| k.starts_with("blk.*.ffn_gate.*")) {
        MoeLayout::PerExpert
    } else {
        MoeLayout::None
    };

    let mut notes = Vec::new();
    if moe_layout == MoeLayout::None {
        notes.push("no expert tensors found: dense model, or an expert naming scheme this probe does not know".into());
    }
    if !has("ffn_gate_inp") && moe_layout != MoeLayout::None {
        notes.push("experts present but no `ffn_gate_inp` router tensor: check the router naming".into());
    }
    let unclaimed_keys: Vec<String> = arch_keys.keys().filter(|k| !claimed.contains(k)).cloned().collect();
    if !unclaimed_keys.is_empty() {
        notes.push(format!("{} `{arch}.*` key(s) are not modelled by any field; review them", unclaimed_keys.len()));
    }

    let (shexp, router_bias, qk_norm, mtp) =
        (has("shexp"), has("exp_probs_b"), has("attn_q_norm") || has("attn_k_norm"), has("nextn") || has("mtp"));
    Ok(Probe {
        path: path.display().to_string(),
        arch,
        fields,
        tensor_groups: groups.into_values().collect(),
        arch_keys,
        unclaimed_keys,
        moe_layout,
        has_shared_expert: shexp,
        has_router_bias: router_bias,
        has_qk_norm: qk_norm,
        has_mtp: mtp,
        notes,
    })
}
