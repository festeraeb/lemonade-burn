//! Measured timings of one card and of the host, the inputs the simulator needs.
//! Produced by `lb-gpu-bench` (crate `lb-q4-gpu`, one JSON object per line) and by the
//! defaults here when a card has not been measured yet (clearly marked `measured: false`).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// LFM2.5-24B-A2B expert: gate+up (1536x2048) + down (2048x1536) in Q4_0 = 9.44 M values x 18/32 B.
pub const EXPERT_BYTES: usize = 2 * 1536 * (2048 / 32 * 18) + 2048 * (1536 / 32 * 18);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CardTimings {
    /// Human name (from the bench: host + device spec).
    pub name: String,
    pub host: String,
    /// `LB_WGPU_DEVICE` spec that selects this card on that host.
    pub device: String,
    /// One resident expert for one token, launch-only cost when batched (ms per expert).
    pub expert_ms: f64,
    /// Same for a 4-token batch (prefill-ish), per expert.
    pub expert_m4_ms: f64,
    /// Host -> device copy bandwidth for an expert-sized blob (MB/s, achieved).
    pub h2d_mb_s: f64,
    /// Same for a 64 MB blob (what a whole-layer prestage looks like).
    pub h2d_bulk_mb_s: f64,
    /// Activation handoff through the host: read back + re-upload one token's row (us).
    pub handoff_us: f64,
    /// Dense part of one short-conv layer (in_proj + out_proj matvecs) on this card, ms.
    pub conv_layer_ms: f64,
    /// Dense part of one attention layer (q, k, v, o matvecs) on this card, ms.
    pub attn_layer_ms: f64,
    /// Dense FFN of the two leading dense layers (gate/up/down 11776 wide), ms.
    pub dense_ffn_ms: f64,
    /// Free VRAM at measurement time (MB), from the driver when available.
    pub vram_free_mb: f64,
    pub measured: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HostTimings {
    pub host: String,
    pub threads: usize,
    /// One expert for one token on the CPU reference path, all threads (ms).
    pub cpu_expert_ms: f64,
    /// Q6_K tied head (128000 x 2048) on the CPU (ms per token).
    pub cpu_head_ms: f64,
    /// CPU dense time per conv layer / attention layer / leading dense FFN (ms), derived
    /// from the CPU expert rate when not measured directly.
    pub cpu_conv_layer_ms: f64,
    pub cpu_attn_layer_ms: f64,
    pub cpu_dense_ffn_ms: f64,
    pub measured: bool,
}

/// Parse the JSON lines written by `lb-gpu-bench`.
pub fn parse_bench(text: &str) -> Result<(CardTimings, HostTimings)> {
    let mut vals: HashMap<String, f64> = HashMap::new();
    let (mut host, mut device, mut gpu) = (String::new(), String::new(), String::new());
    for line in text.lines().filter(|l| l.trim_start().starts_with('{')) {
        let v: serde_json::Value = serde_json::from_str(line).with_context(|| format!("bench line: {line}"))?;
        if let Some(m) = v.get("metric").and_then(|m| m.as_str()) {
            vals.insert(m.to_string(), v["value"].as_f64().unwrap_or(0.0));
        } else {
            host = v["host"].as_str().unwrap_or("").to_string();
            device = v["device"].as_str().unwrap_or("").to_string();
            gpu = v["gpu"].as_str().unwrap_or("").to_string();
        }
    }
    let g = |k: &str| -> Result<f64> { vals.get(k).copied().with_context(|| format!("bench output lacks {k}")) };
    if vals.is_empty() {
        bail!("no bench metrics found");
    }
    // Dense layer costs from the measured matvecs (launch + readback each; the readback is
    // counted once per layer by subtracting the small-matrix floor from the larger ones).
    let in_proj = g("dense_in_proj_6144x2048_ms")?;
    let sq = g("dense_out_proj_2048x2048_ms")?;
    let kv = g("dense_kv_512x2048_ms")?;
    // per-row cost from the two square-ish points; the k/v matvec is near the launch floor
    let per_row = ((in_proj - sq) / (6144.0 - 2048.0)).max(0.0);
    let floor = (sq - per_row * 2048.0).max(0.0);
    let conv_layer_ms = floor + per_row * (6144.0 + 2048.0);
    let attn_layer_ms = floor + per_row * (2048.0 + 512.0 + 512.0 + 2048.0) + 0.2 * kv.max(0.05);
    // dense FFN: gate/up 11776 rows of 2048 each, down 2048 rows of 11776 (5.75x the cols)
    let dense_ffn_ms = floor + per_row * (2.0 * 11776.0 + 2048.0 * (11776.0 / 2048.0));
    let card = CardTimings {
        name: format!("{host}:{device} {gpu}"),
        host: host.clone(),
        device,
        expert_ms: g("expert_ffn_batch4_m1_ms_per_expert")?,
        expert_m4_ms: g("expert_ffn_batch4_m4_ms_per_expert")?,
        h2d_mb_s: g("h2d_expert_mb_s")?,
        h2d_bulk_mb_s: g("h2d_bulk_mb_s")?,
        handoff_us: g("d2h_h2d_act_m1_us")?,
        conv_layer_ms,
        attn_layer_ms,
        dense_ffn_ms,
        vram_free_mb: vals.get("vram_free_mb").copied().unwrap_or(0.0),
        measured: true,
    };
    let cpu_expert_ms = g("cpu_expert_ms_all_threads")?;
    // CPU dense from the expert rate: an expert is 9.44 M Q4 values; scale by values.
    let per_mval = cpu_expert_ms / 9.437184;
    let hostt = HostTimings {
        host,
        threads: 0,
        cpu_expert_ms,
        cpu_head_ms: g("cpu_q6k_head_ms")?,
        cpu_conv_layer_ms: per_mval * (6144.0 * 2048.0 + 2048.0 * 2048.0) / 1e6,
        cpu_attn_layer_ms: per_mval * (2.0 * 2048.0 * 2048.0 + 2.0 * 512.0 * 2048.0) / 1e6,
        cpu_dense_ffn_ms: per_mval * (3.0 * 11776.0 * 2048.0) / 1e6,
        measured: true,
    };
    Ok((card, hostt))
}

pub fn load_bench(path: &Path) -> Result<(CardTimings, HostTimings)> {
    parse_bench(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?)
}

impl CardTimings {
    /// A hypothetical card: scale a measured one's compute and set the bus explicitly.
    /// Used for "what would a faster interconnect / card give" sweeps; `measured: false`.
    pub fn hypothetical(name: &str, base: &CardTimings, compute_scale: f64, h2d_mb_s: f64, vram_free_mb: f64) -> Self {
        CardTimings {
            name: name.to_string(),
            host: base.host.clone(),
            device: "hypothetical".into(),
            expert_ms: base.expert_ms * compute_scale,
            expert_m4_ms: base.expert_m4_ms * compute_scale,
            h2d_mb_s,
            h2d_bulk_mb_s: h2d_mb_s,
            handoff_us: base.handoff_us,
            conv_layer_ms: base.conv_layer_ms * compute_scale,
            attn_layer_ms: base.attn_layer_ms * compute_scale,
            dense_ffn_ms: base.dense_ffn_ms * compute_scale,
            vram_free_mb,
            measured: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bench_lines() {
        let txt = r#"{"host":"h","device":"discrete:2","gpu":"Wgpu","expert_bytes":5308416}
{"metric":"cpu_expert_ms_all_threads","value":1.0,"note":""}
{"metric":"cpu_expert_ms_1_thread","value":20.0,"note":""}
{"metric":"cpu_q6k_head_ms","value":30.0,"note":""}
{"metric":"h2d_expert_mb_s","value":5000,"note":""}
{"metric":"h2d_bulk_mb_s","value":6000,"note":""}
{"metric":"d2h_h2d_act_m1_us","value":300,"note":""}
{"metric":"expert_ffn_m1_ms","value":2.0,"note":""}
{"metric":"expert_ffn_batch4_m1_ms_per_expert","value":0.5,"note":""}
{"metric":"expert_ffn_m4_ms","value":2.5,"note":""}
{"metric":"expert_ffn_batch4_m4_ms_per_expert","value":0.7,"note":""}
{"metric":"dense_in_proj_6144x2048_ms","value":1.2,"note":""}
{"metric":"dense_out_proj_2048x2048_ms","value":0.6,"note":""}
{"metric":"dense_kv_512x2048_ms","value":0.4,"note":""}"#;
        let (c, h) = parse_bench(txt).unwrap();
        assert_eq!(c.host, "h");
        assert!((c.expert_ms - 0.5).abs() < 1e-9);
        assert!(c.conv_layer_ms > c.attn_layer_ms);
        assert!(h.cpu_conv_layer_ms > 0.0 && h.cpu_expert_ms == 1.0);
        assert_eq!(EXPERT_BYTES, 5308416);
    }
}
