//! Quant fit: price the model in the common quant types and say how deep into the memory
//! hierarchy each one has to go on this fleet. Pricing is the pure block-geometry floor
//! (bits per weight from ggml block sizes: bytes per block x 8 / values per block); real
//! mixed files (Unsloth UD-*, *_K_M) keep some tensors higher and are larger, so the file in
//! hand is priced exactly as "as-is" and the others are marked as floors.

use super::model::ModelDesc;
use super::plan::{place, residency, usable_gpus};
use super::*;

/// (file naming, bits per weight, block geometry source)
pub fn specs() -> Vec<(&'static str, f64, &'static str)> {
    vec![
        ("UD-IQ1_S", 50.0 * 8.0 / 256.0, "IQ1_S: 50 B / 256 values (UD mixes keep some tensors higher: floor)"),
        ("IQ2_XXS", 66.0 * 8.0 / 256.0, "IQ2_XXS: 66 B / 256 values"),
        ("Q4_0", 18.0 * 8.0 / 32.0, "Q4_0: 18 B / 32 values"),
        ("Q4_K_M", 144.0 * 8.0 / 256.0, "Q4_K: 144 B / 256 values (the _M mix adds Q6_K tensors: floor)"),
        ("Q8_0", 34.0 * 8.0 / 32.0, "Q8_0: 34 B / 32 values"),
        ("F16", 16.0, "F16: 2 B / value"),
    ]
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QuantRow {
    pub name: String,
    pub bpw: f64,
    pub total_gib: f64,
    pub exact: bool,
    pub level: String,
    pub rank: i32,
    pub node: Option<String>,
    pub kernel_note: Option<String>,
}

pub fn reprice(m: &ModelDesc, bpw: f64) -> ModelDesc {
    let mut r = m.clone();
    let moe = m.moe_layers().max(1) as f64;
    let ppe = if m.n_expert > 0 { m.params_expert as f64 / (m.n_expert as f64 * moe) } else { 0.0 };
    r.expert_bytes = (ppe * bpw / 8.0) as u64;
    let dense_b = (m.dense_total_bytes() + m.head_bytes) as f64;
    let frac_layers = if dense_b > 0.0 { m.dense_total_bytes() as f64 / dense_b } else { 1.0 };
    let dense_params = m.params_dense as f64;
    r.dense_layer_bytes = (dense_params * frac_layers * bpw / 8.0 / m.n_layer as f64) as u64;
    r.head_bytes = (dense_params * (1.0 - frac_layers) * bpw / 8.0) as u64;
    r.total_bytes = r.expert_total_bytes() + r.dense_total_bytes() + r.head_bytes;
    r
}

fn rank_of(res: &str) -> i32 {
    match res {
        "all resident" | "all resident (incl. helper VRAM)" => 0,
        "resident + RAM" => 1,
        "resident + RAM + disk tier" => 2,
        _ => 3,
    }
}

fn level_name(rank: i32) -> &'static str {
    ["all-resident", "resident+RAM", "resident+RAM+tier", "streaming (model-file reads)", "does not fit"][rank.clamp(0, 4) as usize]
}

/// Deepest level needed on the best node (all of its usable cards, with or without a helper).
pub fn fit(fleet: &Fleet, m: &ModelDesc, knobs: &Knobs) -> (i32, Option<String>, Option<String>) {
    let mut best: (i32, Option<String>, Option<String>) = (4, None, None);
    for node in &fleet.nodes {
        let sel = usable_gpus(node);
        if sel.is_empty() {
            continue;
        }
        let mut helpers: Vec<Option<(&NodeDesc, &Link)>> = vec![None];
        for h in &fleet.nodes {
            if h.name != node.name && !usable_gpus(h).is_empty() {
                if let Some(l) = fleet.link(&node.name, &h.name) {
                    helpers.push(Some((h, l)));
                }
            }
        }
        for h in helpers {
            // all cards, then fewer if the smaller cards cannot hold a layer
            for n in (1..=sel.len()).rev() {
                if let Ok(p) = place(fleet, node, &sel[..n], h, m, knobs) {
                    let r = rank_of(residency(&p));
                    if r < best.0 {
                        let note = p.gpus.iter().find(|g| g.has_dp4a == Some(false)).map(|g| format!("{} has no int8 dot: dequantizes to fp16/fp32 before the multiply", g.name));
                        best = (r, Some(node.name.clone()), note);
                    }
                    break;
                }
            }
        }
    }
    best
}

pub fn table(fleet: &Fleet, m: &ModelDesc, knobs: &Knobs) -> Vec<QuantRow> {
    let mut rows = Vec::new();
    let (r, n, k) = fit(fleet, m, knobs);
    rows.push(QuantRow { name: "as-is (file in hand)".into(), bpw: m.total_bytes as f64 * 8.0 / (m.params_dense + m.params_expert).max(1) as f64, total_gib: m.total_bytes as f64 / GIB, exact: true, level: level_name(r).into(), rank: r, node: n, kernel_note: k });
    for (name, bpw, _src) in specs() {
        let rm = reprice(m, bpw);
        let (r, n, k) = fit(fleet, &rm, knobs);
        rows.push(QuantRow { name: name.into(), bpw, total_gib: rm.total_bytes as f64 / GIB, exact: false, level: level_name(r).into(), rank: r, node: n, kernel_note: k });
    }
    rows
}

/// "Remake the quant" lines: only when a smaller quant reaches a shallower level than the file in hand.
pub fn remake_advice(m: &ModelDesc, rows: &[QuantRow], knobs: &Knobs) -> Vec<String> {
    let cur = rows[0].rank;
    let mut out = Vec::new();
    let Some(better) = rows.iter().skip(1).filter(|r| r.rank < cur).max_by(|a, b| a.bpw.partial_cmp(&b.bpw).unwrap()) else {
        return out;
    };
    let params = (m.params_dense + m.params_expert) as f64;
    let src_gib = params * 2.0 / GIB;
    let hours = match knobs.requant_gb_per_hour {
        Some(r) => format!("{:.1} h at the stated {:.0} GB/h", src_gib * 1.073741824 / r, r),
        None => "hours: unknown (no measured quantize rate; pass --requant-gb-h)".into(),
    };
    out.push(format!(
        "remake the quant: {} (about {:.0} GiB, floor) reaches {} instead of {}. Needs: an F16/BF16 source of ~{:.0} GiB on disk plus ~{:.0} GiB for the output, RAM for the largest tensor ({:.1} GiB) plus working buffers, {}. Quality change is not estimated here.",
        better.name,
        better.total_gib,
        better.level,
        rows[0].level,
        src_gib,
        better.total_gib,
        m.largest_tensor_bytes as f64 / GIB,
        hours
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fleet::plan::tests::{node, p100, small_moe};

    fn model() -> ModelDesc {
        let mut m = small_moe();
        m.n_expert = 128;
        m.params_expert = (m.n_expert * 18) as u64 * 8_000_000; // 8 M weights per expert
        m.expert_bytes = (8_000_000.0 * 4.5 / 8.0) as u64; // Q4_0
        m.params_dense = 1_500_000_000;
        m.dense_layer_bytes = 60 << 20;
        m.total_bytes = m.expert_total_bytes() + m.dense_total_bytes() + m.head_bytes;
        m
    }

    #[test]
    fn smaller_quants_reach_shallower_levels() {
        let f = Fleet { nodes: vec![node("a", vec![p100()], 8.0)], ..Default::default() };
        let m = model();
        let rows = table(&f, &m, &Knobs::default());
        let by = |n: &str| rows.iter().find(|r| r.name == n).unwrap().clone();
        assert!(by("UD-IQ1_S").rank <= by("Q4_0").rank);
        assert!(by("Q4_0").rank <= by("F16").rank);
        assert!(by("F16").total_gib > by("Q8_0").total_gib && by("Q8_0").total_gib > by("Q4_0").total_gib);
        assert!(by("Q4_0").kernel_note.as_deref().map_or(false, |s| s.contains("no int8 dot")));
        assert_eq!(rows[0].name, "as-is (file in hand)");
    }

    #[test]
    fn remake_line_states_resources_and_unknown_rate() {
        let f = Fleet { nodes: vec![node("a", vec![p100()], 8.0)], ..Default::default() };
        let m = model();
        let rows = table(&f, &m, &Knobs::default());
        if rows[0].rank > 0 {
            let adv = remake_advice(&m, &rows, &Knobs::default());
            assert!(adv[0].contains("hours: unknown"));
            let mut k = Knobs::default();
            k.requant_gb_per_hour = Some(100.0);
            assert!(remake_advice(&m, &rows, &k)[0].contains("at the stated 100"));
        }
    }
}
