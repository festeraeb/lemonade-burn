//! Human output: a table of the best layouts, the quant fit, and `--explain` text.

use super::plan::{FleetPlan, PlanSet};
use super::quant::QuantRow;
use super::*;
use std::fmt::Write;

fn tok(v: Option<f64>) -> String {
    v.map(|t| if t < 1.0 { format!("{t:.3}") } else { format!("{t:.1}") }).unwrap_or_else(|| "n/a".into())
}

pub fn plan_summary(p: &FleetPlan, explain: bool) -> String {
    let l = &p.layout;
    let e = &l.expect;
    let mut s = String::new();
    let _ = writeln!(s, "node {}  residency: {}", l.node, l.residency);
    let _ = writeln!(s, "  {:<4} {:<26} {:<8} {:>11} {:>9} {:>9} {:>10}", "gpu", "card", "dp4a", "layers", "dense GiB", "KV GiB", "experts/l");
    for c in &l.cards {
        let _ = writeln!(
            s,
            "  {:<4} {:<26} {:<8} {:>5}-{:<5} {:>9.2} {:>9.2} {:>10}",
            c.gpu_index,
            c.name.chars().take(26).collect::<String>(),
            c.has_dp4a.map(|b| if b { "yes" } else { "NO" }).unwrap_or("?"),
            c.layer_start,
            c.layer_end - 1,
            c.dense_mib / 1024.0,
            c.kv_mib / 1024.0,
            c.resident_per_layer
        );
    }
    if let Some(h) = &l.helper {
        let _ = writeln!(s, "  helper {} over {:?} ({} GB/s, {} us): {} experts ({} per layer) on {:?}", h.node, h.link_kind, h.link_gb_s.map(|g| format!("{g:.2}")).unwrap_or("?".into()), h.link_latency_us.map(|g| format!("{g:.0}")).unwrap_or("?".into()), h.experts, h.per_layer, h.gpus);
    }
    let _ = writeln!(s, "  RAM tier {:.1} GiB ({} experts)", l.ram_tier_gib, l.ram_experts);
    for t in &l.tiers {
        let _ = writeln!(s, "  tier {:?} {:.1} GiB ({} experts) @ {:.0} MB/s{}", t.kind, t.gib, t.experts, t.read_mb_s, t.path.as_ref().map(|p| format!("  {p}")).unwrap_or_default());
    }
    if l.file_experts > 0 {
        let _ = writeln!(s, "  model file: {} experts read on demand @ {} MB/s", l.file_experts, l.model_store_mb_s.map(|m| format!("{m:.0}")).unwrap_or("unknown".into()));
    }
    let _ = writeln!(
        s,
        "  expected {} tok/s  (low {} .. high {})   uses/token {:.0}: VRAM {:.0}%  helper {:.0}%  RAM {:.0}%  tier {:.0}%  file {:.0}%",
        tok(e.tok_s),
        tok(e.tok_s_low),
        tok(e.tok_s_high),
        e.uses_per_token,
        e.share_vram * 100.0,
        e.share_helper * 100.0,
        e.share_ram * 100.0,
        e.share_tier * 100.0,
        e.share_file * 100.0
    );
    let _ = writeln!(s, "  ms/token: sim {:.1}  helper {:.1}  disk reads {:.1} serial / {:.1} overlapped", e.ms_sim, e.ms_helper, e.ms_io_serial, e.ms_io_overlap);
    for c in &l.cards {
        if let Some(n) = &c.kernel_note {
            let _ = writeln!(s, "  note: {n}");
        }
    }
    for u in &l.unknowns_used {
        let _ = writeln!(s, "  UNKNOWN: {u}");
    }
    if explain {
        let _ = writeln!(s, "  why:");
        for x in &l.explain {
            let _ = writeln!(s, "    - {x}");
        }
        let _ = writeln!(s, "  basis (measured vs estimated):");
        for x in &e.basis {
            let _ = writeln!(s, "    - {x}");
        }
        let _ = writeln!(s, "    - routing skew: {}", e.skew);
    }
    s
}

pub fn render(m: &model::ModelDesc, set: &PlanSet, quants: &[QuantRow], remake: &[String], top: usize, explain: bool, fleet: &Fleet) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "model {}: {} layers ({} dense lead), {} experts/layer top-{}, expert {:.1} MiB, total {:.1} GiB",
        m.name,
        m.n_layer,
        m.n_dense_lead,
        m.n_expert,
        m.top_k,
        m.expert_bytes as f64 / MIB,
        m.total_bytes as f64 / GIB
    );
    if !fleet.unknowns.is_empty() {
        let _ = writeln!(s, "unknown fleet fields (left null, not guessed): {}", fleet.unknowns.join(", "));
    }
    if set.plans.is_empty() {
        let _ = writeln!(s, "\nDOES NOT FIT. Options:");
        for o in &set.options {
            let _ = writeln!(s, "  - {o}");
        }
    }
    for (i, p) in set.plans.iter().take(top).enumerate() {
        let _ = writeln!(s, "\n#{} {}", i + 1, plan_summary(p, explain && i == 0));
    }
    if explain {
        let _ = writeln!(s, "\nrejected layouts:");
        for r in &set.rejected {
            let _ = writeln!(s, "  - {r}");
        }
    }
    let _ = writeln!(s, "\nquant fit (deepest memory level the quant needs on the best node):");
    let _ = writeln!(s, "  {:<22} {:>6} {:>9}  {:<30} {}", "quant", "bpw", "GiB", "level", "note");
    for q in quants {
        let _ = writeln!(s, "  {:<22} {:>6.2} {:>9.1}  {:<30} {}{}", q.name, q.bpw, q.total_gib, q.level, if q.exact { "exact" } else { "floor" }, q.kernel_note.as_ref().map(|n| format!("; {n}")).unwrap_or_default());
    }
    for r in remake {
        let _ = writeln!(s, "  {r}");
    }
    s
}
