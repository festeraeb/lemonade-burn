//! Build a fleet description from fleet-manifest.json and the nauti fleet matrix/views.
//! Unknown stays `None` and goes into `Fleet::unknowns`; nothing is guessed. Disk tiers are
//! not in either source, so they come from an overlay file (`apply_overlay`).

use super::*;
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

/// Measured 2026-11-07 on the t440<->c4 25 GbE RoCE link: RDMA read 2.84 GB/s (task brief,
/// memory le-chonk-polystrata); small-message RDMA latency 8 us at 4 KB (same note).
pub const ROCE_MEASURED_GB_S: f64 = 2.84;
pub const ROCE_MEASURED_LAT_US: f64 = 8.0;

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(|x| x.to_string())
}

fn canon(manifest_nodes: &[String], matrix_name: &str) -> String {
    manifest_nodes.iter().find(|m| matrix_name.starts_with(m.as_str()) || m.starts_with(matrix_name)).cloned().unwrap_or_else(|| matrix_name.to_string())
}

fn norm_bdf(b: &str) -> String {
    // "00000000:5E:00.0" and "0000:5e:00.0" -> "5e:00.0" (drop the PCI domain)
    let parts: Vec<&str> = b.rsplitn(3, ':').collect();
    format!("{}:{}", parts.get(1).unwrap_or(&""), parts[0]).to_lowercase()
}

pub fn from_files(manifest: &Path, matrix: Option<&Path>) -> Result<Fleet> {
    let m: Value = serde_json::from_str(&std::fs::read_to_string(manifest).with_context(|| manifest.display().to_string())?)?;
    let mx: Option<Value> = match matrix {
        Some(p) if p.exists() => Some(serde_json::from_str(&std::fs::read_to_string(p)?)?),
        _ => None,
    };
    let mut fleet = Fleet::default();
    let nodes_obj = m.get("nodes").and_then(|n| n.as_object()).context("manifest has no nodes")?;
    let names: Vec<String> = nodes_obj.keys().filter(|k| !k.starts_with('_')).cloned().collect();
    for name in &names {
        let mv = &nodes_obj[name];
        let mut node = NodeDesc { name: name.clone(), ..Default::default() };
        // manifest GPUs (compute_cap by bdf)
        let mut man_gpus: Vec<GpuDesc> = Vec::new();
        let mut bdf_cc: Vec<(String, Option<String>)> = Vec::new();
        if let Some(arr) = mv.get("gpus").and_then(|g| g.as_array()) {
            for g in arr {
                let cc = s(g, "compute_cap");
                if let Some(b) = s(g, "bdf") {
                    bdf_cc.push((norm_bdf(&b), cc.clone()));
                }
                man_gpus.push(GpuDesc { name: s(g, "name").unwrap_or_default(), vram_mib: g.get("vram_mb").and_then(|v| v.as_u64()).unwrap_or(0), compute_cap: cc, ..Default::default() });
            }
        }
        if let Some(c) = mv.get("cpu_cap").and_then(|c| c.get("nproc")).and_then(|n| n.as_u64()) {
            node.cpu_threads = Some(c as usize);
        }
        // live matrix overrides the manifest list (what the box reports now)
        let live = mx.as_ref().and_then(|x| x.get("nodes")).and_then(|n| n.as_array()).and_then(|a| a.iter().find(|n| canon(&names, n.get("node").and_then(|x| x.as_str()).unwrap_or("")) == *name));
        if let Some(l) = live {
            let r = &l["resources"];
            node.ram_gib = r.get("memory_bytes").and_then(|v| v.as_f64()).map(|b| (b / GIB * 10.0).round() / 10.0);
            node.cpu_threads = r.get("logical_cpus").and_then(|v| v.as_u64()).map(|v| v as usize).or(node.cpu_threads);
            for g in r.get("gpus").and_then(|g| g.as_array()).into_iter().flatten() {
                let vram = g.get("vram_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                if vram == 0 {
                    continue; // iGPU sharing system RAM
                }
                let bdf = s(g, "pci_bdf").map(|b| norm_bdf(&b)).unwrap_or_default();
                let cc = bdf_cc.iter().find(|(b, _)| *b == bdf).and_then(|(_, c)| c.clone());
                node.gpus.push(GpuDesc { name: s(g, "name").unwrap_or_default(), vram_mib: vram / (1024 * 1024), compute_cap: cc, ..Default::default() });
            }
        } else {
            node.gpus = man_gpus;
            fleet.notes.push(format!("{name}: not in the fleet matrix; GPUs from fleet-manifest.json, RAM unknown"));
        }
        fleet.nodes.push(node);
    }
    if let Some(x) = &mx {
        links_from_matrix(&mut fleet, x, &names);
        if let Some(t) = x.get("generated_at").and_then(|t| t.as_f64()) {
            fleet.notes.push(format!("fleet matrix generated_at={t:.0} (unix s)"));
        }
    }
    fleet.notes.push("disk tiers (optane/ssd/hdd/nfs) are in neither source: supply an overlay".into());
    fleet.collect_unknowns();
    Ok(fleet)
}

fn links_from_matrix(fleet: &mut Fleet, x: &Value, names: &[String]) {
    let Some(icmp) = x.get("icmp_avg_ms_by_path").and_then(|v| v.as_object()) else { return };
    let nodes = x.get("nodes").and_then(|n| n.as_array()).cloned().unwrap_or_default();
    let node_val = |n: &str| nodes.iter().find(|v| v.get("node").and_then(|x| x.as_str()) == Some(n)).cloned();
    let mut keys: Vec<&String> = icmp.keys().collect();
    keys.sort();
    for (i, a) in keys.iter().enumerate() {
        for b in keys.iter().skip(i + 1) {
            let (ca, cb) = (canon(names, a), canon(names, b));
            let (Some(av), Some(bv)) = (node_val(a), node_val(b)) else { continue };
            let paths = &icmp[*a][*b];
            let pick = ["25g", "lan-10.44"].iter().find_map(|p| paths.get(*p).and_then(|v| v.as_f64()).map(|ms| (*p, ms)));
            let Some((path, ms)) = pick else { continue };
            // interface speed: the interface on the shared /24, read from either end (the slower one wins)
            let ip_of = |v: &Value| v["paths"].as_array().and_then(|ps| ps.iter().find(|p| p["path"] == *path)).and_then(|p| p["ip"].as_str()).map(|s| s.to_string());
            let speed_at = |v: &Value, peer_ip: Option<String>| -> Option<f64> {
                let pf = peer_ip.as_deref().and_then(|ip| ip.rsplit_once('.').map(|(p, _)| p.to_string()))?;
                v["resources"]["links_up"].as_array().into_iter().flatten().find(|l| l["ipv4"].as_str().map_or(false, |ip| ip.starts_with(&pf))).and_then(|l| l["speed_mbps"].as_str().and_then(|s| s.parse::<f64>().ok()))
            };
            let speeds: Vec<f64> = [speed_at(&av, ip_of(&bv)), speed_at(&bv, ip_of(&av))].into_iter().flatten().collect();
            let speed = speeds.into_iter().fold(None, |m: Option<f64>, x| Some(m.map_or(x, |y| y.min(x))));
            let roce = path == "25g";
            let (gb_s, src_rate) = if roce {
                (Some(ROCE_MEASURED_GB_S), "RDMA read measured 2.84 GB/s on t440<->c4, applied to this 25G RoCE link".to_string())
            } else {
                (speed.map(|m| m / 8000.0), speed.map(|m| format!("nominal from interface speed {m:.0} Mbps")).unwrap_or_else(|| "interface speed not reported by the matrix".to_string()))
            };
            let lat = if roce { ROCE_MEASURED_LAT_US } else { ms * 1000.0 };
            fleet.links.push(Link {
                a: ca,
                b: cb,
                kind: if roce { LinkKind::Roce } else { LinkKind::Ethernet },
                gb_s,
                latency_us: Some(lat),
                source: format!("{src_rate}; latency {}", if roce { "RDMA 4 KB measured 8 us".to_string() } else { format!("ICMP avg {ms} ms") }),
            });
        }
    }
}

/// Merge an overlay (hand-written partial fleet) into `base`. Nodes match by name; a
/// non-empty `tiers` / `gpus` list replaces the base's; `ram_gib` / `cpu_threads` override
/// when set; links replace by pair.
pub fn apply_overlay(base: &mut Fleet, ov: Fleet) {
    for on in ov.nodes {
        match base.nodes.iter_mut().find(|n| n.name == on.name) {
            Some(n) => {
                if !on.tiers.is_empty() {
                    n.tiers = on.tiers;
                }
                if !on.gpus.is_empty() {
                    n.gpus = on.gpus;
                }
                if on.ram_gib.is_some() {
                    n.ram_gib = on.ram_gib;
                }
                if on.cpu_threads.is_some() {
                    n.cpu_threads = on.cpu_threads;
                }
            }
            None => base.nodes.push(on),
        }
    }
    for l in ov.links {
        base.links.retain(|x| !((x.a == l.a && x.b == l.b) || (x.a == l.b && x.b == l.a)));
        base.links.push(l);
    }
    base.notes.extend(ov.notes);
    base.unknowns.clear();
    base.collect_unknowns();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_and_matrix_merge_with_unknowns() {
        let d = std::env::temp_dir().join(format!("lbtune-load-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let man = serde_json::json!({"nodes": {"t440": {"gpus": [{"name": "Tesla P100-PCIE-16GB", "vram_mb": 16384, "bdf": "00000000:5E:00.0", "compute_cap": "6.0"}], "cpu_cap": {"nproc": 48}},
            "cesarops1": {"gpus": [{"name": "NVIDIA GeForce GTX 1070", "vram_mb": 8192, "bdf": "00000000:86:00.0", "compute_cap": "6.1"}]}}});
        let mat = serde_json::json!({"icmp_avg_ms_by_path": {}, "nodes": [{"node": "t440cesarops", "resources": {"memory_bytes": 75497472000u64, "logical_cpus": 48,
            "gpus": [{"name": "Tesla P100-PCIE-16GB", "pci_bdf": "0000:5e:00.0", "vram_bytes": 17179869184u64}, {"name": "Intel GPU", "pci_bdf": "0000:00:02.0", "vram_bytes": 0}], "links_up": []}, "paths": []}]});
        std::fs::write(d.join("m.json"), man.to_string()).unwrap();
        std::fs::write(d.join("x.json"), mat.to_string()).unwrap();
        let f = from_files(&d.join("m.json"), Some(&d.join("x.json"))).unwrap();
        let t = f.node("t440").unwrap();
        assert_eq!(t.gpus.len(), 1, "iGPU with 0 VRAM skipped");
        assert_eq!(t.gpus[0].compute_cap.as_deref(), Some("6.0"), "cc joined by bdf");
        assert!((t.ram_gib.unwrap() - 70.3).abs() < 0.1);
        let c1 = f.node("cesarops1").unwrap();
        assert!(c1.ram_gib.is_none());
        assert!(f.unknowns.iter().any(|u| u == "cesarops1.ram_gib"));
        std::fs::remove_dir_all(&d).ok();
    }
}
