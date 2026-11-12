//! What cards this box has, from the drivers' own counters (not the graphics API, which
//! under-reports free memory): amdgpu sysfs and `nvidia-smi`. Pure probing, no GPU work.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CardInfo {
    pub vendor: String,
    pub name: String,
    /// Driver-level identifier: `card2` (amdgpu) or the nvidia-smi index.
    pub id: String,
    pub vram_total_mb: f64,
    pub vram_free_mb: f64,
    /// PCIe link as reported now (idle links report a low power state).
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    /// Nominal host<->device bandwidth of that link (MB/s), from gen x width; not measured.
    pub pcie_nominal_mb_s: Option<f64>,
    pub busy_percent: Option<f64>,
}

/// Nominal PCIe throughput per lane in MB/s (after encoding).
fn lane_mb_s(gen: u32) -> f64 {
    match gen {
        1 => 250.0,
        2 => 500.0,
        3 => 985.0,
        4 => 1969.0,
        5 => 3938.0,
        _ => 0.0,
    }
}

fn read_num(p: &Path) -> Option<f64> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}

/// amdgpu (and other DRM) cards from sysfs.
pub fn probe_sysfs(drm_root: &Path) -> Vec<CardInfo> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(drm_root) else { return out };
    let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    for n in names {
        if !n.starts_with("card") || n.contains('-') {
            continue;
        }
        let dev = drm_root.join(&n).join("device");
        let Some(total) = read_num(&dev.join("mem_info_vram_total")) else { continue };
        let used = read_num(&dev.join("mem_info_vram_used")).unwrap_or(0.0);
        let speed = std::fs::read_to_string(dev.join("current_link_speed")).ok().unwrap_or_default();
        let gen = if speed.starts_with("2.5") {
            Some(1)
        } else if speed.starts_with("5.0") {
            Some(2)
        } else if speed.starts_with("8.0") {
            Some(3)
        } else if speed.starts_with("16.0") {
            Some(4)
        } else if speed.starts_with("32.0") {
            Some(5)
        } else {
            None
        };
        let width = read_num(&dev.join("current_link_width")).map(|w| w as u32);
        let vendor = std::fs::read_to_string(dev.join("vendor")).ok().map(|v| v.trim().to_string()).unwrap_or_default();
        out.push(CardInfo {
            vendor: if vendor == "0x1002" { "amd".into() } else { vendor },
            name: n.clone(),
            id: n,
            vram_total_mb: total / 1e6,
            vram_free_mb: (total - used) / 1e6,
            pcie_gen: gen,
            pcie_width: width,
            pcie_nominal_mb_s: match (gen, width) {
                (Some(g), Some(w)) => Some(lane_mb_s(g) * w as f64),
                _ => None,
            },
            busy_percent: read_num(&dev.join("gpu_busy_percent")),
        });
    }
    out
}

/// Parse `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits`.
pub fn parse_nvidia_smi(csv: &str) -> Vec<CardInfo> {
    let mut out = Vec::new();
    for line in csv.lines() {
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if f.len() < 7 {
            continue;
        }
        let num = |i: usize| f[i].parse::<f64>().ok();
        let (used, total) = (num(2).unwrap_or(0.0), num(3).unwrap_or(0.0));
        let gen = num(5).map(|g| g as u32);
        let width = num(6).map(|w| w as u32);
        out.push(CardInfo {
            vendor: "nvidia".into(),
            name: f[1].to_string(),
            id: f[0].to_string(),
            vram_total_mb: total * 1.048576,
            vram_free_mb: (total - used) * 1.048576,
            pcie_gen: gen,
            pcie_width: width,
            pcie_nominal_mb_s: match (gen, width) {
                (Some(g), Some(w)) => Some(lane_mb_s(g) * w as f64),
                _ => None,
            },
            busy_percent: num(4),
        });
    }
    out
}

/// Probe this machine: sysfs first, then nvidia-smi when present.
pub fn probe() -> Vec<CardInfo> {
    let mut cards = probe_sysfs(Path::new("/sys/class/drm"));
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,memory.used,memory.total,utilization.gpu,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader,nounits"])
        .output()
    {
        if o.status.success() {
            let nv = parse_nvidia_smi(&String::from_utf8_lossy(&o.stdout));
            // sysfs lists NVIDIA cards too (no vram counters, so they were skipped above)
            cards.extend(nv);
        }
    }
    cards
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_csv_parses() {
        let csv = "2, NVIDIA GeForce GTX 1070, 2260, 8192, 0, 1, 4\n3, Tesla P100-PCIE-16GB, 10885, 16384, 44, 3, 16\n";
        let c = parse_nvidia_smi(csv);
        assert_eq!(c.len(), 2);
        assert_eq!(c[1].pcie_nominal_mb_s, Some(985.0 * 16.0));
        assert!((c[0].vram_free_mb - (8192.0 - 2260.0) * 1.048576).abs() < 1e-6);
    }

    #[test]
    fn sysfs_probe_reads_amdgpu_layout() {
        let d = std::env::temp_dir().join(format!("lb-calibrate-sysfs-{}", std::process::id()));
        let dev = d.join("card2").join("device");
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::write(dev.join("mem_info_vram_total"), "8589934592").unwrap();
        std::fs::write(dev.join("mem_info_vram_used"), "7939194880").unwrap();
        std::fs::write(dev.join("current_link_speed"), "5.0 GT/s PCIe").unwrap();
        std::fs::write(dev.join("current_link_width"), "4").unwrap();
        std::fs::write(dev.join("vendor"), "0x1002").unwrap();
        let c = probe_sysfs(&d);
        std::fs::remove_dir_all(&d).ok();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].vendor, "amd");
        assert_eq!(c[0].pcie_nominal_mb_s, Some(2000.0));
        assert!((c[0].vram_free_mb - 650.739712).abs() < 1e-3);
    }
}
