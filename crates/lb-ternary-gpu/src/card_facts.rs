//! What the kernel knows about an AMD card, read from sysfs by PCI bus id: chip family, VRAM, how much of it the CPU can map through BAR 0
//! (Resizable BAR), the power cap and the PCIe link. Read-only; nothing here touches the card.
//!
//! Why the engine cares (cesarops1, 2026-11-16..18):
//! - Vega 10 (MI25 / WX 9100, gfx900): chunks above 128 rows hung the compute ring under llama.cpp Vulkan ("ring comp timeout", device
//!   lost, card wedged until a reboot); `-b 128 -ub 128` is the stable setting. [`FamilyLimits::max_batch_rows`] carries that to our prefill.
//! - The power cap moves decode speed: MI25 LFM2-2.6B Q4_K_M tg128 107 / 124 / 129 tok/s at 110 / 140 / 170 W (antigravity-t440's
//!   measurement); a seat that reads 110 W should not be scheduled as if it had 170.
//! - ReBAR (BAR 0 covering VRAM) is what would let a host write land in device memory directly; on c1 only the RX 570 had it, and that card
//!   then failed amdgpu init, so the engine reports it and does not depend on it.

use std::path::{Path, PathBuf};

/// Limits the engine applies per chip family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilyLimits {
    /// Largest number of rows (tokens) one prefill chunk may carry on this chip, when known to matter.
    pub max_batch_rows: Option<usize>,
}

/// Chip family from the PCI vendor/device id (AMD only; `None` for anything not listed).
pub fn amd_family(vendor: u16, device: u16) -> Option<&'static str> {
    if vendor != 0x1002 {
        return None;
    }
    Some(match device {
        0x6860..=0x687f => "gfx900 (Vega 10)",
        0x66a0..=0x66af => "gfx906 (Vega 20)",
        0x67c0..=0x67df | 0x6fdf => "gfx803 (Polaris 10/20)",
        0x67e0..=0x67ff => "gfx803 (Polaris 11)",
        0x6980..=0x699f => "gfx803 (Polaris 12)",
        0x73a0..=0x73bf => "gfx1030 (Navi 21)",
        0x73c0..=0x73df => "gfx1031 (Navi 22)",
        0x73e0..=0x73ff => "gfx1032 (Navi 23)",
        0x7440..=0x745f => "gfx1100 (Navi 31)",
        0x7470..=0x748f => "gfx1101 (Navi 32)",
        _ => return None,
    })
}

pub fn family_limits(vendor: u16, device: u16) -> FamilyLimits {
    let vega10 = vendor == 0x1002 && (0x6860..=0x687f).contains(&device);
    FamilyLimits { max_batch_rows: vega10.then_some(128) }
}

/// The adapter-name fallback when the PCI id is not readable: RADV names Vega 10 "... (RADV VEGA10)".
pub fn family_limits_by_name(adapter_name: &str) -> FamilyLimits {
    FamilyLimits { max_batch_rows: adapter_name.to_ascii_uppercase().contains("VEGA10").then_some(128) }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CardFacts {
    pub bus: String,
    pub vendor: u16,
    pub device: u16,
    pub family: Option<&'static str>,
    /// size of BAR 0 (the VRAM aperture) in bytes
    pub bar0_bytes: Option<u64>,
    pub vram_bytes: Option<u64>,
    pub power_cap_w: Option<f64>,
    pub power_cap_max_w: Option<f64>,
    pub link: Option<String>,
    pub driver: Option<String>,
}

impl CardFacts {
    /// BAR 0 covers all of VRAM (Resizable BAR in effect).
    pub fn rebar(&self) -> Option<bool> {
        Some(self.bar0_bytes? >= self.vram_bytes?)
    }

    pub fn limits(&self) -> FamilyLimits {
        family_limits(self.vendor, self.device)
    }

    /// One line for the startup log.
    pub fn summary(&self) -> String {
        let mib = |b: u64| if b >= 1 << 30 { format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64) } else { format!("{} MiB", b >> 20) };
        let mut s = format!("[{}] {}", self.bus, self.family.unwrap_or("unknown family"));
        match (self.bar0_bytes, self.vram_bytes) {
            (Some(b), Some(v)) => s += &format!(", BAR0 {} of {} VRAM ({})", mib(b), mib(v), if b >= v { "ReBAR" } else { "no ReBAR" }),
            (Some(b), None) => s += &format!(", BAR0 {}", mib(b)),
            _ => {}
        }
        if let Some(w) = self.power_cap_w {
            s += &format!(", power cap {w:.0} W");
            if let Some(m) = self.power_cap_max_w {
                s += &format!(" of {m:.0}");
            }
        }
        if let Some(l) = &self.link {
            s += &format!(", link {l}");
        }
        if self.driver.is_none() {
            s += ", NO DRIVER BOUND";
        }
        if let Some(r) = self.limits().max_batch_rows {
            s += &format!(", prefill chunks capped at {r} rows");
        }
        s
    }
}

fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn read_hex_u16(p: &Path) -> Option<u16> {
    u16::from_str_radix(read_trim(p)?.trim_start_matches("0x"), 16).ok()
}

/// BAR 0 size from the `resource` file (first line: `start end flags` in hex).
fn bar0_from_resource(text: &str) -> Option<u64> {
    let mut it = text.lines().next()?.split_whitespace().map(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok());
    let (start, end) = (it.next()??, it.next()??);
    (end > start).then(|| end - start + 1)
}

/// Facts for the card at `bus` (`0000:23:00.0`) under `sys_root` (normally `/sys`; tests pass a fake tree).
pub fn read_card_facts_at(sys_root: &Path, bus: &str) -> Option<CardFacts> {
    let dev: PathBuf = sys_root.join("bus/pci/devices").join(bus);
    if !dev.is_dir() {
        return None;
    }
    let vendor = read_hex_u16(&dev.join("vendor")).unwrap_or(0);
    let device = read_hex_u16(&dev.join("device")).unwrap_or(0);
    let hwmon = std::fs::read_dir(dev.join("hwmon")).ok().and_then(|mut d| d.next()).and_then(|e| e.ok()).map(|e| e.path());
    let watts = |f: &str| hwmon.as_ref().and_then(|h| read_trim(&h.join(f))).and_then(|v| v.parse::<f64>().ok()).map(|uw| uw / 1e6);
    let link = match (read_trim(&dev.join("current_link_speed")), read_trim(&dev.join("current_link_width"))) {
        (Some(s), Some(w)) => Some(format!("{s} x{w}")),
        _ => None,
    };
    Some(CardFacts {
        bus: bus.to_string(),
        vendor,
        device,
        family: amd_family(vendor, device),
        bar0_bytes: read_trim(&dev.join("resource")).as_deref().and_then(bar0_from_resource),
        vram_bytes: read_trim(&dev.join("mem_info_vram_total")).and_then(|v| v.parse().ok()),
        power_cap_w: watts("power1_cap"),
        power_cap_max_w: watts("power1_cap_max"),
        link,
        driver: std::fs::read_link(dev.join("driver")).ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())),
    })
}

pub fn read_card_facts(bus: &str) -> Option<CardFacts> {
    read_card_facts_at(Path::new("/sys"), bus)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_card(root: &Path, bus: &str, device: &str, bar0: (u64, u64), vram: u64, cap_uw: u64) {
        let d = root.join("bus/pci/devices").join(bus);
        std::fs::create_dir_all(d.join("hwmon/hwmon3")).unwrap();
        std::fs::create_dir_all(root.join("bus/pci/drivers/amdgpu")).unwrap();
        std::fs::write(d.join("vendor"), "0x1002\n").unwrap();
        std::fs::write(d.join("device"), format!("{device}\n")).unwrap();
        std::fs::write(d.join("resource"), format!("0x{:016x} 0x{:016x} 0x000000000014220c\n0x0000000000000000 0x0000000000000000 0x0000000000000000\n", bar0.0, bar0.1)).unwrap();
        std::fs::write(d.join("mem_info_vram_total"), format!("{vram}\n")).unwrap();
        std::fs::write(d.join("hwmon/hwmon3/power1_cap"), format!("{cap_uw}\n")).unwrap();
        std::fs::write(d.join("hwmon/hwmon3/power1_cap_max"), "170000000\n").unwrap();
        std::fs::write(d.join("current_link_speed"), "8.0 GT/s PCIe\n").unwrap();
        std::fs::write(d.join("current_link_width"), "16\n").unwrap();
        std::os::unix::fs::symlink(root.join("bus/pci/drivers/amdgpu"), d.join("driver")).unwrap();
    }

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lb-card-facts-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn mi25_without_rebar_gets_the_128_row_cap() {
        let root = tmp("mi25");
        // the MI25 on c1 as of 2026-11-18: BAR 0 256 MiB at 0x3ffd0000000, 16 GiB HBM2, 170 W cap
        fake_card(&root, "0000:23:00.0", "0x6860", (0x3ffd0000000, 0x3ffdfffffff), 16 << 30, 170_000_000);
        let f = read_card_facts_at(&root, "0000:23:00.0").unwrap();
        assert_eq!(f.family, Some("gfx900 (Vega 10)"));
        assert_eq!(f.bar0_bytes, Some(256 << 20));
        assert_eq!(f.rebar(), Some(false));
        assert_eq!(f.power_cap_w, Some(170.0));
        assert_eq!(f.limits().max_batch_rows, Some(128));
        assert_eq!(f.driver.as_deref(), Some("amdgpu"));
        let s = f.summary();
        assert!(s.contains("BAR0 256 MiB of 16.0 GiB VRAM (no ReBAR)") && s.contains("capped at 128 rows") && s.contains("8.0 GT/s PCIe x16"), "{s}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn rx570_with_a_4g_bar_reads_as_rebar_and_has_no_cap() {
        let root = tmp("rx570");
        fake_card(&root, "0000:27:00.0", "0x67df", (0x3e000000000, 0x3e0ffffffff), 4 << 30, 120_000_000);
        let f = read_card_facts_at(&root, "0000:27:00.0").unwrap();
        assert_eq!(f.family, Some("gfx803 (Polaris 10/20)"));
        assert_eq!(f.rebar(), Some(true));
        assert_eq!(f.limits().max_batch_rows, None);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn families_and_name_fallback() {
        assert_eq!(amd_family(0x1002, 0x73df), Some("gfx1031 (Navi 22)"));
        assert_eq!(amd_family(0x10de, 0x15f8), None);
        assert_eq!(family_limits(0x1002, 0x73df).max_batch_rows, None);
        assert_eq!(family_limits_by_name("AMD Radeon Pro WX 9100 (RADV VEGA10)").max_batch_rows, Some(128));
        assert_eq!(family_limits_by_name("AMD Radeon RX 6700 XT (RADV NAVI22)").max_batch_rows, None);
        assert!(read_card_facts_at(Path::new("/nonexistent"), "0000:00:00.0").is_none());
    }
}
