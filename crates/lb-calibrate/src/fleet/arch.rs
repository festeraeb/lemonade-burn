//! Architecture table. Every number here is an ESTIMATE from vendor specifications or from
//! the facts in the fleet notes (fleet-gpu-inventory.md); none is a measurement of this
//! engine. A card with a `--bench` JSON gets measured timings instead (see plan.rs).

use super::GpuDesc;

#[derive(Clone, Debug, PartialEq)]
pub struct ArchInfo {
    pub arch: &'static str,
    pub compute_cap: Option<&'static str>,
    pub has_dp4a: bool,
    pub fp16_rate: &'static str,
    /// Memory bandwidth, GB/s. Source in `src`.
    pub mem_gb_s: f64,
    pub src: &'static str,
}

/// (name fragment lower-case, info). First match wins, so specific names come first.
pub fn table() -> Vec<(&'static str, ArchInfo)> {
    let a = |arch, cc, dp4a, fp16, bw, src| ArchInfo { arch, compute_cap: cc, has_dp4a: dp4a, fp16_rate: fp16, mem_gb_s: bw, src };
    vec![
        // DP4A exists on cc 6.1/6.2, not 6.0; P100 fp16 full rate, 1070 1/64 (fleet-gpu-inventory.md, verified 2026-11-02).
        ("tesla p100", a("sm_60", Some("6.0"), false, "full", 732.0, "estimate: NVIDIA P100-PCIE-16GB spec 732 GB/s")),
        ("gtx 1070", a("sm_61", Some("6.1"), true, "1/64", 256.0, "estimate: GTX 1070 spec 256 GB/s")),
        ("p106", a("sm_61", Some("6.1"), true, "1/64", 192.0, "estimate: P106-100 spec 192 GB/s")),
        ("p1000", a("sm_61", Some("6.1"), true, "1/64", 80.0, "estimate: Quadro P1000 spec 80 GB/s")),
        ("2060 super", a("sm_75", Some("7.5"), true, "full", 448.0, "estimate: RTX 2060 SUPER spec 448 GB/s")),
        ("rtx 2060", a("sm_75", Some("7.5"), true, "full", 336.0, "estimate: RTX 2060 spec 336 GB/s")),
        ("t10", a("sm_75", Some("7.5"), true, "full", 672.0, "estimate: TU102GL class, 384-bit GDDR6 (672 GB/s); the c4 T10 throttles thermally, so an upper bound")),
        // RDNA2 has v_dot4 (gfx1030+), no WMMA; Polaris has neither packed fp16 nor dot4 (estimate: vendor ISA docs).
        ("6700 xt", a("gfx1031", None, true, "full", 384.0, "estimate: RX 6700 XT spec 384 GB/s")),
        ("0x73df", a("gfx1031", None, true, "full", 384.0, "estimate: PCI id 0x73df = Navi22 = RX 6700 XT; spec 384 GB/s")),
        ("wx 5100", a("gfx803", None, false, "same", 160.0, "estimate: Radeon Pro WX 5100 spec 160 GB/s")),
        ("0x67c7", a("gfx803", None, false, "same", 160.0, "estimate: PCI id 0x67c7 = Ellesmere = WX 5100; spec 160 GB/s")),
        ("rx 570", a("gfx803", None, false, "same", 224.0, "estimate: RX 570 spec 224 GB/s")),
    ]
}

pub fn lookup(name: &str) -> Option<ArchInfo> {
    let n = name.to_lowercase();
    table().into_iter().find(|(frag, _)| n.contains(frag)).map(|(_, i)| i)
}

/// A GPU with the unknowns filled from the table (fields given explicitly win).
pub fn resolve(g: &GpuDesc) -> (GpuDesc, Option<ArchInfo>) {
    let info = lookup(&g.name);
    let mut r = g.clone();
    if let Some(i) = &info {
        r.arch.get_or_insert_with(|| i.arch.to_string());
        if r.compute_cap.is_none() {
            r.compute_cap = i.compute_cap.map(|s| s.to_string());
        }
        r.has_dp4a.get_or_insert(i.has_dp4a);
        r.fp16_rate.get_or_insert_with(|| i.fp16_rate.to_string());
        r.mem_gb_s.get_or_insert(i.mem_gb_s);
    }
    // compute capability alone decides dp4a for NVIDIA when the name is unknown
    if r.has_dp4a.is_none() {
        if let Some(cc) = &r.compute_cap {
            let v: f64 = cc.parse().unwrap_or(0.0);
            r.has_dp4a = Some(v >= 6.1);
        }
    }
    (r, info)
}

/// Host<->device MB/s from the link. Anchors are measured: Gen3 x16 ~12 GB/s, x8 ~6 GB/s
/// (task brief; V2_PIPELINE.md 12.26 GB/s on a real x16); other widths/gens scale linearly by
/// lane count and per-lane rate. `None` link = unknown, the caller must say it assumed.
pub fn pcie_mb_s(gen: Option<u32>, width: Option<u32>) -> Option<f64> {
    let (g, w) = (gen?, width?);
    let lane = match g {
        1 => 250.0,
        2 => 500.0,
        3 => 985.0,
        4 => 1969.0,
        _ => return None,
    };
    // measured/nominal at Gen3: 12000 / (16*985) = 0.76
    Some(lane * w as f64 * 0.76)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p100_has_no_dp4a_1070_has() {
        assert_eq!(lookup("Tesla P100-PCIE-16GB").unwrap().has_dp4a, false);
        assert_eq!(lookup("NVIDIA GeForce GTX 1070").unwrap().has_dp4a, true);
        assert_eq!(lookup("NVIDIA GeForce RTX 2060 SUPER").unwrap().mem_gb_s, 448.0);
        assert_eq!(lookup("AMD 0x73df").unwrap().arch, "gfx1031");
        assert_eq!(lookup("Quadro P1000").unwrap().arch, "sm_61", "P1000 is not a P100");
        assert!(lookup("P106-100").unwrap().has_dp4a == true);
        assert!(lookup("mystery card").is_none());
    }

    #[test]
    fn pcie_anchors() {
        assert!((pcie_mb_s(Some(3), Some(16)).unwrap() - 11981.0).abs() < 50.0);
        assert!((pcie_mb_s(Some(3), Some(8)).unwrap() - 5990.0).abs() < 30.0);
        assert!(pcie_mb_s(None, Some(8)).is_none());
    }
}
