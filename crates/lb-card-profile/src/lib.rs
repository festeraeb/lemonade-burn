//! Card profiles: the per-card tuning of the engine as data.
//!
//! One JSON file per card family (`profiles/*.json`, compiled in so a single binary runs anywhere; `LB_CARD_PROFILE_DIR` adds or replaces
//! profiles by id without a rebuild). A profile says which cards it is for (what the engine can probe: CUDA compute capability, PCI
//! vendor/device range, adapter name) and holds knob values, each with the note or commit that measured it.
//!
//! Order of precedence for every knob, per card: **environment variable > profile > code default**. A knob is named by its env var, so
//! `LB_QW_PREFILL_CHUNK=0` in the environment still wins over a profile that says 256. A card that matches no profile gets the code
//! defaults (the registry [`KNOBS`] lists them) and one log line. Nothing here touches the card.
//!
//! Only knobs marked `profile: true` in [`KNOBS`] are read through a profile; a profile naming any other knob gets a warning and the
//! value is ignored (the knob still works through its env var).

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

/// A knob the engine knows: its env var name, the code default (text, as the code reads it), whether a profile may set it.
pub struct KnobDef {
    pub name: &'static str,
    pub default: &'static str,
    /// read through the card profile at the call site (the rest are env-only today)
    pub profile: bool,
    pub about: &'static str,
}

/// Every knob a profile may carry, plus the card-relevant env-only ones for the inventory (`profile: false`).
pub const KNOBS: &[KnobDef] = &[
    KnobDef { name: "LB_PREFILL_MAX_ROWS", default: "none", profile: true, about: "largest prefill chunk (rows) the card runs safely; caps LB_PREFILL_UBATCH" },
    KnobDef { name: "LB_PREFILL_UBATCH", default: "2048", profile: true, about: "rows per device-prefill chunk (lb-lfm2)" },
    KnobDef { name: "LB_Q4_ROWS", default: "4", profile: true, about: "rows per plane in the Q4_0 rows kernels (1, 4 or 8)" },
    KnobDef { name: "LB_Q4_MATH", default: "f32", profile: true, about: "Q4_0 rows-kernel arithmetic: f32 | f32m | f16 | dp4a | auto" },
    KnobDef { name: "LB_Q4_CUBE", default: "2 planes", profile: true, about: "units per cube for the matvec kernels" },
    KnobDef { name: "LB_H2", default: "off", profile: true, about: "fp16 (half2) prefill GEMM on sm_60 (P100); fails the accuracy gate" },
    KnobDef { name: "LB_NO_TC", default: "off", profile: true, about: "1 forces the f32 prefill GEMM on a tensor-core card" },
    KnobDef { name: "LB_QW_PREFILL_CHUNK", default: "0 (auto 64..1024)", profile: true, about: "rows per qwen35 prefill chunk; on a pipeline the smallest value any stage card's profile gives wins" },
    KnobDef { name: "LB_QW_GRAPH", default: "1", profile: true, about: "CUDA graphs for the qwen35 decode stages (0 off)" },
    KnobDef { name: "LB_QW_FLUSH", default: "1", profile: true, about: "qwen35 stage flush interval" },
    KnobDef { name: "LB_MTP_DEPTH", default: "1", profile: true, about: "tokens the qwen35 nextn head drafts per verify (1..7); needs LB_MTP=1" },
    KnobDef { name: "LB_QW_Q6_LG", default: "0", profile: true, about: "Q6_K rows under 32 half blocks (2048 columns: head, attn_q) run lane groups, no idle lanes; bit-identical (1 on)" },
    KnobDef { name: "LB_QW_ATTN_SPLIT", default: "0", profile: true, about: "qwen35 one-token attention as three launches over many cubes instead of one cube a head; bit-identical (1 on)" },
    KnobDef { name: "LB_QW_ATTN_CUBES", default: "112", profile: true, about: "cubes of the split attention scores kernel (with LB_QW_ATTN_SPLIT=1)" },
    KnobDef { name: "LB_QW_GU_R", default: "2", profile: true, about: "rows a plane takes in the qwen35 routed gate+up kernel (1, 2, 4)" },
    KnobDef { name: "LB_QW_DN_R", default: "1", profile: true, about: "rows a lane group takes in the qwen35 routed down kernel (1, 2, 4, 8); bit-identical for any value" },
    // env-only: documented so the inventory is complete, not read through a profile yet
    KnobDef { name: "QWEN_SPLIT", default: "even over CUDA_VISIBLE_DEVICES", profile: false, about: "qwen35 layers per stage; depends on model + card set, so it is seat placement, not a card property" },
    KnobDef { name: "LB_MTP", default: "off", profile: false, about: "turn the qwen35 MTP draft on (an operator decision per seat)" },
    KnobDef { name: "LB_NO_PINNED", default: "off", profile: false, about: "disable the page-locked staging pool on CUDA" },
    KnobDef { name: "LB_NO_REGISTER", default: "off", profile: false, about: "do not cuMemHostRegister the GGUF mmap" },
    KnobDef { name: "LB_ROUTER_SPLIT_ANY", default: "on", profile: true, about: "E2 router split on plane-range (RDNA) cards; 0 off" },
    KnobDef { name: "LB_NO_PLANE", default: "off", profile: false, about: "do not use subgroup reductions" },
    KnobDef { name: "LB_Q4_XSHARED", default: "0", profile: false, about: "stage x in shared memory in the rows kernels" },
    KnobDef { name: "LB_TC_STAGE2", default: "off", profile: false, about: "two Q4_0 blocks per tensor-core stage (slower on the T10)" },
    KnobDef { name: "LB_GEMM_V4", default: "off", profile: false, about: "4-wide staged f32 GEMM (measured, not adopted)" },
];

pub fn knob_def(name: &str) -> Option<&'static KnobDef> {
    KNOBS.iter().find(|k| k.name == name)
}

/// What the engine probed about one card. Only the fields that are known are set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CardKey {
    /// where it is: `cuda:0`, `0000:23:00.0`, for log lines
    pub label: String,
    /// driver / adapter name
    pub name: String,
    pub cuda_cc: Option<(u32, u32)>,
    /// PCI (vendor, device)
    pub pci: Option<(u16, u16)>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct PciMatch {
    pub vendor: String,
    /// inclusive `[lo, hi]` device id range, hex strings; empty = any device of the vendor
    pub device: Vec<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Match {
    /// `"6.0"` style compute capabilities (any of)
    pub cuda_cc: Vec<String>,
    pub pci: Vec<PciMatch>,
    /// case-insensitive substrings of the adapter name (any of)
    pub name_contains: Vec<String>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum OneOrMany {
    One(Match),
    Many(Vec<Match>),
}

impl Default for OneOrMany {
    fn default() -> Self {
        OneOrMany::Many(vec![])
    }
}

impl OneOrMany {
    fn all(&self) -> &[Match] {
        match self {
            OneOrMany::One(m) => std::slice::from_ref(m),
            OneOrMany::Many(v) => v,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct KnobValue {
    pub value: serde_json::Value,
    /// the note, commit or doc that measured it
    pub source: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Profile {
    pub schema: String,
    pub id: String,
    pub family: String,
    /// one criteria object, or an array of them (any of; an array lets a card match by PCI id or, when that is unreadable, by adapter name)
    #[serde(rename = "match", default)]
    pub matcher: OneOrMany,
    #[serde(default)]
    pub knobs: BTreeMap<String, KnobValue>,
    /// measured findings that are not a profile value (a switch that stays off and why, a probed behaviour)
    #[serde(default)]
    pub notes: Vec<String>,
    /// what was never measured on this family (stays at the default)
    #[serde(default)]
    pub not_measured: Vec<String>,
}

fn parse_hex(s: &str) -> Option<u16> {
    u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

fn parse_cc(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('.')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

impl Profile {
    /// How many criteria groups the profile states, when the card satisfies all of them; `None` when it does not match.
    /// A profile with no criteria matches nothing (it would be a default in disguise).
    pub fn score(&self, key: &CardKey) -> Option<usize> {
        self.matcher.all().iter().filter_map(|m| Self::score_one(m, key)).max()
    }

    fn score_one(m: &Match, key: &CardKey) -> Option<usize> {
        let mut groups = 0;
        if !m.cuda_cc.is_empty() {
            groups += 1;
            let cc = key.cuda_cc?;
            if !m.cuda_cc.iter().filter_map(|s| parse_cc(s)).any(|c| c == cc) {
                return None;
            }
        }
        if !m.pci.is_empty() {
            groups += 1;
            let (v, d) = key.pci?;
            let ok = m.pci.iter().any(|p| {
                parse_hex(&p.vendor) == Some(v)
                    && match p.device.as_slice() {
                        [] => true,
                        [one] => parse_hex(one) == Some(d),
                        [lo, hi, ..] => matches!((parse_hex(lo), parse_hex(hi)), (Some(lo), Some(hi)) if (lo..=hi).contains(&d)),
                    }
            });
            if !ok {
                return None;
            }
        }
        if !m.name_contains.is_empty() {
            groups += 1;
            let n = key.name.to_ascii_uppercase();
            if !m.name_contains.iter().any(|s| !s.is_empty() && n.contains(&s.to_ascii_uppercase())) {
                return None;
            }
        }
        (groups > 0).then_some(groups)
    }
}

/// The profiles in force: the compiled-in ones, with any from `LB_CARD_PROFILE_DIR` added (same id replaces).
#[derive(Debug, Default)]
pub struct ProfileSet {
    pub profiles: Vec<Profile>,
    pub warnings: Vec<String>,
}

const BUILTIN: &[(&str, &str)] = &[
    ("nvidia-p100.json", include_str!("../profiles/nvidia-p100.json")),
    ("nvidia-turing-sm75.json", include_str!("../profiles/nvidia-turing-sm75.json")),
    ("nvidia-pascal-sm61.json", include_str!("../profiles/nvidia-pascal-sm61.json")),
    ("amd-vega10.json", include_str!("../profiles/amd-vega10.json")),
    ("amd-rdna2.json", include_str!("../profiles/amd-rdna2.json")),
    ("amd-polaris.json", include_str!("../profiles/amd-polaris.json")),
];

impl ProfileSet {
    fn add_text(&mut self, origin: &str, text: &str) {
        match serde_json::from_str::<Profile>(text) {
            Ok(p) => {
                if p.schema != "lb-card-profile-v1" {
                    self.warnings.push(format!("{origin}: schema {:?} is not lb-card-profile-v1, skipped", p.schema));
                    return;
                }
                for (k, _) in &p.knobs {
                    match knob_def(k) {
                        None => self.warnings.push(format!("{origin}: unknown knob {k}, ignored")),
                        Some(d) if !d.profile => self.warnings.push(format!("{origin}: knob {k} is env-only today, ignored")),
                        _ => {}
                    }
                }
                self.profiles.retain(|q| q.id != p.id);
                self.profiles.push(p);
            }
            Err(e) => self.warnings.push(format!("{origin}: {e}, skipped")),
        }
    }

    pub fn builtin() -> Self {
        let mut s = Self::default();
        for (n, t) in BUILTIN {
            s.add_text(n, t);
        }
        s
    }

    /// Built-ins plus every `*.json` in `dir` (sorted by name). An unreadable or invalid file is a warning, never a failure: the built-ins stay.
    pub fn with_dir(dir: &Path) -> Self {
        let mut s = Self::builtin();
        let mut files: Vec<_> = match std::fs::read_dir(dir) {
            Ok(d) => d.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect(),
            Err(e) => {
                s.warnings.push(format!("{}: {e}", dir.display()));
                return s;
            }
        };
        files.sort();
        for f in files {
            match std::fs::read_to_string(&f) {
                Ok(t) => s.add_text(&f.display().to_string(), &t),
                Err(e) => s.warnings.push(format!("{}: {e}", f.display())),
            }
        }
        s
    }

    /// The process-wide set (built once; warnings go to stderr once).
    pub fn global() -> &'static ProfileSet {
        static G: OnceLock<ProfileSet> = OnceLock::new();
        G.get_or_init(|| {
            let s = match std::env::var("LB_CARD_PROFILE_DIR") {
                Ok(d) if !d.is_empty() => Self::with_dir(Path::new(&d)),
                _ => Self::builtin(),
            };
            for w in &s.warnings {
                eprintln!("lb-card-profile: {w}");
            }
            s
        })
    }

    /// The best matching profile: most criteria groups satisfied, ties to the earlier one.
    pub fn matching(&self, key: &CardKey) -> Option<&Profile> {
        let mut best: Option<(usize, &Profile)> = None;
        for p in &self.profiles {
            if let Some(s) = p.score(key) {
                if best.is_none_or(|(b, _)| s > b) {
                    best = Some((s, p));
                }
            }
        }
        best.map(|(_, p)| p)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Default,
    Profile,
    Env,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Default => "default",
            Source::Profile => "profile",
            Source::Env => "env",
        }
    }
}

/// One row of the per-card report.
#[derive(Clone, Debug, PartialEq)]
pub struct KnobRow {
    pub name: &'static str,
    pub value: String,
    pub source: Source,
    /// the profile's note for the value, when it comes from the profile (or sits under an env override)
    pub note: Option<String>,
    pub profile_capable: bool,
}

/// A card's knob values: the matched profile's, with env on top.
#[derive(Clone, Debug, Default)]
pub struct Resolved {
    pub key: CardKey,
    pub profile_id: Option<String>,
    pub family: Option<String>,
    vals: BTreeMap<String, (String, String)>,
}

fn render(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => (if *b { "1" } else { "0" }).to_string(),
        other => other.to_string(),
    }
}

impl Resolved {
    pub fn from_profile(key: &CardKey, p: Option<&Profile>) -> Self {
        let mut r = Resolved { key: key.clone(), ..Default::default() };
        if let Some(p) = p {
            r.profile_id = Some(p.id.clone());
            r.family = Some(p.family.clone());
            for (k, v) in &p.knobs {
                if knob_def(k).is_some_and(|d| d.profile) {
                    r.vals.insert(k.clone(), (render(&v.value), v.source.clone()));
                }
            }
        }
        r
    }

    /// The value of `name` with `env` as the environment: env first, then the profile, else `None` (the caller's code default applies).
    pub fn var_with(&self, name: &str, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
        env(name).or_else(|| self.vals.get(name).map(|(v, _)| v.clone()))
    }

    /// [`Resolved::var_with`] over the real environment.
    pub fn var(&self, name: &str) -> Option<String> {
        self.var_with(name, &|n| std::env::var(n).ok())
    }

    pub fn parse<T: std::str::FromStr>(&self, name: &str) -> Option<T> {
        self.var(name).and_then(|v| v.trim().parse().ok())
    }

    /// Every registry knob with its value and where it came from.
    pub fn report_with(&self, env: &dyn Fn(&str) -> Option<String>) -> Vec<KnobRow> {
        KNOBS
            .iter()
            .map(|d| {
                let prof = self.vals.get(d.name);
                let (value, source) = match (env(d.name), prof) {
                    (Some(v), _) => (v, Source::Env),
                    (None, Some((v, _))) => (v.clone(), Source::Profile),
                    (None, None) => (d.default.to_string(), Source::Default),
                };
                KnobRow { name: d.name, value, source, note: prof.map(|(_, n)| n.clone()), profile_capable: d.profile }
            })
            .collect()
    }

    pub fn report(&self) -> Vec<KnobRow> {
        self.report_with(&|n| std::env::var(n).ok())
    }

    /// One log line for the open path.
    pub fn summary(&self) -> String {
        match &self.profile_id {
            Some(id) => {
                let set: Vec<String> = self.report().iter().filter(|r| r.source != Source::Default && r.profile_capable).map(|r| format!("{}={} ({})", r.name, r.value, r.source.as_str())).collect();
                format!("[{}] {} matched profile {id}{}", self.key.label, self.key.name, if set.is_empty() { String::new() } else { format!(": {}", set.join(", ")) })
            }
            None => format!("[{}] {}: no card profile matched, code defaults{}", self.key.label, self.key.name, self.cc_text()),
        }
    }

    fn cc_text(&self) -> String {
        match (self.key.cuda_cc, self.key.pci) {
            (Some((a, b)), _) => format!(" (sm_{a}{b})"),
            (None, Some((v, d))) => format!(" (pci {v:04x}:{d:04x})"),
            _ => String::new(),
        }
    }
}

/// A knob over several cards (the stages of one pipeline): the environment wins; else the smallest numeric value any card's profile
/// gives (the most constrained card decides), `None` when no card's profile sets it.
pub fn min_knob_with(cards: &[Resolved], name: &str, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    if let Some(v) = env(name) {
        return Some(v);
    }
    cards.iter().filter_map(|c| c.var_with(name, &|_| None)).filter_map(|v| v.trim().parse::<i64>().ok().map(|n| (n, v))).min_by_key(|(n, _)| *n).map(|(_, v)| v)
}

pub fn min_knob(cards: &[Resolved], name: &str) -> Option<String> {
    min_knob_with(cards, name, &|n| std::env::var(n).ok())
}

/// Match `key` against the process-wide profiles and log the outcome once for this call.
pub fn resolve(key: &CardKey) -> Resolved {
    let r = Resolved::from_profile(key, ProfileSet::global().matching(key));
    eprintln!("lb-card-profile: {}", r.summary());
    r
}

/// Same, with no log line (for the report tool).
pub fn resolve_quiet(key: &CardKey) -> Resolved {
    Resolved::from_profile(key, ProfileSet::global().matching(key))
}

/// The multi-line report the print tool shows for one card. Profile-capable knobs always; an env-only knob only when the environment sets it.
/// Notes are cut to 110 characters unless `full` (the full text is in the profile's JSON).
pub fn format_report_opts(r: &Resolved, full: bool) -> String {
    let mut s = format!("{} {}\n", r.key.label, r.key.name);
    match (&r.profile_id, &r.family) {
        (Some(id), Some(f)) => s += &format!("  profile: {id} ({f})\n"),
        _ => s += "  profile: none matched, safe defaults\n",
    }
    for row in r.report() {
        if !row.profile_capable && row.source != Source::Env {
            continue;
        }
        let note = match (&row.note, row.source) {
            (Some(n), Source::Profile) if full || n.chars().count() <= 110 => format!("  [{n}]"),
            (Some(n), Source::Profile) => format!("  [{}...]", n.chars().take(107).collect::<String>()),
            _ => String::new(),
        };
        let tag = if row.profile_capable { "" } else { " (env-only)" };
        s += &format!("    {:<22} = {:<20} {:<8}{}{}\n", row.name, row.value, row.source.as_str(), tag, note);
    }
    s
}

pub fn format_report(r: &Resolved) -> String {
    format_report_opts(r, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda(cc: (u32, u32), name: &str) -> CardKey {
        CardKey { label: "cuda:0".into(), name: name.into(), cuda_cc: Some(cc), pci: None }
    }
    fn pci(v: u16, d: u16, name: &str) -> CardKey {
        CardKey { label: "0000:23:00.0".into(), name: name.into(), cuda_cc: None, pci: Some((v, d)) }
    }
    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn builtin_profiles_all_parse_without_warnings() {
        let s = ProfileSet::builtin();
        assert!(s.warnings.is_empty(), "{:?}", s.warnings);
        assert_eq!(s.profiles.len(), BUILTIN.len());
        for p in &s.profiles {
            for (k, v) in &p.knobs {
                assert!(!v.source.trim().is_empty(), "{}: {k} has no source", p.id);
            }
        }
    }

    #[test]
    fn every_profile_file_is_compiled_in() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles");
        let mut on_disk: Vec<String> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".json")).collect();
        on_disk.sort();
        let mut built: Vec<String> = BUILTIN.iter().map(|(n, _)| n.to_string()).collect();
        built.sort();
        assert_eq!(on_disk, built);
    }

    #[test]
    fn p100_matches_by_compute_capability_and_carries_measured_values() {
        let s = ProfileSet::builtin();
        let r = Resolved::from_profile(&cuda((6, 0), "Tesla P100-PCIE-16GB"), s.matching(&cuda((6, 0), "Tesla P100-PCIE-16GB")));
        assert_eq!(r.profile_id.as_deref(), Some("nvidia-p100"));
        assert_eq!(r.var_with("LB_QW_PREFILL_CHUNK", &no_env).as_deref(), Some("256"));
        assert_eq!(r.var_with("LB_MTP_DEPTH", &no_env).as_deref(), Some("1"));
    }

    #[test]
    fn vega10_matches_by_pci_range_and_by_adapter_name() {
        let s = ProfileSet::builtin();
        // MI25 on c1: 1002:6860
        let k = pci(0x1002, 0x6860, "AMD Radeon Pro WX 9100 (RADV VEGA10)");
        assert_eq!(s.matching(&k).map(|p| p.id.as_str()), Some("amd-vega10"));
        // an adapter whose PCI id could not be read still matches by name (RADV names Vega 10 "VEGA10")
        let by_name = CardKey { label: "discrete:2".into(), name: "AMD Radeon Pro WX 9100 (RADV VEGA10)".into(), ..Default::default() };
        let r = Resolved::from_profile(&by_name, s.matching(&by_name));
        assert_eq!(r.var_with("LB_PREFILL_MAX_ROWS", &no_env).as_deref(), Some("128"));
        // Navi 22 is not Vega
        assert_eq!(s.matching(&pci(0x1002, 0x73df, "AMD Radeon RX 6700 XT (RADV NAVI22)")).map(|p| p.id.as_str()), Some("amd-rdna2"));
    }

    #[test]
    fn p1000_is_not_a_p100_by_name_or_by_capability() {
        let s = ProfileSet::builtin();
        // found on c2 by the report tool: "P100" is a substring of "P1000"
        let by_cuda = CardKey { label: "cuda:1".into(), name: "Quadro P1000".into(), cuda_cc: Some((6, 1)), pci: None };
        assert_eq!(s.matching(&by_cuda).map(|p| p.id.as_str()), Some("nvidia-pascal-sm61"));
        let by_vulkan = CardKey { label: "0000:03:00.0".into(), name: "Quadro P1000".into(), ..Default::default() };
        assert_eq!(s.matching(&by_vulkan).map(|p| p.id.as_str()), Some("nvidia-pascal-sm61"));
        let p100_vulkan = CardKey { label: "0000:01:00.0".into(), name: "Tesla P100-PCIE-16GB".into(), ..Default::default() };
        assert_eq!(s.matching(&p100_vulkan).map(|p| p.id.as_str()), Some("nvidia-p100"));
        let t10 = CardKey { label: "0000:0d:00.0".into(), name: "Tesla T10".into(), ..Default::default() };
        assert_eq!(s.matching(&t10).map(|p| p.id.as_str()), Some("nvidia-turing-sm75"));
    }

    #[test]
    fn env_overrides_profile_overrides_default() {
        let s = ProfileSet::builtin();
        let k = cuda((6, 0), "Tesla P100");
        let r = Resolved::from_profile(&k, s.matching(&k));
        let env = |n: &str| (n == "LB_QW_PREFILL_CHUNK").then(|| "0".to_string());
        assert_eq!(r.var_with("LB_QW_PREFILL_CHUNK", &env).as_deref(), Some("0"));
        assert_eq!(r.var_with("LB_QW_PREFILL_CHUNK", &no_env).as_deref(), Some("256"));
        // a knob the profile does not set falls through to the caller's default
        assert_eq!(r.var_with("LB_Q4_MATH", &no_env), None);
        let rows = r.report_with(&env);
        let find = |n: &str| rows.iter().find(|x| x.name == n).unwrap().clone();
        assert_eq!(find("LB_QW_PREFILL_CHUNK").source, Source::Env);
        assert_eq!(find("LB_QW_PREFILL_CHUNK").value, "0");
        assert_eq!(find("LB_MTP_DEPTH").source, Source::Profile);
        assert_eq!(find("LB_Q4_MATH").source, Source::Default);
        assert_eq!(find("LB_Q4_MATH").value, "f32");
    }

    #[test]
    fn unknown_card_gets_defaults_and_says_so() {
        let s = ProfileSet::builtin();
        let k = cuda((9, 0), "NVIDIA H100");
        assert!(s.matching(&k).is_none());
        let r = Resolved::from_profile(&k, None);
        assert_eq!(r.var_with("LB_QW_PREFILL_CHUNK", &no_env), None);
        assert!(r.report_with(&no_env).iter().all(|x| x.source == Source::Default));
        assert!(r.summary().contains("no card profile matched") && r.summary().contains("sm_90"), "{}", r.summary());
        assert!(format_report(&r).contains("safe defaults"));
        // a card with nothing probed matches nothing
        assert!(s.matching(&CardKey::default()).is_none());
    }

    #[test]
    fn most_specific_profile_wins_and_a_dir_can_replace_by_id() {
        let mut s = ProfileSet::default();
        s.add_text("a", r#"{"schema":"lb-card-profile-v1","id":"cc","family":"f","match":{"cuda_cc":["7.5"]},"knobs":{"LB_Q4_ROWS":{"value":8,"source":"x"}}}"#);
        s.add_text("b", r#"{"schema":"lb-card-profile-v1","id":"cc-name","family":"f","match":{"cuda_cc":["7.5"],"name_contains":["t10"]},"knobs":{"LB_Q4_ROWS":{"value":1,"source":"y"}}}"#);
        let t10 = cuda((7, 5), "Tesla T10");
        assert_eq!(s.matching(&t10).unwrap().id, "cc-name");
        assert_eq!(s.matching(&cuda((7, 5), "GeForce RTX 2060")).unwrap().id, "cc");
        // same id replaces
        s.add_text("c", r#"{"schema":"lb-card-profile-v1","id":"cc","family":"f2","match":{"cuda_cc":["7.5"]},"knobs":{}}"#);
        assert_eq!(s.profiles.iter().filter(|p| p.id == "cc").count(), 1);
        assert_eq!(s.matching(&cuda((7, 5), "GeForce RTX 2060")).unwrap().family, "f2");
    }

    #[test]
    fn bad_files_and_unusable_knobs_warn_but_do_not_fail() {
        let mut s = ProfileSet::builtin();
        let n = s.profiles.len();
        s.add_text("junk", "{not json");
        s.add_text("v2", r#"{"schema":"lb-card-profile-v2","id":"z","family":"f","match":{"cuda_cc":["1.0"]}}"#);
        s.add_text("k", r#"{"schema":"lb-card-profile-v1","id":"k","family":"f","match":{"cuda_cc":["1.0"]},"knobs":{"LB_NOT_A_KNOB":{"value":1,"source":"s"},"LB_NO_PINNED":{"value":1,"source":"s"}}}"#);
        assert_eq!(s.profiles.len(), n + 1);
        assert_eq!(s.warnings.len(), 4, "{:?}", s.warnings);
        // ignored knobs do not reach the card
        let k = cuda((1, 0), "x");
        let r = Resolved::from_profile(&k, s.matching(&k));
        assert_eq!(r.var_with("LB_NO_PINNED", &no_env), None);
        // a profile with no criteria matches nothing
        s.add_text("e", r#"{"schema":"lb-card-profile-v1","id":"empty","family":"f","knobs":{}}"#);
        assert!(s.matching(&cuda((2, 2), "x")).is_none());
    }

    #[test]
    fn a_pipeline_takes_the_most_constrained_cards_value() {
        let s = ProfileSet::builtin();
        let p100 = |_: ()| Resolved::from_profile(&cuda((6, 0), "P100"), s.matching(&cuda((6, 0), "P100")));
        let t2060 = Resolved::from_profile(&cuda((7, 5), "RTX 2060"), s.matching(&cuda((7, 5), "RTX 2060")));
        // the 2060 + two P100 seat: only the P100 profiles set a chunk, so 256
        let cards = [t2060.clone(), p100(()), p100(())];
        assert_eq!(min_knob_with(&cards, "LB_QW_PREFILL_CHUNK", &no_env).as_deref(), Some("256"));
        // env wins over every card
        assert_eq!(min_knob_with(&cards, "LB_QW_PREFILL_CHUNK", &|_| Some("512".into())).as_deref(), Some("512"));
        // no card sets it
        assert_eq!(min_knob_with(&[t2060], "LB_QW_PREFILL_CHUNK", &no_env), None);
        assert_eq!(min_knob_with(&[], "LB_QW_PREFILL_CHUNK", &no_env), None);
    }

    #[test]
    fn json_values_render_like_env_text() {
        assert_eq!(render(&serde_json::json!(256)), "256");
        assert_eq!(render(&serde_json::json!(true)), "1");
        assert_eq!(render(&serde_json::json!("auto")), "auto");
    }
}
