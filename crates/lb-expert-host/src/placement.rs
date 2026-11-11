//! Placement as data: which host holds each `(layer, expert)`. One JSON file drives both sides:
//! the trunk reads it to know where to send, a host reads it (with its own name) to know what
//! to hold. It can be rewritten while both run; each side reloads it.
//!
//! ```json
//! { "version": 1,
//!   "hosts": { "a": { "addr": "tcp://10.0.10.204:7411", "deadline_ms": 250 },
//!              "b": { "addr": "rdma://10.0.10.61:7412", "wait": "busy" } },
//!   "default": "local",
//!   "layers": { "*": "a",
//!               "2-9": "local",
//!               "12": { "0-31": "a", "32-63": "b", "7": "local" } } }
//! ```
//! `layers` keys are a layer, a range `lo-hi` (inclusive) or `*`; a value is a host name for
//! the whole layer or a map of expert (or expert range) to host name. The most specific entry
//! wins: an expert entry over its layer's, a single layer over a range, a range over `*`, `*`
//! over `default`. `local` is the trunk itself (the name is reserved). Layers that have no
//! experts are ignored.

use serde::Deserialize;
use std::collections::BTreeMap;

pub const LOCAL: u16 = u16::MAX;

/// How a side waits for a message: sleep in the kernel, spin, or spin for a while then sleep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    Block,
    Busy,
    /// Spin for this many microseconds, then block.
    Spin(u32),
}

impl Wait {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "block" => Ok(Wait::Block),
            "busy" => Ok(Wait::Busy),
            _ => match s.strip_prefix("spin:").and_then(|n| n.parse::<u32>().ok()) {
                Some(us) => Ok(Wait::Spin(us)),
                None => Err(format!("wait {s:?}: expected block, busy or spin:<microseconds>")),
            },
        }
    }

    pub fn name(self) -> String {
        match self {
            Wait::Block => "block".into(),
            Wait::Busy => "busy".into(),
            Wait::Spin(us) => format!("spin:{us}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostSpec {
    pub name: String,
    /// `tcp://host:port`, `unix:///path`, `rdma://host:port`, `inproc://name`.
    pub addr: String,
    /// A reply to a one-row request must arrive within this, or the host is marked down.
    pub deadline_ms: u64,
    /// Added per activation row beyond the first (prefill batches).
    pub deadline_row_ms: u64,
    pub wait: Wait,
    /// Seconds after a mark-down before the trunk tries the host again; 0 = only when the
    /// placement file is reloaded.
    pub retry_s: u64,
}

#[derive(Deserialize)]
struct RawHost {
    addr: String,
    #[serde(default)]
    deadline_ms: Option<u64>,
    #[serde(default)]
    deadline_row_ms: Option<u64>,
    #[serde(default)]
    wait: Option<String>,
    #[serde(default)]
    retry_s: Option<u64>,
}

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    version: Option<u64>,
    #[serde(default)]
    hosts: BTreeMap<String, RawHost>,
    #[serde(default)]
    default: Option<String>,
    #[serde(default)]
    layers: BTreeMap<String, serde_json::Value>,
}

/// The parsed map for a model of `n_layer` x `n_expert`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacementMap {
    pub hosts: Vec<HostSpec>,
    /// `assign[layer][expert]`: index into `hosts`, or [`LOCAL`].
    pub assign: Vec<Vec<u16>>,
}

/// `"7"` or `"3-9"` (inclusive) below `limit`.
fn range(s: &str, limit: usize, what: &str) -> Result<(usize, usize), String> {
    let parse = |t: &str| t.trim().parse::<usize>().map_err(|_| format!("{what} {s:?}: not a number or lo-hi range"));
    let (lo, hi) = match s.split_once('-') {
        Some((a, b)) => (parse(a)?, parse(b)?),
        None => {
            let v = parse(s)?;
            (v, v)
        }
    };
    if lo > hi || hi >= limit {
        return Err(format!("{what} {s:?}: outside 0..{limit}"));
    }
    Ok((lo, hi))
}

impl PlacementMap {
    /// Everything local, no hosts.
    pub fn all_local(n_layer: usize, n_expert: usize) -> Self {
        Self { hosts: Vec::new(), assign: vec![vec![LOCAL; n_expert]; n_layer] }
    }

    pub fn parse(json: &str, n_layer: usize, n_expert: usize) -> Result<Self, String> {
        let raw: Raw = serde_json::from_str(json).map_err(|e| format!("placement JSON: {e}"))?;
        if let Some(v) = raw.version {
            if v != 1 {
                return Err(format!("placement version {v}: this build reads version 1"));
            }
        }
        let mut hosts = Vec::new();
        for (name, h) in &raw.hosts {
            if name == "local" {
                return Err("a host cannot be named \"local\"".into());
            }
            hosts.push(HostSpec {
                name: name.clone(),
                addr: h.addr.clone(),
                deadline_ms: h.deadline_ms.unwrap_or(1000),
                deadline_row_ms: h.deadline_row_ms.unwrap_or(250),
                wait: match &h.wait {
                    Some(w) => Wait::parse(w)?,
                    None => Wait::Block,
                },
                retry_s: h.retry_s.unwrap_or(0),
            });
        }
        let idx = |name: &str| -> Result<u16, String> {
            if name == "local" {
                return Ok(LOCAL);
            }
            hosts.iter().position(|h| h.name == name).map(|i| i as u16).ok_or_else(|| format!("placement names host {name:?}, which is not under \"hosts\""))
        };
        let default = idx(raw.default.as_deref().unwrap_or("local"))?;
        let mut assign = vec![vec![default; n_expert]; n_layer];
        // least specific first: "*", then ranges, then single layers
        let mut keys: Vec<(u8, usize, usize, &serde_json::Value)> = Vec::new();
        for (k, v) in &raw.layers {
            if k == "*" {
                if n_layer > 0 {
                    keys.push((0, 0, n_layer - 1, v));
                }
            } else {
                let (lo, hi) = range(k, n_layer, "layer")?;
                keys.push((if lo == hi { 2 } else { 1 }, lo, hi, v));
            }
        }
        keys.sort_by_key(|k| k.0);
        for (_, lo, hi, v) in keys {
            match v {
                serde_json::Value::String(name) => {
                    let h = idx(name)?;
                    for row in &mut assign[lo..=hi] {
                        row.fill(h);
                    }
                }
                serde_json::Value::Object(map) => {
                    // ranges before single experts
                    let mut ents: Vec<(bool, usize, usize, u16)> = Vec::new();
                    for (ek, ev) in map {
                        let name = ev.as_str().ok_or_else(|| format!("layer {lo}: expert {ek:?} must map to a host name"))?;
                        let (elo, ehi) = range(ek, n_expert, "expert")?;
                        ents.push((elo == ehi, elo, ehi, idx(name)?));
                    }
                    ents.sort_by_key(|e| e.0);
                    for row in &mut assign[lo..=hi] {
                        for &(_, elo, ehi, h) in &ents {
                            row[elo..=ehi].fill(h);
                        }
                    }
                }
                _ => return Err(format!("layers {lo}-{hi}: expected a host name or an expert map")),
            }
        }
        Ok(Self { hosts, assign })
    }

    pub fn host_index(&self, name: &str) -> Option<u16> {
        self.hosts.iter().position(|h| h.name == name).map(|i| i as u16)
    }

    /// The `(layer, expert)` set assigned to host `name`.
    pub fn held_by(&self, name: &str) -> Result<HeldSet, String> {
        let h = self.host_index(name).ok_or_else(|| format!("placement has no host {name:?}"))?;
        let n_expert = self.assign.first().map_or(0, Vec::len);
        let mut held = HeldSet::empty(self.assign.len(), n_expert);
        for (l, row) in self.assign.iter().enumerate() {
            for (e, &a) in row.iter().enumerate() {
                if a == h {
                    held.set(l, e, true);
                }
            }
        }
        Ok(held)
    }

    pub fn n_remote(&self) -> usize {
        self.assign.iter().flatten().filter(|&&a| a != LOCAL).count()
    }
}

/// A set of `(layer, expert)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldSet {
    pub n_layer: usize,
    pub n_expert: usize,
    pub words: usize,
    pub bits: Vec<u64>,
}

impl HeldSet {
    pub fn empty(n_layer: usize, n_expert: usize) -> Self {
        let words = n_expert.div_ceil(64).max(1);
        Self { n_layer, n_expert, words, bits: vec![0; n_layer * words] }
    }

    pub fn all(n_layer: usize, n_expert: usize) -> Self {
        let mut s = Self::empty(n_layer, n_expert);
        for l in 0..n_layer {
            for e in 0..n_expert {
                s.set(l, e, true);
            }
        }
        s
    }

    /// `"2-39:0-63,5:7"`: comma-separated `layers:experts`, each a number or an inclusive range;
    /// `*` for everything.
    pub fn parse_spec(spec: &str, n_layer: usize, n_expert: usize) -> Result<Self, String> {
        if spec.trim() == "*" {
            return Ok(Self::all(n_layer, n_expert));
        }
        let mut s = Self::empty(n_layer, n_expert);
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (ls, es) = part.split_once(':').ok_or_else(|| format!("hold spec {part:?}: expected layers:experts"))?;
            let (llo, lhi) = if ls == "*" { (0, n_layer - 1) } else { range(ls, n_layer, "layer")? };
            let (elo, ehi) = if es == "*" { (0, n_expert - 1) } else { range(es, n_expert, "expert")? };
            for l in llo..=lhi {
                for e in elo..=ehi {
                    s.set(l, e, true);
                }
            }
        }
        Ok(s)
    }

    pub fn set(&mut self, layer: usize, expert: usize, on: bool) {
        let w = &mut self.bits[layer * self.words + expert / 64];
        if on {
            *w |= 1 << (expert % 64);
        } else {
            *w &= !(1 << (expert % 64));
        }
    }

    pub fn holds(&self, layer: usize, expert: usize) -> bool {
        layer < self.n_layer && expert < self.n_expert && self.bits[layer * self.words + expert / 64] >> (expert % 64) & 1 == 1
    }

    pub fn len(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.n_layer).flat_map(move |l| (0..self.n_expert).filter(move |&e| self.holds(l, e)).map(move |e| (l, e)))
    }

    /// Whether every expert of `layer` is in the set.
    pub fn layer_whole(&self, layer: usize) -> bool {
        (0..self.n_expert).all(|e| self.holds(layer, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn most_specific_entry_wins() {
        let j = r#"{"version":1,
            "hosts":{"a":{"addr":"tcp://h:1","deadline_ms":50,"wait":"busy"},"b":{"addr":"unix:///tmp/x"}},
            "default":"local",
            "layers":{"*":"a","1-2":"local","2":{"0-3":"b","1":"a","7":"local"}}}"#;
        let p = PlacementMap::parse(j, 4, 8).unwrap();
        let (a, b) = (p.host_index("a").unwrap(), p.host_index("b").unwrap());
        assert_eq!(p.assign[0], vec![a; 8]);
        assert_eq!(p.assign[1], vec![LOCAL; 8]);
        assert_eq!(p.assign[2], vec![b, a, b, b, LOCAL, LOCAL, LOCAL, LOCAL]);
        assert_eq!(p.assign[3], vec![a; 8]);
        assert_eq!(p.hosts[a as usize].deadline_ms, 50);
        assert_eq!(p.hosts[a as usize].wait, Wait::Busy);
        assert_eq!(p.hosts[b as usize].deadline_ms, 1000);
        let held = p.held_by("b").unwrap();
        assert_eq!(held.iter().collect::<Vec<_>>(), vec![(2, 0), (2, 2), (2, 3)]);
        assert_eq!(p.held_by("a").unwrap().len(), 8 + 1 + 8);
        assert_eq!(p.n_remote(), 8 + 4 + 8);
        assert!(p.held_by("c").is_err());
    }

    #[test]
    fn bad_maps_are_refused_with_the_reason() {
        for (j, want) in [
            (r#"{"layers":{"*":"a"}}"#, "not under"),
            (r#"{"hosts":{"local":{"addr":"x"}}}"#, "cannot be named"),
            (r#"{"hosts":{"a":{"addr":"x"}},"layers":{"9":"a"}}"#, "outside"),
            (r#"{"hosts":{"a":{"addr":"x"}},"layers":{"1":{"8":"a"}}}"#, "outside"),
            (r#"{"hosts":{"a":{"addr":"x","wait":"nap"}}}"#, "wait"),
            (r#"{"version":2}"#, "version"),
            (r#"{"hosts":"#, "JSON"),
        ] {
            let e = PlacementMap::parse(j, 4, 8).unwrap_err();
            assert!(e.contains(want), "{j}: {e}");
        }
        assert_eq!(PlacementMap::parse("{}", 2, 3).unwrap(), PlacementMap::all_local(2, 3));
    }

    #[test]
    fn hold_spec_and_bitmap() {
        let s = HeldSet::parse_spec("1-2:0-1, 3:69", 4, 70).unwrap();
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![(1, 0), (1, 1), (2, 0), (2, 1), (3, 69)]);
        assert!(!s.holds(0, 0) && !s.holds(4, 0) && !s.holds(3, 70));
        assert!(HeldSet::parse_spec("*", 2, 5).unwrap().layer_whole(1));
        assert!(HeldSet::parse_spec("1", 2, 5).is_err());
        assert_eq!(Wait::parse("spin:200").unwrap(), Wait::Spin(200));
    }
}
