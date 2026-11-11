//! Neighbouring experts, for preloading cards (operator ask 2026-11-11: "measure the neighboring
//! experts firing to preload cards").
//!
//! Input is routing data: NautiBurn scrape files (`run-*.json`, schema 1 or 2) or the histogram
//! DB (`activations`, and `activation_near` once migration 006 is in). Output is one JSON report:
//!
//! * **co-fire pairs** per layer: how often two experts fire for the same token;
//! * **transitions** `L -> L+1`: which experts fire at the next MoE layer of the same token,
//!   given one that fired at this layer;
//! * **held-out next-layer coverage**: guess `N` experts of layer `L+1` from layer `L`'s
//!   picks (summed transition counts) vs from plain `L+1` heat, scored on requests the counts
//!   never saw;
//! * **runner-up promotion** (v2 data only): how often an expert ranked `k+1..2k` at token `t`
//!   fires at token `t+1` in the same layer, against the persistence of fired experts and the
//!   rate for every other expert;
//! * **cards touched** per token-layer when a layer's experts are split over `C` cards by
//!   expert id vs grouped by co-firing, and a suggested **placement** in lb-expert-host's
//!   format (`crates/lb-expert-host/src/placement.rs`).
//!
//! Tie-breaks are fixed (count, then the lower expert id) so a run is reproducible.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

/// One MoE layer of one token: the experts that fired (rank order) and the runners-up.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row {
    pub ids: Vec<u16>,
    pub near: Vec<u16>,
}

/// One request (one scrape file): tokens in position order, each `layer -> Row`.
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub id: String,
    pub phase: String,
    /// `(token_pos, layers)` sorted by position.
    pub tokens: Vec<(u32, BTreeMap<u32, Row>)>,
}

#[derive(Clone, Debug, Default)]
pub struct Corpus {
    pub requests: Vec<Request>,
    pub n_expert: usize,
    pub top_k: usize,
    pub model_id: String,
    pub source: String,
}

#[derive(Deserialize)]
struct ScrapeAct {
    layer: u32,
    token_pos: u32,
    expert_ids: Vec<i64>,
    #[serde(default)]
    near_ids: Option<Vec<i64>>,
}

#[derive(Deserialize)]
struct ScrapeFile {
    #[serde(default)]
    model_id: String,
    request_id: String,
    #[serde(default)]
    phase: String,
    activations: Vec<ScrapeAct>,
}

fn to_u16(v: &[i64]) -> Vec<u16> {
    v.iter().map(|&e| e as u16).collect()
}

impl Corpus {
    fn push_rows(&mut self, id: String, phase: String, rows: impl IntoIterator<Item = (u32, u32, Row)>) {
        let mut by_tok: BTreeMap<u32, BTreeMap<u32, Row>> = BTreeMap::new();
        for (pos, layer, row) in rows {
            by_tok.entry(pos).or_default().insert(layer, row);
        }
        if !by_tok.is_empty() {
            self.requests.push(Request { id, phase, tokens: by_tok.into_iter().collect() });
        }
    }

    fn finish_shape(&mut self) {
        let mut max_id = 0usize;
        let mut k = 0usize;
        for r in &self.requests {
            for (_, t) in &r.tokens {
                for row in t.values() {
                    k = k.max(row.ids.len());
                    for &e in row.ids.iter().chain(&row.near) {
                        max_id = max_id.max(e as usize + 1);
                    }
                }
            }
        }
        self.n_expert = self.n_expert.max(max_id);
        if self.top_k == 0 {
            self.top_k = k;
        }
    }

    /// Every `run-*.json` under `dir` (recursive). Unreadable files are skipped and counted.
    pub fn load_scrape_dirs(dirs: &[impl AsRef<Path>], n_expert_hint: usize) -> Result<(Self, usize)> {
        let mut files = Vec::new();
        for d in dirs {
            collect_runs(d.as_ref(), &mut files)?;
        }
        files.sort();
        let mut c = Corpus { n_expert: n_expert_hint, source: "scrape".into(), ..Default::default() };
        let mut skipped = 0;
        for f in files {
            let Ok(text) = std::fs::read_to_string(&f) else {
                skipped += 1;
                continue;
            };
            let Ok(s) = serde_json::from_str::<ScrapeFile>(&text) else {
                skipped += 1;
                continue;
            };
            if c.model_id.is_empty() {
                c.model_id = s.model_id.clone();
            }
            let key = f.to_string_lossy().into_owned();
            let rows = s.activations.into_iter().map(|a| {
                (a.token_pos, a.layer, Row { ids: to_u16(&a.expert_ids), near: a.near_ids.as_deref().map(to_u16).unwrap_or_default() })
            });
            // keyed by file path: two engines can mint the same request id
            c.push_rows(key, s.phase, rows);
            let _ = s.request_id;
        }
        c.finish_shape();
        Ok((c, skipped))
    }

    /// Requests of one class (and model) from the histogram DB, opened read-only. `max_requests`
    /// > 0 takes an evenly spaced sample over the class in request-id (time) order.
    pub fn load_db(db: &Path, class_id: &str, model_id: Option<&str>, max_requests: usize) -> Result<Self> {
        use rusqlite::{params, Connection, OpenFlags};
        let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .with_context(|| format!("open {} read-only", db.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        let model: String = match model_id {
            Some(m) => m.to_string(),
            None => conn
                .query_row(
                    "SELECT model_id FROM requests WHERE class_id = ?1 GROUP BY model_id ORDER BY COUNT(*) DESC LIMIT 1",
                    params![class_id],
                    |r| r.get(0),
                )
                .with_context(|| format!("no requests in class {class_id}"))?,
        };
        let (n_expert, top_k): (i64, i64) = conn
            .query_row("SELECT n_experts, top_k FROM models WHERE model_id = ?1", params![model], |r| Ok((r.get(0)?, r.get(1)?)))
            .with_context(|| format!("model {model} not in models"))?;
        let mut ids: Vec<String> = conn
            .prepare("SELECT request_id FROM requests WHERE class_id = ?1 AND model_id = ?2 ORDER BY request_id")?
            .query_map(params![class_id, model], |r| r.get(0))?
            .collect::<std::result::Result<_, _>>()?;
        if max_requests > 0 && ids.len() > max_requests {
            let step = ids.len() as f64 / max_requests as f64;
            ids = (0..max_requests).map(|i| ids[(i as f64 * step) as usize].clone()).collect();
        }
        let has_near: bool = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='activation_near'", [], |r| r.get::<_, i64>(0))?
            > 0;
        let mut act = conn.prepare(
            "SELECT phase, layer, token_pos, rank, expert_id FROM activations WHERE request_id = ?1 ORDER BY phase, token_pos, layer, rank",
        )?;
        let mut near_q = if has_near {
            Some(conn.prepare(
                "SELECT phase, layer, token_pos, rank, expert_id FROM activation_near WHERE request_id = ?1 ORDER BY phase, token_pos, layer, rank",
            )?)
        } else {
            None
        };
        let mut c = Corpus { n_expert: n_expert as usize, top_k: top_k as usize, model_id: model.clone(), source: format!("db:{class_id}"), ..Default::default() };
        for id in ids {
            // phase -> (pos, layer) -> Row
            let mut by: BTreeMap<String, BTreeMap<(u32, u32), Row>> = BTreeMap::new();
            let rows = act.query_map(params![id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(4)?)))?;
            for x in rows {
                let (ph, l, p, e) = x?;
                by.entry(ph).or_default().entry((p as u32, l as u32)).or_default().ids.push(e as u16);
            }
            if let Some(q) = near_q.as_mut() {
                let rows = q.query_map(params![id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(4)?)))?;
                for x in rows {
                    let (ph, l, p, e) = x?;
                    if let Some(row) = by.get_mut(&ph).and_then(|m| m.get_mut(&(p as u32, l as u32))) {
                        row.near.push(e as u16);
                    }
                }
            }
            for (ph, m) in by {
                let key = format!("{id}/{ph}");
                c.push_rows(key, ph, m.into_iter().map(|((p, l), row)| (p, l, row)));
            }
        }
        c.finish_shape();
        Ok(c)
    }

    pub fn n_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.tokens.len()).sum()
    }

    pub fn has_near(&self) -> bool {
        self.requests.iter().any(|r| r.tokens.iter().any(|(_, t)| t.values().any(|row| !row.near.is_empty())))
    }

    pub fn layers(&self) -> Vec<u32> {
        let mut s: Vec<u32> = self.requests.iter().flat_map(|r| r.tokens.iter().flat_map(|(_, t)| t.keys().copied())).collect();
        s.sort_unstable();
        s.dedup();
        s
    }
}

fn collect_runs(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let p = e?.path();
        if p.is_dir() {
            collect_runs(&p, out)?;
        } else if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("run-") && n.ends_with(".json")) {
            out.push(p);
        }
    }
    Ok(())
}

/// FNV-1a, for the default request split (stable across runs and machines).
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// How requests are held out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitMode {
    /// `fnv1a(id) % 2 == 1`: half, scattered (decode steps of one conversation land on both sides).
    Hash,
    /// The last `fraction` of requests in time order (the epoch inside the id): nothing in the
    /// test set happened before anything the counts saw. The honest one for live traffic.
    Time(f64),
}

/// Sort key for time order: the first number of 9+ digits in the id (an epoch), then the
/// trailing sequence number, then the id.
fn time_key(id: &str) -> (u64, u64, String) {
    let stem = Path::new(id).file_stem().and_then(|s| s.to_str()).unwrap_or(id);
    let epoch = stem.split(|ch: char| !ch.is_ascii_digit()).find(|t| t.len() >= 9).and_then(|t| t.parse().ok()).unwrap_or(0);
    (epoch, request_seq(id).0, id.to_string())
}

/// Train/test split by request: `test_ids` when given (a request is held out when its id, or
/// its file name, is listed), else by `mode`.
pub fn split(c: &Corpus, test_ids: Option<&HashSet<String>>) -> (Vec<usize>, Vec<usize>) {
    split_mode(c, test_ids, SplitMode::Hash)
}

pub fn split_mode(c: &Corpus, test_ids: Option<&HashSet<String>>, mode: SplitMode) -> (Vec<usize>, Vec<usize>) {
    if let (None, SplitMode::Time(f)) = (test_ids, mode) {
        let mut order: Vec<usize> = (0..c.requests.len()).collect();
        order.sort_by_cached_key(|&i| time_key(&c.requests[i].id));
        let n_test = ((c.requests.len() as f64) * f.clamp(0.0, 1.0)).round() as usize;
        let cut = c.requests.len() - n_test;
        let (a, b) = order.split_at(cut);
        return (a.to_vec(), b.to_vec());
    }
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, r) in c.requests.iter().enumerate() {
        let held = match test_ids {
            Some(s) => s.contains(&r.id) || Path::new(&r.id).file_name().and_then(|n| n.to_str()).is_some_and(|n| s.contains(n)),
            None => fnv1a(&r.id) % 2 == 1,
        };
        if held {
            test.push(i);
        } else {
            train.push(i);
        }
    }
    (train, test)
}

/// Counts from the training requests. Dense per layer, indexed by layer slot.
pub struct Counts {
    pub e: usize,
    /// layer -> slot
    pub slot: BTreeMap<u32, usize>,
    pub heat: Vec<Vec<u64>>,
    /// `co[slot][a * e + b]`, a < b
    pub co: Vec<Vec<u32>>,
    /// `next[slot][a * e + b]`: a fired at L, b at L+1 (same token)
    pub next: Vec<Vec<u32>>,
    /// as `next`, but a was a runner-up at L
    pub next_near: Vec<Vec<u32>>,
    pub tokens_per_layer: Vec<u64>,
}

impl Counts {
    pub fn build(c: &Corpus, reqs: &[usize]) -> Self {
        let layers = c.layers();
        let e = c.n_expert;
        let slot: BTreeMap<u32, usize> = layers.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        let nl = layers.len();
        let mut k = Counts {
            e,
            slot,
            heat: vec![vec![0; e]; nl],
            co: vec![vec![0; e * e]; nl],
            next: vec![vec![0; e * e]; nl],
            next_near: vec![Vec::new(); nl],
            tokens_per_layer: vec![0; nl],
        };
        let near = c.has_near();
        if near {
            k.next_near = vec![vec![0; e * e]; nl];
        }
        for &ri in reqs {
            for (_, t) in &c.requests[ri].tokens {
                for (&l, row) in t {
                    let s = k.slot[&l];
                    k.tokens_per_layer[s] += 1;
                    for &a in &row.ids {
                        k.heat[s][a as usize] += 1;
                    }
                    let mut ids = row.ids.clone();
                    ids.sort_unstable();
                    for i in 0..ids.len() {
                        for j in i + 1..ids.len() {
                            k.co[s][ids[i] as usize * e + ids[j] as usize] += 1;
                        }
                    }
                    if let Some(nx) = t.get(&(l + 1)) {
                        for &a in &row.ids {
                            for &b in &nx.ids {
                                k.next[s][a as usize * e + b as usize] += 1;
                            }
                        }
                        if near {
                            for &a in &row.near {
                                for &b in &nx.ids {
                                    k.next_near[s][a as usize * e + b as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        k
    }

    fn pair(&self, s: usize, a: usize, b: usize) -> u32 {
        let (x, y) = if a < b { (a, b) } else { (b, a) };
        self.co[s][x * self.e + y]
    }

    /// Experts of a layer hottest first (count desc, then id asc).
    pub fn heat_order(&self, s: usize) -> Vec<usize> {
        let mut o: Vec<usize> = (0..self.e).collect();
        o.sort_by(|&a, &b| self.heat[s][b].cmp(&self.heat[s][a]).then(a.cmp(&b)));
        o
    }
}

/// Top `n` by score desc, then id asc; zero scores never make the list.
fn top_n(scores: &[u64], n: usize) -> Vec<usize> {
    let mut o: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] > 0).collect();
    o.sort_by(|&a, &b| scores[b].cmp(&scores[a]).then(a.cmp(&b)));
    o.truncate(n);
    o
}

/// Held-out next-layer coverage: share of layer `L+1`'s real picks found among `n` guesses.
/// Returns (from this layer's picks, from plain heat, from picks + runners-up when v2).
pub fn next_layer_coverage(c: &Corpus, k: &Counts, test: &[usize], n: usize) -> (f64, f64, Option<f64>) {
    let (mut hit, mut hit_h, mut hit_n, mut tot) = (0u64, 0u64, 0u64, 0u64);
    let near = !k.next_near.is_empty() && k.next_near[0].len() == k.e * k.e;
    let heat_top: Vec<Vec<usize>> = (0..k.slot.len()).map(|s| top_n(&k.heat[s], n)).collect();
    let mut sc = vec![0u64; k.e];
    let mut scn = vec![0u64; k.e];
    for &ri in test {
        for (_, t) in &c.requests[ri].tokens {
            for (&l, row) in t {
                let Some(nx) = t.get(&(l + 1)) else { continue };
                let (Some(&s), Some(&s1)) = (k.slot.get(&l), k.slot.get(&(l + 1))) else { continue };
                sc.iter_mut().for_each(|x| *x = 0);
                for &a in &row.ids {
                    let r = &k.next[s][a as usize * k.e..(a as usize + 1) * k.e];
                    for (x, &v) in sc.iter_mut().zip(r) {
                        *x += v as u64;
                    }
                }
                let pred = top_n(&sc, n);
                let pred_n = if near && !row.near.is_empty() {
                    scn.copy_from_slice(&sc);
                    for &a in &row.near {
                        let r = &k.next_near[s][a as usize * k.e..(a as usize + 1) * k.e];
                        for (x, &v) in scn.iter_mut().zip(r) {
                            *x += v as u64;
                        }
                    }
                    Some(top_n(&scn, n))
                } else {
                    None
                };
                for &b in &nx.ids {
                    let b = b as usize;
                    tot += 1;
                    hit += pred.contains(&b) as u64;
                    hit_h += heat_top[s1].contains(&b) as u64;
                    if let Some(p) = &pred_n {
                        hit_n += p.contains(&b) as u64;
                    }
                }
            }
        }
    }
    let f = |h: u64| if tot > 0 { h as f64 / tot as f64 } else { 0.0 };
    (f(hit), f(hit_h), near.then(|| f(hit_n)))
}

/// Greedy co-fire grouping of one layer into `cards` groups of equal capacity: experts in heat
/// order, each to the open group it co-fires with most (ties: the emptier group, then the
/// lower index).
pub fn group_by_cofire(k: &Counts, s: usize, cards: usize) -> Vec<usize> {
    let e = k.e;
    let cap = e.div_ceil(cards);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); cards];
    let mut assign = vec![0usize; e];
    for x in k.heat_order(s) {
        let mut best: Option<(u64, usize, usize)> = None; // (cofire, free slots, group)
        for (g, members) in groups.iter().enumerate() {
            if members.len() >= cap {
                continue;
            }
            let score: u64 = members.iter().map(|&m| k.pair(s, x, m) as u64).sum();
            let free = cap - members.len();
            let better = match best {
                None => true,
                Some((bs, bf, _)) => score > bs || (score == bs && free > bf),
            };
            if better {
                best = Some((score, free, g));
            }
        }
        let g = best.expect("capacity covers every expert").2;
        groups[g].push(x);
        assign[x] = g;
    }
    assign
}

/// Mean distinct cards per test token-layer under an assignment `assign[slot][expert]`.
pub fn cards_touched(c: &Corpus, k: &Counts, test: &[usize], assign: &[Vec<usize>]) -> f64 {
    let (mut sum, mut n) = (0usize, 0usize);
    let mut seen: Vec<usize> = Vec::new();
    for &ri in test {
        for (_, t) in &c.requests[ri].tokens {
            for (&l, row) in t {
                let Some(&s) = k.slot.get(&l) else { continue };
                seen.clear();
                for &e in &row.ids {
                    let g = assign[s][e as usize];
                    if !seen.contains(&g) {
                        seen.push(g);
                    }
                }
                sum += seen.len();
                n += 1;
            }
        }
    }
    if n > 0 {
        sum as f64 / n as f64
    } else {
        0.0
    }
}

/// Runner-up promotion over consecutive tokens (`t`, `t+1`) of the same request, same layer.
/// With `chain_decode`, single-token decode requests are also chained in id order (one
/// llama-server slot writes one file per decode step; only valid with one slot).
pub fn promotion(c: &Corpus, chain_decode: bool) -> Option<Value> {
    if !c.has_near() {
        return None;
    }
    let e = c.n_expert as f64;
    let k = c.top_k as f64;
    let (mut near_n, mut near_hit, mut fired_n, mut fired_stay, mut other_n, mut other_hit) = (0u64, 0u64, 0u64, 0u64, 0f64, 0u64);
    let mut by_rank_n = vec![0u64; c.top_k];
    let mut by_rank_hit = vec![0u64; c.top_k];
    let mut pairs = 0u64;
    let mut visit = |a: &BTreeMap<u32, Row>, b: &BTreeMap<u32, Row>| {
        for (l, ra) in a {
            let Some(rb) = b.get(l) else { continue };
            if ra.near.is_empty() {
                continue;
            }
            pairs += 1;
            let next: HashSet<u16> = rb.ids.iter().copied().collect();
            for (i, x) in ra.near.iter().enumerate() {
                near_n += 1;
                let h = next.contains(x) as u64;
                near_hit += h;
                if i < by_rank_n.len() {
                    by_rank_n[i] += 1;
                    by_rank_hit[i] += h;
                }
            }
            for x in &ra.ids {
                fired_n += 1;
                fired_stay += next.contains(x) as u64;
            }
            let known: HashSet<u16> = ra.ids.iter().chain(&ra.near).copied().collect();
            other_n += e - known.len() as f64;
            other_hit += rb.ids.iter().filter(|x| !known.contains(x)).count() as u64;
        }
    };
    for r in &c.requests {
        for w in r.tokens.windows(2) {
            if w[1].0 == w[0].0 + 1 {
                visit(&w[0].1, &w[1].1);
            }
        }
    }
    if chain_decode {
        let mut dec: Vec<&Request> = c.requests.iter().filter(|r| r.phase == "decode" && r.tokens.len() == 1).collect();
        dec.sort_by_key(|r| request_seq(&r.id));
        for w in dec.windows(2) {
            visit(&w[0].tokens[0].1, &w[1].tokens[0].1);
        }
    }
    let f = |h: u64, n: f64| if n > 0.0 { h as f64 / n } else { 0.0 };
    Some(json!({
        "token_layer_pairs": pairs,
        "runner_up_fires_next_token": f(near_hit, near_n as f64),
        "runner_up_fires_next_token_by_rank": (0..c.top_k).map(|i| json!({"rank": c.top_k + i + 1, "rate": f(by_rank_hit[i], by_rank_n[i] as f64), "n": by_rank_n[i]})).collect::<Vec<_>>(),
        "fired_fires_again_next_token": f(fired_stay, fired_n as f64),
        "other_expert_fires_next_token": f(other_hit, other_n),
        "chance_any_expert": k / e,
        "chain_decode": chain_decode,
    }))
}

/// The trailing number of an id (`.../run-<epoch>-<seq>.json` or `nb-server-<epoch>-<seq>`).
fn request_seq(id: &str) -> (u64, String) {
    let stem = Path::new(id).file_stem().and_then(|s| s.to_str()).unwrap_or(id);
    let n = stem.rsplit('-').next().and_then(|x| x.parse::<u64>().ok()).unwrap_or(0);
    (n, id.to_string())
}

/// lb-expert-host placement (version 1) for `cards` cards from per-layer assignments. Host
/// addresses are placeholders: the consumer (W4) fills them in.
pub fn placement_json(c: &Corpus, k: &Counts, cards: usize, assign: &[Vec<usize>]) -> Value {
    let hosts: serde_json::Map<String, Value> =
        (0..cards).map(|g| (format!("card{g}"), json!({"addr": format!("unset://card{g}")}))).collect();
    let mut layers = serde_json::Map::new();
    for (&l, &s) in &k.slot {
        let m: serde_json::Map<String, Value> = (0..k.e).map(|x| (x.to_string(), json!(format!("card{}", assign[s][x])))).collect();
        layers.insert(l.to_string(), Value::Object(m));
    }
    json!({
        "version": 1,
        "_comment": format!("lb-cofire: experts grouped per card by co-firing ({} model {}, {} requests, {} tokens). Host addrs are placeholders.", c.source, c.model_id, c.requests.len(), c.n_tokens()),
        "hosts": hosts,
        "default": "local",
        "layers": layers,
    })
}

pub struct Options {
    pub guesses: Vec<usize>,
    pub cards: Vec<usize>,
    pub top_pairs: usize,
    pub top_next: usize,
    pub chain_decode: bool,
    pub placement_cards: Option<usize>,
    pub split: SplitMode,
}

/// The whole report, plus the placement (when asked for).
pub fn analyse(c: &Corpus, test_ids: Option<&HashSet<String>>, o: &Options) -> Result<(Value, Option<Value>)> {
    if c.requests.is_empty() {
        bail!("no requests loaded");
    }
    if c.n_expert == 0 || c.n_expert > u16::MAX as usize {
        return Err(anyhow!("n_expert {} out of range", c.n_expert));
    }
    let (train, test) = split_mode(c, test_ids, o.split);
    if train.is_empty() || test.is_empty() {
        bail!("split left {} train / {} test requests", train.len(), test.len());
    }
    let k = Counts::build(c, &train);
    let n_test_tokens: usize = test.iter().map(|&i| c.requests[i].tokens.len()).sum();

    let coverage: Vec<Value> = o
        .guesses
        .iter()
        .map(|&n| {
            let (a, h, nn) = next_layer_coverage(c, &k, &test, n);
            json!({"guesses": n, "from_this_layer": a, "from_heat": h, "from_this_layer_plus_runners_up": nn})
        })
        .collect();

    let n_slot = k.slot.len();
    let mut cards_rows = Vec::new();
    let mut placement = None;
    let mut cards_list = o.cards.clone();
    if let Some(pc) = o.placement_cards {
        if !cards_list.contains(&pc) {
            cards_list.push(pc);
        }
    }
    for &cards in &cards_list {
        let by_id: Vec<Vec<usize>> = (0..n_slot).map(|_| (0..k.e).map(|x| x * cards / k.e).collect()).collect();
        let by_co: Vec<Vec<usize>> = (0..n_slot).map(|s| group_by_cofire(&k, s, cards)).collect();
        let kk = c.top_k as f64;
        let random = cards as f64 * (1.0 - (1.0 - 1.0 / cards as f64).powf(kk));
        let a = cards_touched(c, &k, &test, &by_id);
        let b = cards_touched(c, &k, &test, &by_co);
        if o.cards.contains(&cards) {
            cards_rows.push(json!({"cards": cards, "by_expert_id": a, "by_cofire": b, "random_expected": random, "cut_vs_by_id": if a > 0.0 { 1.0 - b / a } else { 0.0 }}));
        }
        if o.placement_cards == Some(cards) {
            placement = Some(placement_json(c, &k, cards, &by_co));
        }
    }

    let mut pairs = serde_json::Map::new();
    let mut trans = serde_json::Map::new();
    for (&l, &s) in &k.slot {
        let mut top: Vec<(u32, usize, usize)> = Vec::new();
        for a in 0..k.e {
            for b in a + 1..k.e {
                let v = k.co[s][a * k.e + b];
                if v > 0 {
                    top.push((v, a, b));
                }
            }
        }
        top.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        let n_pairs = top.len();
        top.truncate(o.top_pairs);
        pairs.insert(
            l.to_string(),
            json!({"tokens": k.tokens_per_layer[s], "nonzero_pairs": n_pairs, "top": top.iter().map(|&(v, a, b)| json!([a, b, v])).collect::<Vec<_>>()}),
        );
        if k.slot.contains_key(&(l + 1)) {
            let mut m = serde_json::Map::new();
            for a in 0..k.e {
                let row: Vec<u64> = k.next[s][a * k.e..(a + 1) * k.e].iter().map(|&v| v as u64).collect();
                let t = top_n(&row, o.top_next);
                if !t.is_empty() {
                    m.insert(a.to_string(), Value::Array(t.iter().map(|&b| json!([b, row[b]])).collect()));
                }
            }
            trans.insert(format!("{l}->{}", l + 1), Value::Object(m));
        }
    }

    let report = json!({
        "tool": "lb-cofire",
        "source": c.source,
        "model_id": c.model_id,
        "n_expert": c.n_expert,
        "top_k": c.top_k,
        "moe_layers": k.slot.keys().collect::<Vec<_>>(),
        "requests": {"total": c.requests.len(), "train": train.len(), "test": test.len()},
        "tokens": {"total": c.n_tokens(), "test": n_test_tokens},
        "has_runners_up": c.has_near(),
        "split": match (test_ids.is_some(), o.split) {
            (true, _) => "explicit test-request list".to_string(),
            (false, SplitMode::Hash) => "fnv1a(request id) % 2 == 1 held out".to_string(),
            (false, SplitMode::Time(f)) => format!("time: the last {:.0}% of requests (epoch in the id) held out", f * 100.0),
        },
        "next_layer_coverage": coverage,
        "cards_touched_per_token_layer": cards_rows,
        "runner_up_promotion": promotion(c, o.chain_decode),
        "cofire_pairs": pairs,
        "transitions": trans,
    });
    Ok((report, placement))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 8 experts, top-2, layers 0..2. Experts pair up (0,1), (2,3), ... and layer L's pair p is
    /// followed by pair p at L+1; runners-up are the next pair, and fire at the next token.
    fn toy() -> Corpus {
        let mut c = Corpus { n_expert: 8, top_k: 2, source: "toy".into(), model_id: "toy".into(), ..Default::default() };
        for r in 0..40u32 {
            let mut rows = Vec::new();
            for pos in 0..6u32 {
                let p = ((r + pos) % 4) as u16;
                let q = (p + 1) % 4;
                for l in 0..3u32 {
                    rows.push((pos, l, Row { ids: vec![2 * p, 2 * p + 1], near: vec![2 * q, 2 * q + 1] }));
                }
            }
            c.push_rows(format!("req-{r}"), "prefill".into(), rows);
        }
        c.finish_shape();
        c
    }

    fn opts() -> Options {
        Options { guesses: vec![2, 4], cards: vec![2, 4], top_pairs: 4, top_next: 2, chain_decode: false, placement_cards: Some(4), split: SplitMode::Hash }
    }

    #[test]
    fn structured_routing_is_found() {
        let c = toy();
        let (rep, placement) = analyse(&c, None, &opts()).unwrap();
        // L's pair names L+1's pair exactly: two guesses cover everything; heat cannot
        let cov = &rep["next_layer_coverage"][0];
        assert_eq!(cov["from_this_layer"].as_f64().unwrap(), 1.0);
        assert!(cov["from_heat"].as_f64().unwrap() < 0.6);
        // pairs always co-fire: grouping by co-fire puts each pair on one card
        let ct = &rep["cards_touched_per_token_layer"][1];
        assert_eq!(ct["cards"], 4);
        assert_eq!(ct["by_cofire"].as_f64().unwrap(), 1.0);
        // runners-up are next token's picks by construction
        let pr = &rep["runner_up_promotion"];
        assert_eq!(pr["runner_up_fires_next_token"].as_f64().unwrap(), 1.0);
        assert_eq!(pr["fired_fires_again_next_token"].as_f64().unwrap(), 0.0);
        // the placement parses as lb-expert-host's and keeps every pair together
        let p = placement.unwrap();
        let map = lb_expert_host::placement::PlacementMap::parse(&p.to_string(), 3, 8).unwrap();
        assert_eq!(map.hosts.len(), 4);
        for l in 0..3 {
            for pair in 0..4 {
                assert_eq!(map.assign[l][2 * pair], map.assign[l][2 * pair + 1]);
            }
        }
    }

    #[test]
    fn explicit_split_and_v1_without_runners_up() {
        let mut c = toy();
        for r in &mut c.requests {
            for (_, t) in &mut r.tokens {
                for row in t.values_mut() {
                    row.near.clear();
                }
            }
        }
        let test: HashSet<String> = (0..10).map(|i| format!("req-{i}")).collect();
        let (rep, _) = analyse(&c, Some(&test), &opts()).unwrap();
        assert_eq!(rep["requests"]["test"], 10);
        assert!(rep["runner_up_promotion"].is_null());
        assert!(rep["next_layer_coverage"][0]["from_this_layer_plus_runners_up"].is_null());
    }

    #[test]
    fn time_split_holds_out_the_newest() {
        let mut c = Corpus::default();
        for (i, id) in ["nb-server-1794000300-3", "nb-server-1794000100-1", "x/run-1794000200-2.json", "nb-server-1794000400-4"].iter().enumerate() {
            c.push_rows(id.to_string(), "decode".into(), [(0u32, 0u32, Row { ids: vec![i as u16], near: vec![] })]);
        }
        let (train, test) = split_mode(&c, None, SplitMode::Time(0.5));
        let ids = |v: &[usize]| v.iter().map(|&i| c.requests[i].id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&train), vec!["nb-server-1794000100-1", "x/run-1794000200-2.json"]);
        assert_eq!(ids(&test), vec!["nb-server-1794000300-3", "nb-server-1794000400-4"]);
    }

    #[test]
    fn scrape_files_v1_and_v2_load() {
        let dir = std::env::temp_dir().join(format!("lb-cofire-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(
            dir.join("run-1-1.json"),
            r#"{"schema_version":1,"model_id":"m","request_id":"a","phase":"decode","activations":[{"layer":0,"token_pos":0,"expert_ids":[3,1]}]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("sub/run-1-2.json"),
            r#"{"schema_version":2,"model_id":"m","request_id":"b","phase":"decode","near_k":2,"activations":[{"layer":0,"token_pos":0,"expert_ids":[3,1],"weights":[0.6,0.4],"scores":[1,0.5],"near_ids":[9,2],"near_scores":[0.1,0]}]}"#,
        )
        .unwrap();
        std::fs::write(dir.join("run-1-3.json"), "not json").unwrap();
        let (c, skipped) = Corpus::load_scrape_dirs(&[&dir], 0).unwrap();
        assert_eq!((c.requests.len(), skipped, c.n_expert, c.top_k), (2, 1, 10, 2));
        assert!(c.has_near());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
