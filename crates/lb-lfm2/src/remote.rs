//! The trunk side of work-to-weights: a `remote` tier in front of the local expert paths.
//!
//! For each routed expert of a MoE layer the trunk asks the placement (data: a JSON map layer ->
//! expert -> host, `lb_expert_host::placement`, reloaded when the file changes) where the expert
//! lives, groups the layer's remote experts into **one message per host**, sends them all, then
//! collects the replies. Whatever a host does not answer correctly and in time stays with the
//! local path, which runs as it always has: a host that misses its deadline, drops the
//! connection or answers with an error is **marked down** and its experts fall back; a host
//! that refuses an expert it does not hold is believed (that expert is local until the next
//! reload or the refusal ages out). The outputs come back unweighted and are summed here, in
//! ascending expert id, with the local ones, so the arithmetic does not depend on where an
//! expert ran: never a wrong answer, never a hang.
//!
//! Hops are counted per forward pass (per token in decode) and logged: `LB_W2W_LOG=<file>`
//! appends one JSON line a pass; [`RemoteTier::take_records`] hands the same records to a
//! harness.

use lb_expert_host::placement::{HostSpec, PlacementMap, LOCAL};
use lb_expert_host::wire::RunReply;
use lb_expert_host::HostClient;
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

/// One expert of a layer with the tokens routed to it: `(expert, [(token row, weight)])`.
pub type ExpertCall = (usize, Vec<(usize, f32)>);
/// An expert's computed output rows, in its tokens' order.
pub type ExpertPart = (usize, Vec<(usize, f32)>, Vec<f32>);

/// How long a refusal is believed before the host is asked again.
const REFUSAL_TTL: Duration = Duration::from_secs(2);
/// How often the placement file's mtime is looked at.
const RELOAD_CHECK: Duration = Duration::from_millis(250);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Default, Debug)]
pub struct RemoteStats {
    /// Messages sent to hosts (one per host per layer pass).
    pub hops: AtomicU64,
    /// Expert calls answered by a host.
    pub experts: AtomicU64,
    /// Expert calls that were placed on a host but ran locally (host down, refused, failed).
    pub fallback_experts: AtomicU64,
    pub refusals: AtomicU64,
    pub timeouts: AtomicU64,
    pub mark_downs: AtomicU64,
    pub reloads: AtomicU64,
    /// Sum of round trips (send to reply decoded), per message.
    pub rt_ns: AtomicU64,
    /// Sum of the hosts' own compute time, as they reported it.
    pub host_ns: AtomicU64,
    /// Wall time the trunk spent in the remote tier (round trips of one layer overlap).
    pub wall_ns: AtomicU64,
}

/// One forward pass (one token in decode) as the remote tier saw it.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct PassRecord {
    pub pos0: usize,
    pub m: usize,
    /// Host round trips in this pass.
    pub hops: u64,
    pub experts_remote: u64,
    pub experts_fallback: u64,
    /// Wall time waiting on hosts (ms).
    pub hop_wall_ms: f64,
    /// Sum over hops of round trip minus the host's compute: wire + encode/decode (ms).
    pub hop_net_ms: f64,
    /// Sum of the hosts' compute (ms).
    pub host_ms: f64,
    /// Hosts that are down at the end of the pass.
    pub down: Vec<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct HostState {
    pub name: String,
    pub addr: String,
    pub up: bool,
    /// Why it is down, when it is.
    pub reason: Option<String>,
    pub transport: Option<String>,
    pub backend: Option<String>,
    /// For a CPU host: whether its expert arithmetic fingerprint equals this process's
    /// (`lb_expert_host::expert::arith_fingerprint`); `None` for a device host or one that is down.
    pub arith_match: Option<bool>,
    pub hops: u64,
    pub experts: u64,
}

struct Link {
    spec: HostSpec,
    client: Option<HostClient>,
    down: Option<(Instant, String)>,
    hops: u64,
    experts: u64,
}

#[derive(Default)]
struct Acc {
    hops: u64,
    experts: u64,
    fallback: u64,
    wall_ns: u64,
    net_ns: u64,
    host_ns: u64,
}

struct Inner {
    map: PlacementMap,
    links: Vec<Link>,
    mtime: Option<SystemTime>,
    last_check: Instant,
    /// `(host, layer, expert)` a host refused, and when.
    refused: HashMap<(u16, u16, u16), Instant>,
    acc: Acc,
    records: Vec<PassRecord>,
    log: Option<std::fs::File>,
}

pub struct RemoteTier {
    fingerprint: [u8; 16],
    n_layer: usize,
    n_expert: usize,
    d: usize,
    /// Which layers have experts.
    moe: Vec<bool>,
    path: Option<PathBuf>,
    inner: Mutex<Inner>,
    pub stats: RemoteStats,
}

/// The message for one host in one layer pass.
struct Batch {
    host: usize,
    ids: Vec<u16>,
    /// Row lists per id, as indices into the compact `x` sent to this host.
    groups: Vec<Vec<u32>>,
    /// The calls, in `ids` order (handed back whole on a failure).
    calls: Vec<ExpertCall>,
    rows: usize,
    sent: Option<Instant>,
}

fn connect(spec: &HostSpec, fingerprint: &[u8; 16], d: usize) -> Result<HostClient, String> {
    let c = HostClient::connect(&spec.addr, spec.wait, CONNECT_TIMEOUT).map_err(|e| format!("connect {}: {e}", spec.addr))?;
    if &c.info.fingerprint != fingerprint {
        return Err(format!("{} serves a different model file (fingerprint {})", spec.addr, lb_expert_host::model::hex(&c.info.fingerprint)));
    }
    if c.info.d as usize != d {
        return Err(format!("{} has activation width {}, the trunk {d}", spec.addr, c.info.d));
    }
    Ok(c)
}

impl RemoteTier {
    /// A tier from placement JSON; hosts are connected now, and one that cannot be reached
    /// starts down (the trunk then runs its experts itself).
    pub fn from_json(json: &str, fingerprint: [u8; 16], moe_layers: &[bool], n_expert: usize, d: usize) -> Result<Self, String> {
        let n_layer = moe_layers.len();
        let map = PlacementMap::parse(json, n_layer, n_expert)?;
        let log = match std::env::var_os("LB_W2W_LOG") {
            Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| format!("LB_W2W_LOG {}: {e}", Path::new(&p).display()))?),
            None => None,
        };
        let t = Self {
            fingerprint,
            n_layer,
            n_expert,
            d,
            moe: moe_layers.to_vec(),
            path: None,
            inner: Mutex::new(Inner { map: PlacementMap::all_local(n_layer, n_expert), links: Vec::new(), mtime: None, last_check: Instant::now(), refused: HashMap::new(), acc: Acc::default(), records: Vec::new(), log }),
            stats: RemoteStats::default(),
        };
        t.install(&mut t.inner.lock().unwrap(), map);
        Ok(t)
    }

    /// A tier from a placement file, reloaded whenever the file's modification time changes
    /// (looked at between forward passes).
    pub fn from_file(path: &Path, fingerprint: [u8; 16], moe_layers: &[bool], n_expert: usize, d: usize) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("placement {}: {e}", path.display()))?;
        let mut t = Self::from_json(&json, fingerprint, moe_layers, n_expert, d)?;
        t.path = Some(path.to_path_buf());
        t.inner.lock().unwrap().mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        Ok(t)
    }

    /// Put `map` in force: keep the connections of hosts whose address did not change, connect
    /// the new ones, forget refusals.
    fn install(&self, inner: &mut Inner, mut map: PlacementMap) {
        // "*" and layer ranges may cover layers that have no experts: nothing is placed there
        for (row, &moe) in map.assign.iter_mut().zip(&self.moe) {
            if !moe {
                row.fill(LOCAL);
            }
        }
        let mut old: Vec<Link> = std::mem::take(&mut inner.links);
        let mut links = Vec::with_capacity(map.hosts.len());
        for spec in &map.hosts {
            let kept = old.iter().position(|l| l.spec.name == spec.name && l.spec.addr == spec.addr && l.spec.wait == spec.wait && l.client.is_some()).map(|i| old.swap_remove(i));
            let mut link = match kept {
                Some(mut l) => {
                    l.spec = spec.clone();
                    // learn what the host holds now; a host that does not answer is reconnected
                    if l.client.as_mut().is_some_and(|c| c.refresh(CONNECT_TIMEOUT).is_err()) {
                        l.client = None;
                    }
                    l
                }
                None => Link { spec: spec.clone(), client: None, down: None, hops: 0, experts: 0 },
            };
            if link.client.is_none() {
                match connect(spec, &self.fingerprint, self.d) {
                    Ok(c) => {
                        link.client = Some(c);
                        link.down = None;
                    }
                    Err(e) => {
                        eprintln!("lb-lfm2 w2w: host {} is down, its experts run locally: {e}", spec.name);
                        self.stats.mark_downs.fetch_add(1, Ordering::Relaxed);
                        link.down = Some((Instant::now(), e));
                    }
                }
            }
            if let Some(c) = &link.client {
                if c.info.backend == "cpu" && c.info.arith != lb_expert_host::expert::arith_fingerprint() {
                    eprintln!("lb-lfm2 w2w: host {}: its CPU arithmetic differs from this process's (C library exp): its experts agree with local ones to rounding, not bit for bit", spec.name);
                }
                let h = links.len() as u16;
                let assigned = map.assign.iter().flatten().filter(|&&a| a == h).count();
                let held = map.assign.iter().enumerate().flat_map(|(l, row)| row.iter().enumerate().filter(move |(_, &a)| a == h).map(move |(e, _)| (l, e))).filter(|&(l, e)| c.info.holds(l, e)).count();
                eprintln!(
                    "lb-lfm2 w2w: host {} at {} ({} over {}, wait {}, deadline {} ms): {assigned} experts placed, {held} held there now",
                    spec.name,
                    spec.addr,
                    c.info.backend,
                    c.transport(),
                    spec.wait.name(),
                    spec.deadline_ms
                );
            }
            links.push(link);
        }
        inner.links = links;
        inner.map = map;
        inner.refused.clear();
    }

    /// Re-read the placement file now. On a file that does not parse the old map stays.
    pub fn reload(&self) -> Result<(), String> {
        let Some(path) = &self.path else { return Err("this tier has no placement file".into()) };
        let json = std::fs::read_to_string(path).map_err(|e| format!("placement {}: {e}", path.display()))?;
        let map = PlacementMap::parse(&json, self.n_layer, self.n_expert)?;
        let mut inner = self.inner.lock().unwrap();
        inner.mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        self.install(&mut inner, map);
        self.stats.reloads.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Between passes: pick up a changed placement file, and try hosts whose `retry_s` is due.
    pub fn begin_pass(&self) {
        let mut reload = false;
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(path) = &self.path {
                if inner.last_check.elapsed() >= RELOAD_CHECK {
                    inner.last_check = Instant::now();
                    let now = std::fs::metadata(path).and_then(|m| m.modified()).ok();
                    reload = now.is_some() && now != inner.mtime;
                }
            }
            if !reload {
                for link in &mut inner.links {
                    let due = link.spec.retry_s > 0 && link.down.as_ref().is_some_and(|(at, _)| at.elapsed() >= Duration::from_secs(link.spec.retry_s));
                    if due {
                        match connect(&link.spec, &self.fingerprint, self.d) {
                            Ok(c) => {
                                eprintln!("lb-lfm2 w2w: host {} is back", link.spec.name);
                                link.client = Some(c);
                                link.down = None;
                            }
                            Err(e) => link.down = Some((Instant::now(), e)),
                        }
                    }
                }
            }
        }
        if reload {
            match self.reload() {
                Ok(()) => eprintln!("lb-lfm2 w2w: placement reloaded"),
                Err(e) => {
                    eprintln!("lb-lfm2 w2w: placement reload failed, the old map stays: {e}");
                    // do not retry every pass: wait for the file to change again
                    if let Some(path) = &self.path {
                        self.inner.lock().unwrap().mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
                    }
                }
            }
        }
    }

    /// Close the pass that ran `m` tokens from `pos0`: one record (and one log line).
    pub fn end_pass(&self, pos0: usize, m: usize) {
        let mut inner = self.inner.lock().unwrap();
        let a = std::mem::take(&mut inner.acc);
        let rec = PassRecord {
            pos0,
            m,
            hops: a.hops,
            experts_remote: a.experts,
            experts_fallback: a.fallback,
            hop_wall_ms: a.wall_ns as f64 / 1e6,
            hop_net_ms: a.net_ns as f64 / 1e6,
            host_ms: a.host_ns as f64 / 1e6,
            down: inner.links.iter().filter(|l| l.down.is_some()).map(|l| l.spec.name.clone()).collect(),
        };
        if let Some(f) = inner.log.as_mut() {
            if let Ok(line) = serde_json::to_string(&rec) {
                let _ = writeln!(f, "{line}");
            }
        }
        // a harness drains these; a server that never does keeps the last few thousand
        if inner.records.len() >= 8192 {
            inner.records.drain(..4096);
        }
        inner.records.push(rec);
    }

    pub fn take_records(&self) -> Vec<PassRecord> {
        std::mem::take(&mut self.inner.lock().unwrap().records)
    }

    pub fn hosts(&self) -> Vec<HostState> {
        let inner = self.inner.lock().unwrap();
        inner
            .links
            .iter()
            .map(|l| HostState {
                name: l.spec.name.clone(),
                addr: l.spec.addr.clone(),
                up: l.down.is_none() && l.client.is_some(),
                reason: l.down.as_ref().map(|(_, r)| r.clone()),
                transport: l.client.as_ref().map(|c| c.transport().to_string()),
                backend: l.client.as_ref().map(|c| c.info.backend.clone()),
                arith_match: l.client.as_ref().filter(|c| c.info.backend == "cpu").map(|c| c.info.arith == lb_expert_host::expert::arith_fingerprint()),
                hops: l.hops,
                experts: l.experts,
            })
            .collect()
    }

    /// Experts the current map places on a host.
    pub fn n_placed(&self) -> usize {
        self.inner.lock().unwrap().map.n_remote()
    }

    fn mark_down(&self, link: &mut Link, why: String) {
        eprintln!("lb-lfm2 w2w: host {} marked down, its experts run locally from here: {why}", link.spec.name);
        link.client = None;
        link.down = Some((Instant::now(), why));
        self.stats.mark_downs.fetch_add(1, Ordering::Relaxed);
    }

    /// Serve what the placement puts on hosts. `experts` is the layer's calls in ascending
    /// expert id; the ones a host answered are removed from it and returned with their outputs,
    /// everything else stays for the local path. `x` is `m` rows of `d`.
    pub fn serve_layer(&self, layer: usize, experts: &mut Vec<ExpertCall>, x: &[f32], m: usize) -> Vec<ExpertPart> {
        let d = self.d;
        debug_assert_eq!(x.len(), m * d);
        let mut inner = self.inner.lock().unwrap();
        let inner = &mut *inner;
        if layer >= inner.map.assign.len() || inner.links.is_empty() {
            return Vec::new();
        }
        let t0 = Instant::now();
        // split: one batch per host that is up, in host order; the rest stays local
        let mut batches: Vec<Batch> = Vec::new();
        let mut local: Vec<ExpertCall> = Vec::with_capacity(experts.len());
        let mut fallback = 0u64;
        for call in experts.drain(..) {
            let e = call.0;
            let h = inner.map.assign[layer].get(e).copied().unwrap_or(LOCAL);
            if h == LOCAL {
                local.push(call);
                continue;
            }
            let up = inner.links[h as usize].client.is_some();
            let refused = match inner.refused.get(&(h, layer as u16, e as u16)) {
                Some(at) if at.elapsed() < REFUSAL_TTL => true,
                Some(_) => {
                    inner.refused.remove(&(h, layer as u16, e as u16));
                    false
                }
                None => false,
            };
            if !up || refused {
                fallback += 1;
                local.push(call);
                continue;
            }
            let b = match batches.iter().position(|b| b.host == h as usize) {
                Some(i) => &mut batches[i],
                None => {
                    batches.push(Batch { host: h as usize, ids: Vec::new(), groups: Vec::new(), calls: Vec::new(), rows: 0, sent: None });
                    batches.last_mut().unwrap()
                }
            };
            b.ids.push(e as u16);
            b.calls.push(call);
        }
        if batches.is_empty() {
            *experts = local;
            if fallback > 0 {
                inner.acc.fallback += fallback;
                self.stats.fallback_experts.fetch_add(fallback, Ordering::Relaxed);
            }
            return Vec::new();
        }
        // send every batch before reading any reply, so the hosts work side by side
        for b in &mut batches {
            // the rows this host needs, compacted (a decode token: row 0, sent as it is)
            let mut used: Vec<usize> = b.calls.iter().flat_map(|(_, toks)| toks.iter().map(|(t, _)| *t)).collect();
            used.sort_unstable();
            used.dedup();
            let pos: HashMap<usize, u32> = used.iter().enumerate().map(|(i, &r)| (r, i as u32)).collect();
            b.groups = b.calls.iter().map(|(_, toks)| toks.iter().map(|(t, _)| pos[t]).collect()).collect();
            b.rows = used.len();
            let xs_owned: Vec<f32>;
            let xs: &[f32] = if used.len() == m {
                x
            } else {
                let mut v = Vec::with_capacity(used.len() * d);
                for &r in &used {
                    v.extend_from_slice(&x[r * d..(r + 1) * d]);
                }
                xs_owned = v;
                &xs_owned
            };
            let groups: Vec<&[u32]> = b.groups.iter().map(|g| g.as_slice()).collect();
            let link = &mut inner.links[b.host];
            let sent = Instant::now();
            let r = link.client.as_mut().expect("up").send_run(layer as u16, &b.ids, &groups, xs, b.rows, d);
            link.hops += 1;
            inner.acc.hops += 1;
            self.stats.hops.fetch_add(1, Ordering::Relaxed);
            match r {
                Ok(()) => b.sent = Some(sent),
                Err(e) => self.mark_down(link, format!("send failed: {e}")),
            }
        }
        // collect
        let mut parts: Vec<ExpertPart> = Vec::new();
        for b in batches {
            let link = &mut inner.links[b.host];
            let failed = |calls: Vec<ExpertCall>, local: &mut Vec<ExpertCall>, acc: &mut Acc| {
                let n = calls.len() as u64;
                acc.fallback += n;
                self.stats.fallback_experts.fetch_add(n, Ordering::Relaxed);
                local.extend(calls);
            };
            let Some(sent) = b.sent else {
                failed(b.calls, &mut local, &mut inner.acc);
                continue;
            };
            let deadline = sent + Duration::from_millis(link.spec.deadline_ms + link.spec.deadline_row_ms * (b.rows as u64).saturating_sub(1));
            let reply = link.client.as_mut().expect("sent").recv_run(Some(deadline));
            let rt = sent.elapsed().as_nanos() as u64;
            self.stats.rt_ns.fetch_add(rt, Ordering::Relaxed);
            match reply {
                Ok(RunReply::Ok { host_ns, n, d: rd, y }) => {
                    let want: usize = b.groups.iter().map(|g| g.len()).sum::<usize>() * d;
                    if n as usize != b.ids.len() || rd as usize != d || y.len() != want {
                        self.mark_down(link, format!("malformed reply for layer {layer}: {n} experts, width {rd}, {} values (wanted {} experts, {want} values)", y.len(), b.ids.len()));
                        failed(b.calls, &mut local, &mut inner.acc);
                        continue;
                    }
                    inner.acc.net_ns += rt.saturating_sub(host_ns);
                    inner.acc.host_ns += host_ns;
                    inner.acc.experts += b.ids.len() as u64;
                    link.experts += b.ids.len() as u64;
                    self.stats.host_ns.fetch_add(host_ns, Ordering::Relaxed);
                    self.stats.experts.fetch_add(b.ids.len() as u64, Ordering::Relaxed);
                    let mut at = 0;
                    for (e, toks) in b.calls {
                        let len = toks.len() * d;
                        parts.push((e, toks, y[at..at + len].to_vec()));
                        at += len;
                    }
                }
                Ok(RunReply::Refused { ids, .. }) => {
                    // the host is fine, the placement is ahead of (or behind) it: believe it
                    self.stats.refusals.fetch_add(ids.len() as u64, Ordering::Relaxed);
                    let now = Instant::now();
                    for id in ids {
                        inner.refused.insert((b.host as u16, layer as u16, id), now);
                    }
                    failed(b.calls, &mut local, &mut inner.acc);
                }
                Ok(RunReply::Error(msg)) => {
                    self.mark_down(link, format!("host error on layer {layer}: {msg}"));
                    failed(b.calls, &mut local, &mut inner.acc);
                }
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::TimedOut {
                        self.stats.timeouts.fetch_add(1, Ordering::Relaxed);
                        self.mark_down(link, format!("no reply for layer {layer} inside {} ms", deadline.duration_since(sent).as_millis()));
                    } else {
                        self.mark_down(link, format!("connection lost on layer {layer}: {e}"));
                    }
                    failed(b.calls, &mut local, &mut inner.acc);
                }
            }
        }
        if fallback > 0 {
            inner.acc.fallback += fallback;
            self.stats.fallback_experts.fetch_add(fallback, Ordering::Relaxed);
        }
        // what stays local goes back in ascending expert id, the order the local path expects
        local.sort_by_key(|(e, _)| *e);
        *experts = local;
        let wall = t0.elapsed().as_nanos() as u64;
        inner.acc.wall_ns += wall;
        self.stats.wall_ns.fetch_add(wall, Ordering::Relaxed);
        parts
    }
}

impl RemoteTier {
    /// Counters since start and the hosts' state, for a status endpoint.
    pub fn status(&self) -> serde_json::Value {
        let o = Ordering::Relaxed;
        let s = &self.stats;
        let hops = s.hops.load(o);
        let ms = |ns: u64| (ns as f64 / 1e3).round() / 1e3;
        serde_json::json!({
            "schema": "lemonade-burn.w2w.v1",
            "placement_file": self.path.as_ref().map(|p| p.display().to_string()),
            "experts_placed_on_hosts": self.n_placed(),
            "hops": hops, "experts_remote": s.experts.load(o), "experts_fallback": s.fallback_experts.load(o),
            "refusals": s.refusals.load(o), "timeouts": s.timeouts.load(o), "mark_downs": s.mark_downs.load(o), "reloads": s.reloads.load(o),
            "round_trip_ms_mean": if hops > 0 { ms(s.rt_ns.load(o) / hops) } else { 0.0 },
            "host_compute_ms_mean": if hops > 0 { ms(s.host_ns.load(o) / hops) } else { 0.0 },
            "wall_ms_total": ms(s.wall_ns.load(o)),
            "hosts": self.hosts(),
        })
    }
}
