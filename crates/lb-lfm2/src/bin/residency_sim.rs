//! Offline residency-policy simulator (docs/ENGINE_RESIDENCY.md). Replays `LB_LA_TRACE` files (one
//! line per layer pass: kind "p" prefill / "d" decode, layer, needed experts) against cache policies
//! with the same number of expert slots and counts decode misses. Misses are routings of a decode
//! pass whose expert is not in the cache at that moment; hit rate = 1 - misses / routings.
//! usage: residency-sim --scrapes DIR --train A.jsonl --test B.jsonl [--test ...] [--budgets 5120,3072]
//! Policies: slru (the real `Slru`), belady (oracle, must admit), belady-bypass (oracle, may refuse
//! to admit), static (top frequency of the TRAIN trace, never evicts), lfu-decay (half-life sweep,
//! admit always / admit-if-hotter), quota (per-layer SLRU, equal or train-proportional slots),
//! topic-seed (SLRU, reseeded from each request's prefill routes).
use lb_lfm2::cache::Slru;
use lb_lfm2::moe::{read_scrape_dir, Histogram};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct Pass {
    decode: bool,
    layer: usize,
    need: Vec<usize>,
}

fn load(path: &str) -> Vec<Pass> {
    let txt = std::fs::read_to_string(path).expect("trace file");
    txt.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let v: Value = serde_json::from_str(line).expect("trace line");
            let u = |x: &Value| x.as_u64().unwrap() as usize;
            Pass { decode: v["k"] == "d", layer: u(&v["l"]), need: v["need"].as_array().unwrap().iter().map(u).collect() }
        })
        .collect()
}

const N_EXPERT: usize = 64;

#[derive(Default, Clone, Debug)]
struct Out {
    routings: u64,
    misses: u64,
    tokens: u64,
    copies_extra: u64,
}

impl Out {
    fn per_tok(&self) -> f64 {
        self.misses as f64 / self.tokens.max(1) as f64
    }
    fn hit(&self) -> f64 {
        1.0 - self.misses as f64 / self.routings.max(1) as f64
    }
}

fn key(l: usize, e: usize) -> usize {
    l * N_EXPERT + e
}

fn moe_layers(p: &[Pass]) -> usize {
    p.iter().filter(|x| x.decode).map(|x| x.layer).collect::<HashSet<_>>().len().max(1)
}

fn count(o: &mut Out, p: &Pass, misses: usize, nl: usize, first: usize) {
    if p.decode {
        o.routings += p.need.len() as u64;
        o.misses += misses as u64;
        if p.layer == first {
            o.tokens += 1;
        }
    }
    let _ = nl;
}

fn seed_slru(c: &mut Slru, order: &[(usize, usize)], slots: usize) {
    for &(l, e) in order.iter().take(slots).rev() {
        c.seed(l, e);
    }
}

fn run_slru(passes: &[Pass], nlay: usize, slots: usize, order: &[(usize, usize)]) -> Out {
    let mut c = Slru::new(nlay, N_EXPERT, slots);
    seed_slru(&mut c, order, slots);
    let first = passes.iter().find(|p| p.decode).map(|p| p.layer).unwrap_or(0);
    let mut o = Out::default();
    for p in passes {
        let plan = c.plan(p.layer, &p.need).expect("plan");
        count(&mut o, p, plan.fills.len(), nlay, first);
    }
    o
}

fn run_belady(passes: &[Pass], nlay: usize, slots: usize, order: &[(usize, usize)], bypass: bool) -> Out {
    let mut uses: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, p) in passes.iter().enumerate() {
        for &e in &p.need {
            uses.entry(key(p.layer, e)).or_default().push(i);
        }
    }
    let next_use = |k: usize, i: usize| -> usize {
        match uses.get(&k) {
            Some(v) => match v.binary_search(&(i + 1)) {
                Ok(j) | Err(j) => v.get(j).copied().unwrap_or(usize::MAX),
            },
            None => usize::MAX,
        }
    };
    let mut cache: HashSet<usize> = order.iter().take(slots).map(|&(l, e)| key(l, e)).collect();
    let first = passes.iter().find(|p| p.decode).map(|p| p.layer).unwrap_or(0);
    let mut o = Out::default();
    for (i, p) in passes.iter().enumerate() {
        let cur: HashSet<usize> = p.need.iter().map(|&e| key(p.layer, e)).collect();
        let mut miss = 0;
        for &e in &p.need {
            let k = key(p.layer, e);
            if cache.contains(&k) {
                continue;
            }
            miss += 1;
            if cache.len() < slots {
                cache.insert(k);
                continue;
            }
            let (mut vk, mut vn) = (usize::MAX, 0usize);
            for &r in &cache {
                if cur.contains(&r) {
                    continue;
                }
                let n = next_use(r, i);
                if vk == usize::MAX || n > vn {
                    vk = r;
                    vn = n;
                }
            }
            if vk == usize::MAX {
                continue;
            }
            if bypass && next_use(k, i) >= vn {
                continue;
            }
            cache.remove(&vk);
            cache.insert(k);
        }
        count(&mut o, p, miss, nlay, first);
    }
    o
}

fn freq(train: &[Pass]) -> HashMap<usize, u64> {
    let mut f = HashMap::new();
    for p in train.iter().filter(|p| p.decode) {
        for &e in &p.need {
            *f.entry(key(p.layer, e)).or_insert(0u64) += 1;
        }
    }
    f
}

fn run_static(passes: &[Pass], nlay: usize, slots: usize, f: &HashMap<usize, u64>) -> Out {
    let mut v: Vec<(&usize, &u64)> = f.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let set: HashSet<usize> = v.iter().take(slots).map(|(k, _)| **k).collect();
    let first = passes.iter().find(|p| p.decode).map(|p| p.layer).unwrap_or(0);
    let mut o = Out::default();
    for p in passes {
        let miss = p.need.iter().filter(|&&e| !set.contains(&key(p.layer, e))).count();
        count(&mut o, p, miss, nlay, first);
    }
    o
}

/// Decayed LFU. Time is the decode token index; a score halves every `half` tokens.
fn run_lfu(passes: &[Pass], nlay: usize, slots: usize, order: &[(usize, usize)], half: f64, admit_hotter: bool) -> Out {
    let first = passes.iter().find(|p| p.decode).map(|p| p.layer).unwrap_or(0);
    let lam = if half.is_finite() { (0.5f64).ln() / half } else { 0.0 };
    // key -> (score at last, last time)
    let mut score: HashMap<usize, (f64, f64)> = HashMap::new();
    let mut cache: HashSet<usize> = HashSet::new();
    for &(l, e) in order.iter().take(slots) {
        cache.insert(key(l, e));
        score.insert(key(l, e), (0.5, 0.0));
    }
    let mut t = 0.0f64;
    let mut o = Out::default();
    let now = |s: &(f64, f64), t: f64| s.0 * (lam * (t - s.1)).exp();
    for p in passes {
        if p.decode && p.layer == first {
            t += 1.0;
        }
        let cur: HashSet<usize> = p.need.iter().map(|&e| key(p.layer, e)).collect();
        let mut miss = 0;
        for &e in &p.need {
            let k = key(p.layer, e);
            let s = score.entry(k).or_insert((0.0, t));
            *s = (now(s, t) + 1.0, t);
            if cache.contains(&k) {
                continue;
            }
            miss += 1;
            if cache.len() < slots {
                cache.insert(k);
                continue;
            }
            let mut best: Option<(usize, f64)> = None;
            for &r in &cache {
                if cur.contains(&r) {
                    continue;
                }
                let sc = now(&score[&r], t);
                if best.map_or(true, |(_, b)| sc < b) {
                    best = Some((r, sc));
                }
            }
            if let Some((vk, vs)) = best {
                if admit_hotter && now(&score[&k], t) <= vs {
                    continue;
                }
                cache.remove(&vk);
                cache.insert(k);
            }
        }
        count(&mut o, p, miss, nlay, first);
    }
    o
}

/// Per-layer SLRU: layer l owns `quota[l]` slots.
fn run_quota(passes: &[Pass], nlay: usize, quota: &[usize], order: &[(usize, usize)]) -> Out {
    let mut caches: Vec<Slru> = quota.iter().map(|&q| Slru::new(nlay, N_EXPERT, q.max(4))).collect();
    let mut seeded = vec![0usize; nlay];
    let mut per: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nlay];
    for &(l, e) in order {
        per[l].push((l, e));
    }
    for l in 0..nlay {
        let q = quota[l].max(4);
        for &(l2, e) in per[l].iter().take(q).rev() {
            caches[l].seed(l2, e);
            seeded[l] += 1;
        }
    }
    let first = passes.iter().find(|p| p.decode).map(|p| p.layer).unwrap_or(0);
    let mut o = Out::default();
    for p in passes {
        // a pass needing more distinct experts than the layer's quota (a prefill chunk) cannot be planned
        // in its own share: it streams through without being kept (every expert is a miss)
        let miss = match caches[p.layer].plan(p.layer, &p.need) {
            Some(plan) => plan.fills.len(),
            None => p.need.len(),
        };
        count(&mut o, p, miss, nlay, first);
    }
    o
}

/// SLRU reseeded at each request from that request's prefill routes: the `m` most-used (layer, expert)
/// keys of the prompt are seeded (hottest last) into the cache before decode. `copies_extra` counts
/// the seeds that were not already resident (each is a copy that the plain policy would not make).
fn run_topic(passes: &[Pass], nlay: usize, slots: usize, order: &[(usize, usize)], m: usize) -> Out {
    let mut c = Slru::new(nlay, N_EXPERT, slots);
    seed_slru(&mut c, order, slots);
    let first = passes.iter().find(|p| p.decode).map(|p| p.layer).unwrap_or(0);
    let mut o = Out::default();
    let mut i = 0;
    while i < passes.len() {
        if !passes[i].decode {
            let mut j = i;
            let mut f: HashMap<(usize, usize), u64> = HashMap::new();
            while j < passes.len() && !passes[j].decode {
                let plan = c.plan(passes[j].layer, &passes[j].need).expect("plan");
                let _ = plan;
                for &e in &passes[j].need {
                    *f.entry((passes[j].layer, e)).or_insert(0) += 1;
                }
                j += 1;
            }
            let mut v: Vec<((usize, usize), u64)> = f.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            for &((l, e), _) in v.iter().take(m).rev() {
                if c.seed(l, e).is_some() {
                    o.copies_extra += 1;
                }
            }
            i = j;
        } else {
            let plan = c.plan(passes[i].layer, &passes[i].need).expect("plan");
            count(&mut o, &passes[i], plan.fills.len(), nlay, first);
            i += 1;
        }
    }
    o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let all = |k: &str| a.iter().enumerate().filter(|(_, x)| *x == k).map(|(i, _)| a[i + 1].clone()).collect::<Vec<_>>();
    let scr = read_scrape_dir(std::path::Path::new(&arg("--scrapes").expect("--scrapes"))).expect("scrapes");
    let train_path = arg("--train").expect("--train");
    let tests = all("--test");
    let budgets: Vec<usize> = arg("--budgets").unwrap_or("5120,3072".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let train = load(&train_path);
    let nlay = train.iter().map(|p| p.layer).max().unwrap() + 1;
    let hist = Histogram::from_scrapes("lfm2", nlay, N_EXPERT, &scr);
    let order = hist.heat_order();
    let f_train = freq(&train);
    let nml = moe_layers(&train);
    let mut used = vec![0usize; nlay];
    for k in f_train.keys() {
        used[k / N_EXPERT] += 1;
    }
    println!("train {train_path}; layers {nlay} ({nml} MoE); tests {tests:?}; seed = scrape heat order\n");
    for &b in &budgets {
        let slots = ((b as f64 * 1048576.0 - 8.0 * 5308416.0) / 5308416.0) as usize;
        println!("=== budget {b} MiB = {slots} slots (of {} experts)", nml * N_EXPERT);
        for t in &tests {
            let test = load(t);
            let row = |name: &str, o: &Out| println!("  {:<34} hit {:.4}  misses/token {:6.2}  ({} tokens){}", name, o.hit(), o.per_tok(), o.tokens, if o.copies_extra > 0 { format!("  extra seed copies/request {:.0}", o.copies_extra as f64 / 5.0) } else { String::new() });
            println!(" test {t}{}", if *t == train_path { "  (SAME as train)" } else { "  (held out)" });
            row("slru (today)", &run_slru(&test, nlay, slots, &order));
            row("belady (oracle, must admit)", &run_belady(&test, nlay, slots, &order, false));
            row("belady-bypass (oracle)", &run_belady(&test, nlay, slots, &order, true));
            row("static top-freq of train", &run_static(&test, nlay, slots, &f_train));
            for h in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 256.0, 1024.0, f64::INFINITY] {
                row(&format!("lfu-decay half-life {h} admit-all"), &run_lfu(&test, nlay, slots, &order, h, false));
                row(&format!("lfu-decay half-life {h} admit-hotter"), &run_lfu(&test, nlay, slots, &order, h, true));
            }
            let eq: Vec<usize> = (0..nlay).map(|_| slots / nml).collect();
            let tot: usize = used.iter().sum();
            let prop: Vec<usize> = (0..nlay).map(|l| (slots * used[l] / tot.max(1)).min(N_EXPERT)).collect();
            row("per-layer quota SLRU, equal", &run_quota(&test, nlay, &eq, &order));
            row("per-layer quota SLRU, train-proportional", &run_quota(&test, nlay, &prop, &order));
            for m in [slots / 4, slots / 2] {
                row(&format!("topic-seed top {m} prompt keys"), &run_topic(&test, nlay, slots, &order, m));
            }
        }
        println!();
    }
}
