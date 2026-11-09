//! Offline look-ahead simulator (engine step E3, `docs/ENGINE_E3_LOOKAHEAD.md`): replays a
//! `LB_LA_TRACE` file against the real [`Slru`] with the keep guard and the miss-only prefetch model,
//! and against Belady's optimum as the ceiling. Hit rate only; tok/s needs the card.
//! usage: la-sim <trace.jsonl> [slots ...]
use lb_lfm2::cache::Slru;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

/// One layer pass from the trace, in decode ("d") or prefill ("p").
struct Pass {
    decode: bool,
    layer: usize,
    need: Vec<usize>,
    /// Guess of a coming layer: (target layer, experts, scores, ref score).
    guess: Option<(usize, Vec<usize>, Vec<f32>, f32)>,
}

fn load(path: &str) -> Vec<Pass> {
    let txt = std::fs::read_to_string(path).expect("trace file");
    txt.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let v: Value = serde_json::from_str(line).expect("trace line");
            let u = |x: &Value| x.as_u64().unwrap() as usize;
            let need = v["need"].as_array().unwrap().iter().map(u).collect();
            let guess = v.get("g").filter(|g| !g.is_null()).map(|g| {
                (
                    u(&g["t"]),
                    g["e"].as_array().unwrap().iter().map(u).collect(),
                    g["s"].as_array().unwrap().iter().map(|s| s.as_f64().unwrap() as f32).collect(),
                    g["r"].as_f64().unwrap_or(f64::NEG_INFINITY) as f32,
                )
            });
            Pass { decode: v["k"] == "d", layer: u(&v["l"]), need, guess }
        })
        .collect()
}

#[derive(Default, Clone, Debug)]
struct Tally {
    decode_tokens: u64,
    routings: u64,
    misses: u64,
    /// Misses the prefetch already had in flight (bus hidden).
    covered: u64,
    issued: u64,
    wasted: u64,
    spared: u64,
}

impl Tally {
    fn show(&self, name: &str, n_tok: u64) -> String {
        let tok = n_tok.max(1) as f64;
        let hit = 1.0 - self.misses as f64 / self.routings.max(1) as f64;
        let uncovered = (self.misses - self.covered) as f64 / tok;
        format!(
            "{name:<32} hit {hit:.4} misses/tok {:6.2} covered/tok {:6.2} left-on-bus/tok {uncovered:6.2} issued/tok {:6.2} wasted/tok {:5.2} spared/tok {:5.2}",
            self.misses as f64 / tok,
            self.covered as f64 / tok,
            self.issued as f64 / tok,
            self.wasted as f64 / tok,
            self.spared as f64 / tok
        )
    }
}

/// Run one policy. `keep_n`: spare the top `keep_n` guesses (0 = off). `promote`: 0/1/2 as
/// `LB_LA_PROMOTE`. `cap`/`conf`: miss-only prefetch of guesses not cached, confidence >= `conf`.
#[allow(clippy::too_many_arguments)]
fn run(passes: &[Pass], n_layer: usize, n_expert: usize, slots: usize, keep_n: usize, promote: u8, cap: usize, conf: f32) -> Tally {
    run_w(passes, n_layer, n_expert, slots, keep_n, promote, cap, conf, 0)
}

/// [`run`] with a cycle-aware eviction window.
#[allow(clippy::too_many_arguments)]
fn run_w(passes: &[Pass], n_layer: usize, n_expert: usize, slots: usize, keep_n: usize, promote: u8, cap: usize, conf: f32, window: usize) -> Tally {
    let mut c = Slru::new(n_layer, n_expert, slots);
    let mut t = Tally::default();
    // expert copies in flight for the coming layer
    let mut inflight: Vec<(usize, usize)> = Vec::new();
    for p in passes {
        let keep: Vec<(usize, usize)> = match (keep_n, &p.guess) {
            (n, Some((t, e, _, _))) if n > 0 => e.iter().take(n).map(|&x| (*t, x)).collect(),
            _ => Vec::new(),
        };
        let (plan, spared) = c.plan_la(p.layer, &p.need, &keep, window, n_layer).expect("plan");
        if p.decode {
            t.decode_tokens += u64::from(p.layer == first_moe(passes));
            t.routings += p.need.len() as u64;
            t.misses += plan.fills.len() as u64;
            t.spared += spared;
            for &(e, _) in &plan.fills {
                if inflight.contains(&(p.layer, e)) {
                    t.covered += 1;
                }
            }
            t.wasted += inflight.iter().filter(|(l, e)| *l == p.layer && !p.need.contains(e)).count() as u64;
        }
        inflight.retain(|(l, _)| *l != p.layer);
        if let (true, Some((tl, ex, sc, r))) = (p.decode, &p.guess) {
            if promote > 0 {
                for &e in ex {
                    c.promote(*tl, e, promote > 1);
                }
            }
            let mut n = 0;
            for (i, &e) in ex.iter().enumerate() {
                if n >= cap || sc[i] - *r < conf {
                    break;
                }
                if c.slot(*tl, e).is_some() {
                    continue;
                }
                inflight.push((*tl, e));
                n += 1;
                t.issued += 1;
            }
        }
    }
    t
}

fn first_moe(passes: &[Pass]) -> usize {
    passes.iter().filter(|p| p.decode).map(|p| p.layer).min().unwrap_or(0)
}

/// Belady's optimum over the same accesses (a pass's experts cannot evict each other).
fn belady(passes: &[Pass], n_expert: usize, slots: usize) -> Tally {
    let key = |l: usize, e: usize| l * n_expert + e;
    let mut next: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, p) in passes.iter().enumerate() {
        for &e in &p.need {
            next.entry(key(p.layer, e)).or_default().push(i);
        }
    }
    let mut cur: HashMap<usize, usize> = next.keys().map(|&k| (k, 0)).collect();
    let next_use = |k: usize, now: usize, cur: &mut HashMap<usize, usize>, next: &HashMap<usize, Vec<usize>>| -> usize {
        let v = &next[&k];
        let c = cur.get_mut(&k).unwrap();
        while *c < v.len() && v[*c] <= now {
            *c += 1;
        }
        v.get(*c).copied().unwrap_or(usize::MAX)
    };
    let mut held: HashMap<usize, usize> = HashMap::new(); // key -> next use
    let mut by_next: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut t = Tally::default();
    for (i, p) in passes.iter().enumerate() {
        let mine: Vec<usize> = p.need.iter().map(|&e| key(p.layer, e)).collect();
        for &k in &mine {
            let n = next_use(k, i, &mut cur, &next);
            if let Some(&old) = held.get(&k) {
                by_next.remove(&(old, k));
                held.insert(k, n);
                by_next.insert((n, k));
                if p.decode {
                    t.routings += 1;
                }
            } else {
                if held.len() >= slots {
                    // furthest next use among those this pass does not need
                    let victim = by_next.iter().rev().find(|(_, k2)| !mine.contains(k2)).copied();
                    if let Some(v) = victim {
                        by_next.remove(&v);
                        held.remove(&v.1);
                    }
                }
                held.insert(k, n);
                by_next.insert((n, k));
                if p.decode {
                    t.routings += 1;
                    t.misses += 1;
                }
            }
        }
    }
    t
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: la-sim <trace.jsonl> [slots ...]");
    let passes = load(path);
    let n_layer = passes.iter().map(|p| p.layer + 1).max().unwrap_or(1).max(40);
    let n_expert = passes.iter().flat_map(|p| p.need.iter().map(|e| e + 1)).max().unwrap_or(64).max(64);
    let tokens = passes.iter().filter(|p| p.decode && p.layer == first_moe(&passes)).count() as u64;
    println!("trace: {} passes, {tokens} decode tokens, {n_layer} layers, {n_expert} experts", passes.len());

    // calibration: P(guessed expert really routed at the target layer | confidence bucket) over the
    // decode passes that follow; and recall on the experts that miss (needs residency, below)
    let mut future: HashMap<usize, Vec<usize>> = HashMap::new();
    for p in &passes {
        if p.decode {
            future.entry(p.layer).or_default();
        }
    }
    let mut buckets: Vec<(f32, f32, u64, u64)> = vec![(0.2, f32::MAX, 0, 0), (0.1, 0.2, 0, 0), (0.05, 0.1, 0, 0), (0.0, 0.05, 0, 0), (-0.02, 0.0, 0, 0), (-0.05, -0.02, 0, 0), (-0.1, -0.05, 0, 0), (f32::MIN, -0.1, 0, 0)];
    for (i, p) in passes.iter().enumerate() {
        if !p.decode {
            continue;
        }
        if let Some((tl, ex, sc, r)) = &p.guess {
            let target = passes[i..].iter().find(|q| q.layer == *tl).map(|q| &q.need);
            if let Some(tn) = target {
                for (j, &e) in ex.iter().enumerate() {
                    let c = sc[j] - *r;
                    if let Some(b) = buckets.iter_mut().find(|b| c >= b.0 && c < b.1) {
                        b.2 += 1;
                        b.3 += u64::from(tn.contains(&e));
                    }
                }
            }
        }
    }
    println!("\nconfidence (score - score[k]) -> P(guessed expert routed):");
    for (lo, hi, n, y) in &buckets {
        println!("  [{lo:>6.2}, {hi:>6.2})  n={n:>7}  P={:.3}", *y as f64 / (*n).max(1) as f64);
    }

    let slot_list: Vec<usize> = if args.len() > 2 { args[2..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![1003, 598] };
    for &slots in &slot_list {
        println!("\n=== {slots} slots ===");
        println!("{}", run(&passes, n_layer, n_expert, slots, 0, 0, 0, 0.0).show("E2 SLRU", tokens));
        println!("{}", belady(&passes, n_expert, slots).show("Belady (ceiling, hit only)", tokens));
        for keep_n in [4usize, 8] {
            for promote in [0u8, 1, 2] {
                println!("{}", run(&passes, n_layer, n_expert, slots, keep_n, promote, 0, 0.0).show(&format!("keep{keep_n} promote{promote}"), tokens));
            }
        }
        for cap in [1usize, 2, 4, 8] {
            for conf in [f32::NEG_INFINITY, -0.05, 0.0, 0.05] {
                println!("{}", run(&passes, n_layer, n_expert, slots, 0, 0, cap, conf).show(&format!("prefetch cap{cap} conf>={conf}"), tokens));
            }
        }
        for cap in [1usize, 2, 4] {
            println!("{}", run(&passes, n_layer, n_expert, slots, 8, 0, cap, f32::NEG_INFINITY).show(&format!("keep8 + prefetch cap{cap}"), tokens));
        }
        for cap in [1usize, 2, 3, 4] {
            for conf in [-0.01f32, 0.0, 0.01, 0.02, 0.03, 0.05] {
                println!("{}", run(&passes, n_layer, n_expert, slots, 0, 0, cap, conf).show(&format!("prefetch cap{cap} conf>={conf}"), tokens));
            }
        }
        for w in [4usize, 16, 64, 200, 450] {
            println!("{}", run_w(&passes, n_layer, n_expert, slots, 0, 0, 0, 0.0, w).show(&format!("cycle window {w}"), tokens));
            println!("{}", run_w(&passes, n_layer, n_expert, slots, 8, 0, 0, 0.0, w).show(&format!("cycle window {w} + keep8"), tokens));
        }
    }
}
