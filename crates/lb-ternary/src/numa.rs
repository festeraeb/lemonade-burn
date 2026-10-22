//! Two-socket (NUMA) execution: one rayon pool per memory node, its threads restricted to
//! that node's CPUs, and helpers to place a buffer on a node by first touch.
//!
//! Enabled with `LB_NUMA="<cpus node 0>;<cpus node 1>"`, e.g. `LB_NUMA="0-19;20-39"`.
//! Weights are split by rows between the nodes (`LB_NUMA_SHARE`, the fraction of rows on
//! the first node, default 0.5); each part is copied by a thread of its node so its pages
//! are allocated there (Linux first-touch; do not run under `numactl --membind`). A GEMV
//! then streams each part from local memory on its own node's cores.

use rayon::ThreadPool;
use std::sync::OnceLock;

pub struct NodePools {
    pools: Vec<ThreadPool>,
    /// Fraction of rows placed on node 0 (the rest on node 1).
    pub share: f64,
}

/// Parse a CPU list like `0-19,40-59`.
pub fn parse_cpus(s: &str) -> Option<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
                if a > b {
                    return None;
                }
                v.extend(a..=b);
            }
            None => v.push(part.parse().ok()?),
        }
    }
    (!v.is_empty()).then_some(v)
}

fn pin_current_thread(cpus: &[usize]) {
    use nix::sched::{sched_setaffinity, CpuSet};
    use nix::unistd::Pid;
    let mut set = CpuSet::new();
    for &c in cpus {
        let _ = set.set(c);
    }
    if let Err(e) = sched_setaffinity(Pid::from_raw(0), &set) {
        eprintln!("lb-ternary numa: could not pin thread to {cpus:?}: {e}");
    }
}

impl NodePools {
    /// One pool per CPU set, one thread per CPU, each thread restricted to its set.
    pub fn new(sets: &[Vec<usize>], share: f64) -> Self {
        assert!(sets.len() == 2, "NodePools supports exactly two nodes");
        let pools = sets
            .iter()
            .enumerate()
            .map(|(n, cpus)| {
                let c = cpus.clone();
                rayon::ThreadPoolBuilder::new()
                    .num_threads(cpus.len())
                    .thread_name(move |i| format!("lb-node{n}-{i}"))
                    .start_handler(move |_| pin_current_thread(&c))
                    .build()
                    .expect("build node pool")
            })
            .collect();
        Self { pools, share: share.clamp(0.0, 1.0) }
    }

    /// The process-wide pools from `LB_NUMA` / `LB_NUMA_SHARE`, if set.
    pub fn global() -> Option<&'static NodePools> {
        static P: OnceLock<Option<NodePools>> = OnceLock::new();
        P.get_or_init(|| {
            let spec = std::env::var("LB_NUMA").ok()?;
            let sets: Option<Vec<Vec<usize>>> = spec.split(';').map(parse_cpus).collect();
            let sets = match sets {
                Some(s) if s.len() == 2 => s,
                _ => {
                    eprintln!("lb-ternary numa: LB_NUMA must be '<cpus>;<cpus>', got {spec:?}; NUMA split disabled");
                    return None;
                }
            };
            let share = std::env::var("LB_NUMA_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
            Some(NodePools::new(&sets, share))
        })
        .as_ref()
    }

    pub fn pool(&self, node: usize) -> &ThreadPool {
        &self.pools[node]
    }

    /// Rows of an `n`-row matrix that go to node 0.
    pub fn split_rows(&self, n: usize) -> usize {
        ((n as f64 * self.share).round() as usize).min(n)
    }

    /// Copy `src` into a new buffer whose pages are first touched (allocated) on `node`.
    pub fn place<T: Clone + Send + Sync>(&self, node: usize, src: &[T]) -> Vec<T> {
        self.pools[node].install(|| src.to_vec())
    }

    /// Run `a` on node 0's pool and `b` on node 1's pool concurrently.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.pools[0].install(|| rayon::join(a, || self.pools[1].install(b)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu_lists() {
        assert_eq!(parse_cpus("0-3,8"), Some(vec![0, 1, 2, 3, 8]));
        assert_eq!(parse_cpus(" 5 "), Some(vec![5]));
        assert_eq!(parse_cpus("3-1"), None);
        assert_eq!(parse_cpus(""), None);
    }

    #[test]
    fn join_runs_both_sides() {
        let p = NodePools::new(&[vec![0], vec![0]], 0.5);
        let (a, b) = p.join(|| 1 + 1, || 2 + 2);
        assert_eq!((a, b), (2, 4));
        assert_eq!(p.split_rows(10), 5);
        assert_eq!(p.place(1, &[1u8, 2, 3]), vec![1, 2, 3]);
    }
}
