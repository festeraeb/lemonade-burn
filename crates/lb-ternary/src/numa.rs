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
    /// Node 0's pool; `None` means node 0 runs on rayon's global pool (pinned to node 0),
    /// which avoids one cross-pool hop per operation.
    pool0: Option<ThreadPool>,
    pool1: ThreadPool,
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
    fn builder(n: usize, cpus: &[usize]) -> rayon::ThreadPoolBuilder {
        let c = cpus.to_vec();
        rayon::ThreadPoolBuilder::new()
            .num_threads(cpus.len())
            .thread_name(move |i| format!("lb-node{n}-{i}"))
            .start_handler(move |_| pin_current_thread(&c))
    }

    /// Two private pools, one thread per CPU, each thread restricted to its set.
    pub fn new(sets: &[Vec<usize>], share: f64) -> Self {
        assert!(sets.len() == 2, "NodePools supports exactly two nodes");
        let pool0 = Some(Self::builder(0, &sets[0]).build().expect("build node pool"));
        let pool1 = Self::builder(1, &sets[1]).build().expect("build node pool");
        Self { pool0, pool1, share: share.clamp(0.0, 1.0) }
    }

    /// Like [`Self::new`] but node 0 becomes rayon's global pool (falls back to a private
    /// pool if the global pool already exists).
    fn new_with_global(sets: &[Vec<usize>], share: f64) -> Self {
        let pool1 = Self::builder(1, &sets[1]).build().expect("build node pool");
        let pool0 = match Self::builder(0, &sets[0]).build_global() {
            Ok(()) => None,
            Err(e) => {
                eprintln!("lb-ternary numa: global rayon pool already built ({e}); using a private node-0 pool");
                Some(Self::builder(0, &sets[0]).build().expect("build node pool"))
            }
        };
        Self { pool0, pool1, share: share.clamp(0.0, 1.0) }
    }

    /// Run `f` on node `node`'s pool.
    pub fn install<R: Send>(&self, node: usize, f: impl FnOnce() -> R + Send) -> R {
        match (node, &self.pool0) {
            (0, Some(p)) => p.install(f),
            // global pool: run here (the caller is the node-0 pinned main thread); its
            // parallel loops use the global = node-0 pool
            (0, None) => f(),
            _ => self.pool1.install(f),
        }
    }

    /// The process-wide pools from `LB_NUMA` / `LB_NUMA_SHARE`, if set. The thread that
    /// first calls this (the model loader / main thread, which runs the non-GEMM ops) is
    /// restricted to node 0's CPUs.
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
            pin_current_thread(&sets[0]);
            Some(NodePools::new_with_global(&sets, share))
        })
        .as_ref()
    }

    /// Rows of an `n`-row matrix that go to node 0.
    pub fn split_rows(&self, n: usize) -> usize {
        ((n as f64 * self.share).round() as usize).min(n)
    }

    /// Copy `src` into a new buffer whose pages are first touched (allocated) on `node`.
    pub fn place<T: Clone + Send + Sync>(&self, node: usize, src: &[T]) -> Vec<T> {
        self.install(node, || src.to_vec())
    }

    /// Run `a` on node 0 and `b` on node 1's pool concurrently. `b` is injected into node
    /// 1's pool straight from the calling thread; `a` runs on the calling thread (its
    /// parallel loops go to node 0's pool), then the call waits for `b`.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let mut rb = None;
        let ra = self.pool1.in_place_scope(|s| {
            s.spawn(|_| rb = Some(b()));
            self.install(0, a)
        });
        (ra, rb.expect("node 1 task finished"))
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
