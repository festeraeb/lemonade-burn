//! Persistent spinning worker pool for the decode hot path.
//!
//! Decode runs ~120 small parallel regions per token (fused q/k/v, o, gate/up, down per
//! layer, plus the LM head), separated by short single-threaded Burn ops. With rayon,
//! idle workers go to sleep in those gaps and every region pays a wake-up. Here workers
//! spin on an epoch counter for `LB_SPIN_US` (default 2000 us) after their last job
//! before parking on a condvar, so back-to-back regions start in well under a microsecond.
//!
//! Jobs are `'static` closures over chunk indices (callers capture `Arc`s), so no
//! lifetime tricks or `unsafe` are needed. Chunks are claimed dynamically from an atomic
//! counter, so a core that is busy with another process does not stall the region.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

type Task = Arc<dyn Fn(usize) + Send + Sync + 'static>;

struct Job {
    task: Task,
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    panicked: AtomicBool,
}

impl Job {
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n {
                return;
            }
            if catch_unwind(AssertUnwindSafe(|| (self.task)(i))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
            self.done.fetch_add(1, Ordering::Release);
        }
    }
}

struct Shared {
    epoch: AtomicU64,
    job: Mutex<Option<Arc<Job>>>,
    sleep_lock: Mutex<()>,
    wake: Condvar,
    sleepers: AtomicUsize,
    shutdown: AtomicBool,
    spin: Duration,
}

/// A fixed set of `threads - 1` worker threads; the calling thread is the last worker.
pub struct SpinPool {
    shared: Arc<Shared>,
    threads: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl SpinPool {
    pub fn new(threads: usize, spin: Duration) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            job: Mutex::new(None),
            sleep_lock: Mutex::new(()),
            wake: Condvar::new(),
            sleepers: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            spin,
        });
        let handles = (1..threads)
            .map(|i| {
                let sh = shared.clone();
                std::thread::Builder::new().name(format!("lb-spin-{i}")).spawn(move || worker(&sh)).expect("spawn pool worker")
            })
            .collect();
        Self { shared, threads, handles }
    }

    /// Process-wide pool: `LB_THREADS`, else `RAYON_NUM_THREADS`, else all CPUs;
    /// spin time `LB_SPIN_US` (default 2000).
    pub fn global() -> &'static SpinPool {
        static POOL: OnceLock<SpinPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0);
            let threads = env("LB_THREADS")
                .or_else(|| env("RAYON_NUM_THREADS"))
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            let spin = Duration::from_micros(env("LB_SPIN_US").unwrap_or(2000) as u64);
            SpinPool::new(threads, spin)
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `task(i)` for every `i in 0..n` across the pool; returns when all are done.
    /// Panics if any task panicked.
    pub fn run(&self, n: usize, task: impl Fn(usize) + Send + Sync + 'static) {
        if n == 0 {
            return;
        }
        let job = Arc::new(Job {
            task: Arc::new(task),
            n,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        });
        if self.threads > 1 && n > 1 {
            *self.shared.job.lock().unwrap() = Some(job.clone());
            self.shared.epoch.fetch_add(1, Ordering::SeqCst);
            if self.shared.sleepers.load(Ordering::SeqCst) > 0 {
                let _g = self.shared.sleep_lock.lock().unwrap();
                self.shared.wake.notify_all();
            }
        }
        job.work();
        while job.done.load(Ordering::Acquire) < n {
            std::hint::spin_loop();
        }
        if self.threads > 1 && n > 1 {
            // drop the pool's reference so captured buffers are freed promptly
            let mut slot = self.shared.job.lock().unwrap();
            if slot.as_ref().is_some_and(|j| Arc::ptr_eq(j, &job)) {
                *slot = None;
            }
        }
        assert!(!job.panicked.load(Ordering::Relaxed), "a SpinPool task panicked");
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.epoch.fetch_add(1, Ordering::SeqCst);
        {
            let _g = self.shared.sleep_lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker(sh: &Shared) {
    let mut seen = sh.epoch.load(Ordering::SeqCst);
    let mut last = Instant::now();
    let mut idle_iters = 0u32;
    loop {
        let e = sh.epoch.load(Ordering::Acquire);
        if sh.shutdown.load(Ordering::Relaxed) {
            return;
        }
        if e != seen {
            seen = e;
            let job = sh.job.lock().unwrap().clone();
            if let Some(j) = job {
                j.work();
            }
            last = Instant::now();
            idle_iters = 0;
            continue;
        }
        idle_iters += 1;
        if idle_iters % 256 != 0 || last.elapsed() < sh.spin {
            std::hint::spin_loop();
            continue;
        }
        // park until the epoch moves
        let mut g = sh.sleep_lock.lock().unwrap();
        sh.sleepers.fetch_add(1, Ordering::SeqCst);
        while sh.epoch.load(Ordering::SeqCst) == seen {
            g = sh.wake.wait(g).unwrap();
        }
        sh.sleepers.fetch_sub(1, Ordering::SeqCst);
        drop(g);
        last = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_chunk_once() {
        for threads in [1, 2, 5] {
            let pool = SpinPool::new(threads, Duration::from_micros(200));
            for round in 0..200 {
                let n = 1 + round % 37;
                let hits: Arc<Vec<AtomicUsize>> = Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
                let h = hits.clone();
                pool.run(n, move |i| {
                    h[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|c| c.load(Ordering::Relaxed) == 1), "threads {threads} round {round}");
                if round % 50 == 0 {
                    // let workers park, then make sure they wake up again
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        }
    }

    #[test]
    fn task_panic_is_reported() {
        let pool = SpinPool::new(3, Duration::from_micros(100));
        let r = catch_unwind(AssertUnwindSafe(|| pool.run(8, |i| assert!(i != 5))));
        assert!(r.is_err());
        // pool still usable
        let c = Arc::new(AtomicUsize::new(0));
        let c2 = c.clone();
        pool.run(10, move |_| {
            c2.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(c.load(Ordering::Relaxed), 10);
    }
}
