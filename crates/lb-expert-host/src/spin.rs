//! A small spinning worker pool for the CPU-miss path (`LB_CPU_MISS`), and the batch expert
//! kernels that run on it.
//!
//! A decode token asks the CPU for a few experts in each of ~14 layers, 0.3-1 ms apart. A rayon
//! pool sleeps between those calls and pays a futex wake-up on every parallel region (and an expert
//! has several), which cost more than the matvec itself. Here the workers spin on one generation
//! counter for a while after the last job (so consecutive layers of a token find them awake), then
//! block on a condvar so an idle server burns nothing. One `run` is one barrier: tasks are handed
//! out through one atomic counter, the caller works too, and `run` returns when every worker is
//! done with it.

use lb_gguf::kernels_f32x::dot_q4_0_f32x_at;
use lb_gguf::kernels_i8::{dot_q4_0_i8_at, quantize_act_q8, simd_level};
use lb_gguf::{GgmlType, QuantMat};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

struct Shared {
    gen: AtomicUsize,
    next: AtomicUsize,
    n_tasks: AtomicUsize,
    pending: AtomicUsize,
    job_data: AtomicUsize,
    job_vt: AtomicUsize,
    sleepers: AtomicUsize,
    stop: AtomicBool,
    lock: Mutex<()>,
    cv: Condvar,
    spin_for: Duration,
}

pub struct SpinPool {
    shared: Arc<Shared>,
    n_workers: usize,
    // one caller at a time
    gate: Mutex<()>,
}

impl SpinPool {
    /// `threads` counts the caller, so `threads - 1` workers are started.
    pub fn new(threads: usize, spin_for: Duration) -> Self {
        // Never more threads than CPUs this process may run on (`available_parallelism` reads the scheduler affinity and
        // cgroup quota on Linux): spinning workers beyond that fight each other for cores and the pool collapses
        // (measured on c4, 64 threads pinned to one NUMA node: 3 tok/s against 27).
        let allowed = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(threads).max(1);
        if threads > allowed {
            eprintln!("lb-expert-host: spin pool clamped from {threads} to {allowed} threads (process CPU affinity)");
        }
        let threads = threads.min(allowed);
        let n_workers = threads.saturating_sub(1);
        let shared = Arc::new(Shared {
            gen: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            n_tasks: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            job_data: AtomicUsize::new(0),
            job_vt: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            spin_for,
        });
        for i in 0..n_workers {
            let s = shared.clone();
            std::thread::Builder::new().name(format!("lb-spin-{i}")).spawn(move || worker(s)).expect("spawn spin worker");
        }
        Self { shared, n_workers, gate: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.n_workers + 1
    }

    /// Run `f(0..n_tasks)` across the pool and the caller; returns when all tasks are done.
    pub fn run(&self, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_tasks == 0 {
            return;
        }
        if self.n_workers == 0 || n_tasks == 1 {
            (0..n_tasks).for_each(f);
            return;
        }
        let _g = self.gate.lock().expect("spin pool gate");
        let s = &*self.shared;
        // SAFETY: the pointer is only used until `pending` reaches 0 below, while `f` is borrowed.
        let (data, vt): (usize, usize) = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), (usize, usize)>(f) };
        s.job_data.store(data, Ordering::Relaxed);
        s.job_vt.store(vt, Ordering::Relaxed);
        s.next.store(0, Ordering::Relaxed);
        s.n_tasks.store(n_tasks, Ordering::Relaxed);
        s.pending.store(self.n_workers, Ordering::Relaxed);
        s.gen.fetch_add(1, Ordering::SeqCst);
        if s.sleepers.load(Ordering::SeqCst) > 0 {
            let _l = s.lock.lock().expect("lock");
            s.cv.notify_all();
        }
        loop {
            let t = s.next.fetch_add(1, Ordering::Relaxed);
            if t >= n_tasks {
                break;
            }
            f(t);
        }
        while s.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

fn worker(s: Arc<Shared>) {
    let mut seen = 0usize;
    loop {
        // wait for a new generation: spin, then block
        let mut t0: Option<Instant> = None;
        let mut spins = 0u32;
        loop {
            let g = s.gen.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if s.stop.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins & 255 == 0 {
                let t = *t0.get_or_insert_with(Instant::now);
                if t.elapsed() > s.spin_for {
                    let l = s.lock.lock().expect("lock");
                    s.sleepers.fetch_add(1, Ordering::SeqCst);
                    if s.gen.load(Ordering::SeqCst) == seen && !s.stop.load(Ordering::Relaxed) {
                        let _ = s.cv.wait_timeout(l, Duration::from_millis(50)).expect("cv");
                    }
                    s.sleepers.fetch_sub(1, Ordering::SeqCst);
                    t0 = None;
                }
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: published by `run` before the generation bump this worker just saw.
        let f: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute::<(usize, usize), &(dyn Fn(usize) + Sync)>((s.job_data.load(Ordering::Relaxed), s.job_vt.load(Ordering::Relaxed))) };
        let n = s.n_tasks.load(Ordering::Relaxed);
        loop {
            let t = s.next.fetch_add(1, Ordering::Relaxed);
            if t >= n {
                break;
            }
            f(t);
        }
        s.pending.fetch_sub(1, Ordering::Release);
    }
}

/// The process-wide pool: `LB_CPU_MISS_THREADS`, else `LB_THREADS`, else 20 threads.
pub fn global() -> &'static SpinPool {
    static P: OnceLock<SpinPool> = OnceLock::new();
    P.get_or_init(|| {
        let n = ["LB_CPU_MISS_THREADS", "LB_THREADS"].iter().find_map(|k| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok())).unwrap_or(20).max(1);
        let spin_ms = std::env::var("LB_CPU_MISS_SPIN_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(20u64);
        SpinPool::new(n, Duration::from_millis(spin_ms))
    })
}

struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

/// One SwiGLU expert per entry of `mats` on the same activation row `x` (`m == 1`), all of them in
/// two barriers on `pool`: gate and up with the SiLU product, then down. `int8` picks the
/// int8-activation kernel (activations rounded to 8 bits) instead of the exact f32 one. Returns
/// one `n_embd` row per expert. All matrices must be Q4_0 (the caller checks).
pub fn swiglu_batch(pool: &SpinPool, mats: &[(QuantMat<'_>, QuantMat<'_>, QuantMat<'_>)], x: &[f32], int8: bool) -> Vec<Vec<f32>> {
    let level = simd_level();
    let n = mats.len();
    let (n_ff, n_embd) = (mats[0].0.rows, mats[0].2.rows);
    debug_assert!(mats.iter().all(|(g, u, d)| g.ty == GgmlType::Q4_0 && u.ty == GgmlType::Q4_0 && d.ty == GgmlType::Q4_0 && g.rows == n_ff && d.rows == n_embd));
    const CH: usize = 32;
    let mut h: Vec<Vec<f32>> = (0..n).map(|_| vec![0f32; n_ff]).collect();
    let hp: Vec<Out> = h.iter_mut().map(|v| Out(v.as_mut_ptr())).collect();
    let per1 = n_ff.div_ceil(CH);
    let xa = int8.then(|| quantize_act_q8(x));
    pool.run(n * per1, &|t| {
        let (j, c) = (t / per1, t % per1);
        let (g, u, _) = &mats[j];
        let (rbg, rbu) = (g.row_bytes(), u.row_bytes());
        for r in c * CH..((c + 1) * CH).min(n_ff) {
            let (gv, uv) = match &xa {
                Some(a) => (dot_q4_0_i8_at(level, &g.bytes[r * rbg..(r + 1) * rbg], a), dot_q4_0_i8_at(level, &u.bytes[r * rbu..(r + 1) * rbu], a)),
                None => (dot_q4_0_f32x_at(level, &g.bytes[r * rbg..(r + 1) * rbg], x), dot_q4_0_f32x_at(level, &u.bytes[r * rbu..(r + 1) * rbu], x)),
            };
            // SAFETY: each (expert, row) is written by exactly one task
            unsafe { *hp[j].0.add(r) = gv / (1.0 + (-gv).exp()) * uv };
        }
    });
    let ha: Vec<_> = if int8 { h.iter().map(|v| Some(quantize_act_q8(v))).collect() } else { vec![None; n] };
    let mut out: Vec<Vec<f32>> = (0..n).map(|_| vec![0f32; n_embd]).collect();
    let op: Vec<Out> = out.iter_mut().map(|v| Out(v.as_mut_ptr())).collect();
    let per2 = n_embd.div_ceil(CH);
    pool.run(n * per2, &|t| {
        let (j, c) = (t / per2, t % per2);
        let d = &mats[j].2;
        let rb = d.row_bytes();
        for r in c * CH..((c + 1) * CH).min(n_embd) {
            let row = &d.bytes[r * rb..(r + 1) * rb];
            let v = match &ha[j] {
                Some(a) => dot_q4_0_i8_at(level, row, a),
                None => dot_q4_0_f32x_at(level, row, &h[j]),
            };
            // SAFETY: each (expert, row) is written by exactly one task
            unsafe { *op[j].0.add(r) = v };
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_covers_every_task_once_across_many_barriers() {
        let pool = SpinPool::new(6, Duration::from_millis(1));
        for round in 0..200usize {
            let n = 1 + round % 97;
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|t| {
                hits[t].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "round {round}");
            if round % 50 == 0 {
                // let the workers go to sleep, then wake them
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}
