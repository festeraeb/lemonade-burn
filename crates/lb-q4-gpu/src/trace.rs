//! Launch trace for the decode token (diagnostic, off by default): which kernels a token launches
//! and, in sync mode, what each costs. `Gpu::tr(name, tag)` is called right after a kernel launch
//! is queued. In count mode it only records the name; in sync mode it also waits for the device and
//! records the time since the previous mark, which is host queue time plus device time plus the
//! sync round trip (the caller subtracts the measured floor of an empty launch).
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Default)]
pub struct Trace {
    on: AtomicBool,
    sync: AtomicBool,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    label: String,
    last: Option<Instant>,
    rows: Vec<Row>,
}

/// One traced launch: the layer label active at the time, the kernel, a shape tag (output rows
/// where it applies), and microseconds since the previous mark (0 in count mode).
#[derive(Clone, Debug)]
pub struct Row {
    pub label: String,
    pub name: &'static str,
    pub tag: usize,
    pub us: f64,
}

impl Trace {
    pub fn start(&self, sync: bool) {
        let mut g = self.inner.lock().expect("trace");
        g.rows.clear();
        g.last = Some(Instant::now());
        self.sync.store(sync, Ordering::Relaxed);
        self.on.store(true, Ordering::Release);
    }

    /// Stop and take the rows.
    pub fn take(&self) -> Vec<Row> {
        self.on.store(false, Ordering::Release);
        std::mem::take(&mut self.inner.lock().expect("trace").rows)
    }

    pub fn is_on(&self) -> bool {
        self.on.load(Ordering::Acquire)
    }

    pub fn label(&self, l: &str) {
        if self.is_on() {
            self.inner.lock().expect("trace").label = l.to_string();
        }
    }

    pub fn sync_mode(&self) -> bool {
        self.sync.load(Ordering::Relaxed)
    }

    pub fn push(&self, name: &'static str, tag: usize) {
        let mut g = self.inner.lock().expect("trace");
        let now = Instant::now();
        let us = g.last.map(|t| now.duration_since(t).as_secs_f64() * 1e6).unwrap_or(0.0);
        let label = g.label.clone();
        g.rows.push(Row { label, name, tag, us });
        g.last = Some(now);
    }

    /// After a sync, restart the interval clock (the sync itself is not the next launch's).
    pub fn restart(&self) {
        self.inner.lock().expect("trace").last = Some(Instant::now());
    }
}
