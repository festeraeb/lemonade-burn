//! Launch trace for the decode token (diagnostic, off by default): which kernels a token launches
//! and, in device-time mode, what each costs on the device. `Gpu::tr(name, tag)` is called right
//! after a kernel launch is queued. In count mode it only records the name. In device-time mode a
//! cubecl profile window (CUDA events on the stream) is closed and the next one opened at every
//! mark, so each launch is bracketed by two events: its device time, with the host not waiting
//! (the work runs pipelined as in a real token). `us` is the host time since the previous mark
//! (the queue time of that launch).
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use cubecl::client::ProfileWindow;
use cubecl::profile::ProfileDuration;

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
    durs: Vec<Option<ProfileDuration>>,
    window: Option<ProfileWindow>,
}

/// One traced launch: the layer label active at the time, the kernel, a shape tag (output rows
/// where it applies), host microseconds since the previous mark, and the device microseconds of
/// that launch (NaN in count mode).
#[derive(Clone, Debug)]
pub struct Row {
    pub label: String,
    pub name: &'static str,
    pub tag: usize,
    pub us: f64,
    pub dev_us: f64,
}

impl Trace {
    /// `device`: bracket every launch with events (see the module docs).
    pub fn start(&self, device: bool) {
        let mut g = self.inner.lock().expect("trace");
        g.rows.clear();
        g.durs.clear();
        g.window = None;
        g.last = Some(Instant::now());
        self.sync.store(device, Ordering::Relaxed);
        self.on.store(true, Ordering::Release);
    }

    /// Stop and take the rows; resolves the event durations (waits for the device).
    pub fn take(&self) -> Vec<Row> {
        self.on.store(false, Ordering::Release);
        let (mut rows, durs) = {
            let mut g = self.inner.lock().expect("trace");
            g.window = None;
            (std::mem::take(&mut g.rows), std::mem::take(&mut g.durs))
        };
        for (r, d) in rows.iter_mut().zip(durs) {
            if let Some(d) = d {
                if let Some(t) = cubecl::future::block_on(d.resolve()) {
                    r.dev_us = t.duration().as_secs_f64() * 1e6;
                }
            }
        }
        rows
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

    /// Record one launch. `dur`: the closed event window of this launch (device mode).
    pub fn push(&self, name: &'static str, tag: usize, dur: Option<ProfileDuration>) {
        let mut g = self.inner.lock().expect("trace");
        let now = Instant::now();
        let us = g.last.map(|t| now.duration_since(t).as_secs_f64() * 1e6).unwrap_or(0.0);
        let label = g.label.clone();
        g.rows.push(Row { label, name, tag, us, dev_us: f64::NAN });
        g.durs.push(dur);
        g.last = Some(now);
    }

    /// The open window (device mode), taken out so the caller can close it.
    pub fn take_window(&self) -> Option<ProfileWindow> {
        self.inner.lock().expect("trace").window.take()
    }

    pub fn set_window(&self, w: Option<ProfileWindow>) {
        self.inner.lock().expect("trace").window = w;
    }
}
