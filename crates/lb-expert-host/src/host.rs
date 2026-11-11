//! The expert host: holds a set of `(layer, expert)` and computes them on request.
//!
//! The host never weights or sums: it returns each expert's raw output. An expert it does not
//! hold is **refused by name**, never answered with zeros and never loaded on the side, so a
//! stale placement shows up as a counted refusal at the trunk (which then runs the expert
//! itself), not as a wrong token.

use crate::expert::swiglu;
use crate::model::ModelExperts;
use crate::placement::HeldSet;
use crate::transport::{Conn, Listener};
use crate::wire::{self, HostInfo, Kind, Run};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Where a host's experts are computed.
pub trait Backend: Send {
    fn name(&self) -> &'static str;
    /// Make `held` the loaded set (load what is new, release what is gone). Returns device
    /// bytes in use afterwards (0 for a backend that computes from the file mapping).
    fn load(&mut self, model: &ModelExperts, held: &HeldSet) -> Result<usize, String>;
    /// Run `ids[i]` on rows `groups[i]` of `x` (`m` rows of `model.d`); append each output
    /// (`groups[i].len()` rows of `model.d`) to `out` in order. Every id is held (the host
    /// checked).
    fn run(&mut self, model: &ModelExperts, layer: usize, ids: &[u16], groups: &[Vec<u32>], x: &[f32], out: &mut Vec<u8>) -> Result<(), String>;
}

/// The CPU reference backend: experts computed from the file mapping with the same kernel as
/// the trunk's one-process path. Nothing is loaded, so "held" is only the set it answers for.
#[derive(Default)]
pub struct CpuBackend;

fn gather(x: &[f32], rows: &[u32], d: usize) -> Vec<f32> {
    let mut xs = Vec::with_capacity(rows.len() * d);
    for &r in rows {
        xs.extend_from_slice(&x[r as usize * d..(r as usize + 1) * d]);
    }
    xs
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn load(&mut self, _model: &ModelExperts, _held: &HeldSet) -> Result<usize, String> {
        Ok(0)
    }

    fn run(&mut self, model: &ModelExperts, layer: usize, ids: &[u16], groups: &[Vec<u32>], x: &[f32], out: &mut Vec<u8>) -> Result<(), String> {
        let le = model.layers[layer].as_ref().ok_or_else(|| format!("layer {layer} has no experts"))?;
        let d = model.d;
        // experts side by side, as the one-process path runs its CPU experts
        let ys: Vec<Vec<f32>> = ids
            .par_iter()
            .zip(groups.par_iter())
            .map(|(&e, rows)| {
                let e = e as usize;
                swiglu(model.mat(&le.gate, e), model.mat(&le.up, e), model.mat(&le.down, e), &gather(x, rows, d), rows.len())
            })
            .collect();
        for y in &ys {
            wire::put_f32s(out, y);
        }
        Ok(())
    }
}

/// Experts resident on a device through `lb-q4-gpu` (CUDA with the `cuda` feature and
/// `LB_CUDA_DEVICE`, otherwise the wgpu adapter named by `LB_WGPU_DEVICE`: that is the seam for
/// Polaris / Intel / other people's cards, the same code on another runtime). A layer held whole
/// is uploaded as one stacked buffer per projection, otherwise per expert, exactly as the
/// trunk's own resident store does, and an expert runs through the same `GpuExpert::launch`.
#[cfg(feature = "gpu")]
pub struct GpuBackend {
    gpu: lb_q4_gpu::Gpu,
    resident: std::collections::HashMap<(usize, usize), lb_q4_gpu::GpuExpert>,
    stacks: std::collections::HashMap<usize, lb_q4_gpu::GpuExpertLayer>,
    bytes: usize,
}

#[cfg(feature = "gpu")]
impl GpuBackend {
    /// Open the device named by `LB_CUDA_DEVICE` / `LB_WGPU_DEVICE`.
    pub fn open() -> Result<Self, String> {
        Ok(Self { gpu: lb_q4_gpu::Gpu::open()?, resident: Default::default(), stacks: Default::default(), bytes: 0 })
    }
}

#[cfg(feature = "gpu")]
impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        if cfg!(feature = "cuda") && std::env::var_os("LB_CUDA_DEVICE").is_some() {
            "cuda"
        } else {
            "wgpu"
        }
    }

    fn load(&mut self, model: &ModelExperts, held: &HeldSet) -> Result<usize, String> {
        // release what is no longer held (a stack goes when its layer is no longer whole)
        let gone_stacks: Vec<usize> = self.stacks.keys().copied().filter(|&l| !held.layer_whole(l)).collect();
        for l in gone_stacks {
            if let Some(s) = self.stacks.remove(&l) {
                self.bytes -= s.bytes_on_device();
            }
            self.resident.retain(|&(rl, _), _| rl != l);
        }
        let gone: Vec<(usize, usize)> = self.resident.keys().copied().filter(|&(l, e)| !held.holds(l, e)).collect();
        for k in gone {
            if let Some(ex) = self.resident.remove(&k) {
                self.bytes -= ex.bytes_on_device();
            }
        }
        for (l, layer) in model.layers.iter().enumerate() {
            let Some(le) = layer else { continue };
            if held.layer_whole(l) && !self.stacks.contains_key(&l) && (0..model.n_expert).all(|e| !self.resident.contains_key(&(l, e))) {
                let stack = lb_q4_gpu::GpuExpertLayer::upload_stacked(&self.gpu, model.mat_all(&le.gate), model.mat_all(&le.up), model.mat_all(&le.down), model.n_expert)?;
                for e in 0..model.n_expert {
                    self.resident.insert((l, e), stack.expert(&self.gpu, e)?);
                }
                self.bytes += stack.bytes_on_device();
                self.stacks.insert(l, stack);
                continue;
            }
            for e in 0..model.n_expert {
                if !held.holds(l, e) || self.resident.contains_key(&(l, e)) {
                    continue;
                }
                let ex = lb_q4_gpu::GpuExpert::upload_per_matrix(&self.gpu, model.mat(&le.gate, e), model.mat(&le.up, e), model.mat(&le.down, e))?;
                self.bytes += ex.bytes_on_device();
                self.resident.insert((l, e), ex);
            }
        }
        self.gpu.sync()?;
        Ok(self.bytes)
    }

    fn run(&mut self, model: &ModelExperts, layer: usize, ids: &[u16], groups: &[Vec<u32>], x: &[f32], out: &mut Vec<u8>) -> Result<(), String> {
        let d = model.d;
        // one upload per distinct row set (a decode token: one upload for the whole message)
        let mut uploaded: std::collections::HashMap<&[u32], lb_q4_gpu::cubecl_server::Handle> = std::collections::HashMap::new();
        let mut launched = Vec::with_capacity(ids.len());
        for (&e, rows) in ids.iter().zip(groups) {
            let ex = self.resident.get(&(layer, e as usize)).ok_or_else(|| format!("expert ({layer}, {e}) is held but not on the device"))?;
            let xh = uploaded.entry(rows.as_slice()).or_insert_with(|| self.gpu.upload_f32(&gather(x, rows, d))).clone();
            launched.push(ex.launch(&self.gpu, &xh, rows.len()));
        }
        for y in self.gpu.read_all(launched)? {
            wire::put_f32s(out, &y);
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct HostStats {
    pub runs: AtomicU64,
    pub experts: AtomicU64,
    pub refused: AtomicU64,
    pub errors: AtomicU64,
    pub compute_ns: AtomicU64,
}

pub struct Host {
    pub name: String,
    pub model: ModelExperts,
    held: RwLock<HeldSet>,
    backend: Mutex<Box<dyn Backend>>,
    backend_name: &'static str,
    pub stats: HostStats,
    /// Test hook: a host that stops answering (requests are read and dropped).
    pub mute: AtomicBool,
    /// Test hook: a host that dies: the next request on a connection closes it unanswered,
    /// which is what a trunk sees when the host process is killed.
    pub die: AtomicBool,
}

impl Host {
    /// A host named `name` holding `held`, with the experts loaded into `backend`.
    pub fn new(name: &str, model: ModelExperts, held: HeldSet, mut backend: Box<dyn Backend>) -> Result<Self, String> {
        if held.n_layer != model.n_layer() || held.n_expert != model.n_expert {
            return Err(format!("held set is {}x{}, the model {}x{}", held.n_layer, held.n_expert, model.n_layer(), model.n_expert));
        }
        if let Some((l, e)) = held.iter().find(|&(l, _)| model.layers[l].is_none()) {
            return Err(format!("held set names ({l}, {e}) but layer {l} has no experts"));
        }
        backend.load(&model, &held)?;
        let backend_name = backend.name();
        Ok(Self { name: name.to_string(), model, held: RwLock::new(held), backend: Mutex::new(backend), backend_name, stats: HostStats::default(), mute: AtomicBool::new(false), die: AtomicBool::new(false) })
    }

    pub fn held(&self) -> HeldSet {
        self.held.read().unwrap().clone()
    }

    /// Replace the held set (a placement reload): the backend loads and releases first, then
    /// the host starts answering for the new set.
    pub fn set_held(&self, held: HeldSet) -> Result<usize, String> {
        if held.n_layer != self.model.n_layer() || held.n_expert != self.model.n_expert {
            return Err("held set shape differs from the model".into());
        }
        if let Some((l, e)) = held.iter().find(|&(l, _)| self.model.layers[l].is_none()) {
            return Err(format!("held set names ({l}, {e}) but layer {l} has no experts"));
        }
        // answer for nothing that is about to be released while the device changes
        let mut guard = self.held.write().unwrap();
        let bytes = self.backend.lock().unwrap().load(&self.model, &held)?;
        *guard = held;
        Ok(bytes)
    }

    pub fn info(&self) -> HostInfo {
        let held = self.held.read().unwrap();
        HostInfo {
            fingerprint: self.model.fingerprint,
            d: self.model.d as u32,
            n_ff: self.model.n_ff as u32,
            n_layer: held.n_layer as u16,
            n_expert: held.n_expert as u16,
            held_words: held.words as u16,
            held: held.bits.clone(),
            arith: if self.backend_name == "cpu" { crate::expert::arith_fingerprint() } else { 0 },
            backend: self.backend_name.to_string(),
            name: self.name.clone(),
        }
    }

    /// Answer one request frame into `out`. Never panics on a bad frame: it answers `Error`.
    pub fn handle(&self, frame: &[u8], run: &mut Run, out: &mut Vec<u8>) {
        let (kind, seq) = match wire::peek(frame) {
            Ok(k) => k,
            Err(e) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                return wire::encode_error(out, 0, &e.to_string());
            }
        };
        match kind {
            Kind::Hello => wire::encode_hello_ok(out, seq, &self.info()),
            Kind::Ping => wire::encode_simple(out, Kind::Pong, seq),
            Kind::Run => {
                if let Err(e) = wire::decode_run(frame, run) {
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                    return wire::encode_error(out, seq, &e.to_string());
                }
                let layer = run.layer as usize;
                if run.d as usize != self.model.d {
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                    return wire::encode_error(out, seq, &format!("activation width {} is not the model's {}", run.d, self.model.d));
                }
                // the held set stays readable (not replaceable) until the answer is computed
                let held = self.held.read().unwrap();
                let missing: Vec<u16> = run.ids.iter().copied().filter(|&e| !held.holds(layer, e as usize)).collect();
                if !missing.is_empty() {
                    self.stats.refused.fetch_add(missing.len() as u64, Ordering::Relaxed);
                    return wire::encode_refused(out, seq, run.layer, &missing);
                }
                let t = Instant::now();
                wire::begin_run_ok(out, seq, 0, run.ids.len() as u16, run.d);
                let r = self.backend.lock().unwrap().run(&self.model, layer, &run.ids, &run.groups, &run.x, out);
                drop(held);
                let ns = t.elapsed().as_nanos() as u64;
                match r {
                    Ok(()) => {
                        wire::set_host_ns(out, ns);
                        self.stats.runs.fetch_add(1, Ordering::Relaxed);
                        self.stats.experts.fetch_add(run.ids.len() as u64, Ordering::Relaxed);
                        self.stats.compute_ns.fetch_add(ns, Ordering::Relaxed);
                    }
                    Err(e) => {
                        self.stats.errors.fetch_add(1, Ordering::Relaxed);
                        wire::encode_error(out, seq, &e);
                    }
                }
            }
            other => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                wire::encode_error(out, seq, &format!("{other:?} is not a request"));
            }
        }
    }

    /// Serve one connection until the peer goes away.
    pub fn serve_conn(&self, mut conn: Box<dyn Conn>) -> std::io::Result<()> {
        let (mut buf, mut out, mut run) = (Vec::new(), Vec::new(), Run::default());
        loop {
            conn.recv(&mut buf, None)?;
            if self.die.load(Ordering::Relaxed) {
                return Err(std::io::ErrorKind::ConnectionAborted.into());
            }
            if self.mute.load(Ordering::Relaxed) {
                continue;
            }
            self.handle(&buf, &mut run, &mut out);
            conn.send(&out)?;
        }
    }
}

/// Accept connections for ever, one thread each. Returns when the listener fails.
pub fn serve(host: Arc<Host>, mut listener: Box<dyn Listener>) -> std::io::Result<()> {
    loop {
        let conn = listener.accept()?;
        let h = host.clone();
        std::thread::Builder::new().name(format!("w2w-{}", conn.kind())).spawn(move || {
            let kind = conn.kind();
            if let Err(e) = h.serve_conn(conn) {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    eprintln!("lb-expert-host {}: {kind} connection ended: {e}", h.name);
                }
            }
        })?;
    }
}

/// [`serve`] on its own thread (tests, and a host inside the trunk's process).
pub fn spawn(host: Arc<Host>, listener: Box<dyn Listener>) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("w2w-accept".into())
        .spawn(move || {
            let _ = serve(host, listener);
        })
        .expect("spawn")
}
