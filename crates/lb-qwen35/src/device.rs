//! Device (CubeCL/CUDA) layers of the `qwen35moe` decoder: gated delta-net, gated attention, and the MoE block with the gated shared
//! expert, built from the file-density kernels in `lb-q4-gpu` (`kquant.rs`, `qwen_ops.rs`). Verified layer by layer against the CPU
//! decoder in `tests/device_layers.rs`.

use crate::{Config, Layer, Model};
use anyhow::{anyhow, Result};
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuNativeExperts, GpuQ5KNative, GpuQ6KNative, GpuQ8Native};
use std::sync::Arc;

type Handle = lb_q4_gpu::cubecl_server::Handle;

pub enum DMat {
    Q5(GpuQ5KNative),
    Q6(GpuQ6KNative),
    /// One device copy of the Q8_0 matrix serves decode (matvec) and prefill (GEMM); `sh_fused` shares it too.
    Q8(Arc<GpuQ8Native>),
}

impl DMat {
    fn new(gpu: &Gpu, q: &QuantMat<'_>) -> Result<Self> {
        let r = match q.ty {
            GgmlType::Q5K => GpuQ5KNative::upload(gpu, *q).map(DMat::Q5),
            GgmlType::Q6K => GpuQ6KNative::upload(gpu, *q).map(DMat::Q6),
            GgmlType::Q8_0 => GpuQ8Native::upload(gpu, *q).map(|b| DMat::Q8(Arc::new(b))),
            t => return Err(anyhow!("no device kernel for {}", t.name())),
        };
        r.map_err(|e| anyhow!(e))
    }
    fn launch(&self, gpu: &Gpu, x: &Handle) -> Handle {
        match self {
            DMat::Q5(m) => m.launch(gpu, x, 1),
            DMat::Q6(m) => m.launch(gpu, x, 1),
            DMat::Q8(m) => m.launch(gpu, x, 1),
        }
    }
    /// Tiled GEMM over `m` token rows (prefill): `[m, rows]`.
    fn gemm(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        match self {
            DMat::Q5(a) => a.gemm(gpu, x, m),
            DMat::Q6(a) => a.gemm(gpu, x, m),
            DMat::Q8(b) => b.gemm(gpu, x, m),
        }
    }
    /// With the activation available both ways (the producer wrote the permuted copy), no permute launch is needed.
    fn launch2(&self, gpu: &Gpu, x: &Handle, xp: &Handle) -> Handle {
        match self {
            DMat::Q5(m) => m.launch_permuted(gpu, xp, 1),
            DMat::Q6(m) => m.launch_permuted(gpu, xp, 1),
            DMat::Q8(m) => m.launch_permuted(gpu, xp, 1),
        }
    }
}

struct DLin {
    qkv: DMat,
    z: DMat,
    out: DMat,
    ba: Handle,
    conv: Handle,
    a: Handle,
    dt: Handle,
    norm: Handle,
    conv_state: Handle,
    s: Handle,
}

struct DAttn {
    q: DMat,
    k: DMat,
    v: DMat,
    o: DMat,
    qn: Handle,
    kn: Handle,
    kc: Handle,
    vc: Handle,
}

pub struct DevLayer {
    attn_norm: Handle,
    post_norm: Handle,
    lin: Option<DLin>,
    attn: Option<DAttn>,
    router_sg: Handle,
    experts: GpuNativeExperts,
    sh_gate: DMat,
    sh_up: DMat,
    sh_down: DMat,
    /// All three shared matrices are Q8_0 (Qwen3.6, Tiel-Coder): gate+up+SiLU and down-into-combine as two launches.
    sh_fused: Option<(Arc<GpuQ8Native>, Arc<GpuQ8Native>, Arc<GpuQ8Native>)>,
}

/// `LB_QW_SKIP=experts,shared,core,dense,route` skips those parts of every layer (timing ablation only: the results are wrong).
fn skip(name: &str) -> bool {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("LB_QW_SKIP").unwrap_or_default()).split(',').any(|x| x == name)
}

fn shf_on() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("LB_QW_SHF").is_ok())
}

/// `LB_QW_PTIME=1`: wait for the device after every phase of a prefill layer and add the time to a per-phase table (`ptime_report`).
pub fn ptime_on() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("LB_QW_PTIME").is_ok())
}

static PTIME: std::sync::Mutex<Vec<(&'static str, f64)>> = std::sync::Mutex::new(Vec::new());

pub fn ptime_add(name: &'static str, s: f64) {
    let mut t = PTIME.lock().unwrap();
    if let Some(e) = t.iter_mut().find(|e| e.0 == name) {
        e.1 += s;
    } else {
        t.push((name, s));
    }
}

pub fn ptime_report() -> Vec<(&'static str, f64)> {
    std::mem::take(&mut *PTIME.lock().unwrap())
}

/// One layer's host copy of its recurrent state / KV prefix (see [`DevModel::snapshot`]).
#[derive(Default, Clone)]
pub struct LayerSnap {
    conv: Vec<f32>,
    s: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
}

pub struct DevModel {
    pp2: Handle,
    pp1: Handle,
    zeros: Handle,
    pub cfg: Config,
    pub cap: usize,
    pub layers: Vec<DevLayer>,
    theta_scale: f32,
    pos_h: Handle,
}

impl DevModel {
    /// Upload layers `range` of `m` to `gpu`; attention caches hold `cap` positions.
    pub fn upload(gpu: &Gpu, m: &Model<'_>, range: std::ops::Range<usize>, cap: usize) -> Result<Self> {
        let c = m.cfg.clone();
        let mut layers = Vec::new();
        for il in range {
            layers.push(Self::upload_layer(gpu, &c, &m.layers[il], cap)?);
        }
        let theta_scale = c.rope_base.powf(-2.0 / c.n_rot as f32);
        let zeros = gpu.upload_f32(&vec![0.0f32; 16384]);
        Ok(DevModel { pp2: gpu.upload_u32(vec![0, 0]), pp1: gpu.upload_u32(vec![0]), zeros, cfg: c, cap, layers, theta_scale, pos_h: gpu.upload_u32(vec![0]) })
    }

    fn upload_layer(gpu: &Gpu, c: &Config, l: &Layer<'_>, cap: usize) -> Result<DevLayer> {
        let f = |v: &[f32]| gpu.upload_f32(v);
        let (lin, attn) = match (&l.lin, &l.attn) {
            (Some(w), _) => (
                Some(DLin {
                    qkv: DMat::new(gpu, &w.qkv.0)?,
                    z: DMat::new(gpu, &w.z.0)?,
                    out: DMat::new(gpu, &w.out.0)?,
                    ba: f(&[w.beta.as_slice(), w.alpha.as_slice()].concat()),
                    conv: f(&w.conv),
                    a: f(&w.a),
                    dt: f(&w.dt_bias),
                    norm: f(&w.norm),
                    conv_state: f(&vec![0.0; 2 * (c.d_conv - 1) * c.conv_channels()]),
                    s: f(&vec![0.0; c.n_v_heads * c.head_v() * c.head_k]),
                }),
                None,
            ),
            (_, Some(w)) => (
                None,
                Some(DAttn {
                    q: DMat::new(gpu, &w.q.0)?,
                    k: DMat::new(gpu, &w.k.0)?,
                    v: DMat::new(gpu, &w.v.0)?,
                    o: DMat::new(gpu, &w.o.0)?,
                    qn: f(&w.q_norm),
                    kn: f(&w.k_norm),
                    kc: f(&vec![0.0; cap * c.n_head_kv * c.head_dim]),
                    vc: f(&vec![0.0; cap * c.n_head_kv * c.head_dim]),
                }),
            ),
            _ => unreachable!(),
        };
        let sh_gate = DMat::new(gpu, &l.sh_gate.0)?;
        let sh_up = DMat::new(gpu, &l.sh_up.0)?;
        let sh_down = DMat::new(gpu, &l.sh_down.0)?;
        Ok(DevLayer {
            attn_norm: f(&l.attn_norm),
            post_norm: f(&l.post_norm),
            lin,
            attn,
            router_sg: f(&[l.router.as_slice(), l.sh_gate_inp.as_slice()].concat()),
            experts: GpuNativeExperts::upload(gpu, c.n_expert, l.gate_exps, l.up_exps, l.down_exps).map_err(|e| anyhow!(e))?,
            sh_fused: match (&sh_gate, &sh_up, &sh_down) {
                (DMat::Q8(g), DMat::Q8(u), DMat::Q8(d)) => Some((g.clone(), u.clone(), d.clone())),
                _ => None,
            },
            sh_gate,
            sh_up,
            sh_down,
        })
    }

    /// Time the parts of layer `i` (microseconds per call, `reps` calls queued then one sync), for finding what limits a token.
    pub fn bench_parts(&self, gpu: &Gpu, i: usize, reps: usize) -> Vec<(&'static str, f64)> {
        let c = &self.cfg;
        let l = &self.layers[i];
        let x = gpu.upload_f32(&vec![0.1f32; c.n_embd]);
        // device time: record `reps` calls as one CUDA graph and replay it, so host launch cost is not in the number
        let time = |name: &'static str, f: &dyn Fn() -> Handle| -> (&'static str, f64) {
            gpu.client.graph_prepare().expect("graph_prepare");
            let mut h0 = f();
            for _ in 1..reps {
                h0 = f(); // fill the scratch pool with as many buffers as the capture will take
            }
            gpu.read_all(vec![(h0, 1)]).unwrap();
            gpu.client.start_capture().expect("start_capture");
            let mut last = None;
            for _ in 0..reps {
                last = Some(f());
            }
            let graph = gpu.client.stop_capture().expect("stop_capture");
            let held = gpu.take_in_flight();
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                unsafe { graph.replay() }.expect("replay");
            }
            gpu.read_all(vec![(last.unwrap(), 1)]).unwrap();
            let us = t0.elapsed().as_secs_f64() / 3.0 / reps as f64 * 1e6;
            drop(held);
            (name, us)
        };
        let mut v = Vec::new();
        v.push(time("rms", &|| gpu.qw_rms(&x, &l.attn_norm, 1, c.n_embd, c.eps).0));
        if let Some(d) = &l.lin {
            v.push(time("qkv matvec", &|| d.qkv.launch(gpu, &x)));
            v.push(time("z matvec", &|| d.z.launch(gpu, &x)));
            let zin = gpu.upload_f32(&vec![0.1f32; c.d_inner]);
            v.push(time("out matvec", &|| d.out.launch(gpu, &zin)));
            v.push(time("beta+alpha f32 matvec", &|| gpu.qw_f32_matvec(&d.ba, &x, 2 * c.n_v_heads, c.n_embd)));
            let qkv = gpu.upload_f32(&vec![0.1f32; c.conv_channels()]);
            let z = gpu.upload_f32(&vec![0.1f32; c.d_inner]);
            let ba = gpu.upload_f32(&vec![0.1f32; 2 * c.n_v_heads]);
            v.push(time("delta-net core", &|| gpu.qw_delta_net(&qkv, &z, &ba, &d.conv, &d.dt, &d.a, &d.norm, &d.conv_state, &d.s, &self.pos_h, (c.conv_channels(), c.d_conv, c.n_k_heads, c.n_v_heads, c.head_k, c.head_v()), c.eps).0));
        } else if let Some(a) = &l.attn {
            v.push(time("q matvec", &|| a.q.launch(gpu, &x)));
            v.push(time("k matvec", &|| a.k.launch(gpu, &x)));
            let oin = gpu.upload_f32(&vec![0.1f32; c.n_head * c.head_dim]);
            v.push(time("o matvec", &|| a.o.launch(gpu, &oin)));
        }
        let logits = gpu.upload_f32(&(0..c.n_expert).map(|i| (i as f32 * 0.37).sin()).collect::<Vec<_>>());
        v.push(time("router+shared-gate matvec", &|| gpu.qw_f32_matvec(&l.router_sg, &x, c.n_expert + 1, c.n_embd)));
        let (routes, _w) = gpu.qw_route(&logits, c.n_expert, c.n_used);
        v.push(time("route top-k", &|| gpu.qw_route(&logits, c.n_expert, c.n_used).1));
        v.push(time("routed experts", &|| l.experts.launch(gpu, &x, &routes, c.n_used)));
        v.push(time("shared gate+up", &|| {
            let _g = l.sh_gate.launch(gpu, &x);
            l.sh_up.launch(gpu, &x)
        }));
        let shin = gpu.upload_f32(&vec![0.1f32; c.ff_shexp]);
        v.push(time("shared down", &|| l.sh_down.launch(gpu, &shin)));
        v
    }

    /// Set the chunk the prefill kernels see: positions `p0 .. p0 + m`.
    pub fn set_chunk(&self, gpu: &Gpu, p0: usize, m: usize) {
        gpu.qw_write_u32(&self.pp2, &[p0 as u32, m as u32]);
        gpu.qw_write_u32(&self.pp1, &[p0 as u32]);
    }

    /// One layer for a chunk of `m` tokens (`x` `[m, n_embd]`): tiled GEMMs, batched glue kernels, routed experts grouped by expert on the host (one
    /// read of the routes a layer). Leaves the recurrent state and KV cache as `m` single-token steps would. Returns the layer output `[m, n_embd]`.
    pub fn prefill_layer(&self, gpu: &Gpu, i: usize, x: &Handle, m: usize) -> Result<Handle> {
        let c = &self.cfg;
        let l = &self.layers[i];
        let t0 = std::cell::Cell::new(std::time::Instant::now());
        let ph = |name: &'static str, hd: &Handle| {
            if ptime_on() {
                let _ = gpu.read_all(vec![(hd.clone(), 1)]);
                ptime_add(name, t0.get().elapsed().as_secs_f64());
                t0.set(std::time::Instant::now());
            }
        };
        let h = gpu.qp_rms_rows(x, &l.attn_norm, m, c.n_embd, c.eps);
        ph("norm", &h);
        let y = if let Some(d) = &l.lin {
            let qkv = d.qkv.gemm(gpu, &h, m);
            let z = d.z.gemm(gpu, &h, m);
            let ba = gpu.qp_f32_rows(&d.ba, &h, m, 2 * c.n_v_heads, c.n_embd);
            ph("dn proj gemm", &ba);
            let core = gpu.qp_delta_net(&qkv, &z, &ba, &d.conv, &d.dt, &d.a, &d.norm, &d.conv_state, &d.s, &self.pp2, m, (c.conv_channels(), c.n_k_heads, c.n_v_heads, c.head_k), c.eps);
            ph("dn core", &core);
            let y = d.out.gemm(gpu, &core, m);
            ph("dn out gemm", &y);
            y
        } else {
            let a = l.attn.as_ref().unwrap();
            let qf = a.q.gemm(gpu, &h, m);
            let k = a.k.gemm(gpu, &h, m);
            let v = a.v.gemm(gpu, &h, m);
            ph("attn proj gemm", &v);
            let o = gpu.qp_attention(&qf, &k, &v, &a.qn, &a.kn, &a.kc, &a.vc, &self.pp1, m, (c.n_head, c.n_head_kv, c.head_dim, c.n_rot / 2, self.cap), c.eps, self.theta_scale);
            ph("attn core", &o);
            let y = a.o.gemm(gpu, &o, m);
            ph("attn out gemm", &y);
            y
        };
        let x1 = gpu.qw_add(x, &y, m * c.n_embd);
        let h2 = gpu.qp_rms_rows(&x1, &l.post_norm, m, c.n_embd, c.eps);
        let logits = gpu.qp_f32_rows(&l.router_sg, &h2, m, c.n_expert + 1, c.n_embd);
        let (routes, weights) = gpu.qp_route_rows(&logits, m, c.n_expert, c.n_used);
        ph("norm+router", &weights);
        let rbits = gpu.read_all(vec![(routes, 2 * m * c.n_used)]).map_err(|e| anyhow!(e))?.remove(0);
        t0.set(std::time::Instant::now());
        let rv: Vec<(u32, u32)> = rbits.chunks_exact(2).map(|p| (p[0].to_bits(), p[1].to_bits())).collect();
        let down = l.experts.gemm(gpu, &h2, m, &rv);
        ph("experts gemm", &down);
        let (g, u, dn) = l.sh_fused.as_ref().ok_or_else(|| anyhow!("prefill needs Q8_0 shared experts"))?;
        let gg = g.gemm(gpu, &h2, m);
        let uu = u.gemm(gpu, &h2, m);
        let sh = gpu.qw_silu_mul(&gg, &uu, m * c.ff_shexp);
        let shout = dn.gemm(gpu, &sh, m);
        ph("shared gemm", &shout);
        Ok(gpu.qp_combine_rows(&x1, &down, &weights, &shout, &logits, m, c.n_embd, c.n_used, c.n_expert))
    }

    /// Copy the recurrent state and the first `len` KV positions of every layer to the host (a rewind point for prefix reuse).
    pub fn snapshot(&self, gpu: &Gpu, len: usize) -> Result<Vec<LayerSnap>> {
        let c = &self.cfg;
        let mut out = Vec::new();
        for l in &self.layers {
            let mut snap = LayerSnap::default();
            if let Some(d) = &l.lin {
                let r = gpu.read_all(vec![(d.conv_state.clone(), 2 * (c.d_conv - 1) * c.conv_channels()), (d.s.clone(), c.n_v_heads * c.head_v() * c.head_k)]).map_err(|e| anyhow!(e))?;
                let mut it = r.into_iter();
                snap.conv = it.next().unwrap();
                snap.s = it.next().unwrap();
            }
            if let Some(a) = &l.attn {
                let n = len * c.n_head_kv * c.head_dim;
                if n > 0 {
                    let r = gpu.read_all(vec![(a.kc.clone(), n), (a.vc.clone(), n)]).map_err(|e| anyhow!(e))?;
                    let mut it = r.into_iter();
                    snap.kc = it.next().unwrap();
                    snap.vc = it.next().unwrap();
                }
            }
            out.push(snap);
        }
        Ok(out)
    }

    /// Put a [`snapshot`](Self::snapshot) back (the KV positions beyond it are zeroed: attention never reads them).
    pub fn restore(&self, gpu: &Gpu, snaps: &[LayerSnap]) {
        let c = &self.cfg;
        let full = self.cap * c.n_head_kv * c.head_dim;
        for (l, snap) in self.layers.iter().zip(snaps) {
            if let Some(d) = &l.lin {
                gpu.write_f32(&d.conv_state, &snap.conv);
                gpu.write_f32(&d.s, &snap.s);
            }
            if let Some(a) = &l.attn {
                let mut k = snap.kc.clone();
                k.resize(full, 0.0);
                gpu.write_f32(&a.kc, &k);
                let mut v = snap.vc.clone();
                v.resize(full, 0.0);
                gpu.write_f32(&a.vc, &v);
            }
        }
    }

    /// Every persistent device buffer of the block (states, KV caches).
    pub fn persistent(&self) -> Vec<Handle> {
        let mut v = Vec::new();
        for l in &self.layers {
            if let Some(d) = &l.lin {
                v.push(d.conv_state.clone());
                v.push(d.s.clone());
            }
            if let Some(a) = &l.attn {
                v.push(a.kc.clone());
                v.push(a.vc.clone());
            }
        }
        v
    }

    /// Zero every layer's recurrent state and KV cache (a new sequence).
    pub fn reset(&self, gpu: &Gpu) {
        let c = &self.cfg;
        for l in &self.layers {
            if let Some(d) = &l.lin {
                gpu.write_f32(&d.conv_state, &vec![0.0; 2 * (c.d_conv - 1) * c.conv_channels()]);
                gpu.write_f32(&d.s, &vec![0.0; c.n_v_heads * c.head_v() * c.head_k]);
            }
            if let Some(a) = &l.attn {
                gpu.write_f32(&a.kc, &vec![0.0; self.cap * c.n_head_kv * c.head_dim]);
                gpu.write_f32(&a.vc, &vec![0.0; self.cap * c.n_head_kv * c.head_dim]);
            }
        }
    }

    /// Set the token position every attention layer of this block reads (call once a token, outside any captured graph).
    pub fn set_pos(&self, gpu: &Gpu, pos: usize) {
        gpu.qw_write_u32(&self.pos_h, &[pos as u32]);
    }

    /// One layer for one token at `pos`. Returns (`x` after the mixer, layer output), both on the device.
    pub fn layer_forward(&self, gpu: &Gpu, i: usize, x: &Handle, _pos: usize) -> (Handle, Handle) {
        let c = &self.cfg;
        let l = &self.layers[i];
        let (h, hp) = gpu.qw_rms(x, &l.attn_norm, 1, c.n_embd, c.eps);
        let y = if skip("dense") {
            self.zeros.clone()
        } else if let Some(d) = &l.lin {
            let qkv = if skip("qkv") { self.zeros.clone() } else { d.qkv.launch2(gpu, &h, &hp) };
            let z = if skip("z") { self.zeros.clone() } else { d.z.launch2(gpu, &h, &hp) };
            let ba = gpu.qw_f32_matvec(&d.ba, &h, 2 * c.n_v_heads, c.n_embd);
            let core = if skip("core") { (self.zeros.clone(), self.zeros.clone()) } else { gpu.qw_delta_net(&qkv, &z, &ba, &d.conv, &d.dt, &d.a, &d.norm, &d.conv_state, &d.s, &self.pos_h, (c.conv_channels(), c.d_conv, c.n_k_heads, c.n_v_heads, c.head_k, c.head_v()), c.eps) };
            if skip("out") { self.zeros.clone() } else { d.out.launch2(gpu, &core.0, &core.1) }
        } else {
            let a = l.attn.as_ref().unwrap();
            let qf = if skip("aq") { self.zeros.clone() } else { a.q.launch2(gpu, &h, &hp) };
            let k = a.k.launch2(gpu, &h, &hp);
            let v = a.v.launch2(gpu, &h, &hp);
            let o = gpu.qw_attention(&qf, &k, &v, &a.qn, &a.kn, &a.kc, &a.vc, &self.pos_h, (c.n_head, c.n_head_kv, c.head_dim, c.n_rot / 2, self.cap), c.eps, self.theta_scale);
            if skip("ao") { self.zeros.clone() } else { a.o.launch(gpu, &o) }
        };
        let (x1, h2, h2p) = gpu.qw_add_rms(x, &y, &l.post_norm, c.n_embd, c.eps);
        let logits = gpu.qw_f32_matvec(&l.router_sg, &h2, c.n_expert + 1, c.n_embd);
        let (routes, weights) = gpu.qw_route(&logits, c.n_expert, c.n_used);
        let h2p2 = h2p.clone();
        let down = if skip("experts") { self.zeros.clone() } else { l.experts.launch_permuted(gpu, h2p, &routes, c.n_used) };
        let out = if let (Some((g, u, dn)), false, true) = (&l.sh_fused, skip("shared"), shf_on()) {
            let hs = GpuQ8Native::gate_up_silu(gpu, g, u, &h2p2);
            dn.combine_shared(gpu, &x1, &down, &weights, &logits, c.n_expert + 1, c.n_expert, &hs, c.n_used)
        } else {
            let shout = if skip("shared") {
                self.zeros.clone()
            } else {
                let g = l.sh_gate.launch(gpu, &h2);
                let u = l.sh_up.launch(gpu, &h2);
                let sh = gpu.qw_silu_mul(&g, &u, c.ff_shexp);
                l.sh_down.launch(gpu, &sh)
            };
            gpu.qw_moe_combine(&x1, &down, &weights, &shout, &logits, c.n_expert + 1, c.n_expert, c.n_embd, c.n_used)
        };
        (x1, out)
    }
}
