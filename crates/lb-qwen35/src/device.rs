//! Device (CubeCL/CUDA) layers of the `qwen35moe` decoder: gated delta-net, gated attention, and the MoE block with the gated shared
//! expert, built from the file-density kernels in `lb-q4-gpu` (`kquant.rs`, `qwen_ops.rs`). Verified layer by layer against the CPU
//! decoder in `tests/device_layers.rs`.

use crate::{Config, Layer, Model};
use anyhow::{anyhow, Result};
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuNativeExperts, GpuQ5KNative, GpuQ6KNative, GpuQ8Mat};

type Handle = lb_q4_gpu::cubecl_server::Handle;

pub enum DMat {
    Q5(GpuQ5KNative),
    Q6(GpuQ6KNative),
    /// One device copy of the Q8_0 matrix (decode layout) serves decode (rows matvec) and prefill (`GpuQ8Mat::gemm`).
    Q8(GpuQ8Mat),
}

impl DMat {
    pub(crate) fn new(gpu: &Gpu, q: &QuantMat<'_>) -> Result<Self> {
        let r = match q.ty {
            GgmlType::Q5K => GpuQ5KNative::upload(gpu, *q).map(DMat::Q5),
            GgmlType::Q6K => GpuQ6KNative::upload(gpu, *q).map(DMat::Q6),
            GgmlType::Q8_0 => GpuQ8Mat::upload(gpu, *q).map(DMat::Q8),
            t => return Err(anyhow!("no device kernel for {}", t.name())),
        };
        r.map_err(|e| anyhow!(e))
    }
    pub(crate) fn launch(&self, gpu: &Gpu, x: &Handle) -> Handle {
        match self {
            DMat::Q5(m) => m.launch(gpu, x, 1),
            DMat::Q6(m) => m.launch(gpu, x, 1),
            DMat::Q8(m) => m.launch(gpu, x, 1),
        }
    }
    /// The decode matvec over `m` token rows (`[m, rows]`): every plane runs one token's rows, no tile padding. For a verify chunk of a few tokens
    /// (records into a CUDA graph; the tiled GEMM does not).
    pub(crate) fn launch_m(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        match self {
            DMat::Q5(a) => a.launch(gpu, x, m),
            DMat::Q6(a) => a.launch(gpu, x, m),
            DMat::Q8(a) => a.launch_mt(gpu, x, m),
        }
    }
    /// Tiled GEMM over `m` token rows (prefill): `[m, rows]`.
    pub(crate) fn gemm(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
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
            DMat::Q8(m) => m.launch(gpu, x, 1),
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
}

/// `LB_QW_SKIP=experts,shared,core,dense,route` skips those parts of every layer (timing ablation only: the results are wrong).
fn skip(name: &str) -> bool {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("LB_QW_SKIP").unwrap_or_default()).split(',').any(|x| x == name)
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

/// Device time of `f` in microseconds a call: `reps` calls recorded as one CUDA graph and replayed three times, so host launch cost is not in
/// the number (the kernels' own time plus the in-graph launch gaps).
pub fn graph_time_us(gpu: &Gpu, reps: usize, f: &dyn Fn() -> Handle) -> f64 {
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
    us
}

/// One layer's host copy of its recurrent state / KV prefix (see [`DevModel::snapshot`]).
#[derive(Default, Clone)]
pub struct LayerSnap {
    conv: Vec<f32>,
    s: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
}

/// Per delta-net layer scratch for speculative verification: the layer's inputs of the last verified chunk (to replay the recurrent update for a
/// shorter prefix) and the state as it was before the chunk.
pub struct DnKeep {
    qkv: Handle,
    z: Handle,
    ba: Handle,
    ck_conv: Handle,
    ck_s: Handle,
}

pub struct DevModel {
    pub spec: Vec<Option<DnKeep>>,
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
    pub fn upload(gpu: &Gpu, m: &Model<'_>, range: std::ops::Range<usize>, cap: usize, spec_m: usize) -> Result<Self> {
        let ls: Vec<&Layer<'_>> = range.map(|il| &m.layers[il]).collect();
        Self::upload_layers(gpu, &m.cfg, &ls, cap, spec_m)
    }

    /// As [`upload`](Self::upload) for explicit layers (the nextn layer is not in `Model::layers`). `spec_m` > 0 allocates the speculative-verify
    /// scratch for chunks of up to that many tokens.
    pub fn upload_layers(gpu: &Gpu, cfg: &Config, ls: &[&Layer<'_>], cap: usize, spec_m: usize) -> Result<Self> {
        let c = cfg.clone();
        let mut layers = Vec::new();
        let mut spec = Vec::new();
        for l in ls {
            layers.push(Self::upload_layer(gpu, &c, l, cap)?);
            spec.push(if spec_m > 0 && l.lin.is_some() {
                Some(DnKeep {
                    qkv: gpu.upload_f32(&vec![0.0; spec_m * c.conv_channels()]),
                    z: gpu.upload_f32(&vec![0.0; spec_m * c.d_inner]),
                    ba: gpu.upload_f32(&vec![0.0; spec_m * 2 * c.n_v_heads]),
                    ck_conv: gpu.upload_f32(&vec![0.0; 2 * (c.d_conv - 1) * c.conv_channels()]),
                    ck_s: gpu.upload_f32(&vec![0.0; c.n_v_heads * c.head_v() * c.head_k]),
                })
            } else {
                None
            });
        }
        let theta_scale = c.rope_base.powf(-2.0 / c.n_rot as f32);
        let zeros = gpu.upload_f32(&vec![0.0f32; 16384]);
        Ok(DevModel { spec, pp2: gpu.upload_u32(vec![0, 0]), pp1: gpu.upload_u32(vec![0]), zeros, cfg: c, cap, layers, theta_scale, pos_h: gpu.upload_u32(vec![0]) })
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
            experts: Self::upload_experts(gpu, c, l)?,
            sh_gate,
            sh_up,
            sh_down,
        })
    }

    /// The routed experts of a layer. A Q8_0 gate/up pair (the nextn layer of the Q5_K_M file) has no device kernel and is requantised to Q6_K first.
    fn upload_experts(gpu: &Gpu, c: &Config, l: &Layer<'_>) -> Result<GpuNativeExperts> {
        use rayon::prelude::*;
        let conv = |q: QuantMat<'_>| -> Vec<u8> {
            let rb_in = q.ty.row_bytes(q.cols).unwrap();
            let rb_out = GgmlType::Q6K.row_bytes(q.cols).unwrap();
            let mut out = vec![0u8; q.rows * rb_out];
            out.par_chunks_mut(rb_out).enumerate().for_each_init(|| vec![0f32; q.cols], |buf, (r, o)| {
                q.ty.dequant_row(&q.bytes[r * rb_in..(r + 1) * rb_in], buf);
                o.copy_from_slice(&lb_gguf::types::quantize_q6_k(buf));
            });
            out
        };
        if l.gate_exps.ty == GgmlType::Q8_0 && l.up_exps.ty == GgmlType::Q8_0 {
            let (g, u) = (conv(l.gate_exps), conv(l.up_exps));
            let gq = QuantMat::new(GgmlType::Q6K, l.gate_exps.rows, l.gate_exps.cols, &g);
            let uq = QuantMat::new(GgmlType::Q6K, l.up_exps.rows, l.up_exps.cols, &u);
            return GpuNativeExperts::upload(gpu, c.n_expert, gq, uq, l.down_exps).map_err(|e| anyhow!(e));
        }
        GpuNativeExperts::upload(gpu, c.n_expert, l.gate_exps, l.up_exps, l.down_exps).map_err(|e| anyhow!(e))
    }

    /// Time the parts of layer `i` (microseconds per call, `reps` calls queued then one sync), for finding what limits a token.
    pub fn bench_parts(&self, gpu: &Gpu, i: usize, reps: usize) -> Vec<(&'static str, f64)> {
        let c = &self.cfg;
        let l = &self.layers[i];
        let x = gpu.upload_f32(&vec![0.1f32; c.n_embd]);
        let time = |name: &'static str, f: &dyn Fn() -> Handle| -> (&'static str, f64) { (name, graph_time_us(gpu, reps, f)) };
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
            // the one-token attention (norms, rope, cache append, softmax attention) at two context lengths (writes cache entry n-1: timing only)
            let qf = gpu.upload_f32(&(0..c.n_head * 2 * c.head_dim).map(|i| (i as f32 * 0.013).sin()).collect::<Vec<_>>());
            let kr = gpu.upload_f32(&(0..c.n_head_kv * c.head_dim).map(|i| (i as f32 * 0.029).cos()).collect::<Vec<_>>());
            for (n, name) in [(64usize, "attention core n=64"), (4096, "attention core n=4096")] {
                if n <= self.cap {
                    let ph = gpu.upload_u32(vec![(n - 1) as u32]);
                    v.push(time(name, &|| gpu.qw_attention(&qf, &kr, &kr, &a.qn, &a.kn, &a.kc, &a.vc, &ph, (c.n_head, c.n_head_kv, c.head_dim, c.n_rot / 2, self.cap), c.eps, self.theta_scale)));
                }
            }
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
    pub fn prefill_layer(&self, gpu: &Gpu, i: usize, x: &Handle, m: usize, rec: bool) -> Result<Handle> {
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
            if rec {
                if let Some(k) = &self.spec[i] {
                    gpu.qw_copy(&qkv, &k.qkv, m * c.conv_channels());
                    gpu.qw_copy(&z, &k.z, m * c.d_inner);
                    gpu.qw_copy(&ba, &k.ba, m * 2 * c.n_v_heads);
                }
            }
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
        let gg = l.sh_gate.gemm(gpu, &h2, m);
        let uu = l.sh_up.gemm(gpu, &h2, m);
        let sh = gpu.qw_silu_mul(&gg, &uu, m * c.ff_shexp);
        let shout = l.sh_down.gemm(gpu, &sh, m);
        ph("shared gemm", &shout);
        Ok(gpu.qp_combine_rows(&x1, &down, &weights, &shout, &logits, m, c.n_embd, c.n_used, c.n_expert))
    }

    /// The verify-pass layer: [`prefill_layer`](Self::prefill_layer)'s arithmetic for a few tokens with every step on the device (no routes read back;
    /// the routed experts run the row-indexed decode kernels), so a stage's whole pass can be recorded as one CUDA graph. Records the delta-net inputs.
    pub fn verify_layer(&self, gpu: &Gpu, i: usize, x: &Handle, m: usize) -> Handle {
        let c = &self.cfg;
        let l = &self.layers[i];
        let stop = std::env::var("LB_VG_STOP").unwrap_or_default();
        let h = gpu.qp_rms_rows(x, &l.attn_norm, m, c.n_embd, c.eps);
        if stop == "rms" { return x.clone(); }
        let y = if let Some(d) = &l.lin {
            let qkv = d.qkv.launch_m(gpu, &h, m);
            if stop == "qkv gemm" { return x.clone(); }
            let z = d.z.launch_m(gpu, &h, m);
            if stop == "z gemm" { return x.clone(); }
            let ba = gpu.qp_f32_rows(&d.ba, &h, m, 2 * c.n_v_heads, c.n_embd);
            if stop == "ba" { return x.clone(); }
            if let Some(k) = &self.spec[i] {
                gpu.qw_copy(&qkv, &k.qkv, m * c.conv_channels());
                gpu.qw_copy(&z, &k.z, m * c.d_inner);
                gpu.qw_copy(&ba, &k.ba, m * 2 * c.n_v_heads);
                if stop == "copy keep" { return x.clone(); }
            }
            let core = gpu.qp_delta_net(&qkv, &z, &ba, &d.conv, &d.dt, &d.a, &d.norm, &d.conv_state, &d.s, &self.pp2, m, (c.conv_channels(), c.n_k_heads, c.n_v_heads, c.head_k), c.eps);
            if stop == "dn core" { return x.clone(); }
            d.out.launch_m(gpu, &core, m)
        } else {
            let a = l.attn.as_ref().unwrap();
            let qf = a.q.launch_m(gpu, &h, m);
            let k = a.k.launch_m(gpu, &h, m);
            let v = a.v.launch_m(gpu, &h, m);
            if stop == "attn proj" { return x.clone(); }
            let o = gpu.qp_attention(&qf, &k, &v, &a.qn, &a.kn, &a.kc, &a.vc, &self.pp1, m, (c.n_head, c.n_head_kv, c.head_dim, c.n_rot / 2, self.cap), c.eps, self.theta_scale);
            if stop == "attn core" { return x.clone(); }
            a.o.launch_m(gpu, &o, m)
        };
        if stop == "mixer out" { return x.clone(); }
        let x1 = gpu.qw_add(x, &y, m * c.n_embd);
        let h2 = gpu.qp_rms_rows(&x1, &l.post_norm, m, c.n_embd, c.eps);
        let logits = gpu.qp_f32_rows(&l.router_sg, &h2, m, c.n_expert + 1, c.n_embd);
        let (routes, weights) = gpu.qp_route_rows(&logits, m, c.n_expert, c.n_used);
        if stop == "route" { return x.clone(); }
        let h2p = gpu.qw_permute_x(&h2, m * c.n_embd);
        if stop == "permute" { return x.clone(); }
        let down = l.experts.launch_permuted_m(gpu, h2p, &routes, m * c.n_used, m);
        if stop == "experts" { return x.clone(); }
        let gg = l.sh_gate.launch_m(gpu, &h2, m);
        let uu = l.sh_up.launch_m(gpu, &h2, m);
        let sh = gpu.qw_silu_mul(&gg, &uu, m * c.ff_shexp);
        let shout = l.sh_down.launch_m(gpu, &sh, m);
        gpu.qp_combine_rows(&x1, &down, &weights, &shout, &logits, m, c.n_embd, c.n_used, c.n_expert)
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

    /// Save every delta-net layer's recurrent state (the point a speculative chunk can be rolled back to).
    pub fn spec_checkpoint(&self, gpu: &Gpu) {
        let c = &self.cfg;
        for (l, k) in self.layers.iter().zip(&self.spec) {
            if let (Some(d), Some(k)) = (&l.lin, k) {
                gpu.qw_copy(&d.conv_state, &k.ck_conv, 2 * (c.d_conv - 1) * c.conv_channels());
                gpu.qw_copy(&d.s, &k.ck_s, c.n_v_heads * c.head_v() * c.head_k);
            }
        }
    }

    /// Put every delta-net layer's recurrent state back to the checkpoint (a whole chunk undone).
    pub fn spec_restore(&self, gpu: &Gpu) {
        let c = &self.cfg;
        for (l, k) in self.layers.iter().zip(&self.spec) {
            if let (Some(d), Some(k)) = (&l.lin, k) {
                gpu.qw_copy(&k.ck_conv, &d.conv_state, 2 * (c.d_conv - 1) * c.conv_channels());
                gpu.qw_copy(&k.ck_s, &d.s, c.n_v_heads * c.head_v() * c.head_k);
            }
        }
    }

    /// Roll the recurrent state back to the checkpoint and re-apply the first `keep` tokens of the last verified chunk (from its recorded layer
    /// inputs), so the state is exactly that after `keep` tokens. The attention KV caches need nothing: positions past `keep` are overwritten
    /// before they are read.
    pub fn spec_replay(&self, gpu: &Gpu, p0: usize, keep: usize) {
        let c = &self.cfg;
        self.set_chunk(gpu, p0, keep);
        for (l, k) in self.layers.iter().zip(&self.spec) {
            if let (Some(d), Some(k)) = (&l.lin, k) {
                gpu.qw_copy(&k.ck_conv, &d.conv_state, 2 * (c.d_conv - 1) * c.conv_channels());
                gpu.qw_copy(&k.ck_s, &d.s, c.n_v_heads * c.head_v() * c.head_k);
                let _ = gpu.qp_delta_net(&k.qkv, &k.z, &k.ba, &d.conv, &d.dt, &d.a, &d.norm, &d.conv_state, &d.s, &self.pp2, keep, (c.conv_channels(), c.n_k_heads, c.n_v_heads, c.head_k), c.eps);
            }
        }
    }

    /// The cache-writing half of an attention layer for one token (the nextn layer's catch-up: its K/V entry needs only the attention input).
    pub fn kv_only(&self, gpu: &Gpu, i: usize, x: &Handle) {
        let c = &self.cfg;
        let l = &self.layers[i];
        let a = l.attn.as_ref().expect("attention layer");
        let (h, hp) = gpu.qw_rms(x, &l.attn_norm, 1, c.n_embd, c.eps);
        let qf = a.q.launch2(gpu, &h, &hp);
        let k = a.k.launch2(gpu, &h, &hp);
        let v = a.v.launch2(gpu, &h, &hp);
        let _ = gpu.qw_attention(&qf, &k, &v, &a.qn, &a.kn, &a.kc, &a.vc, &self.pos_h, (c.n_head, c.n_head_kv, c.head_dim, c.n_rot / 2, self.cap), c.eps, self.theta_scale);
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
        for k in self.spec.iter().flatten() {
            v.extend([k.qkv.clone(), k.z.clone(), k.ba.clone(), k.ck_conv.clone(), k.ck_s.clone()]);
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
        let down = if skip("experts") { self.zeros.clone() } else { l.experts.launch_permuted(gpu, h2p, &routes, c.n_used) };
        let shout = if skip("shared") {
            self.zeros.clone()
        } else {
            let fused = match (&l.sh_gate, &l.sh_up) {
                (DMat::Q8(g), DMat::Q8(u)) if gpu.kq.sh_fuse => lb_q4_gpu::GpuQ8Mat::gate_up_silu(gpu, g, u, &h2),
                _ => None,
            };
            let sh = fused.unwrap_or_else(|| {
                let g = l.sh_gate.launch(gpu, &h2);
                let u = l.sh_up.launch(gpu, &h2);
                gpu.qw_silu_mul(&g, &u, c.ff_shexp)
            });
            l.sh_down.launch(gpu, &sh)
        };
        let out = gpu.qw_moe_combine(&x1, &down, &weights, &shout, &logits, c.n_expert + 1, c.n_expert, c.n_embd, c.n_used);
        (x1, out)
    }
}
