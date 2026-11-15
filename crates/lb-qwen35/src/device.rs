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
    Q8(GpuQ8Mat),
}

impl DMat {
    fn new(gpu: &Gpu, q: &QuantMat<'_>) -> Result<Self> {
        let r = match q.ty {
            GgmlType::Q5K => GpuQ5KNative::upload(gpu, *q).map(DMat::Q5),
            GgmlType::Q6K => GpuQ6KNative::upload(gpu, *q).map(DMat::Q6),
            GgmlType::Q8_0 => GpuQ8Mat::upload(gpu, *q).map(DMat::Q8),
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
}

struct DLin {
    qkv: DMat,
    z: DMat,
    out: DMat,
    beta: Handle,
    alpha: Handle,
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
    pos_h: Handle,
}

pub struct DevLayer {
    attn_norm: Handle,
    post_norm: Handle,
    lin: Option<DLin>,
    attn: Option<DAttn>,
    router: Handle,
    experts: GpuNativeExperts,
    sh_gate: DMat,
    sh_up: DMat,
    sh_down: DMat,
    sh_gate_inp: Handle,
}

pub struct DevModel {
    pub cfg: Config,
    pub cap: usize,
    pub layers: Vec<DevLayer>,
    theta_scale: f32,
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
        Ok(DevModel { cfg: c, cap, layers, theta_scale })
    }

    fn upload_layer(gpu: &Gpu, c: &Config, l: &Layer<'_>, cap: usize) -> Result<DevLayer> {
        let f = |v: &[f32]| gpu.upload_f32(v);
        let (lin, attn) = match (&l.lin, &l.attn) {
            (Some(w), _) => (
                Some(DLin {
                    qkv: DMat::new(gpu, &w.qkv.0)?,
                    z: DMat::new(gpu, &w.z.0)?,
                    out: DMat::new(gpu, &w.out.0)?,
                    beta: f(&w.beta),
                    alpha: f(&w.alpha),
                    conv: f(&w.conv),
                    a: f(&w.a),
                    dt: f(&w.dt_bias),
                    norm: f(&w.norm),
                    conv_state: f(&vec![0.0; (c.d_conv - 1) * c.conv_channels()]),
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
                    pos_h: gpu.upload_u32(vec![0]),
                }),
            ),
            _ => unreachable!(),
        };
        Ok(DevLayer {
            attn_norm: f(&l.attn_norm),
            post_norm: f(&l.post_norm),
            lin,
            attn,
            router: f(&l.router),
            experts: GpuNativeExperts::upload(gpu, c.n_expert, l.gate_exps, l.up_exps, l.down_exps).map_err(|e| anyhow!(e))?,
            sh_gate: DMat::new(gpu, &l.sh_gate.0)?,
            sh_up: DMat::new(gpu, &l.sh_up.0)?,
            sh_down: DMat::new(gpu, &l.sh_down.0)?,
            sh_gate_inp: f(&l.sh_gate_inp),
        })
    }

    /// Zero every layer's recurrent state and KV cache (a new sequence).
    pub fn reset(&self, gpu: &Gpu) {
        let c = &self.cfg;
        for l in &self.layers {
            if let Some(d) = &l.lin {
                gpu.write_f32(&d.conv_state, &vec![0.0; (c.d_conv - 1) * c.conv_channels()]);
                gpu.write_f32(&d.s, &vec![0.0; c.n_v_heads * c.head_v() * c.head_k]);
            }
            if let Some(a) = &l.attn {
                gpu.write_f32(&a.kc, &vec![0.0; self.cap * c.n_head_kv * c.head_dim]);
                gpu.write_f32(&a.vc, &vec![0.0; self.cap * c.n_head_kv * c.head_dim]);
            }
        }
    }

    /// One layer for one token at `pos`. Returns (`x` after the mixer, layer output), both on the device.
    pub fn layer_forward(&self, gpu: &Gpu, i: usize, x: &Handle, pos: usize) -> (Handle, Handle) {
        let c = &self.cfg;
        let l = &self.layers[i];
        let h = gpu.qw_rms(x, &l.attn_norm, 1, c.n_embd, c.eps);
        let y = if let Some(d) = &l.lin {
            let qkv = d.qkv.launch(gpu, &h);
            let z = d.z.launch(gpu, &h);
            let braw = gpu.qw_f32_matvec(&d.beta, &h, c.n_v_heads, c.n_embd);
            let araw = gpu.qw_f32_matvec(&d.alpha, &h, c.n_v_heads, c.n_embd);
            let core = gpu.qw_delta_net(&qkv, &z, &braw, &araw, &d.conv, &d.dt, &d.a, &d.norm, &d.conv_state, &d.s, (c.conv_channels(), c.d_conv, c.n_k_heads, c.n_v_heads, c.head_k, c.head_v()), c.eps);
            d.out.launch(gpu, &core)
        } else {
            let a = l.attn.as_ref().unwrap();
            let qf = a.q.launch(gpu, &h);
            let k = a.k.launch(gpu, &h);
            let v = a.v.launch(gpu, &h);
            gpu.qw_write_u32(&a.pos_h, &[pos as u32]);
            let o = gpu.qw_attention(&qf, &k, &v, &a.qn, &a.kn, &a.kc, &a.vc, &a.pos_h, (c.n_head, c.n_head_kv, c.head_dim, c.n_rot / 2, self.cap), c.eps, self.theta_scale);
            a.o.launch(gpu, &o)
        };
        let x1 = gpu.qw_add(x, &y, c.n_embd);
        let h2 = gpu.qw_rms(&x1, &l.post_norm, 1, c.n_embd, c.eps);
        let logits = gpu.qw_f32_matvec(&l.router, &h2, c.n_expert, c.n_embd);
        let (routes, weights) = gpu.qw_route(&logits, c.n_expert, c.n_used);
        let down = l.experts.launch(gpu, &h2, &routes, c.n_used);
        let sg = gpu.qw_f32_matvec(&l.sh_gate_inp, &h2, 1, c.n_embd);
        let g = l.sh_gate.launch(gpu, &h2);
        let u = l.sh_up.launch(gpu, &h2);
        let sh = gpu.qw_silu_mul(&g, &u, c.ff_shexp);
        let shout = l.sh_down.launch(gpu, &sh);
        let out = gpu.qw_moe_combine(&x1, &down, &weights, &shout, &sg, c.n_embd, c.n_used);
        (x1, out)
    }
}
