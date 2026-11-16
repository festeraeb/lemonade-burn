//! The prefill GEMMs against the CPU reference on real Qwen3.6 tensors: dense Q5_K / Q6_K / Q8_0 over 70 token rows, and the grouped routed
//! experts (down Q5_K in layer 20, Q6_K in layer 4) over 40 tokens with 8 routed experts each.
mod common;
use common::{assert_close, Rng};
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuNativeExperts, GpuQ5KNative, GpuQ6KNative, GpuQ8Native};

const PATH: &str = "/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf";

#[test]
fn dense_gemms_match_matvec_reference() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        return;
    }
    let g = lb_gguf::Gguf::open(path).unwrap();
    let gpu = Gpu::open().unwrap();
    for (name, ty) in [("blk.0.attn_gate.weight", GgmlType::Q5K), ("output.weight", GgmlType::Q6K), ("blk.0.attn_qkv.weight", GgmlType::Q8_0), ("blk.3.attn_q.weight", GgmlType::Q6K)] {
        let t = g.tensor(name).unwrap();
        let cols = t.dims[0];
        let rows = 300usize;
        let w = QuantMat::new(ty, rows, cols, &g.tensor_bytes(t)[..rows * ty.row_bytes(cols).unwrap()]);
        let m = 70usize;
        let mut rng = Rng(5);
        let x: Vec<f32> = (0..m * cols).map(|_| rng.unit()).collect();
        let mut want = vec![0f32; m * rows];
        matvec_f32(w, &x, m, &mut want);
        let xh = gpu.upload_f32(&x);
        let out = match ty {
            GgmlType::Q5K => GpuQ5KNative::upload(&gpu, w).unwrap().gemm(&gpu, &xh, m),
            GgmlType::Q6K => GpuQ6KNative::upload(&gpu, w).unwrap().gemm(&gpu, &xh, m),
            _ => GpuQ8Native::upload(&gpu, w).unwrap().gemm(&gpu, &xh, m),
        };
        let got = gpu.read_all(vec![(out, m * rows)]).unwrap().remove(0);
        assert_close(&got, &want, cols, rows, &format!("gemm {} {name}", ty.name()));
    }
}

fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

#[test]
fn grouped_expert_gemm_matches_reference() {
    let path = std::path::Path::new(PATH);
    if !path.exists() {
        return;
    }
    let g = lb_gguf::Gguf::open(path).unwrap();
    let gpu = Gpu::open().unwrap();
    for il in [20usize, 4] {
        let (tg, tu, td) = (g.tensor(&format!("blk.{il}.ffn_gate_exps.weight")).unwrap(), g.tensor(&format!("blk.{il}.ffn_up_exps.weight")).unwrap(), g.tensor(&format!("blk.{il}.ffn_down_exps.weight")).unwrap());
        let (embd, ff, ne) = (tg.dims[0], tg.dims[1], tg.dims[2]);
        let gate = QuantMat::new(GgmlType::Q5K, ne * ff, embd, g.tensor_bytes(tg));
        let up = QuantMat::new(GgmlType::Q5K, ne * ff, embd, g.tensor_bytes(tu));
        let down = QuantMat::new(td.ty, ne * embd, ff, g.tensor_bytes(td));
        let ex = GpuNativeExperts::upload(&gpu, ne, gate, up, down).unwrap();
        let m = 40usize;
        let mut rng = Rng(11);
        let x: Vec<f32> = (0..m * embd).map(|_| rng.unit()).collect();
        // 8 routed experts per token, drawn so that several tokens share experts
        let mut routes = Vec::new();
        for t in 0..m {
            for k in 0..8u32 {
                routes.push((((rng.byte() as u32 * 3 + k * 17) % 256), t as u32));
            }
        }
        let out = ex.gemm(&gpu, &gpu.upload_f32(&x), m, &routes);
        let got = gpu.read_all(vec![(out, routes.len() * embd)]).unwrap().remove(0);
        let dot = |mat: &QuantMat<'_>, r: usize, v: &[f32]| -> f32 {
            let mut b = vec![0f32; mat.cols];
            mat.dequant_row(r, &mut b);
            b.iter().zip(v).map(|(a, c)| *a as f64 * *c as f64).sum::<f64>() as f32
        };
        let mut want = Vec::new();
        for &(e, t) in &routes {
            let xv = &x[t as usize * embd..(t as usize + 1) * embd];
            let h: Vec<f32> = (0..ff).map(|r| silu(dot(&gate, e as usize * ff + r, xv)) * dot(&up, e as usize * ff + r, xv)).collect();
            want.extend((0..embd).map(|r| dot(&down, e as usize * embd + r, &h)));
        }
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("grouped experts layer {il} (down {}): max |gpu-cpu| {worst:e} on outputs up to {scale:e} (ratio {:e})", td.ty.name(), worst / scale);
        assert!(worst <= 2e-5 * scale);
    }
}
