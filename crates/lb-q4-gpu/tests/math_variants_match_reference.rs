//! X7a: the Q4_0 rows kernels in each math (`Q4Math`: f32, f32m, f16, dp4a) against
//!   1. the f32 reference kernel on the same upload, and
//!   2. an exact host emulation of what the path computes per block (`emulate`).
//!
//! Tolerances (written in docs/WORK_TO_WEIGHTS.md "## X7" before the first run):
//!   * f32m: **bit-identical** to f32 (same values, same FMAs in the same order).
//!   * f16, dp4a vs their host emulation: |gpu - emu| <= 2e-6 * sum_b |block_b| + 1e-6 per output,
//!     i.e. the per-block numbers are reproduced exactly and only the f32 order across blocks
//!     (lane stride + plane reduce) differs.
//!   * f16 vs f32: normalised error max|gpu - f32| / max|f32| <= 5e-3 (activations rounded to
//!     11 bits, 16-term sums in half).
//!   * dp4a vs f32: normalised error <= 3e-2 (activations rounded to 8 bits per block of 32).
//!
//! The dp4a path runs on any card: CubeCL lowers `dp4a` to `__dp4a` on sm_61+ and to a portable
//! byte polyfill elsewhere (sm_60), so a P100 checks its arithmetic, not its speed.
//! Runs only when a device is named (`LB_WGPU_DEVICE` / `LB_CUDA_DEVICE`).
mod common;
use common::Rng;
use half::f16;
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert, GpuExpertLayer, GpuQ4Mat, Q4Math};

fn gpu() -> Option<Gpu> {
    if std::env::var_os("LB_WGPU_DEVICE").is_none() && std::env::var_os("LB_CUDA_DEVICE").is_none() {
        eprintln!("skipped: no device named (set LB_WGPU_DEVICE or LB_CUDA_DEVICE)");
        return None;
    }
    let g = Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}"));
    if !g.use_plane {
        eprintln!("skipped: the rows kernels (where the math variants live) need plane ops");
        return None;
    }
    eprintln!("device {} compute capability {:?}, auto math {:?}", g.name, g.compute_capability, Q4Math::for_card(g.compute_capability));
    Some(g)
}

fn q4(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

/// Activations with the awkward cases in them: a block of zeros, a block with one large outlier,
/// a few channels 100x the rest (what RMSNorm outputs and SwiGLU products look like).
fn activations(rng: &mut Rng, m: usize, cols: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 2.0).collect();
    for t in 0..m {
        let row = &mut x[t * cols..(t + 1) * cols];
        for v in row[..32].iter_mut() {
            *v = 0.0;
        }
        if cols >= 96 {
            row[70] = 900.0;
        }
        for c in (5..cols).step_by(97) {
            row[c] *= 100.0;
        }
    }
    x
}

fn f16_bits_to_f32(b: u16) -> f32 {
    f16::from_bits(b).to_f32()
}

/// Exactly fused half fma: a*b + c computed exactly (f64 holds it), rounded once to half.
fn hfma(a: f16, b: f16, c: f16) -> f16 {
    f16::from_f64(a.to_f64() * b.to_f64() + c.to_f64())
}

/// One block's value in `math`, the way the kernel computes it, and its f32 magnitude.
fn block_value(math: Q4Math, blk: &[u8], x: &[f32]) -> f32 {
    let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    // q[v] = nibble of value v (0..15): byte i low nibble = value i, high = value i + 16
    let mut q = [0i32; 32];
    for i in 0..16 {
        q[i] = (blk[2 + i] & 15) as i32;
        q[i + 16] = (blk[2 + i] >> 4) as i32;
    }
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    match math {
        Q4Math::F16 => {
            let ebits = (amax.to_bits() >> 23) & 255;
            let inv = f32::from_bits((253 - ebits) << 23);
            let s_x = f32::from_bits((ebits + 1) << 23);
            let xh: Vec<f16> = x.iter().map(|v| f16::from_f32(v * inv)).collect();
            let wq = |v: usize| f16::from_f32((q[v] - 8) as f32);
            // a01 holds values (4j, 4j+1), a23 (4j+2, 4j+3); per j: the low half (j) then the high (j+4)
            let (mut a01, mut a23) = ([f16::ZERO; 2], [f16::ZERO; 2]);
            for j in 0..4 {
                for (jj, _) in [(j, 0), (j + 4, 1)] {
                    for c in 0..2 {
                        a01[c] = hfma(wq(4 * jj + c), xh[4 * jj + c], a01[c]);
                        a23[c] = hfma(wq(4 * jj + 2 + c), xh[4 * jj + 2 + c], a23[c]);
                    }
                }
            }
            let blk = (a01[0].to_f32() + a01[1].to_f32()) + (a23[0].to_f32() + a23[1].to_f32());
            d * s_x * blk
        }
        Q4Math::Dp4a => {
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            let dx = amax / 127.0;
            let qx: Vec<i32> = x.iter().map(|v| (v * inv).round() as i32).collect();
            let sumq: i32 = qx.iter().sum();
            let isum: i32 = (0..32).map(|v| q[v] * qx[v]).sum();
            d * dx * (isum - 8 * sumq) as f32
        }
        Q4Math::F32 | Q4Math::F32Magic => (0..32).map(|v| d * (q[v] - 8) as f32 * x[v]).sum(),
    }
}

/// Host emulation of a whole matvec in `math`: per output, (value, sum of |block values|).
fn emulate(math: Q4Math, w: &[u8], rows: usize, cols: usize, x: &[f32], m: usize) -> Vec<(f32, f32)> {
    let bpr = cols / 32;
    let mut out = Vec::with_capacity(m * rows);
    for t in 0..m {
        for r in 0..rows {
            let (mut s, mut mag) = (0f64, 0f64);
            for b in 0..bpr {
                let blk = &w[(r * bpr + b) * 18..(r * bpr + b + 1) * 18];
                let v = block_value(math, blk, &x[t * cols + b * 32..t * cols + b * 32 + 32]);
                s += v as f64;
                mag += v.abs() as f64;
            }
            out.push((s as f32, mag as f32));
        }
    }
    out
}

fn norm_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale
}

fn tol_vs_f32(math: Q4Math) -> f32 {
    match math {
        Q4Math::F16 => 5e-3,
        Q4Math::Dp4a => 3e-2,
        _ => 0.0,
    }
}

const LOWP: [Q4Math; 2] = [Q4Math::F16, Q4Math::Dp4a];

#[test]
fn matvec_each_math_on_the_model_shapes() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(71);
    gpu.set_q4_rows(4);
    for &(rows, cols) in &[(1536usize, 2048usize), (2048, 1536), (6144, 2048), (512, 2048), (2048, 11776), (64, 64)] {
        let w = q4(&mut rng, rows, cols);
        let mat = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, rows, cols, &w)).unwrap();
        for m in [1usize, 3] {
            for xs in [false, true] {
                gpu.set_q4_xshared(xs);
                let x = activations(&mut rng, m, cols);
                gpu.set_q4_math(Q4Math::F32);
                let want = mat.matvec(&gpu, &x, m).unwrap();
                gpu.set_q4_math(Q4Math::F32Magic);
                let got = mat.matvec(&gpu, &x, m).unwrap();
                let nd = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                assert_eq!(nd, 0, "f32m {rows}x{cols} m={m} xs={xs}: {nd} outputs differ in bits from f32");
                for math in LOWP {
                    gpu.set_q4_math(math);
                    let got = mat.matvec(&gpu, &x, m).unwrap();
                    let emu = emulate(math, &w, rows, cols, &x, m);
                    let mut worst = 0f32;
                    for (i, (g, (e, mag))) in got.iter().zip(&emu).enumerate() {
                        let tol = 2e-6 * mag + 1e-6;
                        assert!((g - e).abs() <= tol, "{math:?} {rows}x{cols} m={m} xs={xs} out {i}: gpu {g} emu {e} (|block| sum {mag})");
                        worst = worst.max((g - e).abs() / (mag + 1e-30));
                    }
                    let ne = norm_err(&got, &want);
                    eprintln!("{:>4} {rows}x{cols} m={m} xs={xs}: vs emulation {worst:.2e} (rel to |block| sum), vs f32 {ne:.2e}", math.name());
                    assert!(ne <= tol_vs_f32(math), "{math:?} {rows}x{cols} m={m}: normalised error vs f32 {ne:e} > {}", tol_vs_f32(math));
                }
            }
        }
    }
    gpu.set_q4_math(Q4Math::F32);
    gpu.set_q4_xshared(false);
}

#[test]
fn expert_ffn_and_routed_layer_each_math() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(73);
    gpu.set_q4_rows(4);
    let (n_ff, n_embd) = (1536usize, 2048usize);
    let (g, u, d) = (q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_embd, n_ff));
    let ex = GpuExpert::upload(&gpu, QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &g), QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &u), QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &d)).unwrap();
    let x = activations(&mut rng, 1, n_embd);
    gpu.set_q4_math(Q4Math::F32);
    let want = ex.ffn(&gpu, &x, 1).unwrap();
    gpu.set_q4_math(Q4Math::F32Magic);
    assert!(ex.ffn(&gpu, &x, 1).unwrap().iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "f32m expert ffn not bit-identical");
    for math in LOWP {
        gpu.set_q4_math(math);
        let ne = norm_err(&ex.ffn(&gpu, &x, 1).unwrap(), &want);
        eprintln!("{:>4} expert ffn (gate/up/silu + down): vs f32 {ne:.2e}", math.name());
        // two matvecs in series: the error of the first feeds the second
        assert!(ne <= 2.0 * tol_vs_f32(math), "{math:?} expert ffn normalised error {ne:e}");
    }

    let n_exp = 6usize;
    let (g, u, d) = (q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_embd, n_ff));
    let layer = GpuExpertLayer::upload_stacked(
        &gpu,
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &g),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &u),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_embd, n_ff, &d),
        n_exp,
    )
    .unwrap();
    let xh = gpu.upload_f32(&x);
    let rh = gpu.upload_u32(vec![9, 9, 3, 0, 5, 0, 0, 0, 2, 0]);
    let run = |gpu: &Gpu| {
        let (o, n) = layer.launch_device(gpu, &xh, 1, &rh, 5, 1, 4);
        gpu.read_all(vec![(o, n)]).unwrap().remove(0)
    };
    gpu.set_q4_math(Q4Math::F32);
    let want = run(&gpu);
    gpu.set_q4_math(Q4Math::F32Magic);
    assert!(run(&gpu).iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "f32m routed layer not bit-identical");
    for math in LOWP {
        gpu.set_q4_math(math);
        let ne = norm_err(&run(&gpu), &want);
        eprintln!("{:>4} routed layer, 4 slots: vs f32 {ne:.2e}", math.name());
        assert!(ne <= 2.0 * tol_vs_f32(math), "{math:?} routed layer normalised error {ne:e}");
    }
    gpu.set_q4_math(Q4Math::F32);
}

#[test]
fn auto_math_follows_the_card() {
    assert_eq!(Q4Math::for_card(Some((6, 0))), Q4Math::F16, "P100");
    assert_eq!(Q4Math::for_card(Some((6, 1))), Q4Math::Dp4a, "GTX 1070 / P106 / P40");
    assert_eq!(Q4Math::for_card(Some((7, 5))), Q4Math::Dp4a, "Turing");
    assert_eq!(Q4Math::for_card(None), Q4Math::F32, "unknown / wgpu");
    for m in Q4Math::ALL {
        assert_eq!(Q4Math::parse(m.name()), Some(m));
        assert_eq!(Q4Math::from_code(m.code()), m);
    }
}
