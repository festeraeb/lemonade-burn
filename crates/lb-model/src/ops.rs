//! Custom Burn operations, exposed as a Burn 0.22 backend extension.
//!
//! `#[backend_extension(Flex)]` generates the `Dispatch` routing, so model code keeps using
//! the ordinary backend-agnostic `Tensor<D>` API and only these two ops drop to the
//! backend primitive. The Flex (CPU) implementation calls the `lb-ternary` kernels on the
//! tensor's contiguous storage. A CubeCL (wgpu) implementation would be added as another
//! selector (`Cube: cfg(feature = "wgpu")`) with a `#[cube]` kernel over the same packed
//! bytes viewed as u32 words; it is not implemented yet.

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{backend_extension, Backend, Dispatch, Flex};
use burn::tensor::{DType, Int, Shape, Tensor, TensorData};
use burn_flex::FlexTensor;
use lb_ternary::numa::NodePools;
use lb_ternary::{quant::quantize_acts_into, QuantActs, TernaryView};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[backend_extension(Flex)]
pub trait TernaryBackend: Backend {
    /// `x[m, cols] (f32) @ W^T` where `packed` is `[rows, padded(cols)/4]` u8 in the
    /// `lb_ternary::pack` layout and `scale` is the WeightQuant scale. Activations are
    /// quantized per row to int8 (ActQuant) inside the op. Returns `[m, rows]` f32.
    fn ternary_linear(x: FloatTensor<Self>, packed: IntTensor<Self>, scale: f32, cols: usize) -> FloatTensor<Self>;

    /// Two ternary linears sharing the input `x` (gate/up): the activations are quantized
    /// once and both matrices run in one parallel region. Each output equals
    /// `ternary_linear(x, p_i, scales[i], cols)` bit for bit.
    fn ternary_linear2(x: FloatTensor<Self>, p0: IntTensor<Self>, p1: IntTensor<Self>, scales: [f32; 2], cols: usize) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// Three ternary linears sharing the input `x` (q/k/v), like [`Self::ternary_linear2`].
    fn ternary_linear3(
        x: FloatTensor<Self>,
        p0: IntTensor<Self>,
        p1: IntTensor<Self>,
        p2: IntTensor<Self>,
        scales: [f32; 3],
        cols: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    /// Two-socket variants (see `lb_ternary::numa`): every matrix is stored as two row
    /// parts, `*0` on node 0 and `*1` on node 1, run concurrently on the node pools.
    /// Outputs are identical (bit for bit) to the single-part ops on the whole matrix.
    fn ternary_linear_numa(x: FloatTensor<Self>, a0: IntTensor<Self>, a1: IntTensor<Self>, scale: f32, cols: usize) -> FloatTensor<Self>;

    /// Two-socket [`Self::ternary_linear2`].
    fn ternary_linear2_numa(
        x: FloatTensor<Self>,
        a0: IntTensor<Self>,
        b0: IntTensor<Self>,
        a1: IntTensor<Self>,
        b1: IntTensor<Self>,
        scales: [f32; 2],
        cols: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// Two-socket [`Self::ternary_linear3`].
    #[allow(clippy::too_many_arguments)]
    fn ternary_linear3_numa(
        x: FloatTensor<Self>,
        a0: IntTensor<Self>,
        b0: IntTensor<Self>,
        c0: IntTensor<Self>,
        a1: IntTensor<Self>,
        b1: IntTensor<Self>,
        c1: IntTensor<Self>,
        scales: [f32; 3],
        cols: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    /// Two-socket [`Self::bf16_linear`]: rows of `w` split into `w0` (node 0) and `w1`.
    fn bf16_linear_numa(x: FloatTensor<Self>, w0: FloatTensor<Self>, w1: FloatTensor<Self>) -> FloatTensor<Self>;

    /// `x[m, k] (f32) @ W^T` with `W[n, k]` stored as bf16 (the tied LM head).
    fn bf16_linear(x: FloatTensor<Self>, w: FloatTensor<Self>) -> FloatTensor<Self>;
}

/// Cumulative wall time (ns) and call counts of the custom ops, for profiling.
pub static PROF_TERNARY_NS: AtomicU64 = AtomicU64::new(0);
pub static PROF_TERNARY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static PROF_BF16_NS: AtomicU64 = AtomicU64::new(0);
/// Ternary op time spent in single-token (decode) calls.
pub static PROF_TERNARY_M1_NS: AtomicU64 = AtomicU64::new(0);
/// Time inside the activation quantizer (all ternary calls).
pub static PROF_QUANT_NS: AtomicU64 = AtomicU64::new(0);
/// Time inside the GEMM/GEMV kernel call (all ternary calls).
pub static PROF_GEMM_NS: AtomicU64 = AtomicU64::new(0);

pub fn prof_reset() {
    PROF_TERNARY_NS.store(0, Ordering::Relaxed);
    PROF_TERNARY_CALLS.store(0, Ordering::Relaxed);
    PROF_BF16_NS.store(0, Ordering::Relaxed);
    PROF_TERNARY_M1_NS.store(0, Ordering::Relaxed);
    PROF_QUANT_NS.store(0, Ordering::Relaxed);
    PROF_GEMM_NS.store(0, Ordering::Relaxed);
}

thread_local! {
    static QBUF: RefCell<QuantActs> = RefCell::new(QuantActs::default());
}

fn dims2(shape: &Shape) -> (usize, usize) {
    assert_eq!(shape.num_dims(), 2, "expected a rank-2 tensor");
    (shape[0], shape[1])
}

fn contiguous(t: FlexTensor) -> FlexTensor {
    if t.is_contiguous() {
        t
    } else {
        t.to_contiguous()
    }
}

impl TernaryBackend for Flex {
    fn ternary_linear(x: FlexTensor, packed: FlexTensor, scale: f32, cols: usize) -> FlexTensor {
        flex_ternary_multi(x, vec![packed], &[scale], cols).pop().unwrap()
    }

    fn ternary_linear2(x: FlexTensor, p0: FlexTensor, p1: FlexTensor, scales: [f32; 2], cols: usize) -> (FlexTensor, FlexTensor) {
        let mut v = flex_ternary_multi(x, vec![p0, p1], &scales, cols).into_iter();
        (v.next().unwrap(), v.next().unwrap())
    }

    fn ternary_linear3(
        x: FlexTensor,
        p0: FlexTensor,
        p1: FlexTensor,
        p2: FlexTensor,
        scales: [f32; 3],
        cols: usize,
    ) -> (FlexTensor, FlexTensor, FlexTensor) {
        let mut v = flex_ternary_multi(x, vec![p0, p1, p2], &scales, cols).into_iter();
        (v.next().unwrap(), v.next().unwrap(), v.next().unwrap())
    }

    fn ternary_linear_numa(x: FlexTensor, a0: FlexTensor, a1: FlexTensor, scale: f32, cols: usize) -> FlexTensor {
        flex_ternary_numa(x, vec![a0], vec![a1], &[scale], cols).pop().unwrap()
    }

    fn ternary_linear2_numa(
        x: FlexTensor,
        a0: FlexTensor,
        b0: FlexTensor,
        a1: FlexTensor,
        b1: FlexTensor,
        scales: [f32; 2],
        cols: usize,
    ) -> (FlexTensor, FlexTensor) {
        let mut v = flex_ternary_numa(x, vec![a0, b0], vec![a1, b1], &scales, cols).into_iter();
        (v.next().unwrap(), v.next().unwrap())
    }

    fn ternary_linear3_numa(
        x: FlexTensor,
        a0: FlexTensor,
        b0: FlexTensor,
        c0: FlexTensor,
        a1: FlexTensor,
        b1: FlexTensor,
        c1: FlexTensor,
        scales: [f32; 3],
        cols: usize,
    ) -> (FlexTensor, FlexTensor, FlexTensor) {
        let mut v = flex_ternary_numa(x, vec![a0, b0, c0], vec![a1, b1, c1], &scales, cols).into_iter();
        (v.next().unwrap(), v.next().unwrap(), v.next().unwrap())
    }

    fn bf16_linear_numa(x: FlexTensor, w0: FlexTensor, w1: FlexTensor) -> FlexTensor {
        let t0 = Instant::now();
        let np = NodePools::global().expect("bf16_linear_numa needs LB_NUMA");
        let x = contiguous(x);
        let (w0, w1) = (contiguous(w0), contiguous(w1));
        assert_eq!(x.dtype(), DType::F32);
        assert!(w0.dtype() == DType::BF16 && w1.dtype() == DType::BF16, "bf16_linear expects bf16 weights");
        let (m, k) = dims2(x.layout().shape());
        let (n0, k0) = dims2(w0.layout().shape());
        let (n1, k1) = dims2(w1.layout().shape());
        assert!(k == k0 && k == k1);
        let xs: &[f32] = x.as_slice::<f32>().expect("contiguous f32");
        let (ws0, ws1) = (bf16_bits(w0.bytes()), bf16_bits(w1.bytes()));
        let n = n0 + n1;
        let mut out = vec![0f32; m * n];
        let kernel = lb_ternary::Kernel::detect();
        for (i, row) in out.chunks_mut(n).enumerate() {
            let xi = &xs[i * k..(i + 1) * k];
            let (o0, o1) = row.split_at_mut(n0);
            np.join(
                || lb_ternary::bf16::bf16_gemv(ws0, n0, k, xi, o0, kernel),
                || lb_ternary::bf16::bf16_gemv(ws1, n1, k, xi, o1, kernel),
            );
        }
        let r = FlexTensor::from_data(TensorData::new(out, [m, n]));
        PROF_BF16_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    fn bf16_linear(x: FlexTensor, w: FlexTensor) -> FlexTensor {
        let t0 = Instant::now();
        let x = contiguous(x);
        let w = contiguous(w);
        assert_eq!(x.dtype(), DType::F32);
        assert_eq!(w.dtype(), DType::BF16, "bf16_linear expects bf16 weights");
        let (m, k) = dims2(x.layout().shape());
        let (n, k2) = dims2(w.layout().shape());
        assert_eq!(k, k2);
        let xs: &[f32] = x.as_slice::<f32>().expect("contiguous f32");
        let ws: &[u16] = bf16_bits(w.bytes());
        let mut out = vec![0f32; m * n];
        let kernel = lb_ternary::Kernel::detect();
        for i in 0..m {
            lb_ternary::bf16::bf16_gemv(ws, n, k, &xs[i * k..(i + 1) * k], &mut out[i * n..(i + 1) * n], kernel);
        }
        let r = FlexTensor::from_data(TensorData::new(out, [m, n]));
        PROF_BF16_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }
}

/// bf16 storage viewed as its u16 bit patterns.
fn bf16_bits(b: &[u8]) -> &[u16] {
    // bf16 storage is 2-byte aligned plain little-endian u16 bit patterns
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) }
}

fn ternary_views<'a>(packed: &'a [FlexTensor], scales: &[f32], cols: usize) -> Vec<TernaryView<'a>> {
    packed
        .iter()
        .zip(scales)
        .map(|(p, &scale)| {
            assert_eq!(p.dtype(), DType::U8, "ternary_linear expects u8 packed weights");
            let (rows, _rb) = dims2(p.layout().shape());
            TernaryView::new(rows, cols, p.as_slice::<u8>().expect("contiguous u8"), scale)
        })
        .collect()
}

fn prof_ternary(t0: Instant, m: usize) {
    let dt = t0.elapsed().as_nanos() as u64;
    PROF_TERNARY_NS.fetch_add(dt, Ordering::Relaxed);
    if m == 1 {
        PROF_TERNARY_M1_NS.fetch_add(dt, Ordering::Relaxed);
    }
    PROF_TERNARY_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Two-socket implementation: quantize once on node 0, then each node runs the fused GEMM
/// over its row parts; part outputs are stitched back into `[m, rows]` per matrix.
fn flex_ternary_numa(x: FlexTensor, parts0: Vec<FlexTensor>, parts1: Vec<FlexTensor>, scales: &[f32], cols: usize) -> Vec<FlexTensor> {
    let np = NodePools::global().expect("ternary_linear_numa needs LB_NUMA");
    flex_ternary_numa_with(np, x, parts0, parts1, scales, cols)
}

fn flex_ternary_numa_with(np: &NodePools, x: FlexTensor, parts0: Vec<FlexTensor>, parts1: Vec<FlexTensor>, scales: &[f32], cols: usize) -> Vec<FlexTensor> {
    let t0 = Instant::now();
    assert!(parts0.len() == scales.len() && parts1.len() == scales.len());
    let x = contiguous(x);
    let parts0: Vec<FlexTensor> = parts0.into_iter().map(contiguous).collect();
    let parts1: Vec<FlexTensor> = parts1.into_iter().map(contiguous).collect();
    assert_eq!(x.dtype(), DType::F32, "ternary_linear expects f32 activations");
    let (m, k) = dims2(x.layout().shape());
    assert_eq!(k, cols);
    let xs: &[f32] = x.as_slice::<f32>().expect("contiguous f32");
    let (v0, v1) = (ternary_views(&parts0, scales, cols), ternary_views(&parts1, scales, cols));
    let kernel = lb_ternary::Kernel::detect();
    let tq = Instant::now();
    let q = lb_ternary::quantize_acts(xs, m, k); // calling thread is on node 0
    let tg = Instant::now();
    let mut b0: Vec<Vec<f32>> = v0.iter().map(|v| vec![0f32; m * v.rows]).collect();
    let mut b1: Vec<Vec<f32>> = v1.iter().map(|v| vec![0f32; m * v.rows]).collect();
    {
        let mut o0: Vec<&mut [f32]> = b0.iter_mut().map(|b| b.as_mut_slice()).collect();
        let mut o1: Vec<&mut [f32]> = b1.iter_mut().map(|b| b.as_mut_slice()).collect();
        np.join(
            || lb_ternary::ternary_gemm_multi(&v0, &q, kernel, &mut o0),
            || lb_ternary::ternary_gemm_multi(&v1, &q, kernel, &mut o1),
        );
    }
    PROF_QUANT_NS.fetch_add((tg - tq).as_nanos() as u64, Ordering::Relaxed);
    PROF_GEMM_NS.fetch_add(tg.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let r = b0
        .into_iter()
        .zip(b1)
        .map(|(p0, p1)| {
            let (n0, n1) = (p0.len() / m, p1.len() / m);
            let mut out = Vec::with_capacity(m * (n0 + n1));
            for t in 0..m {
                out.extend_from_slice(&p0[t * n0..(t + 1) * n0]);
                out.extend_from_slice(&p1[t * n1..(t + 1) * n1]);
            }
            FlexTensor::from_data(TensorData::new(out, [m, n0 + n1]))
        })
        .collect();
    prof_ternary(t0, m);
    r
}

/// Flex implementation shared by the single and fused ternary ops: quantize `x` once,
/// then one fused GEMM over all `packed` matrices.
fn flex_ternary_multi(x: FlexTensor, packed: Vec<FlexTensor>, scales: &[f32], cols: usize) -> Vec<FlexTensor> {
    let t0 = Instant::now();
    assert_eq!(packed.len(), scales.len());
    let x = contiguous(x);
    let packed: Vec<FlexTensor> = packed.into_iter().map(contiguous).collect();
    assert_eq!(x.dtype(), DType::F32, "ternary_linear expects f32 activations");
    let (m, k) = dims2(x.layout().shape());
    assert_eq!(k, cols);
    let xs: &[f32] = x.as_slice::<f32>().expect("contiguous f32");
    let views = ternary_views(&packed, scales, cols);
    let mut bufs: Vec<Vec<f32>> = views.iter().map(|v| vec![0f32; m * v.rows]).collect();
    QBUF.with(|q| {
        let mut q = q.borrow_mut();
        let tq = Instant::now();
        quantize_acts_into(xs, m, k, &mut q);
        let tg = Instant::now();
        let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
        lb_ternary::ternary_gemm_multi(&views, &q, lb_ternary::Kernel::detect(), &mut outs);
        PROF_QUANT_NS.fetch_add((tg - tq).as_nanos() as u64, Ordering::Relaxed);
        PROF_GEMM_NS.fetch_add(tg.elapsed().as_nanos() as u64, Ordering::Relaxed);
    });
    let r: Vec<FlexTensor> = bufs
        .into_iter()
        .zip(&views)
        .map(|(b, v)| FlexTensor::from_data(TensorData::new(b, [m, v.rows])))
        .collect();
    prof_ternary(t0, m);
    r
}

/// High-level wrapper: `x[m, cols] -> [m, rows]`.
pub fn ternary_linear(x: Tensor<2>, packed: &Tensor<2, Int>, scale: f32, cols: usize) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as TernaryBackend>::ternary_linear(
        x.into_dispatch(),
        packed.clone().into_dispatch(),
        scale,
        cols,
    ))
}

/// High-level wrapper for [`TernaryBackend::ternary_linear2`].
pub fn ternary_linear2(x: Tensor<2>, p: [&Tensor<2, Int>; 2], scales: [f32; 2], cols: usize) -> [Tensor<2>; 2] {
    let (a, b) = <Dispatch as TernaryBackend>::ternary_linear2(x.into_dispatch(), p[0].clone().into_dispatch(), p[1].clone().into_dispatch(), scales, cols);
    [Tensor::from_dispatch(a), Tensor::from_dispatch(b)]
}

/// High-level wrapper for [`TernaryBackend::ternary_linear3`].
pub fn ternary_linear3(x: Tensor<2>, p: [&Tensor<2, Int>; 3], scales: [f32; 3], cols: usize) -> [Tensor<2>; 3] {
    let (a, b, c) = <Dispatch as TernaryBackend>::ternary_linear3(
        x.into_dispatch(),
        p[0].clone().into_dispatch(),
        p[1].clone().into_dispatch(),
        p[2].clone().into_dispatch(),
        scales,
        cols,
    );
    [Tensor::from_dispatch(a), Tensor::from_dispatch(b), Tensor::from_dispatch(c)]
}

/// Two-socket wrapper: `parts[i] = (node-0 part, node-1 part)` of matrix `i`.
pub fn ternary_linear_numa_n<const N: usize>(x: Tensor<2>, parts: [(&Tensor<2, Int>, &Tensor<2, Int>); N], scales: [f32; N], cols: usize) -> [Tensor<2>; N] {
    let d = |t: &Tensor<2, Int>| t.clone().into_dispatch();
    let x = x.into_dispatch();
    let out: Vec<Tensor<2>> = match N {
        1 => vec![Tensor::from_dispatch(<Dispatch as TernaryBackend>::ternary_linear_numa(x, d(parts[0].0), d(parts[0].1), scales[0], cols))],
        2 => {
            let (a, b) = <Dispatch as TernaryBackend>::ternary_linear2_numa(x, d(parts[0].0), d(parts[1].0), d(parts[0].1), d(parts[1].1), [scales[0], scales[1]], cols);
            vec![Tensor::from_dispatch(a), Tensor::from_dispatch(b)]
        }
        3 => {
            let (a, b, c) = <Dispatch as TernaryBackend>::ternary_linear3_numa(
                x,
                d(parts[0].0),
                d(parts[1].0),
                d(parts[2].0),
                d(parts[0].1),
                d(parts[1].1),
                d(parts[2].1),
                [scales[0], scales[1], scales[2]],
                cols,
            );
            vec![Tensor::from_dispatch(a), Tensor::from_dispatch(b), Tensor::from_dispatch(c)]
        }
        _ => unreachable!("1 to 3 matrices"),
    };
    out.try_into().unwrap_or_else(|_| unreachable!())
}

/// Two-socket wrapper for the LM head (rows split into `w0` on node 0 and `w1`).
pub fn bf16_linear_numa(x: Tensor<2>, w0: &Tensor<2>, w1: &Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as TernaryBackend>::bf16_linear_numa(x.into_dispatch(), w0.clone().into_dispatch(), w1.clone().into_dispatch()))
}

/// High-level wrapper: `x[m, k] @ w[n, k]^T` with bf16 `w`.
pub fn bf16_linear(x: Tensor<2>, w: &Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as TernaryBackend>::bf16_linear(x.into_dispatch(), w.clone().into_dispatch()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lb_ternary::PackedTernary;

    fn u8_tensor(bytes: Vec<u8>, rows: usize) -> FlexTensor {
        let rb = bytes.len() / rows;
        FlexTensor::from_data(TensorData::from_bytes_vec(bytes, [rows, rb], DType::U8))
    }

    /// The two-socket path (row parts on two pools, stitched back) is bit-identical to the
    /// single fused op, for decode and prefill shapes and uneven splits.
    #[test]
    fn numa_split_matches_single_bit_for_bit() {
        let np = NodePools::new(&[vec![0], vec![0]], 0.5);
        let cols = 512;
        let mut s = 99u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mats: Vec<(PackedTernary, usize)> = [(96usize, 0.8f32, 37usize), (40, 1.4, 1), (130, 0.6, 129)]
            .iter()
            .map(|&(rows, sc, split)| {
                let t: Vec<i8> = (0..rows * cols).map(|_| (rnd() % 3) as i8 - 1).collect();
                (PackedTernary::from_ternary(rows, cols, &t, sc), split)
            })
            .collect();
        let scales: Vec<f32> = mats.iter().map(|(p, _)| p.scale).collect();
        for m in [1usize, 3] {
            let xv: Vec<f32> = (0..m * cols).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
            let x = || FlexTensor::from_data(TensorData::new(xv.clone(), [m, cols]));
            let whole: Vec<FlexTensor> = mats.iter().map(|(p, _)| u8_tensor(p.data.clone(), p.rows)).collect();
            let expect = flex_ternary_multi(x(), whole, &scales, cols);
            let rb = cols / 4;
            let p0: Vec<FlexTensor> = mats.iter().map(|(p, sp)| u8_tensor(p.data[..sp * rb].to_vec(), *sp)).collect();
            let p1: Vec<FlexTensor> = mats.iter().map(|(p, sp)| u8_tensor(p.data[sp * rb..].to_vec(), p.rows - sp)).collect();
            let got = flex_ternary_numa_with(&np, x(), p0, p1, &scales, cols);
            for (g, e) in got.iter().zip(&expect) {
                assert_eq!(g.layout().shape(), e.layout().shape());
                let (g, e) = (g.as_slice::<f32>().unwrap(), e.as_slice::<f32>().unwrap());
                assert!(g.iter().zip(e).all(|(a, b)| a.to_bits() == b.to_bits()), "m={m}");
            }
        }
    }
}
