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
use burn_std::Bytes;
use lb_ternary::{quant::quantize_acts_into, QuantActs, SharedBytes, SharedTernary, SpinPool, TernaryView};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
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
        let kernel = lb_ternary::Kernel::detect();
        if m == 1 && use_spin_pool() {
            let w16: Arc<dyn AsRef<[u16]> + Send + Sync> = Arc::new(Bf16Storage(w.data_arc()));
            let out = lb_ternary::bf16::bf16_gemv_pool(SpinPool::global(), w16, n, k, Arc::new(xs.to_vec()), kernel);
            let r = FlexTensor::from_data(TensorData::new(out, [m, n]));
            PROF_BF16_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return r;
        }
        let ws: &[u16] = bf16_bits(w.bytes());
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            lb_ternary::bf16::bf16_gemv(ws, n, k, &xs[i * k..(i + 1) * k], &mut out[i * n..(i + 1) * n], kernel);
        }
        let r = FlexTensor::from_data(TensorData::new(out, [m, n]));
        PROF_BF16_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }
}

/// Single-token ops run on the spinning [`SpinPool`] unless `LB_POOL=rayon`.
fn use_spin_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LB_POOL").map(|v| v != "rayon").unwrap_or(true))
}

/// bf16 storage viewed as its u16 bit patterns.
fn bf16_bits(b: &[u8]) -> &[u16] {
    // bf16 storage is 2-byte aligned plain little-endian u16 bit patterns
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) }
}

/// A Burn tensor's shared storage, as owned bytes for pool tasks (whole buffer; the
/// tensors passed here are contiguous at offset 0, checked by the callers).
struct TensorStorage(Arc<Bytes>);
impl AsRef<[u8]> for TensorStorage {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// bf16 tensor storage as u16 bit patterns for pool tasks.
struct Bf16Storage(Arc<Bytes>);
impl AsRef<[u16]> for Bf16Storage {
    fn as_ref(&self) -> &[u16] {
        bf16_bits(&self.0)
    }
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
    let views: Vec<TernaryView<'_>> = packed
        .iter()
        .zip(scales)
        .map(|(p, &scale)| {
            assert_eq!(p.dtype(), DType::U8, "ternary_linear expects u8 packed weights");
            let (rows, _rb) = dims2(p.layout().shape());
            TernaryView::new(rows, cols, p.as_slice::<u8>().expect("contiguous u8"), scale)
        })
        .collect();
    let kernel = lb_ternary::Kernel::detect();
    let bufs: Vec<Vec<f32>> = if m == 1 && use_spin_pool() {
        let tq = Instant::now();
        let mut q = QuantActs::default();
        quantize_acts_into(xs, m, k, &mut q);
        let tg = Instant::now();
        let mats: Vec<SharedTernary> = packed
            .iter()
            .zip(&views)
            .map(|(p, v)| {
                let data: SharedBytes = Arc::new(TensorStorage(p.data_arc()));
                assert_eq!(data.as_ref().as_ref().len(), v.data.len(), "packed tensor must own its whole buffer");
                SharedTernary { data, rows: v.rows, cols, scale: v.scale }
            })
            .collect();
        let out = lb_ternary::ternary_gemv_pool(SpinPool::global(), &mats, Arc::new(q), kernel);
        PROF_QUANT_NS.fetch_add((tg - tq).as_nanos() as u64, Ordering::Relaxed);
        PROF_GEMM_NS.fetch_add(tg.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    } else {
        flex_gemm_rayon(xs, m, k, &views, kernel)
    };
    let r: Vec<FlexTensor> = bufs
        .into_iter()
        .zip(&views)
        .map(|(b, v)| FlexTensor::from_data(TensorData::new(b, [m, v.rows])))
        .collect();
    let dt = t0.elapsed().as_nanos() as u64;
    PROF_TERNARY_NS.fetch_add(dt, Ordering::Relaxed);
    if m == 1 {
        PROF_TERNARY_M1_NS.fetch_add(dt, Ordering::Relaxed);
    }
    PROF_TERNARY_CALLS.fetch_add(1, Ordering::Relaxed);
    r
}

/// Rayon path (prefill, or `LB_POOL=rayon`): quantize into the thread-local buffer, then
/// one fused GEMM.
fn flex_gemm_rayon(xs: &[f32], m: usize, k: usize, views: &[TernaryView<'_>], kernel: lb_ternary::Kernel) -> Vec<Vec<f32>> {
    let mut bufs: Vec<Vec<f32>> = views.iter().map(|v| vec![0f32; m * v.rows]).collect();
    QBUF.with(|q| {
        let mut q = q.borrow_mut();
        let tq = Instant::now();
        quantize_acts_into(xs, m, k, &mut q);
        let tg = Instant::now();
        let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
        lb_ternary::ternary_gemm_multi(views, &q, kernel, &mut outs);
        PROF_QUANT_NS.fetch_add((tg - tq).as_nanos() as u64, Ordering::Relaxed);
        PROF_GEMM_NS.fetch_add(tg.elapsed().as_nanos() as u64, Ordering::Relaxed);
    });
    bufs
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

/// High-level wrapper: `x[m, k] @ w[n, k]^T` with bf16 `w`.
pub fn bf16_linear(x: Tensor<2>, w: &Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as TernaryBackend>::bf16_linear(x.into_dispatch(), w.clone().into_dispatch()))
}
