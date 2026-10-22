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

    /// `x[m, k] (f32) @ W^T` with `W[n, k]` stored as bf16 (the tied LM head).
    fn bf16_linear(x: FloatTensor<Self>, w: FloatTensor<Self>) -> FloatTensor<Self>;
}

/// Cumulative wall time (ns) and call counts of the custom ops, for profiling.
pub static PROF_TERNARY_NS: AtomicU64 = AtomicU64::new(0);
pub static PROF_TERNARY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static PROF_BF16_NS: AtomicU64 = AtomicU64::new(0);

pub fn prof_reset() {
    PROF_TERNARY_NS.store(0, Ordering::Relaxed);
    PROF_TERNARY_CALLS.store(0, Ordering::Relaxed);
    PROF_BF16_NS.store(0, Ordering::Relaxed);
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
        let t0 = Instant::now();
        let x = contiguous(x);
        let packed = contiguous(packed);
        assert_eq!(x.dtype(), DType::F32, "ternary_linear expects f32 activations");
        assert_eq!(packed.dtype(), DType::U8, "ternary_linear expects u8 packed weights");
        let (m, k) = dims2(x.layout().shape());
        assert_eq!(k, cols);
        let (rows, _rb) = dims2(packed.layout().shape());
        let xs: &[f32] = x.as_slice::<f32>().expect("contiguous f32");
        let ws: &[u8] = packed.as_slice::<u8>().expect("contiguous u8");
        let view = TernaryView::new(rows, cols, ws, scale);
        let mut out = vec![0f32; m * rows];
        QBUF.with(|q| {
            let mut q = q.borrow_mut();
            quantize_acts_into(xs, m, k, &mut q);
            lb_ternary::ternary_gemm(view, &q, lb_ternary::Kernel::detect(), &mut out);
        });
        let r = FlexTensor::from_data(TensorData::new(out, [m, rows]));
        PROF_TERNARY_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        PROF_TERNARY_CALLS.fetch_add(1, Ordering::Relaxed);
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
        let wb = w.bytes();
        // bf16 storage is 2-byte aligned plain little-endian u16 bit patterns
        let ws: &[u16] = unsafe { std::slice::from_raw_parts(wb.as_ptr() as *const u16, wb.len() / 2) };
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

/// High-level wrapper: `x[m, cols] -> [m, rows]`.
pub fn ternary_linear(x: Tensor<2>, packed: &Tensor<2, Int>, scale: f32, cols: usize) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as TernaryBackend>::ternary_linear(
        x.into_dispatch(),
        packed.clone().into_dispatch(),
        scale,
        cols,
    ))
}

/// High-level wrapper: `x[m, k] @ w[n, k]^T` with bf16 `w`.
pub fn bf16_linear(x: Tensor<2>, w: &Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as TernaryBackend>::bf16_linear(x.into_dispatch(), w.clone().into_dispatch()))
}
