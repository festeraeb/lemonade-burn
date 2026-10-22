//! Ternary (BitNet b1.58, W1.58A8) building blocks.
//!
//! * [`pack`]: 2-bit packed weight layout (4 weights per byte) and absmean ternarization
//!   exactly as `transformers.integrations.bitnet.WeightQuant` defines it.
//! * [`quant`]: per-token absmax int8 activation quantization (`ActQuant`).
//! * [`gemm`]: int8 x ternary GEMV/GEMM with scalar, AVX2 and AVX-512 (BW / VNNI) paths,
//!   selected at runtime.
//! * [`bf16`]: f32 x bf16 GEMV used for the (unquantized, tied) LM head.

pub mod bf16;
pub mod gemm;
pub mod pack;
pub mod quant;

pub use gemm::{ternary_gemm, ternary_gemm_i32, ternary_gemm_multi, Kernel};
pub use pack::{PackedTernary, TernaryView, BLOCK, BLOCK_BYTES};
pub use quant::{quantize_acts, QuantActs};
