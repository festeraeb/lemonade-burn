//! Ternary (BitNet b1.58) kernels.
//!
//! Weight layout ("LB2" packing, see [`pack`]): each output row of a `[n, k]` weight
//! matrix is stored as `k / 128` blocks of 32 bytes. Byte `j` of block `b` holds in bits
//! `2i..2i+1` the code of the weight at column `b * 128 + i * 32 + j` (`i` in `0..4`).
//! The code is `w + 1`, i.e. `0 => -1`, `1 => 0`, `2 => +1` (`3` is never produced).
//!
//! Storing `w + 1` (unsigned 0..=2) lets SIMD paths use `u8 x i8` multiply-add
//! instructions (`vpmaddubsw`, `vpdpbusd`) directly on the decoded codes:
//! `sum((w + 1) * x) - sum(x) == sum(w * x)`, all in exact integer arithmetic.
//!
//! Activations are quantized per row (token) to int8 with absmax scaling exactly as
//! `transformers.integrations.bitnet.ActQuant` does (see [`quant`]).

pub mod bf16;
pub mod gemv;
pub mod linear;
pub mod pack;
pub mod quant;
pub mod reference;

/// Weights per packed block.
pub const BLOCK: usize = 128;
/// Bytes per packed block.
pub const BLOCK_BYTES: usize = 32;

/// Round `k` up to a multiple of [`BLOCK`].
pub const fn padded_k(k: usize) -> usize {
    k.div_ceil(BLOCK) * BLOCK
}

pub use gemv::{detect_isa, Isa};
pub use linear::{ternary_linear, TernaryWeight};
