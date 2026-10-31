//! GGUF v2/v3 reader (memory-mapped) with block dequantization and CPU matvec kernels.
//!
//! Written from the GGUF file-format description (header, key/value table, tensor table,
//! aligned data section) and the published block layouts of the quant types we need:
//!
//! * `F32`, `F16`: plain little-endian floats.
//! * `Q8_0`: blocks of 32 values, one f16 scale `d` then 32 signed bytes; value = `d * q`.
//! * `Q4_0`: blocks of 32 values, one f16 scale `d` then 16 bytes of nibbles. Byte `j`
//!   holds value `j` in its low nibble and value `j + 16` in its high nibble, both as
//!   unsigned 0..15 with an offset of 8: value = `d * (nibble - 8)`.
//! * `Q6_K`: super-blocks of 256 values, 210 bytes: 128 bytes of low nibbles (`ql`),
//!   64 bytes of 2-bit highs (`qh`), 16 signed 8-bit sub-scales, one f16 super scale `d`.
//!   Each 128-value half is four groups of 32 laid out as in [`dequant_q6_k_block`].
//!
//! Tensor shapes follow the GGUF convention: `dims[0]` is the contiguous (fastest) axis,
//! so a linear layer's weight `[in, out]` in GGUF terms is `out` rows of `in` columns here.

pub mod kernels;
pub mod reader;
pub mod types;

pub use kernels::{matvec_f32, QuantMat};
pub use reader::{Gguf, GgufError, MetaValue, TensorInfo};
pub use types::GgmlType;
