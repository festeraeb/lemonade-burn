//! Bring a new GGUF model family into the engine: read everything the file states, say what is
//! still unknown, and write a scaffold an engineer finishes.
//!
//! The rule the whole crate follows: **never guess silently.** Every field is either resolved
//! from a named GGUF key (and carries that key as evidence), explicitly defaulted with the
//! default recorded, or reported missing. A missing field becomes a `todo!()` in the scaffold,
//! so an unfinished adapter fails to build rather than running with invented numbers.

pub mod probe;
pub mod scaffold;

// Known gaps, found by running the probe on the qwen35moe family (2026-11-01):
// * `n_vocab` and `n_ff` report MISSING although `token_embd`/`output` dims state the vocab
//   outright. Deriving a field from a tensor shape (with the tensor name as evidence) is the
//   next step; it is deliberately not a silent default.
// * The probe reports quant type ids it cannot name (`type#12` = Q4_K, `type#13` = Q5_K).
//   `lb_gguf::GgmlType` knows only F32/F16/Q4_0/Q8_0/Q6_K, so those files cannot be loaded yet.
// * It does not yet emit `weights.rs`; the tensor-group listing is the input for that.
