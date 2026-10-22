//! lemonade-burn ternary checkpoint format, version 1.
//!
//! A standard safetensors file (`model.lbt.safetensors`) next to the source model's
//! `config.json` and `tokenizer.json`:
//!
//! * `<linear>.qweight`  U8  `[out, padded(in)/4]`  2-bit codes, layout in `lb_ternary::pack`
//! * `<linear>.wscale`   F32 `[1]`                 per-tensor `1/mean|W|` (WeightQuant scale)
//! * `*norm.weight`      F32                       RMSNorm gains (bf16 widened, exact)
//! * `model.embed_tokens.weight` BF16 `[vocab, hidden]` (also the tied LM head)
//!
//! `<linear>` is the HF module path without `.weight`, e.g.
//! `model.layers.3.mlp.gate_proj`.

pub const FORMAT: &str = "lemonade-burn-ternary";
pub const VERSION: &str = "1";
pub const FILE_NAME: &str = "model.lbt.safetensors";

pub const LINEAR_SUFFIXES: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// Is `name` (a source tensor name such as `model.layers.0.mlp.up_proj.weight`) one of the
/// BitLinear weights that the model quantizes online?
pub fn is_bitlinear_weight(name: &str) -> bool {
    name.starts_with("model.layers.")
        && LINEAR_SUFFIXES.iter().any(|s| name.ends_with(&format!("{s}.weight")))
}

pub fn qweight_name(linear: &str) -> String {
    format!("{linear}.qweight")
}

pub fn wscale_name(linear: &str) -> String {
    format!("{linear}.wscale")
}

/// bf16 bytes (little endian) -> f32.
pub fn bf16_bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}
