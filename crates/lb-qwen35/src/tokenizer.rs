//! Tokenizer for `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = qwen35`, built from the GGUF's own token and merge tables
//! (no external tokenizer.json): byte-level BPE behind the qwen35 pre-tokenizer regex from llama.cpp's `llama-vocab.cpp`
//! (MIT). Control (type 3) and user-defined (type 4) tokens are matched in text, like `llama_tokenize(parse_special=true)`.

use anyhow::{anyhow, Result};
use lb_gguf::Gguf;
use std::collections::HashMap;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

const QWEN35_PRE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub fn from_gguf(g: &Gguf) -> Result<Tokenizer> {
    let pre = g.meta_str("tokenizer.ggml.pre").unwrap_or("");
    if g.meta_str("tokenizer.ggml.model") != Some("gpt2") || pre != "qwen35" {
        return Err(anyhow!("tokenizer model/pre {:?}/{pre:?} not supported (gpt2/qwen35)", g.meta_str("tokenizer.ggml.model")));
    }
    let toks = g.meta.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).ok_or_else(|| anyhow!("no tokens"))?;
    let merges = g.meta.get("tokenizer.ggml.merges").and_then(|v| v.as_array()).ok_or_else(|| anyhow!("no merges"))?;
    let types = g.meta.get("tokenizer.ggml.token_type").and_then(|v| v.as_array());
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.as_str().unwrap_or("").to_string(), i as u32)).collect();
    let merges: Vec<(String, String)> = merges
        .iter()
        .map(|m| {
            let s = m.as_str().unwrap_or("");
            let (a, b) = s.split_once(' ').unwrap_or((s, ""));
            (a.to_string(), b.to_string())
        })
        .collect();
    let bpe = BPE::builder().vocab_and_merges(vocab.into_iter().collect::<ahash::AHashMap<_, _>>(), merges).build().map_err(|e| anyhow!("bpe: {e}"))?;
    let mut tk = Tokenizer::new(bpe);
    let split = Split::new(SplitPattern::Regex(QWEN35_PRE.to_string()), SplitDelimiterBehavior::Isolated, false).map_err(|e| anyhow!("split: {e}"))?;
    tk.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), ByteLevel::new(false, false, false).into()])));
    tk.with_decoder(Some(ByteLevelDec::default()));
    if let Some(types) = types {
        let mut special = Vec::new();
        let mut user = Vec::new();
        for (i, t) in types.iter().enumerate() {
            match t.as_u64() {
                Some(3) => special.push(AddedToken::from(toks[i].as_str().unwrap_or("").to_string(), true)),
                Some(4) => user.push(AddedToken::from(toks[i].as_str().unwrap_or("").to_string(), false)),
                _ => {}
            }
        }
        tk.add_special_tokens(special);
        tk.add_tokens(user);
    }
    Ok(tk)
}

pub fn encode(tk: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tk.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
}

pub fn decode(tk: &Tokenizer, ids: &[u32]) -> Result<String> {
    tk.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
}
