//! Tokenizer for `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = joyai-llm` (DeepSeek V4.1), built from the GGUF's own
//! token and merge tables: byte-level BPE behind the pre-tokenizer of llama.cpp's `llama-vocab.cpp` (MIT). `joyai-llm`
//! shares the DEEPSEEK3_LLM case (llama-vocab.cpp:327-335): three regexes applied one after the other, each splitting
//! the pieces the previous one left (matches isolated), which a `Sequence` of `Split(Isolated)` reproduces.
//! Control (type 3) and user-defined (type 4) tokens are matched in text, like `llama_tokenize(parse_special=true)`.

use anyhow::{anyhow, Result};
use lb_gguf::Gguf;
use std::collections::HashMap;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// The DEEPSEEK3_LLM / JOYAI_LLM regexes (llama-vocab.cpp:331-333), C escapes resolved.
pub const JOYAI_PRE: [&str; 3] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

pub fn from_gguf(g: &Gguf) -> Result<Tokenizer> {
    let pre = g.meta_str("tokenizer.ggml.pre").unwrap_or("");
    if g.meta_str("tokenizer.ggml.model") != Some("gpt2") || !matches!(pre, "joyai-llm" | "deepseek-v3") {
        return Err(anyhow!("tokenizer model/pre {:?}/{pre:?} not supported (gpt2/joyai-llm)", g.meta_str("tokenizer.ggml.model")));
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
    let mut seq = Vec::new();
    for re in JOYAI_PRE {
        let split = Split::new(SplitPattern::Regex(re.to_string()), SplitDelimiterBehavior::Isolated, false).map_err(|e| anyhow!("split {re}: {e}"))?;
        seq.push(split.into());
    }
    seq.push(ByteLevel::new(false, false, false).into());
    tk.with_pre_tokenizer(Some(Sequence::new(seq)));
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
        let _ = tk.add_special_tokens(special);
        let _ = tk.add_tokens(user);
    }
    Ok(tk)
}

pub fn encode(tk: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tk.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
}

pub fn decode(tk: &Tokenizer, ids: &[u32]) -> Result<String> {
    tk.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

    fn pieces(text: &str) -> Vec<String> {
        let mut seq = Vec::new();
        for re in JOYAI_PRE {
            seq.push(Split::new(SplitPattern::Regex(re.to_string()), SplitDelimiterBehavior::Isolated, false).unwrap().into());
        }
        let pre = Sequence::new(seq);
        let mut p = PreTokenizedString::from(text);
        pre.pre_tokenize(&mut p).unwrap();
        p.get_splits(OffsetReferential::Original, OffsetType::Byte).into_iter().map(|(s, _, _)| s.to_string()).collect()
    }

    #[test]
    fn joyai_pre_splits() {
        // digits in runs of at most 3, split first; then CJK runs; then the word/punctuation regex
        assert_eq!(pieces("abc 12345"), vec!["abc", " ", "123", "45"]);
        assert_eq!(pieces("Hello, world!"), vec!["Hello", ",", " world", "!"]);
        assert_eq!(pieces("x.foo"), vec!["x", ".foo"]);
        assert_eq!(pieces("中文abc"), vec!["中文", "abc"]);
    }
}
