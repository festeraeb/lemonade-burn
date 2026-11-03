//! Tokenizer and chat template for BitNet b1.58 2B4T.
//!
//! The model's `tokenizer_config.json` template renders each message as
//! `Role: content.trim()<|eot_id|>` and appends `Assistant: `; the BOS token
//! `<|begin_of_text|>` is prepended (the template itself does not emit it).

use anyhow::{anyhow, Result};
use std::path::Path;

pub const EOT: u32 = 128009; // <|eot_id|>
pub const EOS: u32 = 128001; // <|end_of_text|>

/// How chat messages are rendered to text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatStyle {
    /// BitNet 2B4T: `Role: content<|eot_id|>` ... `Assistant: `.
    BitNet,
    /// ChatML (LFM2): `<|im_start|>role\ncontent<|im_end|>\n` ... `<|im_start|>assistant\n`.
    ChatMl,
}

pub struct TextCodec {
    pub tok: tokenizers::Tokenizer,
    pub bos: u32,
    pub style: ChatStyle,
    /// Token ids that end a generation.
    pub stops: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl TextCodec {
    /// BitNet codec: `dir/tokenizer.json`, BitNet chat style and stop ids.
    pub fn load(dir: &Path, bos: u32) -> Result<Self> {
        Self::load_with(&dir.join("tokenizer.json"), bos, ChatStyle::BitNet, vec![EOT, EOS])
    }

    /// Any HF `tokenizer.json` with an explicit chat style and stop ids.
    pub fn load_with(tokenizer_json: &Path, bos: u32, style: ChatStyle, stops: Vec<u32>) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow!("tokenizer {}: {e}", tokenizer_json.display()))?;
        Ok(Self { tok, bos, style, stops })
    }

    /// Plain-text prompt; BOS is added like `tokenizer(text)` does.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tok.encode(text, true).map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        if ids.first() != Some(&self.bos) {
            ids.insert(0, self.bos);
        }
        Ok(ids)
    }

    pub fn render_chat(&self, messages: &[ChatMessage]) -> String {
        if self.style == ChatStyle::ChatMl {
            let mut s = String::new();
            for m in messages {
                s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content.trim()));
            }
            s.push_str("<|im_start|>assistant\n");
            return s;
        }
        let mut s = String::new();
        for m in messages {
            let mut role = m.role.clone();
            if let Some(f) = role.get_mut(0..1) {
                f.make_ascii_uppercase();
            }
            // jinja `capitalize` also lowercases the rest
            let role = format!("{}{}", &role[..role.len().min(1)], role.get(1..).unwrap_or("").to_lowercase());
            s.push_str(&format!("{role}: {}<|eot_id|>", m.content.trim()));
        }
        s.push_str("Assistant: ");
        s
    }

    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let text = self.render_chat(messages);
        let enc = self.tok.encode(text.as_str(), false).map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids = vec![self.bos];
        ids.extend_from_slice(enc.get_ids());
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, true).map_err(|e| anyhow!("decode: {e}"))
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stops.contains(&id)
    }
}

/// Incremental detokenizer: emits only text that is stable (no partial UTF-8).
pub struct StreamDecoder {
    ids: Vec<u32>,
    emitted: usize,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self { ids: Vec::new(), emitted: 0 }
    }
    pub fn push(&mut self, codec: &TextCodec, id: u32) -> Result<String> {
        self.ids.push(id);
        let text = codec.decode(&self.ids)?;
        if text.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        let new = text.get(self.emitted..).unwrap_or("").to_string();
        self.emitted = text.len();
        Ok(new)
    }
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}
