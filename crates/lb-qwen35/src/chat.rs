//! Chat template for text-only conversations (system / user / assistant), as the GGUF's Jinja template renders them
//! (`tokenizer.chat_template`, checked against a jinja2 render in the tests). Not implemented: tool calls, tool responses,
//! images, and the template's reasoning-content stripping of earlier assistant turns beyond the plain case.

pub struct Msg<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// `enable_thinking == false` opens an empty think block (`<think>\n\n</think>\n\n`), otherwise `<think>\n`.
pub fn render(msgs: &[Msg<'_>], add_generation_prompt: bool, enable_thinking: bool) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str("<|im_start|>");
        s.push_str(m.role);
        s.push('\n');
        s.push_str(m.content.trim());
        s.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        s.push_str("<|im_start|>assistant\n");
        s.push_str(if enable_thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_jinja_render() {
        let u = |c| Msg { role: "user", content: c };
        assert_eq!(render(&[u("Hello there")], true, false), "<|im_start|>user\nHello there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(
            render(&[Msg { role: "system", content: "Be brief." }, u("Hi")], true, true),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            render(&[u("Hi"), Msg { role: "assistant", content: "Hello!" }, u("Again")], true, false),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
