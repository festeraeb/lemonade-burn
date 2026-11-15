//! qwen35-tok <model.gguf> <text-lines.txt> <out-ids.txt> : tokenise (\n in a line is a newline) with the GGUF-built tokenizer.
use anyhow::Result;
use std::io::Write;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[1]))?;
    let tk = lb_qwen35::tokenizer::from_gguf(&g)?;
    let mut out = std::fs::File::create(&a[3])?;
    for line in std::fs::read_to_string(&a[2])?.lines() {
        let text = line.replace("\\n", "\n");
        let ids = lb_qwen35::tokenizer::encode(&tk, &text)?;
        writeln!(out, "{}", ids.iter().map(|t| format!("{t} ")).collect::<String>())?;
    }
    Ok(())
}
