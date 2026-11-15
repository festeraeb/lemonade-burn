//! qwen35-hidden <model.gguf> <ids-line.txt> <ref_hidden.bin> : per-layer comparison of intermediate tensors against llama.cpp's
//! graph callbacks (`ref_hidden`), to find the first layer/sub-layer where the decoders part.
use anyhow::Result;
use std::collections::HashMap;
use std::io::Read;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[1]))?;
    let m = lb_qwen35::Model::load(&g)?;
    let ids: Vec<u32> = std::fs::read_to_string(&a[2])?.split_whitespace().map(|t| t.parse().unwrap()).collect();
    // reference records
    let mut raw = Vec::new();
    std::fs::File::open(&a[3])?.read_to_end(&mut raw)?;
    let mut refs: HashMap<String, (usize, Vec<f32>)> = HashMap::new();
    let mut p = 0;
    while p < raw.len() {
        let e = raw[p..].iter().position(|&b| b == 0).unwrap();
        let name = String::from_utf8_lossy(&raw[p..p + e]).to_string();
        p += e + 1;
        let nt = i32::from_le_bytes(raw[p..p + 4].try_into()?) as usize;
        let ne = i32::from_le_bytes(raw[p + 4..p + 8].try_into()?) as usize;
        p += 8;
        let v: Vec<f32> = raw[p..p + nt * ne * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        p += nt * ne * 4;
        refs.insert(name, (ne, v));
    }
    let mut st = m.new_state();
    for (pos, &t) in ids.iter().enumerate() {
        let mut rows: Vec<(String, usize, f64, f64)> = Vec::new();
        m.forward_ex(&mut st, t, false, &mut |name, il, v| {
            let rname = match name {
                "attn_out" => if m.cfg.is_recurrent(il) { format!("linear_attn_out-{il}") } else { format!("attn_output-{il}") },
                n => format!("{n}-{il}"),
            };
            if let Some((ne, rv)) = refs.get(&rname) {
                let r = &rv[pos * ne..(pos + 1) * ne];
                let num: f64 = v.iter().zip(r).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
                let den: f64 = r.iter().map(|y| (*y as f64).powi(2)).sum();
                let maxd = v.iter().zip(r).fold(0f32, |mm, (x, y)| mm.max((x - y).abs())) as f64;
                rows.push((rname, il, (num / den.max(1e-30)).sqrt(), maxd));
            }
        });
        println!("--- token position {pos} (id {t}): relative L2 error vs llama.cpp (first 12 layers' sub-steps, then every 4th)");
        for (n, il, rel, maxd) in &rows {
            if *il < 12 || il % 4 == 3 {
                println!("{n:<22} rel_l2 {rel:.3e} max_abs {maxd:.3e}");
            }
        }
    }
    Ok(())
}
