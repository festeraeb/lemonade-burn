//! X7b: write the weights lemonade-burn holds **after** transcode-on-residency back out as a GGUF,
//! so an independent evaluator (llama.cpp's `llama-perplexity --kl-divergence`) can score exactly
//! them against the same base logits as the untranscoded file.
//!
//! Every tensor the engine would re-encode is re-encoded with the engine's own function
//! (`lb_q4_gpu::transcode::q4_0_bytes`): quantized matrices with whole Q4_0 blocks that are not
//! Q4_0 already, except Q6_K (the engine keeps a Q6_K kernel for the head) and F32/F16 (norms,
//! router, conv taps: the engine keeps them as they are). Metadata is copied as read (including
//! `general.file_type`, which then no longer describes every tensor; llama.cpp loads by tensor type).
//!
//! usage: transcode_gguf <in.gguf> <out.gguf>
use lb_gguf::reader::{Gguf, MetaValue};
use lb_gguf::{GgmlType, QuantMat};
use std::io::{BufWriter, Write};

fn type_id(ty: GgmlType) -> u32 {
    match ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q4_0 => 2,
        GgmlType::Q8_0 => 8,
        GgmlType::Q2K => 10,
        GgmlType::Q3K => 11,
        GgmlType::Q4K => 12,
        GgmlType::Q5K => 13,
        GgmlType::Q6K => 14,
        GgmlType::IQ4NL => 20,
        GgmlType::IQ4XS => 23,
        GgmlType::Other(id) => id,
    }
}

fn meta_type(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U8(_) => 0,
        MetaValue::I8(_) => 1,
        MetaValue::U16(_) => 2,
        MetaValue::I16(_) => 3,
        MetaValue::U32(_) => 4,
        MetaValue::I32(_) => 5,
        MetaValue::F32(_) => 6,
        MetaValue::Bool(_) => 7,
        MetaValue::Str(_) => 8,
        MetaValue::Array(_) => 9,
        MetaValue::U64(_) => 10,
        MetaValue::I64(_) => 11,
        MetaValue::F64(_) => 12,
    }
}

fn put_str(w: &mut Vec<u8>, s: &str) {
    w.extend_from_slice(&(s.len() as u64).to_le_bytes());
    w.extend_from_slice(s.as_bytes());
}

fn put_val(w: &mut Vec<u8>, v: &MetaValue) -> Result<(), String> {
    match v {
        MetaValue::U8(x) => w.push(*x),
        MetaValue::I8(x) => w.push(*x as u8),
        MetaValue::U16(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I16(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U32(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I32(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F32(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Bool(x) => w.push(*x as u8),
        MetaValue::Str(s) => put_str(w, s),
        MetaValue::U64(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I64(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F64(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Array(a) => {
            let et = a.first().map(meta_type).ok_or("empty metadata array: element type unknown")?;
            w.extend_from_slice(&et.to_le_bytes());
            w.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for e in a {
                put_val(w, e)?;
            }
        }
    }
    Ok(())
}

fn engine_transcodes(ty: GgmlType, dims: &[usize]) -> bool {
    dims.len() >= 2 && dims[0] % 32 == 0 && !matches!(ty, GgmlType::Q4_0 | GgmlType::Q6K | GgmlType::F32 | GgmlType::F16) && lb_q4_gpu::transcode::transcodable(ty)
}

fn main() -> Result<(), String> {
    let mut a: Vec<String> = std::env::args().collect();
    let to_q8 = a.iter().any(|x| x == "--q8");
    a.retain(|x| x != "--q8");
    if a.len() != 3 {
        return Err("usage: transcode_gguf [--q8] <in.gguf> <out.gguf>".into());
    }
    let g = Gguf::open(std::path::Path::new(&a[1])).map_err(|e| e.to_string())?;
    let align = g.alignment;
    // new tensor table: (name, dims, type, bytes)
    let mut table: Vec<(String, Vec<usize>, GgmlType, std::borrow::Cow<[u8]>)> = Vec::new();
    let t0 = std::time::Instant::now();
    let (mut n_tc, mut src_b, mut dst_b) = (0usize, 0usize, 0usize);
    for t in &g.tensors {
        let nb = t.n_bytes().ok_or_else(|| format!("{}: unknown type {:?}", t.name, t.ty))?;
        let bytes = &g.tensor_bytes(t)[..nb];
        if engine_transcodes(t.ty, &t.dims) {
            let cols = t.dims[0];
            let rows = t.n_elems() / cols;
            let m = QuantMat::new(t.ty, rows, cols, bytes);
            let (ty, q) = if to_q8 {
                // the plan's other target: Q8_0 (8.5 bpw, ~1e-3 relative); the engine has no Q8_0 kernel
                // yet, so this is a quality measurement only
                let mut out = Vec::with_capacity(rows * cols / 32 * 34);
                let mut buf = vec![0f32; cols];
                for r in 0..rows {
                    m.dequant_row(r, &mut buf);
                    out.extend_from_slice(&lb_gguf::types::quantize_q8_0(&buf));
                }
                (GgmlType::Q8_0, out)
            } else {
                (GgmlType::Q4_0, lb_q4_gpu::transcode::q4_0_bytes(&m)?.into_owned())
            };
            n_tc += 1;
            src_b += nb;
            dst_b += q.len();
            table.push((t.name.clone(), t.dims.clone(), ty, std::borrow::Cow::Owned(q)));
        } else {
            table.push((t.name.clone(), t.dims.clone(), t.ty, std::borrow::Cow::Borrowed(bytes)));
        }
    }
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(table.len() as u64).to_le_bytes());
    head.extend_from_slice(&(g.meta_order.len() as u64).to_le_bytes());
    for k in &g.meta_order {
        let v = &g.meta[k];
        put_str(&mut head, k);
        head.extend_from_slice(&meta_type(v).to_le_bytes());
        put_val(&mut head, v)?;
    }
    let mut off = 0usize;
    for (name, dims, ty, b) in &table {
        put_str(&mut head, name);
        head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            head.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        head.extend_from_slice(&type_id(*ty).to_le_bytes());
        head.extend_from_slice(&(off as u64).to_le_bytes());
        off = (off + b.len()).div_ceil(align) * align;
    }
    let pad = head.len().div_ceil(align) * align - head.len();
    head.extend(std::iter::repeat_n(0u8, pad));
    let mut w = BufWriter::with_capacity(1 << 24, std::fs::File::create(&a[2]).map_err(|e| e.to_string())?);
    w.write_all(&head).map_err(|e| e.to_string())?;
    let mut pos = 0usize;
    for (_, _, _, b) in &table {
        w.write_all(b).map_err(|e| e.to_string())?;
        pos += b.len();
        let p = pos.div_ceil(align) * align - pos;
        w.write_all(&vec![0u8; p]).map_err(|e| e.to_string())?;
        pos += p;
    }
    w.flush().map_err(|e| e.to_string())?;
    println!(
        "{{\"in\":{:?},\"out\":{:?},\"target\":\"{}\",\"tensors\":{},\"transcoded\":{n_tc},\"src_bytes\":{src_b},\"dst_bytes\":{dst_b},\"secs\":{:.1},\"search\":{}}}",
        a[1],
        a[2],
        if to_q8 { "Q8_0" } else { "Q4_0" },
        table.len(),
        t0.elapsed().as_secs_f64(),
        std::env::var_os("LB_TRANSCODE_SIMPLE").is_none()
    );
    Ok(())
}
