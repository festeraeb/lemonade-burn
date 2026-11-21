//! ggml_ref_check <libggml-base.so> <model .gguf (shard 1 of a split set is fine)> [rows_per_tensor=64] [max_tensors_per_type=6]
//!
//! Bit-exactness check of lb-gguf's dequantisers against llama.cpp's own: loads `libggml-base.so` at run
//! time, and for every quant type present in the file picks up to `max_tensors_per_type` tensors (spread over
//! the file) and `rows_per_tensor` rows of each (spread over the tensor, so different experts of a 3-D expert
//! tensor are covered), dequantises each row with `ggml_get_type_traits(type)->to_float` and with
//! `GgmlType::dequant_row`, and compares the f32 bit patterns. One JSON line per type; exit 1 on any mismatch.
use lb_gguf::{GgmlType, GgufSet};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};

#[repr(C)]
struct GgmlTypeTraits {
    type_name: *const c_char,
    blck_size: i64,
    blck_size_interleave: i64,
    type_size: usize,
    is_quantized: bool,
    to_float: Option<unsafe extern "C" fn(*const c_void, *mut f32, i64)>,
    from_float_ref: Option<unsafe extern "C" fn(*const f32, *mut c_void, i64)>,
}

fn type_id(t: GgmlType) -> Option<i32> {
    (0..64u32).find(|&i| GgmlType::from_id(i) == t).map(|i| i as i32)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rows_per = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let max_t = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(6usize);
    // SAFETY: loading a C library whose two symbols we call with the documented signatures (ggml.h).
    let lib = unsafe { libloading::Library::new(&a[1]) }.expect("load libggml-base");
    let traits: libloading::Symbol<unsafe extern "C" fn(i32) -> *const GgmlTypeTraits> = unsafe { lib.get(b"ggml_get_type_traits\0") }.expect("ggml_get_type_traits");
    let row_size: libloading::Symbol<unsafe extern "C" fn(i32, i64) -> usize> = unsafe { lib.get(b"ggml_row_size\0") }.expect("ggml_row_size");
    let set = GgufSet::open(std::path::Path::new(&a[2])).expect("open gguf");

    let mut by_ty: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    let all: Vec<_> = set.tensors().collect();
    for (i, (_, t)) in all.iter().enumerate() {
        by_ty.entry(t.ty.name()).or_default().push((i, t.n_elems()));
    }
    let mut bad_total = 0usize;
    for (tyname, list) in &by_ty {
        // spread the picks over the file
        let step = (list.len() as f64 / max_t as f64).max(1.0);
        let picks: Vec<usize> = (0..max_t.min(list.len())).map(|k| list[((k as f64 * step) as usize).min(list.len() - 1)].0).collect();
        let (mut values, mut bad, mut rows_done) = (0usize, 0usize, 0usize);
        let mut first_bad = String::from("null");
        let mut names = Vec::new();
        for &ti in &picks {
            let (g, t) = all[ti];
            let id = type_id(t.ty).expect("type id");
            let cols = t.dims[0];
            let nrows = t.n_elems() / cols;
            let rb = t.ty.row_bytes(cols).expect("row bytes");
            let rs = unsafe { row_size(id, cols as i64) };
            assert_eq!(rs, rb, "{}: row size lb {} vs ggml {}", t.name, rb, rs);
            let tr = unsafe { &*traits(id) };
            let bytes = g.tensor_bytes(t);
            names.push(t.name.clone());
            let n = rows_per.min(nrows);
            let mut want = vec![0f32; cols];
            let mut got = vec![0f32; cols];
            for k in 0..n {
                // spread rows: first, last, and evenly in between (u128 to avoid overflow on the 768M-row table)
                let r = if n == 1 { 0 } else { ((k as u128 * (nrows as u128 - 1)) / (n as u128 - 1)) as usize };
                let src = &bytes[r * rb..(r + 1) * rb];
                match t.ty {
                    GgmlType::F32 => want.copy_from_slice(&src.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()),
                    _ => unsafe { (tr.to_float.expect("no to_float"))(src.as_ptr() as *const c_void, want.as_mut_ptr(), cols as i64) },
                }
                t.ty.dequant_row(src, &mut got);
                for (j, (x, y)) in got.iter().zip(&want).enumerate() {
                    if x.to_bits() != y.to_bits() {
                        if bad == 0 {
                            first_bad = format!("{{\"tensor\":{:?},\"row\":{r},\"col\":{j},\"lb\":{x},\"ggml\":{y}}}", t.name);
                        }
                        bad += 1;
                    }
                }
                values += cols;
                rows_done += 1;
            }
        }
        bad_total += bad;
        println!("{{\"type\":\"{tyname}\",\"tensors\":{},\"rows\":{rows_done},\"values\":{values},\"bit_mismatches\":{bad},\"first_mismatch\":{first_bad},\"names\":{names:?}}}", picks.len());
    }
    std::process::exit(if bad_total == 0 { 0 } else { 1 });
}
