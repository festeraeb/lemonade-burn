//! Hostile GGUF headers must produce `GgufError`, never a panic or an unbounded allocation.

use lb_gguf::Gguf;
use std::path::PathBuf;

fn write(name: &str, bytes: &[u8]) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn header(n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(n_tensors.to_le_bytes());
    b.extend(n_kv.to_le_bytes());
    b
}

fn string(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

fn must_err(name: &str, bytes: Vec<u8>) {
    let p = write(name, &bytes);
    assert!(Gguf::open(&p).is_err(), "{name}: expected an error");
}

#[test]
fn huge_string_length_is_an_error() {
    let mut b = header(0, 1);
    b.extend(u64::MAX.to_le_bytes()); // key length
    must_err("huge_string.gguf", b);
}

#[test]
fn deeply_nested_arrays_are_an_error() {
    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend(9u32.to_le_bytes()); // array
    for _ in 0..200_000 {
        b.extend(9u32.to_le_bytes()); // element type: array
        b.extend(1u64.to_le_bytes()); // one element
    }
    must_err("nested_arrays.gguf", b);
}

#[test]
fn huge_array_and_entry_counts_are_errors() {
    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend(9u32.to_le_bytes());
    b.extend(4u32.to_le_bytes());
    b.extend(u64::MAX.to_le_bytes());
    must_err("huge_array.gguf", b);
    must_err("huge_kv.gguf", header(0, u64::MAX));
    must_err("huge_tensors.gguf", header(u64::MAX, 0));
}

fn one_tensor(dims: &[u64], ty: u32, offset: u64) -> Vec<u8> {
    let mut b = header(1, 0);
    string(&mut b, "t");
    b.extend((dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend(d.to_le_bytes());
    }
    b.extend(ty.to_le_bytes());
    b.extend(offset.to_le_bytes());
    b
}

#[test]
fn tensor_offset_overflow_is_an_error() {
    must_err("offset_max.gguf", one_tensor(&[32], 0, u64::MAX)); // F32
    must_err("offset_wrap.gguf", one_tensor(&[32], 0, u64::MAX - 8));
}

#[test]
fn tensor_size_overflow_is_an_error() {
    must_err("dims_overflow.gguf", one_tensor(&[u64::MAX, 2], 0, 0));
    must_err("bytes_overflow.gguf", one_tensor(&[u64::MAX / 2], 0, 0));
}

#[test]
fn zero_alignment_is_an_error() {
    let mut b = header(0, 1);
    string(&mut b, "general.alignment");
    b.extend(4u32.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    must_err("align0.gguf", b);
}
