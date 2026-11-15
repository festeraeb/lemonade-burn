//! dequant_hash <gguf> <tensors.tsv> <out.hash>
//! For every tensor of the tsv (name, ggml type id, cols, offset, rows) and every row, dequantise with this crate and write
//! the FNV-1a64 of the f32 bit patterns, one u64 per row. `ggml_dequant.c` (llama.cpp's own `to_float`) writes the same
//! file; `cmp` of the two is the bit-exactness check.
use lb_gguf::{Gguf, GgmlType};
use rayon::prelude::*;
use std::io::Write;

fn fnv(v: &[f32]) -> u64 {
    let mut h = 1469598103934665603u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1])).unwrap();
    let tsv = std::fs::read_to_string(&a[2]).unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[3]).unwrap());
    for line in tsv.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (name, ty, ncols, nrows) = (f[0], GgmlType::from_id(f[1].parse().unwrap()), f[2].parse::<usize>().unwrap(), f[4].parse::<usize>().unwrap());
        let t = g.tensor(name).unwrap();
        assert_eq!(t.ty, ty, "{name}");
        let bytes = g.tensor_bytes(t);
        let rb = ty.row_bytes(ncols).unwrap();
        let hs: Vec<u64> = (0..nrows)
            .into_par_iter()
            .map_init(|| vec![0f32; ncols], |buf, r| {
                ty.dequant_row(&bytes[r * rb..(r + 1) * rb], buf);
                fnv(buf)
            })
            .collect();
        for h in hs {
            out.write_all(&h.to_le_bytes()).unwrap();
        }
    }
}
