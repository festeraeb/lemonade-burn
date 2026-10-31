//! Parse a real LFM2.5 GGUF on disk (skipped when the file is absent) and compare a few
//! dequantized values with numbers produced by the gguf-py reference (recorded in
//! `tests/oracle_iq3xxs.txt`, generated once with gguf-py's `dequantize`).
use lb_gguf::{Gguf, GgmlType};
use std::path::Path;

const FILE: &str = "/mnt/t440/raid0/models/lfm25/LFM2.5-24B-A2B-UD-IQ3_XXS.gguf";

#[test]
fn parses_lfm25_metadata_and_tensors() {
    if !Path::new(FILE).exists() {
        eprintln!("skip: {FILE} absent");
        return;
    }
    let g = Gguf::open(Path::new(FILE)).unwrap();
    assert_eq!(g.meta_str("general.architecture"), Some("lfm2moe"));
    assert_eq!(g.meta_u64("lfm2moe.block_count"), Some(40));
    assert_eq!(g.meta_u64("lfm2moe.expert_count"), Some(64));
    let kv = g.meta_u64_array("lfm2moe.attention.head_count_kv").unwrap();
    assert_eq!(kv.len(), 40);
    assert_eq!(kv[2], 8);
    assert_eq!(g.tensors.len(), 428);
    let e = g.tensor("token_embd.weight").unwrap();
    assert_eq!(e.ty, GgmlType::Q6K);
    assert_eq!(e.dims, vec![2048, 128000]);
    let n = g.tensor("blk.0.attn_norm.weight").unwrap();
    assert_eq!(n.ty, GgmlType::F32);
    assert_eq!(n.dims, vec![2048]);
}

#[test]
fn dequant_matches_gguf_py_oracle() {
    if !Path::new(FILE).exists() {
        eprintln!("skip: {FILE} absent");
        return;
    }
    let oracle = include_str!("oracle_iq3xxs.txt");
    let g = Gguf::open(Path::new(FILE)).unwrap();
    for line in oracle.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        // name row count v0 v1 ...
        let mut it = line.split_whitespace();
        let name = it.next().unwrap();
        let row: usize = it.next().unwrap().parse().unwrap();
        let want: Vec<f32> = it.map(|v| v.parse().unwrap()).collect();
        let t = g.tensor(name).unwrap();
        let cols = t.dims[0];
        let rb = t.ty.row_bytes(cols).unwrap();
        let bytes = g.tensor_bytes(t);
        let mut got = vec![0f32; cols];
        t.ty.dequant_row(&bytes[row * rb..(row + 1) * rb], &mut got);
        for (i, w) in want.iter().enumerate() {
            let gv = got[i];
            assert!((gv - w).abs() <= 1e-5 * (1.0 + w.abs()), "{name} row {row} [{i}]: {gv} vs {w}");
        }
    }
}
