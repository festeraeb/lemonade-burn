//! Bit-exact check of lb-gguf's dequantizers against a reference dump (llama.cpp's own `to_float`
//! on the same bytes, written by an external harness): `<prefix>.bin` = raw blocks, `<prefix>.f32` =
//! the reference values. usage: iq_check <type name, e.g. IQ3_S> <cols> <prefix>...  -> one JSON line each
use lb_gguf::GgmlType;

fn ty(name: &str) -> GgmlType {
    (0..64).map(GgmlType::from_id).find(|t| t.name() == name).unwrap_or_else(|| panic!("unknown type {name}"))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (t, cols) = (ty(&a[1]), a[2].parse::<usize>().unwrap());
    let rb = t.row_bytes(cols).expect("cols not a whole number of blocks");
    let mut bad_total = 0usize;
    for pfx in &a[3..] {
        let raw = std::fs::read(format!("{pfx}.bin")).unwrap();
        let want: Vec<f32> = std::fs::read(format!("{pfx}.f32")).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let rows = raw.len() / rb;
        assert_eq!(want.len(), rows * cols, "{pfx}: value count");
        let mut got = vec![0f32; rows * cols];
        for r in 0..rows {
            t.dequant_row(&raw[r * rb..(r + 1) * rb], &mut got[r * cols..(r + 1) * cols]);
        }
        let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
        let first = got.iter().zip(&want).position(|(g, w)| g.to_bits() != w.to_bits());
        bad_total += bad;
        println!("{{\"type\":\"{}\",\"prefix\":{:?},\"rows\":{rows},\"cols\":{cols},\"values\":{},\"bit_mismatches\":{bad},\"first_mismatch\":{}}}", t.name(), pfx, got.len(), first.map(|i| format!("[{i},{},{}]", got[i], want[i])).unwrap_or_else(|| "null".into()));
    }
    std::process::exit(if bad_total == 0 { 0 } else { 1 });
}
