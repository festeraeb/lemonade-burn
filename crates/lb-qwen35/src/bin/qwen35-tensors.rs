//! qwen35-tensors <model.gguf> <prefix> : list tensors whose name starts with the prefix (type, dims).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = lb_gguf::Gguf::open(std::path::Path::new(&a[1])).unwrap();
    for t in &g.tensors {
        let n = t.name.clone();
        if n.starts_with(&a[2]) {
            println!("{n} {} {:?}", t.ty.name(), t.dims);
        }
    }
}
