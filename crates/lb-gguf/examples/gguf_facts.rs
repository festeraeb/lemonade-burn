//! gguf_facts <shard-1 or single .gguf> [--tensors]
//! Metadata (arrays summarised), a quant-type table (count, bytes) and a per-tensor-role table
//! (`blk.N.` folded to `blk.*.`) for a single or split GGUF. `--tensors` lists every tensor; `--key K` prints one value in full.
use lb_gguf::split::GgufSet;
use lb_gguf::MetaValue;
use std::collections::BTreeMap;

fn show(v: &MetaValue) -> String {
    match v {
        MetaValue::Array(a) => {
            let n = a.len();
            let head: Vec<String> = a.iter().take(if n > 64 { 8 } else { n }).map(show).collect();
            format!("[{}{}] (len {n})", head.join(", "), if n > 64 { ", ..." } else { "" })
        }
        MetaValue::Str(s) if s.len() > 200 => format!("{:?}... ({} bytes)", &s[..s.char_indices().nth(200).map(|x| x.0).unwrap_or(s.len())], s.len()),
        MetaValue::Str(s) => format!("{s:?}"),
        other => format!("{other:?}"),
    }
}

fn fold(name: &str) -> String {
    let mut out = Vec::new();
    let parts: Vec<&str> = name.split('.').collect();
    let mut i = 0;
    while i < parts.len() {
        if parts[i] == "blk" && i + 1 < parts.len() && parts[i + 1].parse::<usize>().is_ok() {
            out.push("blk.*".to_string());
            i += 2;
        } else {
            out.push(parts[i].to_string());
            i += 1;
        }
    }
    out.join(".")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let set = GgufSet::open(std::path::Path::new(&a[1])).expect("open");
    let list = a.iter().any(|x| x == "--tensors");
    if let Some(i) = a.iter().position(|x| x == "--key") {
        // print one metadata value in full and stop
        match set.meta().get(&a[i + 1]) {
            Some(lb_gguf::MetaValue::Str(s)) => println!("{s}"),
            Some(v) => println!("{v:?}"),
            None => eprintln!("no key {}", a[i + 1]),
        }
        return;
    }
    println!("# shards: {}", set.paths.len());
    for (p, g) in set.paths.iter().zip(&set.shards) {
        println!("#   {} : {} tensors, GGUF v{}", p.display(), g.tensors.len(), g.version);
    }
    println!("\n## metadata (shard 1)");
    let g0 = set.first();
    for k in &g0.meta_order {
        println!("{k} = {}", show(&g0.meta[k]));
    }
    let mut by_ty: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut by_role: BTreeMap<String, (String, Vec<usize>, usize, usize)> = BTreeMap::new();
    let mut total = 0usize;
    for (_, t) in set.tensors() {
        let nb = t.n_bytes().unwrap_or(0);
        total += nb;
        let e = by_ty.entry(t.ty.name()).or_default();
        e.0 += 1;
        e.1 += nb;
        e.2 += t.n_elems();
        let r = by_role.entry(fold(&t.name)).or_insert_with(|| (t.ty.name(), t.dims.clone(), 0, 0));
        r.2 += 1;
        r.3 += nb;
        if r.0 != t.ty.name() && !r.0.contains(&t.ty.name()) {
            r.0 = format!("{}|{}", r.0, t.ty.name());
        }
        if list {
            println!("T {}\t{}\t{:?}\t{}", t.name, t.ty.name(), t.dims, nb);
        }
    }
    if a.iter().any(|x| x == "--layers") {
        // bytes per layer, split into experts (`*_exps`) and the rest, for placement tables
        let mut per: BTreeMap<usize, (usize, usize, usize)> = BTreeMap::new();
        let mut other = 0usize;
        for (_, t) in set.tensors() {
            let nb = t.n_bytes().unwrap_or(0);
            let p: Vec<&str> = t.name.split('.').collect();
            match (p.first(), p.get(1).and_then(|x| x.parse::<usize>().ok())) {
                (Some(&"blk"), Some(il)) => {
                    let e = per.entry(il).or_default();
                    if t.name.contains("_exps") { e.0 += nb } else { e.1 += nb }
                    e.2 += 1;
                }
                _ => other += nb,
            }
        }
        println!("\n## per layer: layer | expert bytes | non-expert bytes | total | tensors");
        for (il, (ex, ne, n)) in &per {
            println!("{il} | {ex} | {ne} | {} | {n}", ex + ne);
        }
        println!("outside layers (embeddings, head, tables): {other}");
    }
    println!("\n## quant types: type | tensors | bytes | GiB | elements | bits/weight");
    for (ty, (n, b, e)) in &by_ty {
        println!("{ty} | {n} | {b} | {:.3} | {e} | {:.3}", *b as f64 / (1u64 << 30) as f64, 8.0 * *b as f64 / *e as f64);
    }
    println!("total bytes {total} ({:.3} GiB)", total as f64 / (1u64 << 30) as f64);
    println!("\n## tensor roles: role | type(s) | dims of first | count | bytes");
    for (role, (ty, dims, n, b)) in &by_role {
        println!("{role} | {ty} | {dims:?} | {n} | {b}");
    }
}
