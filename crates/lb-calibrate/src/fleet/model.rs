//! Model description. Built from GGUF shards (tensor shapes and types via lb-gguf, never
//! reading tensor data) or from JSON for synthetic tests.

use anyhow::{bail, Context, Result};
use lb_gguf::{Gguf, GgmlType};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelDesc {
    pub name: String,
    pub arch: String,
    pub n_layer: usize,
    /// Leading layers without routed experts.
    pub n_dense_lead: usize,
    /// Routed experts per MoE layer; 0 = dense model.
    pub n_expert: usize,
    pub top_k: usize,
    /// Mean bytes of one routed expert (gate+up+down) as stored.
    pub expert_bytes: u64,
    /// Mean non-expert bytes per layer (attention/conv, norms, shared experts, dense FFN).
    pub dense_layer_bytes: u64,
    /// Embedding + output head bytes (kept in host RAM: the engine runs the head on the CPU).
    pub head_bytes: u64,
    pub total_bytes: u64,
    /// Element counts, for re-pricing the model in other quants.
    pub params_dense: u64,
    pub params_expert: u64,
    pub largest_tensor_bytes: u64,
    /// Hidden width (activation payload for helper round trips); null if unknown.
    #[serde(default)]
    pub hidden: Option<usize>,
    /// KV bytes per token summed over layers (f16 K and V); null when the arch's KV layout is
    /// not the plain one (MLA etc.): the per-card reserve then has to cover it.
    #[serde(default)]
    pub kv_bytes_per_token: Option<u64>,
    /// Tensor types present with their byte share, e.g. {"IQ1_S": 2.1e11}.
    #[serde(default)]
    pub type_bytes: BTreeMap<String, u64>,
    #[serde(default)]
    pub files: Vec<String>,
}

impl ModelDesc {
    pub fn moe_layers(&self) -> usize {
        if self.n_expert == 0 {
            0
        } else {
            self.n_layer - self.n_dense_lead
        }
    }
    pub fn expert_total_bytes(&self) -> u64 {
        self.expert_bytes * (self.n_expert * self.moe_layers()) as u64
    }
    pub fn dense_total_bytes(&self) -> u64 {
        self.dense_layer_bytes * self.n_layer as u64
    }
    pub fn kv_bytes(&self, ctx: usize) -> u64 {
        self.kv_bytes_per_token.unwrap_or(0) * ctx as u64
    }
    pub fn load_json(p: &Path) -> Result<Self> {
        serde_json::from_str(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?).context("model json")
    }
}

/// (name, values per block, bytes per block) for a ggml type id. Block sizes are those of
/// ggml-common.h (not measured here); types lb-gguf already knows go through its table.
fn type_info(ty: GgmlType) -> Option<(String, usize, usize)> {
    if let Some((b, s)) = ty.block() {
        return Some((ty.name(), b, s));
    }
    let GgmlType::Other(id) = ty else { return None };
    let (n, b, s) = match id {
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        15 => ("Q8_K", 256, 292),
        16 => ("IQ2_XXS", 256, 66),
        17 => ("IQ2_XS", 256, 74),
        18 => ("IQ3_XXS", 256, 98),
        19 => ("IQ1_S", 256, 50),
        20 => ("IQ4_NL", 32, 18),
        21 => ("IQ3_S", 256, 110),
        22 => ("IQ2_S", 256, 82),
        23 => ("IQ4_XS", 256, 136),
        24 => ("I8", 1, 1),
        25 => ("I16", 1, 2),
        26 => ("I32", 1, 4),
        27 => ("I64", 1, 8),
        28 => ("F64", 1, 8),
        29 => ("IQ1_M", 256, 56),
        30 => ("BF16", 1, 2),
        _ => return None,
    };
    Some((n.to_string(), b, s))
}

/// `name-00001-of-00006.gguf` -> all six shard paths; any other path -> itself.
pub fn expand_shards(p: &Path) -> Result<Vec<PathBuf>> {
    let fname = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
    if let Some(i) = fname.rfind("-of-") {
        let total = &fname[i + 4..fname.len() - ".gguf".len().min(fname.len())];
        if let (Ok(n), true) = (total.parse::<usize>(), i >= 6) {
            let stem = &fname[..i - 6];
            let width = total.len();
            let mut out = Vec::new();
            for k in 1..=n {
                let q = p.with_file_name(format!("{stem}-{:0w$}-of-{total}.gguf", k, w = width));
                if !q.exists() {
                    bail!("shard {} missing", q.display());
                }
                out.push(q);
            }
            return Ok(out);
        }
    }
    Ok(vec![p.to_path_buf()])
}

/// Read shapes only from the given file (and its sibling shards).
pub fn from_gguf(path: &Path) -> Result<ModelDesc> {
    let files = expand_shards(path)?;
    let mut meta_g: Option<Gguf> = None;
    let (mut total, mut largest) = (0u64, 0u64);
    let mut type_bytes: BTreeMap<String, u64> = BTreeMap::new();
    // per layer: expert bytes, dense bytes, expert tensors seen
    let mut exp_bytes: BTreeMap<usize, u64> = BTreeMap::new();
    let mut dense_bytes: BTreeMap<usize, u64> = BTreeMap::new();
    let (mut head, mut p_dense, mut p_exp) = (0u64, 0u64, 0u64);
    let mut n_expert_t = 0usize;
    for f in &files {
        let g = Gguf::open(f).with_context(|| f.display().to_string())?;
        for t in &g.tensors {
            let Some((tn, blk, bs)) = type_info(t.ty) else { bail!("{}: unknown tensor type", t.name) };
            let elems = t.n_elems() as u64;
            let bytes = elems / blk as u64 * bs as u64;
            total += bytes;
            largest = largest.max(bytes);
            *type_bytes.entry(tn).or_default() += bytes;
            let layer = t.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<usize>().ok());
            match layer {
                Some(l) if t.name.contains("_exps.") => {
                    *exp_bytes.entry(l).or_default() += bytes;
                    p_exp += elems;
                    n_expert_t = n_expert_t.max(*t.dims.last().unwrap_or(&0));
                }
                Some(l) => {
                    *dense_bytes.entry(l).or_default() += bytes;
                    p_dense += elems;
                }
                None => {
                    head += bytes;
                    p_dense += elems;
                }
            }
        }
        if g.meta.contains_key("general.architecture") {
            meta_g = Some(g);
        }
    }
    let g = meta_g.context("no shard carries the metadata")?;
    let arch = g.meta_str("general.architecture").context("general.architecture")?.to_string();
    let k = |key: &str| g.meta_u64(&format!("{arch}.{key}"));
    let n_layer = k("block_count").context("block_count")? as usize;
    let n_expert = k("expert_count").map(|v| v as usize).unwrap_or(n_expert_t);
    let top_k = k("expert_used_count").unwrap_or(0) as usize;
    let moe_layers: Vec<usize> = exp_bytes.keys().copied().collect();
    let n_dense_lead = if moe_layers.is_empty() { n_layer } else { n_layer - moe_layers.len() };
    let expert_bytes = if moe_layers.is_empty() || n_expert == 0 {
        0
    } else {
        exp_bytes.values().sum::<u64>() / (moe_layers.len() * n_expert) as u64
    };
    let dense_layer_bytes = dense_bytes.values().sum::<u64>() / n_layer.max(1) as u64;
    // KV: plain layout only (head_count_kv x head_dim, f16 K and V) over attention layers.
    let hidden = k("embedding_length").map(|v| v as usize);
    let kv_bytes_per_token = if g.meta.contains_key(&format!("{arch}.attention.kv_lora_rank")) {
        None
    } else {
        let heads = k("attention.head_count");
        let kvh: Option<Vec<u64>> = g.meta_u64_array(&format!("{arch}.attention.head_count_kv")).or_else(|| k("attention.head_count_kv").map(|v| vec![v; n_layer]));
        let hd = k("attention.key_length").or_else(|| match (hidden, heads) {
            (Some(h), Some(n)) if n > 0 => Some(h as u64 / n),
            _ => None,
        });
        match (kvh, hd) {
            (Some(v), Some(hd)) => Some(v.iter().sum::<u64>() * hd * 2 * 2),
            _ => None,
        }
    };
    let name = g.meta_str("general.name").map(|s| s.to_string()).unwrap_or_else(|| path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string());
    Ok(ModelDesc {
        name,
        arch,
        n_layer,
        n_dense_lead,
        n_expert,
        top_k,
        expert_bytes,
        dense_layer_bytes,
        head_bytes: head,
        total_bytes: total,
        params_dense: p_dense,
        params_expert: p_exp,
        largest_tensor_bytes: largest,
        hidden,
        kv_bytes_per_token,
        type_bytes,
        files: files.iter().map(|f| f.display().to_string()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_names_expand() {
        let d = std::env::temp_dir().join(format!("lbtune-shards-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for i in 1..=3 {
            std::fs::write(d.join(format!("M-0000{i}-of-00003.gguf")), b"x").unwrap();
        }
        let v = expand_shards(&d.join("M-00001-of-00003.gguf")).unwrap();
        assert_eq!(v.len(), 3);
        assert!(v[2].ends_with("M-00003-of-00003.gguf"));
        std::fs::remove_dir_all(&d).ok();
        assert_eq!(expand_shards(Path::new("/x/plain.gguf")).unwrap().len(), 1);
    }

    #[test]
    fn iq_block_sizes() {
        assert_eq!(type_info(GgmlType::Other(19)).unwrap().2, 50); // IQ1_S 256 values in 50 B = 1.5625 bpw
        assert_eq!(type_info(GgmlType::Q4_0).unwrap().2, 18);
        assert!(type_info(GgmlType::Other(9999)).is_none());
    }
}
