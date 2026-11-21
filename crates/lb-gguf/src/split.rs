//! A model stored as several GGUF shards (`<name>-00001-of-0000N.gguf`, the `gguf-split` layout).
//!
//! Every shard is a complete GGUF file with its own header and tensor table; the first shard carries
//! the model metadata (`split.count`, `split.no`, `split.tensors.count` say how the set is cut). This
//! opens all shards (each memory-mapped, nothing is read until a tensor is touched) and looks tensors
//! up by name across them. A single unsplit file is a set of one.

use crate::reader::{Gguf, GgufError, MetaValue, TensorInfo};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct GgufSet {
    pub shards: Vec<Gguf>,
    pub paths: Vec<PathBuf>,
    by_name: HashMap<String, (usize, usize)>,
}

/// `foo-00001-of-00006.gguf` -> paths of shards 1..=6; anything else -> just the path given.
pub fn shard_paths(first: &Path) -> Vec<PathBuf> {
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    // match "-NNNNN-of-MMMMM.gguf"
    if let Some(stem) = name.strip_suffix(".gguf") {
        let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
        // rsplitn: [MMMMM, "of", NNNNN, prefix]
        if parts.len() == 4 && parts[1] == "of" && parts[0].len() == 5 && parts[2].len() == 5 {
            if let (Ok(_), Ok(total)) = (parts[2].parse::<usize>(), parts[0].parse::<usize>()) {
                let dir = first.parent().unwrap_or(Path::new("."));
                return (1..=total).map(|i| dir.join(format!("{}-{:05}-of-{:05}.gguf", parts[3], i, total))).collect();
            }
        }
    }
    vec![first.to_path_buf()]
}

impl GgufSet {
    /// Open `path` and, when it is shard 1 of a split set, every other shard.
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let paths = shard_paths(path);
        let mut shards = Vec::with_capacity(paths.len());
        let mut by_name = HashMap::new();
        for (si, p) in paths.iter().enumerate() {
            let g = Gguf::open(p)?;
            for (ti, t) in g.tensors.iter().enumerate() {
                if by_name.insert(t.name.clone(), (si, ti)).is_some() {
                    return Err(GgufError::Format(format!("{}: tensor {} appears in two shards", p.display(), t.name)));
                }
            }
            shards.push(g);
        }
        if let Some(n) = shards[0].meta_u64("split.count") {
            if n as usize != shards.len() {
                return Err(GgufError::Format(format!("split.count = {n} but {} shards found", shards.len())));
            }
        }
        Ok(Self { shards, paths, by_name })
    }

    /// Model metadata (from shard 1).
    pub fn meta(&self) -> &HashMap<String, MetaValue> {
        &self.shards[0].meta
    }
    pub fn first(&self) -> &Gguf {
        &self.shards[0]
    }
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.shards[0].meta_u64(key)
    }
    pub fn meta_f64(&self, key: &str) -> Option<f64> {
        self.shards[0].meta_f64(key)
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.shards[0].meta_str(key)
    }
    pub fn meta_u64_array(&self, key: &str) -> Option<Vec<u64>> {
        self.shards[0].meta_u64_array(key)
    }

    /// The shard holding `name` and its tensor entry.
    pub fn find(&self, name: &str) -> Option<(&Gguf, &TensorInfo)> {
        let &(s, t) = self.by_name.get(name)?;
        Some((&self.shards[s], &self.shards[s].tensors[t]))
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.find(name).map(|(_, t)| t)
    }
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        self.find(name).map(|(g, t)| g.tensor_bytes(t))
    }
    pub fn tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        self.find(name).map(|(g, t)| g.tensor_f32(t))
    }
    /// All tensors in shard order.
    pub fn tensors(&self) -> impl Iterator<Item = (&Gguf, &TensorInfo)> {
        self.shards.iter().flat_map(|g| g.tensors.iter().map(move |t| (g, t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shard_name_expansion() {
        let p = shard_paths(Path::new("/m/X-Q2_K-00001-of-00003.gguf"));
        assert_eq!(p.len(), 3);
        assert_eq!(p[2], PathBuf::from("/m/X-Q2_K-00003-of-00003.gguf"));
        assert_eq!(shard_paths(Path::new("/m/plain.gguf")), vec![PathBuf::from("/m/plain.gguf")]);
    }
}
