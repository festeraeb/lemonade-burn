//! On-disk format of a capture run, one *chunk* (= one prompt: its prefill plus the greedy
//! decode that follows) at a time. Layout under the run directory:
//!
//! ```text
//! states/<id>.f32    f32 LE, token-major: for t in tokens, for l in layers: x_mid[n_embd], x_out[n_embd]
//! chunks/<id>.json   ChunkMeta (tokens, phase split, the engine's routes); written LAST
//! progress.jsonl     one appended line per finished chunk (a log; the files above decide "done")
//! ```
//!
//! A chunk is done when its meta file exists and its states file has exactly the size the meta
//! names, so a run killed mid-chunk redoes only that chunk. Both files are written to a `.tmp`
//! name and renamed into place.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChunkMeta {
    pub id: String,
    pub kind: String,
    pub model: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_expert: usize,
    pub k: usize,
    /// Tokens whose hidden states are captured, in order: the prompt (prefill), then every
    /// generated token that was fed back (decode).
    pub tokens: Vec<u32>,
    /// The first `n_prefill` tokens were one prefill pass; the rest were decode passes.
    pub n_prefill: usize,
    /// `routes[t][layer]`: the experts the engine chose (empty for dense layers).
    pub routes: Vec<Vec<Vec<u16>>>,
    /// Matching normalized routing weights.
    pub weights: Vec<Vec<Vec<f32>>>,
    /// The token the model produced after the last captured one (not itself captured).
    pub last_token: u32,
    pub text_head: String,
    pub host: String,
    pub secs: f64,
}

impl ChunkMeta {
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }
    pub fn states_len_bytes(&self) -> u64 {
        (self.n_tokens() * self.n_layer * 2 * self.n_embd * 4) as u64
    }
    pub fn is_decode(&self, t: usize) -> bool {
        t >= self.n_prefill
    }
}

pub struct RunDir {
    pub root: PathBuf,
}

/// 0 = `x_mid` (after the layer's operator), 1 = `x_out` (after its FFN).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Mid = 0,
    Out = 1,
}

impl RunDir {
    pub fn new(root: &Path) -> Result<Self> {
        for d in ["states", "chunks"] {
            std::fs::create_dir_all(root.join(d)).with_context(|| format!("mkdir {}", root.join(d).display()))?;
        }
        Ok(Self { root: root.to_path_buf() })
    }

    pub fn meta_path(&self, id: &str) -> PathBuf {
        self.root.join("chunks").join(format!("{id}.json"))
    }
    pub fn states_path(&self, id: &str) -> PathBuf {
        self.root.join("states").join(format!("{id}.f32"))
    }

    /// The chunk's meta when the chunk is complete on disk, else `None`.
    pub fn done(&self, id: &str) -> Option<ChunkMeta> {
        let meta: ChunkMeta = serde_json::from_slice(&std::fs::read(self.meta_path(id)).ok()?).ok()?;
        let len = std::fs::metadata(self.states_path(id)).ok()?.len();
        (len == meta.states_len_bytes()).then_some(meta)
    }

    /// Every complete chunk, sorted by id.
    pub fn list_done(&self) -> Result<Vec<ChunkMeta>> {
        let mut out = Vec::new();
        for ent in std::fs::read_dir(self.root.join("chunks"))? {
            let p = ent?.path();
            if p.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let id = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
            if let Some(m) = self.done(&id) {
                out.push(m);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Write a finished chunk: states first, meta last (each via `.tmp` + rename), then one
    /// line on `progress.jsonl`.
    pub fn write_chunk(&self, meta: &ChunkMeta, states: &[f32]) -> Result<()> {
        if states.len() as u64 * 4 != meta.states_len_bytes() {
            bail!("chunk {}: {} state floats, meta says {}", meta.id, states.len(), meta.states_len_bytes() / 4);
        }
        let sp = self.states_path(&meta.id);
        let tmp = sp.with_extension("f32.tmp");
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            for v in states {
                f.write_all(&v.to_le_bytes())?;
            }
            f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        }
        std::fs::rename(&tmp, &sp)?;
        let mp = self.meta_path(&meta.id);
        let tmp = mp.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(meta)?)?;
        std::fs::rename(&tmp, &mp)?;
        let line = serde_json::json!({
            "id": meta.id, "kind": meta.kind, "n_prefill": meta.n_prefill,
            "n_decode": meta.n_tokens() - meta.n_prefill, "secs": meta.secs, "host": meta.host,
            "ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        });
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.root.join("progress.jsonl"))?;
        writeln!(f, "{line}")?;
        Ok(())
    }

    /// Memory-map a chunk's states.
    pub fn open_states(&self, meta: &ChunkMeta) -> Result<States> {
        let f = std::fs::File::open(self.states_path(&meta.id))?;
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if map.len() as u64 != meta.states_len_bytes() {
            bail!("chunk {}: states file size changed", meta.id);
        }
        Ok(States { map, n_layer: meta.n_layer, n_embd: meta.n_embd })
    }
}

pub struct States {
    map: memmap2::Mmap,
    n_layer: usize,
    n_embd: usize,
}

impl States {
    /// One `n_embd` row: token `t`, layer `l`, `which` state.
    pub fn row(&self, t: usize, l: usize, which: Which) -> Vec<f32> {
        let off = ((t * self.n_layer + l) * 2 + which as usize) * self.n_embd * 4;
        self.map[off..off + self.n_embd * 4].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_meta(id: &str) -> ChunkMeta {
        ChunkMeta {
            id: id.into(),
            kind: "test".into(),
            model: "tiny".into(),
            n_layer: 2,
            n_embd: 3,
            n_expert: 4,
            k: 2,
            tokens: vec![5, 6],
            n_prefill: 1,
            routes: vec![vec![vec![], vec![0, 1]], vec![vec![], vec![2, 3]]],
            weights: vec![vec![vec![], vec![0.5, 0.5]], vec![vec![], vec![0.7, 0.3]]],
            last_token: 7,
            text_head: String::new(),
            host: "test".into(),
            secs: 0.0,
        }
    }

    #[test]
    fn chunk_round_trip_and_done_rule() {
        let dir = std::env::temp_dir().join(format!("lbep-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let run = RunDir::new(&dir).unwrap();
        let meta = tiny_meta("c0");
        assert!(run.done("c0").is_none());
        // 2 tokens x 2 layers x 2 states x 3 floats
        let states: Vec<f32> = (0..24).map(|i| i as f32).collect();
        assert!(run.write_chunk(&meta, &states[..23]).is_err());
        run.write_chunk(&meta, &states).unwrap();
        let back = run.done("c0").expect("complete");
        assert_eq!(back, meta);
        let st = run.open_states(&back).unwrap();
        assert_eq!(st.row(0, 0, Which::Mid), vec![0.0, 1.0, 2.0]);
        assert_eq!(st.row(0, 0, Which::Out), vec![3.0, 4.0, 5.0]);
        assert_eq!(st.row(1, 1, Which::Out), vec![21.0, 22.0, 23.0]);
        // a truncated states file means "not done"
        std::fs::write(run.states_path("c0"), [0u8; 8]).unwrap();
        assert!(run.done("c0").is_none());
        assert_eq!(run.list_done().unwrap().len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
