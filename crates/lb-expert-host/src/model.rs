//! The expert tensors of a GGUF MoE model, found by name, with nothing else of the model
//! loaded: `blk.N.ffn_gate_exps.weight`, `ffn_up_exps`, `ffn_down_exps`, each a stack of
//! `n_expert` equally shaped matrices that stay in the file mapping.

use anyhow::{bail, Context, Result};
use lb_gguf::{GgmlType, Gguf, QuantMat};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

/// A stack of `n` equally shaped quantized matrices in the file's data section.
#[derive(Clone, Debug)]
pub struct Stack {
    pub ty: GgmlType,
    pub n: usize,
    pub rows: usize,
    pub cols: usize,
    pub start: usize,
    pub per_expert: usize,
}

#[derive(Clone, Debug)]
pub struct LayerExperts {
    pub gate: Stack,
    pub up: Stack,
    pub down: Stack,
}

pub struct ModelExperts {
    pub gguf: Arc<Gguf>,
    /// By layer index; `None` for a layer with no experts.
    pub layers: Vec<Option<LayerExperts>>,
    pub n_expert: usize,
    /// Activation width (the expert's input and output).
    pub d: usize,
    /// Expert hidden width.
    pub n_ff: usize,
    pub fingerprint: [u8; 16],
}

/// Identity of a model file for the hello: its length, its first MiB (header, metadata and
/// tensor table) and 4 KiB from the start of every expert stack. Cheap, and two files that
/// differ in any expert tensor's first rows, shape, offset or type differ here. It is not a hash
/// of the whole file: a trunk and a host are expected to open the same file.
pub fn fingerprint(g: &Gguf) -> [u8; 16] {
    let map = g.mapping();
    let mut h = Sha256::new();
    h.update((map.len() as u64).to_le_bytes());
    h.update(&map[..map.len().min(1 << 20)]);
    for t in &g.tensors {
        if t.name.ends_with("_exps.weight") {
            if let Some(nb) = t.n_bytes() {
                let b = g.tensor_bytes_at(t.offset, nb.min(4096));
                h.update(b);
            }
        }
    }
    let d = h.finalize();
    d[..16].try_into().unwrap()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl ModelExperts {
    pub fn open(path: &Path) -> Result<Self> {
        let g = Gguf::open(path).with_context(|| format!("open {}", path.display()))?;
        Self::from_gguf(Arc::new(g))
    }

    pub fn from_gguf(gguf: Arc<Gguf>) -> Result<Self> {
        let g = &*gguf;
        let arch = g.meta_str("general.architecture").context("general.architecture")?.to_string();
        let n_layer = g.meta_u64(&format!("{arch}.block_count")).with_context(|| format!("{arch}.block_count"))? as usize;
        let n_expert = g.meta_u64(&format!("{arch}.expert_count")).with_context(|| format!("{arch}.expert_count (not a MoE model?)"))? as usize;
        let stack = |name: &str| -> Result<Option<Stack>> {
            let Some(t) = g.tensor(name) else { return Ok(None) };
            if t.dims.len() != 3 || t.dims[2] != n_expert {
                bail!("{name}: dims {:?}, expected [cols, rows, {n_expert}]", t.dims);
            }
            let (cols, rows) = (t.dims[0], t.dims[1]);
            let rb = t.ty.row_bytes(cols).with_context(|| format!("{name}: {cols} columns of {} are not whole blocks", t.ty.name()))?;
            Ok(Some(Stack { ty: t.ty, n: n_expert, rows, cols, start: t.offset, per_expert: rb * rows }))
        };
        let mut layers = Vec::with_capacity(n_layer);
        let (mut d, mut n_ff) = (0usize, 0usize);
        for l in 0..n_layer {
            let gate = stack(&format!("blk.{l}.ffn_gate_exps.weight"))?;
            let up = stack(&format!("blk.{l}.ffn_up_exps.weight"))?;
            let down = stack(&format!("blk.{l}.ffn_down_exps.weight"))?;
            match (gate, up, down) {
                (Some(gate), Some(up), Some(down)) => {
                    if gate.cols != up.cols || gate.rows != up.rows || down.cols != gate.rows || down.rows != gate.cols {
                        bail!("layer {l}: expert shapes gate {}x{} up {}x{} down {}x{} are not a SwiGLU FFN", gate.rows, gate.cols, up.rows, up.cols, down.rows, down.cols);
                    }
                    if d != 0 && (d != gate.cols || n_ff != gate.rows) {
                        bail!("layer {l}: expert shape differs from earlier layers");
                    }
                    d = gate.cols;
                    n_ff = gate.rows;
                    layers.push(Some(LayerExperts { gate, up, down }));
                }
                (None, None, None) => layers.push(None),
                _ => bail!("layer {l}: some of ffn_gate_exps / ffn_up_exps / ffn_down_exps are missing"),
            }
        }
        if d == 0 {
            bail!("no expert tensors (blk.N.ffn_*_exps.weight) in this file");
        }
        let fingerprint = fingerprint(g);
        Ok(Self { gguf, layers, n_expert, d, n_ff, fingerprint })
    }

    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    pub fn mat(&self, s: &Stack, e: usize) -> QuantMat<'_> {
        assert!(e < s.n);
        QuantMat::new(s.ty, s.rows, s.cols, self.gguf.tensor_bytes_at(s.start + e * s.per_expert, s.per_expert))
    }

    /// The whole stack as one matrix `[n * rows, cols]`.
    pub fn mat_all(&self, s: &Stack) -> QuantMat<'_> {
        QuantMat::new(s.ty, s.n * s.rows, s.cols, self.gguf.tensor_bytes_at(s.start, s.n * s.per_expert))
    }

    /// File bytes of one expert (gate + up + down) of `layer`; 0 for a layer without experts.
    pub fn expert_bytes(&self, layer: usize) -> usize {
        self.layers.get(layer).and_then(|l| l.as_ref()).map_or(0, |l| l.gate.per_expert + l.up.per_expert + l.down.per_expert)
    }
}
