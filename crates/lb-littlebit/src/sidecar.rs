//! The sidecar file: a safetensors file beside a GGUF that replaces some of its matrices with
//! factorised layers. Keys, for a GGUF tensor `N` and path `p` (0 primary, 1 residual):
//!
//! * `N.lb{p}.u`     u32 `[out, words(r)]`   signs of U
//! * `N.lb{p}.v`     u32 `[r, words(in)]`    signs of Vᵀ
//! * `N.lb{p}.s_out` f16 `[out]`, `N.lb{p}.s_lat` f16 `[r]`, `N.lb{p}.s_in` f16 `[in]`
//!
//! Header metadata `format` = [`FORMAT`]; anything else in the header is informational.

use crate::{words, Linear, Path};
use anyhow::{bail, Context, Result};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use std::collections::{BTreeMap, HashMap};
use std::path::Path as FsPath;

pub const FORMAT: &str = "lemonade-burn littlebit v1";

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Write `layers` (GGUF tensor name → layer) with `meta` added to the header.
pub fn write(path: &FsPath, layers: &BTreeMap<String, Linear>, meta: &BTreeMap<String, String>) -> Result<()> {
    let mut bufs: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, l) in layers {
        for (pi, p) in l.paths.iter().enumerate() {
            p.check().with_context(|| format!("{name} path {pi}"))?;
            let k = |s: &str| format!("{name}.lb{pi}.{s}");
            bufs.push((k("u"), Dtype::U32, vec![p.out, words(p.r)], u32_bytes(&p.u)));
            bufs.push((k("v"), Dtype::U32, vec![p.r, words(p.inp)], u32_bytes(&p.v)));
            bufs.push((k("s_out"), Dtype::F16, vec![p.out], f16_bytes(&p.s_out)));
            bufs.push((k("s_lat"), Dtype::F16, vec![p.r], f16_bytes(&p.s_lat)));
            bufs.push((k("s_in"), Dtype::F16, vec![p.inp], f16_bytes(&p.s_in)));
        }
    }
    let views: Vec<(String, TensorView<'_>)> =
        bufs.iter().map(|(n, d, s, b)| Ok((n.clone(), TensorView::new(*d, s.clone(), b)?))).collect::<Result<_>>()?;
    let mut info: HashMap<String, String> = meta.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    info.insert("format".into(), FORMAT.into());
    let bytes = safetensors::serialize(views, Some(info))?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read every layer in the file, keyed by GGUF tensor name.
pub fn read(path: &FsPath) -> Result<BTreeMap<String, Linear>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let (_, md) = SafeTensors::read_metadata(&bytes)?;
    let fmt = md.metadata().as_ref().and_then(|m| m.get("format").cloned()).unwrap_or_default();
    if fmt != FORMAT {
        bail!("{}: format {fmt:?}, expected {FORMAT:?}", path.display());
    }
    let st = SafeTensors::deserialize(&bytes)?;
    let mut names: Vec<(String, usize)> = Vec::new();
    for k in st.names() {
        if let Some(stem) = k.strip_suffix(".u") {
            if let Some((name, p)) = stem.rsplit_once(".lb") {
                names.push((name.to_string(), p.parse()?));
            }
        }
    }
    names.sort();
    let mut out: BTreeMap<String, Linear> = BTreeMap::new();
    for (name, pi) in names {
        let k = |s: &str| format!("{name}.lb{pi}.{s}");
        let get = |s: &str, dt: Dtype| -> Result<(Vec<usize>, Vec<u8>)> {
            let t = st.tensor(&k(s)).with_context(|| format!("{} missing", k(s)))?;
            if t.dtype() != dt {
                bail!("{}: dtype {:?}, expected {dt:?}", k(s), t.dtype());
            }
            Ok((t.shape().to_vec(), t.data().to_vec()))
        };
        let u32s = |b: &[u8]| b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<u32>>();
        let f16s = |b: &[u8]| b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect::<Vec<f32>>();
        let (su, u) = get("u", Dtype::U32)?;
        let (sv, v) = get("v", Dtype::U32)?;
        let (_, so) = get("s_out", Dtype::F16)?;
        let (_, sl) = get("s_lat", Dtype::F16)?;
        let (_, si) = get("s_in", Dtype::F16)?;
        let (s_out, s_lat, s_in) = (f16s(&so), f16s(&sl), f16s(&si));
        let p = Path { out: su[0], inp: s_in.len(), r: sv[0], u: u32s(&u), v: u32s(&v), s_out, s_lat, s_in };
        p.check().with_context(|| format!("{name} path {pi}"))?;
        let l = out.entry(name.clone()).or_insert(Linear { out: p.out, inp: p.inp, paths: Vec::new() });
        if l.out != p.out || l.inp != p.inp {
            bail!("{name}: paths disagree on shape");
        }
        l.paths.push(p);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn write_then_read_is_exact_up_to_f16_scales() {
        let mut seed = 3;
        let p = crate::tests::random_path(33, 65, 17, &mut seed);
        let l = crate::Linear { out: 33, inp: 65, paths: vec![p.clone()] };
        let dir = std::env::temp_dir().join(format!("lb-littlebit-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.safetensors");
        let mut m = std::collections::BTreeMap::new();
        m.insert("blk.0.ffn_up.weight".to_string(), l);
        super::write(&f, &m, &Default::default()).unwrap();
        let back = super::read(&f).unwrap();
        let q = &back["blk.0.ffn_up.weight"].paths[0];
        assert_eq!(q.u, p.u);
        assert_eq!(q.v, p.v);
        for (a, b) in q.s_in.iter().zip(&p.s_in) {
            assert!((a - b).abs() <= 1e-3 * b.abs());
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
