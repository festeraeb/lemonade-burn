//! Memory-mapped GGUF file: header, metadata key/values, tensor table, data section.

use crate::types::GgmlType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    Format(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io: {e}"),
            GgufError::Format(s) => write!(f, "gguf: {s}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

fn bad<T>(s: impl Into<String>) -> Result<T, GgufError> {
    Err(GgufError::Format(s.into()))
}

/// A metadata value. Arrays keep their element type; nested arrays are not supported.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            MetaValue::U8(v) => v as u64,
            MetaValue::U16(v) => v as u64,
            MetaValue::U32(v) => v as u64,
            MetaValue::U64(v) => v,
            MetaValue::I8(v) if v >= 0 => v as u64,
            MetaValue::I16(v) if v >= 0 => v as u64,
            MetaValue::I32(v) if v >= 0 => v as u64,
            MetaValue::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            MetaValue::F32(v) => v as f64,
            MetaValue::F64(v) => v,
            _ => return self.as_u64().map(|v| v as f64),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: `dims[0]` is the contiguous axis.
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    /// Byte offset from the start of the data section.
    pub offset: usize,
}

impl TensorInfo {
    pub fn n_elems(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn n_bytes(&self) -> Option<usize> {
        let (bv, bb) = self.ty.block()?;
        Some(self.n_elems() / bv * bb)
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.pos + n > self.buf.len() {
            return bad(format!("truncated at byte {} (+{n})", self.pos));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<MetaValue, GgufError> {
        Ok(match ty {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.u8()? as i8),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.u16()? as i16),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(f32::from_bits(self.u32()?)),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                MetaValue::Array(v)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(f64::from_bits(self.u64()?)),
            other => return bad(format!("unknown metadata value type {other}")),
        })
    }
}

/// An opened GGUF file. Tensor data is served as slices of the mapping (zero copy).
pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub meta: HashMap<String, MetaValue>,
    /// Metadata keys in file order (the HashMap loses it).
    pub meta_order: Vec<String>,
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    pub data_start: usize,
    pub alignment: usize,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only mapping of a file we do not write; callers must not truncate it.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self, GgufError> {
        let mut c = Cursor { buf: &mmap, pos: 0 };
        if c.take(4)? != b"GGUF" {
            return bad("not a GGUF file (bad magic)");
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return bad(format!("unsupported GGUF version {version}"));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut meta = HashMap::with_capacity(n_kv);
        let mut meta_order = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let val = c.value(ty)?;
            meta_order.push(key.clone());
            meta.insert(key, val);
        }
        let alignment = meta.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut by_name = HashMap::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            if nd > 4 {
                return bad(format!("{name}: {nd} dims"));
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()? as usize);
            }
            let ty = GgmlType::from_id(c.u32()?);
            let offset = c.u64()? as usize;
            by_name.insert(name.clone(), i);
            tensors.push(TensorInfo { name, dims, ty, offset });
        }
        let data_start = c.pos.div_ceil(alignment) * alignment;
        for t in &tensors {
            if let Some(nb) = t.n_bytes() {
                if data_start + t.offset + nb > mmap.len() {
                    return bad(format!("{}: data runs past end of file", t.name));
                }
            }
        }
        Ok(Self { mmap, version, meta, meta_order, tensors, by_name, data_start, alignment })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Raw bytes of a tensor's data.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let nb = t.n_bytes().unwrap_or_else(|| panic!("{}: unknown type {}", t.name, t.ty.name()));
        &self.mmap[self.data_start + t.offset..self.data_start + t.offset + nb]
    }

    /// Raw bytes at `[start, start + len)` of the data section (a sub-range of a tensor).
    pub fn tensor_bytes_at(&self, start: usize, len: usize) -> &[u8] {
        &self.mmap[self.data_start + start..self.data_start + start + len]
    }

    /// Whole tensor dequantized to f32, in GGUF (row-major over `dims` reversed) order.
    pub fn tensor_f32(&self, t: &TensorInfo) -> Vec<f32> {
        let n = t.n_elems();
        let mut out = vec![0f32; n];
        let row = t.dims[0];
        let rb = t.ty.row_bytes(row).expect("row not a whole number of blocks");
        let bytes = self.tensor_bytes(t);
        for (r, o) in out.chunks_exact_mut(row).enumerate() {
            t.ty.dequant_row(&bytes[r * rb..(r + 1) * rb], o);
        }
        out
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.meta.get(key)?.as_u64()
    }
    pub fn meta_f64(&self, key: &str) -> Option<f64> {
        self.meta.get(key)?.as_f64()
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta.get(key)?.as_str()
    }
    pub fn meta_u64_array(&self, key: &str) -> Option<Vec<u64>> {
        self.meta.get(key)?.as_array()?.iter().map(|v| v.as_u64()).collect()
    }
}
