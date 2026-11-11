//! The wire format between a trunk and an expert host. One message is one frame body; the
//! transport carries the length. Everything is little-endian.
//!
//! ```text
//! header   magic u32 "W2W1" | kind u8 | pad [u8; 3] | seq u64                         16 bytes
//! Hello    (no payload)
//! HelloOk  fingerprint [u8; 16] | d u32 | n_ff u32 | n_layer u16 | n_expert u16
//!          | held_words u16 | pad u16 | held [u64; n_layer * held_words] | arith u64 | backend str | name str
//! Run      layer u16 | n u16 | m u32 | d u32 | ids [u16; n] (+ pad to 4) | counts [u32; n]
//!          | rows [u32; sum(counts)] | x [f32; m * d]
//! RunOk    host_ns u64 | n u16 | pad u16 | d u32 | y [f32; sum(counts) * d]
//! Refused  layer u16 | n u16 | ids [u16; n]          experts of the request the host does not hold
//! Error    str
//! Ping / Pong  (no payload)
//! ```
//! `str` is `u16` length then UTF-8.
//!
//! A `Run` asks for `n` experts of one layer. `x` is `m` activation rows of width `d`; expert
//! `ids[i]` runs on the rows `rows[..]` of its group (`counts[i]` of them, in that order, as one
//! batch) and its output, `counts[i]` rows of `d`, comes back **unweighted** in the same order.
//! The trunk applies the router weights and sums in ascending expert id itself, so the float
//! order is the one-process path's. A decode token is `m = 1`, every group `[0]`: 8 KB out,
//! `n` x 8 KB back on LFM2.5 (d = 2048).
//!
//! The group shape matters for parity, not only for size: the CPU kernel takes a different
//! (fused) path for a one-row batch than for a many-row one, so an expert must see exactly the
//! rows the one-process path would have given it, in one call.

use std::fmt;

pub const MAGIC: u32 = u32::from_le_bytes(*b"W2W1");
pub const HEADER: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Hello = 1,
    HelloOk = 2,
    Run = 3,
    RunOk = 4,
    Refused = 5,
    Error = 6,
    Ping = 7,
    Pong = 8,
}

impl Kind {
    fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            1 => Kind::Hello,
            2 => Kind::HelloOk,
            3 => Kind::Run,
            4 => Kind::RunOk,
            5 => Kind::Refused,
            6 => Kind::Error,
            7 => Kind::Ping,
            8 => Kind::Pong,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireError(pub String);

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wire: {}", self.0)
    }
}
impl std::error::Error for WireError {}

fn bad<T>(s: impl Into<String>) -> Result<T, WireError> {
    Err(WireError(s.into()))
}

/// What a host says about itself in `HelloOk`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostInfo {
    /// [`crate::model::fingerprint`] of the host's model file: a trunk on another file must not
    /// use this host.
    pub fingerprint: [u8; 16],
    pub d: u32,
    pub n_ff: u32,
    pub n_layer: u16,
    pub n_expert: u16,
    /// `held[layer * words + e / 64] >> (e % 64) & 1`.
    pub held_words: u16,
    pub held: Vec<u64>,
    /// [`crate::expert::arith_fingerprint`] of a CPU backend; 0 for a device backend.
    pub arith: u64,
    pub backend: String,
    pub name: String,
}

impl HostInfo {
    pub fn holds(&self, layer: usize, expert: usize) -> bool {
        let w = self.held_words as usize;
        if layer >= self.n_layer as usize || expert >= self.n_expert as usize || w == 0 {
            return false;
        }
        self.held[layer * w + expert / 64] >> (expert % 64) & 1 == 1
    }

    pub fn n_held(&self) -> usize {
        self.held.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// A decoded `Run`, borrowing nothing: the host's working copy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Run {
    pub layer: u16,
    pub m: u32,
    pub d: u32,
    pub ids: Vec<u16>,
    /// Row indices of each expert's batch, in `ids` order.
    pub groups: Vec<Vec<u32>>,
    pub x: Vec<f32>,
}

/// A decoded reply to a `Run`.
#[derive(Clone, Debug, PartialEq)]
pub enum RunReply {
    /// `y` is every group's output rows concatenated in request order; `host_ns` is the time the
    /// host spent computing (round trip minus this is the hop).
    Ok { host_ns: u64, n: u16, d: u32, y: Vec<f32> },
    Refused { layer: u16, ids: Vec<u16> },
    Error(String),
}

fn header(out: &mut Vec<u8>, kind: Kind, seq: u64) {
    out.clear();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.push(kind as u8);
    out.extend_from_slice(&[0u8; 3]);
    out.extend_from_slice(&seq.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    let b = &s.as_bytes()[..s.len().min(u16::MAX as usize)];
    out.extend_from_slice(&(b.len() as u16).to_le_bytes());
    out.extend_from_slice(b);
}

/// Append `v` as little-endian bytes.
pub fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 has no padding and any bit pattern is a valid u8; the slice covers exactly
        // `v`'s bytes, and on a little-endian target those are the wire bytes.
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        out.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Read `n` little-endian f32 from `b` (exactly `4 n` bytes) onto the end of `out`.
pub fn get_f32s(b: &[u8], n: usize, out: &mut Vec<f32>) {
    assert_eq!(b.len(), n * 4);
    #[cfg(target_endian = "little")]
    {
        out.reserve(n);
        let at = out.len();
        // SAFETY: `reserve` made room for `n` more f32; the source is `4 n` initialized bytes and
        // every bit pattern is a valid f32; the regions do not overlap (one is `out`'s spare
        // capacity).
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), out.as_mut_ptr().add(at) as *mut u8, n * 4);
            out.set_len(at + n);
        }
    }
    #[cfg(not(target_endian = "little"))]
    out.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
}

struct Rd<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        if self.b.len() - self.pos < n {
            return bad(format!("truncated: wanted {n} bytes at {}, frame is {}", self.pos, self.b.len()));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, WireError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32, WireError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Result<u64, WireError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String, WireError> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn end(&self) -> Result<(), WireError> {
        if self.pos != self.b.len() {
            return bad(format!("{} trailing bytes", self.b.len() - self.pos));
        }
        Ok(())
    }
}

/// Kind and sequence number of a frame.
pub fn peek(frame: &[u8]) -> Result<(Kind, u64), WireError> {
    if frame.len() < HEADER {
        return bad(format!("frame of {} bytes is shorter than the header", frame.len()));
    }
    if u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) != MAGIC {
        return bad("bad magic");
    }
    let kind = Kind::from_u8(frame[4]).ok_or_else(|| WireError(format!("unknown kind {}", frame[4])))?;
    Ok((kind, u64::from_le_bytes(frame[8..16].try_into().unwrap())))
}

pub fn encode_simple(out: &mut Vec<u8>, kind: Kind, seq: u64) {
    header(out, kind, seq);
}

pub fn encode_error(out: &mut Vec<u8>, seq: u64, msg: &str) {
    header(out, Kind::Error, seq);
    put_str(out, msg);
}

pub fn encode_hello_ok(out: &mut Vec<u8>, seq: u64, h: &HostInfo) {
    header(out, Kind::HelloOk, seq);
    out.extend_from_slice(&h.fingerprint);
    out.extend_from_slice(&h.d.to_le_bytes());
    out.extend_from_slice(&h.n_ff.to_le_bytes());
    out.extend_from_slice(&h.n_layer.to_le_bytes());
    out.extend_from_slice(&h.n_expert.to_le_bytes());
    out.extend_from_slice(&h.held_words.to_le_bytes());
    out.extend_from_slice(&[0u8; 2]);
    for w in &h.held {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&h.arith.to_le_bytes());
    put_str(out, &h.backend);
    put_str(out, &h.name);
}

pub fn decode_hello_ok(frame: &[u8]) -> Result<HostInfo, WireError> {
    let mut r = Rd { b: frame, pos: HEADER };
    let fingerprint: [u8; 16] = r.take(16)?.try_into().unwrap();
    let (d, n_ff, n_layer, n_expert, held_words) = (r.u32()?, r.u32()?, r.u16()?, r.u16()?, r.u16()?);
    r.take(2)?;
    let n = n_layer as usize * held_words as usize;
    let mut held = Vec::with_capacity(n);
    for _ in 0..n {
        held.push(r.u64()?);
    }
    let arith = r.u64()?;
    let (backend, name) = (r.str()?, r.str()?);
    r.end()?;
    Ok(HostInfo { fingerprint, d, n_ff, n_layer, n_expert, held_words, held, arith, backend, name })
}

/// Encode a `Run`. `groups[i]` are the row indices (into `x`'s `m` rows) of expert `ids[i]`.
pub fn encode_run(out: &mut Vec<u8>, seq: u64, layer: u16, ids: &[u16], groups: &[&[u32]], x: &[f32], m: usize, d: usize) {
    assert_eq!(ids.len(), groups.len());
    assert_eq!(x.len(), m * d);
    header(out, Kind::Run, seq);
    out.extend_from_slice(&layer.to_le_bytes());
    out.extend_from_slice(&(ids.len() as u16).to_le_bytes());
    out.extend_from_slice(&(m as u32).to_le_bytes());
    out.extend_from_slice(&(d as u32).to_le_bytes());
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    if ids.len() % 2 == 1 {
        out.extend_from_slice(&[0u8; 2]);
    }
    for g in groups {
        out.extend_from_slice(&(g.len() as u32).to_le_bytes());
    }
    for g in groups {
        for r in *g {
            out.extend_from_slice(&r.to_le_bytes());
        }
    }
    put_f32s(out, x);
}

/// Decode a `Run` into `run` (its vectors are reused). Checks every row index against `m`.
pub fn decode_run(frame: &[u8], run: &mut Run) -> Result<(), WireError> {
    let mut r = Rd { b: frame, pos: HEADER };
    run.layer = r.u16()?;
    let n = r.u16()? as usize;
    run.m = r.u32()?;
    run.d = r.u32()?;
    run.ids.clear();
    for _ in 0..n {
        run.ids.push(r.u16()?);
    }
    if n % 2 == 1 {
        r.take(2)?;
    }
    let mut counts = Vec::with_capacity(n);
    for _ in 0..n {
        counts.push(r.u32()? as usize);
    }
    run.groups.resize_with(n, Vec::new);
    for (g, &c) in run.groups.iter_mut().zip(&counts) {
        g.clear();
        if c == 0 {
            return bad("an expert with no rows");
        }
        for _ in 0..c {
            let row = r.u32()?;
            if row >= run.m {
                return bad(format!("row {row} of a {}-row request", run.m));
            }
            g.push(row);
        }
    }
    let nx = (run.m as usize).checked_mul(run.d as usize).ok_or_else(|| WireError("m * d overflows".into()))?;
    let xb = r.take(nx.checked_mul(4).ok_or_else(|| WireError("m * d overflows".into()))?)?;
    run.x.clear();
    get_f32s(xb, nx, &mut run.x);
    r.end()
}

/// Start a `RunOk`; the caller appends `y` with [`put_f32s`] (per group, in request order).
pub fn begin_run_ok(out: &mut Vec<u8>, seq: u64, host_ns: u64, n: u16, d: u32) {
    header(out, Kind::RunOk, seq);
    out.extend_from_slice(&host_ns.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&[0u8; 2]);
    out.extend_from_slice(&d.to_le_bytes());
}

/// Overwrite the `host_ns` of a `RunOk` built with [`begin_run_ok`].
pub fn set_host_ns(out: &mut [u8], host_ns: u64) {
    out[HEADER..HEADER + 8].copy_from_slice(&host_ns.to_le_bytes());
}

pub fn encode_refused(out: &mut Vec<u8>, seq: u64, layer: u16, ids: &[u16]) {
    header(out, Kind::Refused, seq);
    out.extend_from_slice(&layer.to_le_bytes());
    out.extend_from_slice(&(ids.len() as u16).to_le_bytes());
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
}

/// Decode the reply to a `Run`: `RunOk`, `Refused` or `Error`.
pub fn decode_run_reply(frame: &[u8]) -> Result<RunReply, WireError> {
    let (kind, _) = peek(frame)?;
    let mut r = Rd { b: frame, pos: HEADER };
    match kind {
        Kind::RunOk => {
            let host_ns = r.u64()?;
            let n = r.u16()?;
            r.take(2)?;
            let d = r.u32()?;
            let rest = &frame[r.pos..];
            if rest.len() % 4 != 0 || d == 0 || (rest.len() / 4) % d as usize != 0 {
                return bad(format!("RunOk payload of {} bytes is not rows of {d}", rest.len()));
            }
            let mut y = Vec::new();
            get_f32s(rest, rest.len() / 4, &mut y);
            Ok(RunReply::Ok { host_ns, n, d, y })
        }
        Kind::Refused => {
            let layer = r.u16()?;
            let n = r.u16()? as usize;
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                ids.push(r.u16()?);
            }
            r.end()?;
            Ok(RunReply::Refused { layer, ids })
        }
        Kind::Error => Ok(RunReply::Error(r.str()?)),
        other => bad(format!("{other:?} is not a reply to Run")),
    }
}

pub fn decode_error(frame: &[u8]) -> String {
    let mut r = Rd { b: frame, pos: HEADER };
    r.str().unwrap_or_else(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_round_trips_decode_and_prefill_shapes() {
        let d = 8;
        let x: Vec<f32> = (0..3 * d).map(|i| i as f32 * 0.5 - 3.0).collect();
        let mut out = Vec::new();
        // prefill shape: three rows, uneven groups, odd id count (exercises the pad)
        let groups: [&[u32]; 3] = [&[0, 2], &[1], &[2, 1, 0]];
        encode_run(&mut out, 77, 5, &[3, 9, 60], &groups, &x, 3, d);
        assert_eq!(peek(&out).unwrap(), (Kind::Run, 77));
        let mut run = Run::default();
        decode_run(&out, &mut run).unwrap();
        assert_eq!((run.layer, run.m, run.d), (5, 3, d as u32));
        assert_eq!(run.ids, vec![3, 9, 60]);
        assert_eq!(run.groups, vec![vec![0, 2], vec![1], vec![2, 1, 0]]);
        assert_eq!(run.x, x);
        // decode shape: the plan's 8 KB + ids
        let x1 = vec![1.25f32; 2048];
        let g: [&[u32]; 4] = [&[0], &[0], &[0], &[0]];
        encode_run(&mut out, 1, 2, &[1, 2, 3, 4], &g, &x1, 1, 2048);
        assert_eq!(out.len(), HEADER + 12 + 8 + 16 + 16 + 8192);
        decode_run(&out, &mut run).unwrap();
        assert_eq!(run.x, x1);
    }

    #[test]
    fn a_bad_row_or_a_short_frame_is_an_error_not_a_panic() {
        let mut out = Vec::new();
        let g: [&[u32]; 1] = [&[4]];
        encode_run(&mut out, 1, 0, &[0], &g, &[0.0; 8], 2, 4);
        let mut run = Run::default();
        assert!(decode_run(&out, &mut run).unwrap_err().0.contains("row 4"));
        let g: [&[u32]; 1] = [&[1]];
        encode_run(&mut out, 1, 0, &[0], &g, &[0.0; 8], 2, 4);
        for cut in [0, 5, HEADER, out.len() - 1] {
            assert!(peek(&out[..cut]).is_err() || decode_run(&out[..cut], &mut run).is_err(), "cut {cut}");
        }
        out.push(0);
        assert!(decode_run(&out, &mut run).is_err());
    }

    #[test]
    fn replies_round_trip() {
        let mut out = Vec::new();
        begin_run_ok(&mut out, 9, 0, 2, 4);
        put_f32s(&mut out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        set_host_ns(&mut out, 1234);
        assert_eq!(decode_run_reply(&out).unwrap(), RunReply::Ok { host_ns: 1234, n: 2, d: 4, y: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] });
        encode_refused(&mut out, 9, 7, &[5, 6]);
        assert_eq!(decode_run_reply(&out).unwrap(), RunReply::Refused { layer: 7, ids: vec![5, 6] });
        encode_error(&mut out, 9, "no such layer");
        assert_eq!(decode_run_reply(&out).unwrap(), RunReply::Error("no such layer".into()));
        let info = HostInfo { fingerprint: [7; 16], d: 2048, n_ff: 1536, n_layer: 3, n_expert: 70, held_words: 2, held: vec![1, 0, 0, 32, u64::MAX, 3], arith: 99, backend: "cpu".into(), name: "a".into() };
        encode_hello_ok(&mut out, 0, &info);
        let back = decode_hello_ok(&out).unwrap();
        assert_eq!(back, info);
        assert!(back.holds(0, 0) && !back.holds(0, 1) && back.holds(1, 69) && !back.holds(1, 68) && back.holds(2, 65) && !back.holds(3, 0));
        assert_eq!(back.n_held(), 1 + 1 + 64 + 2);
    }
}
