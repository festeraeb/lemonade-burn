//! GGML tensor element types we understand, with their block geometry and dequantizers.

use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q6K,
    /// Any other type id: readable as raw bytes, not dequantizable here.
    Other(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            8 => GgmlType::Q8_0,
            14 => GgmlType::Q6K,
            other => GgmlType::Other(other),
        }
    }

    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Q4_0 => "Q4_0".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q6K => "Q6_K".into(),
            GgmlType::Other(id) => format!("type#{id}"),
        }
    }

    /// Values per block and bytes per block.
    pub fn block(self) -> Option<(usize, usize)> {
        Some(match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 => (1, 2),
            GgmlType::Q4_0 => (32, 18),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q6K => (256, 210),
            GgmlType::Other(_) => return None,
        })
    }

    /// Bytes of one row of `n` values (`n` must be a whole number of blocks).
    pub fn row_bytes(self, n: usize) -> Option<usize> {
        let (bv, bb) = self.block()?;
        if n % bv != 0 {
            return None;
        }
        Some(n / bv * bb)
    }

    /// Dequantize one row of `n` values from `src` (exactly `row_bytes(n)` bytes) into `dst`.
    pub fn dequant_row(self, src: &[u8], dst: &mut [f32]) {
        let n = dst.len();
        let (bv, bb) = self.block().expect("dequant of unsupported type");
        assert_eq!(n % bv, 0);
        assert_eq!(src.len(), n / bv * bb, "row byte length");
        match self {
            GgmlType::F32 => {
                for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                    *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            GgmlType::F16 => {
                for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                    *d = f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            GgmlType::Q8_0 => {
                for (o, b) in dst.chunks_exact_mut(32).zip(src.chunks_exact(34)) {
                    dequant_q8_0_block(b, o);
                }
            }
            GgmlType::Q4_0 => {
                for (o, b) in dst.chunks_exact_mut(32).zip(src.chunks_exact(18)) {
                    dequant_q4_0_block(b, o);
                }
            }
            GgmlType::Q6K => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(210)) {
                    dequant_q6_k_block(b, o);
                }
            }
            GgmlType::Other(id) => panic!("cannot dequantize ggml type {id}"),
        }
    }
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Q8_0 block: `d` (f16) then 32 signed bytes.
pub fn dequant_q8_0_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for j in 0..32 {
        out[j] = d * (b[2 + j] as i8) as f32;
    }
}

/// Q4_0 block: `d` (f16) then 16 nibble bytes; low nibble = value `j`, high = value `j + 16`.
pub fn dequant_q4_0_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for j in 0..16 {
        let byte = b[2 + j];
        out[j] = d * ((byte & 0x0f) as i32 - 8) as f32;
        out[j + 16] = d * ((byte >> 4) as i32 - 8) as f32;
    }
}

/// Q6_K super-block of 256 values. Layout: `ql[128]`, `qh[64]`, `scales[16]` (i8), `d` (f16).
/// For each 128-value half `h` (ql offset 64h, qh offset 32h, scales offset 8h) and
/// `l` in 0..32, the four values at `l`, `l+32`, `l+64`, `l+96` are
/// `(ql[l] & 15 | (qh[l] >> 0 & 3) << 4)`, `(ql[l+32] & 15 | (qh[l] >> 2 & 3) << 4)`,
/// `(ql[l] >> 4 | (qh[l] >> 4 & 3) << 4)`, `(ql[l+32] >> 4 | (qh[l] >> 6 & 3) << 4)`,
/// each minus 32, times `d * scales[g]` where `g` = `l/16` plus 0, 2, 4, 6 respectively.
pub fn dequant_q6_k_block(b: &[u8], out: &mut [f32]) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = f16_at(b, 208);
    for h in 0..2 {
        let (ql, qh, sc) = (&ql[64 * h..], &qh[32 * h..], &sc[8 * h..]);
        let o = &mut out[128 * h..128 * (h + 1)];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// Quantize a row of f32 to Q4_0 bytes (for tests and synthetic weights): per block of 32,
/// `d = max_by_abs / -8`, `q = round(x / d) + 8` clamped to 0..15. Values must be a
/// whole number of blocks.
pub fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % 32, 0);
    let mut out = Vec::with_capacity(x.len() / 32 * 18);
    for blk in x.chunks_exact(32) {
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in blk {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let dh = f16::from_f32(d);
        out.extend_from_slice(&dh.to_le_bytes());
        for j in 0..16 {
            let q0 = ((blk[j] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
            let q1 = ((blk[j + 16] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
            out.push(q0 | (q1 << 4));
        }
    }
    out
}

/// Quantize a row of f32 to Q8_0 bytes (tests): per block of 32, `d = amax / 127`.
pub fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % 32, 0);
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in blk {
            out.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_round_trip_is_within_half_step() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 37 % 61) as f32 - 30.0) * 0.1).collect();
        let bytes = quantize_q4_0(&x);
        assert_eq!(bytes.len(), 36);
        let mut y = vec![0f32; 64];
        GgmlType::Q4_0.dequant_row(&bytes, &mut y);
        for (a, b) in x.iter().zip(&y) {
            // one step of |d| = 3/8: the far end of the block clamps at +7 steps
            assert!((a - b).abs() <= 0.375 + 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn q8_0_round_trip() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let bytes = quantize_q8_0(&x);
        let mut y = vec![0f32; 32];
        GgmlType::Q8_0.dequant_row(&bytes, &mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_0_nibble_placement() {
        // d = 1.0, byte j = (j+1) | ((15 - j) << 4) for j in 0..16
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        for j in 0..16u8 {
            b[2 + j as usize] = ((j + 1) & 15) | ((15 - j) << 4);
        }
        let mut y = vec![0f32; 32];
        dequant_q4_0_block(&b, &mut y);
        for j in 0..16 {
            assert_eq!(y[j], ((j as i32 + 1) & 15) as f32 - 8.0);
            assert_eq!(y[j + 16], (15 - j as i32) as f32 - 8.0);
        }
    }
}
