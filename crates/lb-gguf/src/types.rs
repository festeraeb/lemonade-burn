//! GGML tensor element types we understand, with their block geometry and dequantizers.

use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    /// i-quant, 32 values: f16 `d`, 16 nibble bytes indexing the non-linear 16-level table
    /// [`IQ4_LEVELS`] (low nibble = value j, high = value j + 16).
    IQ4NL,
    /// i-quant, 256 values in 136 bytes: f16 `d`, u16 `scales_h`, `scales_l[4]`, `qs[128]`; eight
    /// sub-blocks of 32, each with a 6-bit scale (`ls - 32`) times `d`, values from [`IQ4_LEVELS`].
    IQ4XS,
    /// i-quants on the lattice tables in `iq_grids.rs` (all 256 values a super-block).
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ1S,
    IQ1M,
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
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            16 => GgmlType::IQ2XXS,
            17 => GgmlType::IQ2XS,
            18 => GgmlType::IQ3XXS,
            19 => GgmlType::IQ1S,
            20 => GgmlType::IQ4NL,
            21 => GgmlType::IQ3S,
            22 => GgmlType::IQ2S,
            23 => GgmlType::IQ4XS,
            29 => GgmlType::IQ1M,
            other => GgmlType::Other(other),
        }
    }

    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Q4_0 => "Q4_0".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q2K => "Q2_K".into(),
            GgmlType::Q3K => "Q3_K".into(),
            GgmlType::Q4K => "Q4_K".into(),
            GgmlType::Q5K => "Q5_K".into(),
            GgmlType::Q6K => "Q6_K".into(),
            GgmlType::IQ4NL => "IQ4_NL".into(),
            GgmlType::IQ4XS => "IQ4_XS".into(),
            GgmlType::IQ2XXS => "IQ2_XXS".into(),
            GgmlType::IQ2XS => "IQ2_XS".into(),
            GgmlType::IQ2S => "IQ2_S".into(),
            GgmlType::IQ3XXS => "IQ3_XXS".into(),
            GgmlType::IQ3S => "IQ3_S".into(),
            GgmlType::IQ1S => "IQ1_S".into(),
            GgmlType::IQ1M => "IQ1_M".into(),
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
            // K-quants: 256 values a block. d/dmin are f16; the 12-byte `scales` of Q4_K/Q5_K
            // pack eight 6-bit scales and eight 6-bit mins (see `k4_scale_min`).
            GgmlType::Q2K => (256, 84),
            GgmlType::Q3K => (256, 110),
            GgmlType::Q4K => (256, 144),
            GgmlType::Q5K => (256, 176),
            GgmlType::Q6K => (256, 210),
            GgmlType::IQ4NL => (32, 18),
            GgmlType::IQ4XS => (256, 136),
            // d f16 | qs[32] u16
            GgmlType::IQ2XXS => (256, 66),
            // d | qs[32] u16 | scales[8]
            GgmlType::IQ2XS => (256, 74),
            // d | qs[64] (32 index bytes, 32 sign bytes) | qh[8] | scales[8]
            GgmlType::IQ2S => (256, 82),
            // d | qs[96] (64 index bytes, 32 bytes of scale+signs words)
            GgmlType::IQ3XXS => (256, 98),
            // d | qs[64] | qh[8] | signs[32] | scales[4]
            GgmlType::IQ3S => (256, 110),
            // d | qs[32] | qh[8] u16
            GgmlType::IQ1S => (256, 50),
            // qs[32] | qh[16] | scales[8] (the f16 scale is spread over the top nibbles of `scales`)
            GgmlType::IQ1M => (256, 56),
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
            GgmlType::Q2K => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(84)) {
                    dequant_q2_k_block(b, o);
                }
            }
            GgmlType::Q3K => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(110)) {
                    dequant_q3_k_block(b, o);
                }
            }
            GgmlType::Q4K => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(144)) {
                    dequant_q4_k_block(b, o);
                }
            }
            GgmlType::Q5K => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(176)) {
                    dequant_q5_k_block(b, o);
                }
            }
            GgmlType::Q6K => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(210)) {
                    dequant_q6_k_block(b, o);
                }
            }
            GgmlType::IQ4NL => {
                for (o, b) in dst.chunks_exact_mut(32).zip(src.chunks_exact(18)) {
                    dequant_iq4_nl_block(b, o);
                }
            }
            GgmlType::IQ4XS => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(136)) {
                    dequant_iq4_xs_block(b, o);
                }
            }
            t @ (GgmlType::IQ2XXS | GgmlType::IQ2XS | GgmlType::IQ2S | GgmlType::IQ3XXS | GgmlType::IQ3S | GgmlType::IQ1S | GgmlType::IQ1M) => {
                for (o, b) in dst.chunks_exact_mut(256).zip(src.chunks_exact(bb)) {
                    match t {
                        GgmlType::IQ2XXS => dequant_iq2_xxs_block(b, o),
                        GgmlType::IQ2XS => dequant_iq2_xs_block(b, o),
                        GgmlType::IQ2S => dequant_iq2_s_block(b, o),
                        GgmlType::IQ3XXS => dequant_iq3_xxs_block(b, o),
                        GgmlType::IQ3S => dequant_iq3_s_block(b, o),
                        GgmlType::IQ1S => dequant_iq1_s_block(b, o),
                        _ => dequant_iq1_m_block(b, o),
                    }
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
/// Unpack scale `j` and min `j` from the 12-byte 6-bit packing Q4_K and Q5_K share.
///
/// Bytes 0..4 hold the low 6 bits of scales 0..4, bytes 4..8 the low 6 bits of mins 0..4, and
/// bytes 8..12 hold the low nibbles of scales/mins 4..8 with their top 2 bits taken from the
/// high bits of bytes 0..8. This mirrors ggml's `get_scale_min_k4`.
#[inline]
fn k4_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q4_K: `d: f16, dmin: f16, scales[12], qs[128]` = 144 bytes for 256 values.
/// Each 64-value half uses two (scale, min) pairs: low nibbles then high nibbles.
pub fn dequant_q4_k_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    let mut is = 0usize;
    for j in 0..4 {
        let q = &qs[32 * j..32 * (j + 1)];
        let o = &mut out[64 * j..64 * (j + 1)];
        let (sc1, m1) = k4_scale_min(is, scales);
        let (sc2, m2) = k4_scale_min(is + 1, scales);
        let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
        for l in 0..32 {
            o[l] = d1 * (q[l] & 0x0f) as f32 - min1;
            o[l + 32] = d2 * (q[l] >> 4) as f32 - min2;
        }
        is += 2;
    }
}

/// Q5_K: `d: f16, dmin: f16, scales[12], qh[32], qs[128]` = 176 bytes for 256 values.
/// `qh` supplies the 5th bit: one bit per value, two bits of each byte consumed per 64-value half.
pub fn dequant_q5_k_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    let (mut is, mut u1, mut u2) = (0usize, 1u8, 2u8);
    for j in 0..4 {
        let q = &qs[32 * j..32 * (j + 1)];
        let o = &mut out[64 * j..64 * (j + 1)];
        let (sc1, m1) = k4_scale_min(is, scales);
        let (sc2, m2) = k4_scale_min(is + 1, scales);
        let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16u32 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16u32 } else { 0 };
            o[l] = d1 * ((q[l] & 0x0f) as u32 + hi1) as f32 - min1;
            o[l + 32] = d2 * ((q[l] >> 4) as u32 + hi2) as f32 - min2;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Q2_K: `scales[16], qs[64], d: f16, dmin: f16` = 84 bytes for 256 values.
/// Each `scales` byte packs a 4-bit scale (low nibble) and a 4-bit min (high nibble).
/// Two scales are consumed per 2-bit shift, one for each 16-value half of the 32-byte group.
pub fn dequant_q2_k_block(b: &[u8], out: &mut [f32]) {
    let scales = &b[0..16];
    let qs = &b[16..80];
    let d = f16_at(b, 80);
    let dmin = f16_at(b, 82);
    let (mut is, mut y) = (0usize, 0usize);
    for n in 0..2 {
        let q = &qs[32 * n..32 * (n + 1)];
        for j in 0..4 {
            let shift = 2 * j;
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0x0f) as f32, dmin * (sc >> 4) as f32);
                for l in 0..16 {
                    out[y] = dl * ((q[16 * half + l] >> shift) & 3) as f32 - ml;
                    y += 1;
                }
            }
        }
    }
}

/// Q3_K: `hmask[32], qs[64], scales[12], d: f16` = 110 bytes for 256 values.
///
/// `scales` holds sixteen 6-bit values in ggml's interleaved packing: the low nibbles sit in
/// bytes 0..8 and the high 2 bits are distributed through bytes 8..12, unpacked here with the
/// same `kmask` word arithmetic ggml uses. Each unpacked scale is biased by -32. `hmask`
/// carries the third bit of every quant, inverted: a set bit means "do not subtract 4".
pub fn dequant_q3_k_block(b: &[u8], out: &mut [f32]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let hmask = &b[0..32];
    let qs = &b[32..96];
    let sc = &b[96..108];
    let d = f16_at(b, 108);

    let w = |i: usize| u32::from_le_bytes([sc[4 * i], sc[4 * i + 1], sc[4 * i + 2], sc[4 * i + 3]]);
    let (a0, a1, tmp) = (w(0), w(1), w(2));
    let aux = [
        (a0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4),
        (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut scales = [0i32; 16];
    for (i, a) in aux.iter().enumerate() {
        for (j, byte) in a.to_le_bytes().iter().enumerate() {
            scales[4 * i + j] = *byte as i32 - 32;
        }
    }

    let (mut is, mut y, mut m) = (0usize, 0usize, 1u8);
    for n in 0..2 {
        let q = &qs[32 * n..32 * (n + 1)];
        for j in 0..4 {
            let shift = 2 * j;
            for half in 0..2 {
                let dl = d * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let i = 16 * half + l;
                    let hi = if hmask[i] & m != 0 { 0 } else { 4 };
                    out[y] = dl * (((q[i] >> shift) & 3) as i32 - hi) as f32;
                    y += 1;
                }
            }
            m <<= 1;
        }
    }
}

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

#[cfg(test)]
mod kquant_tests {
    use super::*;

    /// Block geometry must match ggml exactly: a wrong byte count makes every tensor after
    /// the first unreadable, so this is the cheapest guard against a typo.
    #[test]
    fn k_quant_block_geometry() {
        for (ty, want) in [
            (GgmlType::Q2K, (256, 84)),
            (GgmlType::Q3K, (256, 110)),
            (GgmlType::Q4K, (256, 144)),
            (GgmlType::Q5K, (256, 176)),
            (GgmlType::Q6K, (256, 210)),
        ] {
            assert_eq!(ty.block(), Some(want), "{}", ty.name());
            // one row of 2048 (the expert FFN width we actually load)
            assert_eq!(ty.row_bytes(2048), Some(2048 / want.0 * want.1));
            assert_eq!(ty.row_bytes(100), None, "{} must reject a partial block", ty.name());
        }
        for id in [10u32, 11, 12, 13, 14] {
            assert!(GgmlType::from_id(id).block().is_some(), "type {id} unknown");
        }
    }

    /// Q4_K/Q5_K share a 12-byte packing of eight 6-bit scales and eight 6-bit mins.
    /// Scales 0..4 and mins 0..4 are plain 6-bit fields; 4..8 take their top 2 bits from
    /// the high bits of the first eight bytes.
    #[test]
    fn k4_scale_min_packing() {
        let mut q = [0u8; 12];
        q[0] = 0b1100_0000 | 5; // scale 0 = 5, and bits 6..8 feed scale 4's high bits
        q[4] = 0b1100_0000 | 9; // min 0 = 9, bits 6..8 feed min 4's high bits
        q[8] = 0x7; // scale 4 low nibble = 7, min 4 low nibble = 0
        assert_eq!(k4_scale_min(0, &q), (5, 9));
        let (s4, m4) = k4_scale_min(4, &q);
        assert_eq!(s4, 0x7 | (0b11 << 4), "scale 4 takes its high bits from q[0] >> 6");
        assert_eq!(m4, 0x0 | (0b11 << 4), "min 4 takes its high bits from q[4] >> 6");
    }

    /// A dequantized block must be finite and use the whole output slice.
    #[test]
    fn k_quant_blocks_fill_and_are_finite() {
        for (ty, len) in [(GgmlType::Q2K, 84), (GgmlType::Q3K, 110), (GgmlType::Q4K, 144), (GgmlType::Q5K, 176)] {
            // a non-trivial bit pattern rather than zeros, so a skipped region shows up
            let src: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11)).collect();
            let mut out = vec![f32::NAN; 256];
            ty.dequant_row(&src, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "{} produced a non-finite value", ty.name());
        }
    }
}

/// The 16 levels of the IQ4 formats (IQ4_NL, IQ4_XS), part of the GGUF format definition: a
/// nibble `n` decodes to `IQ4_LEVELS[n] * scale`. Non-uniform (denser near zero), which is what
/// buys IQ4 its quality over Q4_0 at the same 4 bits.
pub const IQ4_LEVELS: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// IQ4_NL block: f16 `d`, then 16 bytes; byte j's low nibble is value j, high nibble value j + 16.
pub fn dequant_iq4_nl_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for j in 0..16 {
        out[j] = d * IQ4_LEVELS[(b[2 + j] & 15) as usize] as f32;
        out[j + 16] = d * IQ4_LEVELS[(b[2 + j] >> 4) as usize] as f32;
    }
}

/// IQ4_XS super-block (256 values, 136 bytes): `d` f16 @0, `scales_h` u16 @2, `scales_l[4]` @4,
/// `qs[128]` @8. Sub-block `ib` (32 values) has the 6-bit scale
/// `ls = (scales_l[ib/2] >> 4*(ib%2)) & 15 | ((scales_h >> 2*ib) & 3) << 4`, multiplier
/// `d * (ls - 32)`, and its 16 bytes at `qs[16*ib..]` laid out as an IQ4_NL block's.
pub fn dequant_iq4_xs_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let sh = u16::from_le_bytes([b[2], b[3]]) as u32;
    for ib in 0..8 {
        let lo = (b[4 + ib / 2] >> (4 * (ib % 2))) as u32 & 15;
        let ls = lo | (((sh >> (2 * ib)) & 3) << 4);
        let dl = d * (ls as i32 - 32) as f32;
        let q = &b[8 + 16 * ib..8 + 16 * ib + 16];
        let o = &mut out[32 * ib..32 * ib + 32];
        for j in 0..16 {
            o[j] = dl * IQ4_LEVELS[(q[j] & 15) as usize] as f32;
            o[j + 16] = dl * IQ4_LEVELS[(q[j] >> 4) as usize] as f32;
        }
    }
}

// ---- lattice i-quants (tables in `iq_grids.rs`) ----
//
// Every decoder keeps the reference's float operation order (scale first, then times the grid
// value, then the sign), so the results are the same f32 bits as llama.cpp's dequantizer; checked on
// real blocks of our quantised files (`tests/iq_matches_llama.rs`).

use crate::iq_grids::{IQ1S_DELTA, IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

#[inline]
fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

#[inline]
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Byte `j` (0..8) of a 64-bit grid entry, or (0..4) of a 32-bit one: the lattice point's coordinate.
#[inline]
fn gbyte(g: u64, j: usize) -> u8 {
    (g >> (8 * j)) as u8
}

#[inline]
fn sgn(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Eight values `dl * grid[j] * sign(j)` of one 64-bit grid entry.
#[inline]
fn put8(out: &mut [f32], dl: f32, g: u64, signs: u8) {
    for j in 0..8 {
        out[j] = dl * gbyte(g, j) as f32 * sgn(signs, j);
    }
}

/// IQ2_XXS: per 32 values, two u32: four 8-bit grid indices, then 4 x 7 sign bits + a 4-bit scale.
pub fn dequant_iq2_xxs_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let idx = u32_at(b, 2 + 8 * ib);
        let ss = u32_at(b, 2 + 8 * ib + 4);
        let db = d * (0.5 + (ss >> 28) as f32) * 0.25;
        for l in 0..4 {
            let g = IQ2XXS_GRID[(idx >> (8 * l)) as u8 as usize];
            let signs = KSIGNS_IQ2XS[((ss >> (7 * l)) & 127) as usize];
            put8(&mut out[32 * ib + 8 * l..], db, g, signs);
        }
    }
}

/// IQ2_XS: per 8 values, one u16: 9-bit grid index + 7 sign bits; per 32 values, two 4-bit scales.
pub fn dequant_iq2_xs_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let sc = b[66 + ib];
        let db = [d * (0.5 + (sc & 15) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
        for l in 0..4 {
            let q = u16_at(b, 2 + 2 * (4 * ib + l));
            let g = IQ2XS_GRID[(q & 511) as usize];
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            put8(&mut out[32 * ib + 8 * l..], db[l / 2], g, signs);
        }
    }
}

/// IQ2_S: 10-bit grid indices (8 low bits in `qs`, 2 high bits in `qh`), explicit sign bytes.
pub fn dequant_iq2_s_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for ib in 0..8 {
        let sc = scales[ib];
        let db = [d * (0.5 + (sc & 15) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
        for l in 0..4 {
            let hi = ((qh[ib] as usize) << (8 - 2 * l)) & 0x300;
            let g = IQ2S_GRID[qs[4 * ib + l] as usize | hi];
            put8(&mut out[32 * ib + 8 * l..], db[l / 2], g, signs[4 * ib + l]);
        }
    }
}

/// IQ3_XXS: 8-bit indices into a 4-value grid, a u32 of 4 x 7 sign bits + 4-bit scale per 32.
pub fn dequant_iq3_xxs_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let qs = &b[2..66];
    for ib in 0..8 {
        let ss = u32_at(b, 66 + 4 * ib);
        let db = d * (0.5 + (ss >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((ss >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize] as u64;
            let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize] as u64;
            let o = &mut out[32 * ib + 8 * l..];
            for j in 0..4 {
                o[j] = db * gbyte(g1, j) as f32 * sgn(signs, j);
                o[j + 4] = db * gbyte(g2, j) as f32 * sgn(signs, j + 4);
            }
        }
    }
}

/// IQ3_S: 9-bit indices (8 low bits in `qs`, 1 high bit in `qh`) into a 4-value grid, explicit
/// sign bytes, one 4-bit scale per 32 values (`d * (1 + 2 * s)`).
pub fn dequant_iq3_s_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
    for ib in 0..8 {
        let s4 = if ib % 2 == 0 { scales[ib / 2] & 15 } else { scales[ib / 2] >> 4 };
        let db = d * (1 + 2 * s4 as i32) as f32;
        let h = qh[ib] as usize;
        for l in 0..4 {
            let g1 = IQ3S_GRID[qs[8 * ib + 2 * l] as usize | ((h << (8 - 2 * l)) & 256)] as u64;
            let g2 = IQ3S_GRID[qs[8 * ib + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256)] as u64;
            let sg = signs[4 * ib + l];
            let o = &mut out[32 * ib + 8 * l..];
            for j in 0..4 {
                o[j] = db * gbyte(g1, j) as f32 * sgn(sg, j);
                o[j + 4] = db * gbyte(g2, j) as f32 * sgn(sg, j + 4);
            }
        }
    }
}

/// IQ1_S: 11-bit indices (8 bits in `qs`, 3 in `qh`) into a {-1, 0, +1} grid, per 32 values a
/// 3-bit scale and a sign for the +-1/8 offset.
pub fn dequant_iq1_s_block(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let qs = &b[2..34];
    for ib in 0..8 {
        let h = u16_at(b, 34 + 2 * ib) as usize;
        let dl = d * (2 * ((h >> 12) & 7) + 1) as f32;
        let delta = if h & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
        for l in 0..4 {
            let g = IQ1S_GRID[qs[4 * ib + l] as usize | (((h >> (3 * l)) & 7) << 8)];
            for j in 0..8 {
                out[32 * ib + 8 * l + j] = dl * (gbyte(g, j) as i8 as f32 + delta);
            }
        }
    }
}

/// IQ1_M: as IQ1_S's grid, a 3-bit scale per 16 values and an offset sign per 8; the block's f16
/// scale is the top nibbles of the four u16 scale words.
pub fn dequant_iq1_m_block(b: &[u8], out: &mut [f32]) {
    let (qs, qh) = (&b[0..32], &b[32..48]);
    let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = f16::from_bits(bits).to_f32();
    for ib in 0..8 {
        let s = sc[ib / 2] as usize;
        let dl1 = d * (2 * ((s >> (6 * (ib % 2))) & 7) + 1) as f32;
        let dl2 = d * (2 * ((s >> (6 * (ib % 2) + 3)) & 7) + 1) as f32;
        let (h0, h1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
        let idx = [qs[4 * ib] as usize | ((h0 << 8) & 0x700), qs[4 * ib + 1] as usize | ((h0 << 4) & 0x700), qs[4 * ib + 2] as usize | ((h1 << 8) & 0x700), qs[4 * ib + 3] as usize | ((h1 << 4) & 0x700)];
        let dsgn = |m: bool| if m { -IQ1S_DELTA } else { IQ1S_DELTA };
        let delta = [dsgn(h0 & 0x08 != 0), dsgn(h0 & 0x80 != 0), dsgn(h1 & 0x08 != 0), dsgn(h1 & 0x80 != 0)];
        for l in 0..4 {
            let dl = if l < 2 { dl1 } else { dl2 };
            let g = IQ1S_GRID[idx[l]];
            for j in 0..8 {
                out[32 * ib + 8 * l + j] = dl * (gbyte(g, j) as i8 as f32 + delta[l]);
            }
        }
    }
}
