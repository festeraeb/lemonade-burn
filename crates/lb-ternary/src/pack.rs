//! Packed ternary layout.
//!
//! A weight matrix `W[rows, cols]` (PyTorch `nn.Linear` layout: `rows = out_features`,
//! `cols = in_features`) is stored row-major. Each row is split into blocks of
//! [`BLOCK`] = 128 weights stored in [`BLOCK_BYTES`] = 32 bytes. Inside a block, byte `j`
//! (0..32) holds four 2-bit codes; bits `2k..2k+1` (k = 0..4) hold the code for weight
//! `block * 128 + k * 32 + j`. The code is `w + 1`, so `{-1, 0, +1} -> {0, 1, 2}` (3 is
//! never produced and decodes to +2; packers must not emit it).
//!
//! With this layout `(byte >> 2k) & 3` over 32 consecutive bytes yields the codes of 32
//! *consecutive* weights, which lines up with 32 consecutive int8 activations, so SIMD
//! decode is a shift + mask (no shuffles). Columns are padded to a multiple of 128 with
//! code 1 (weight 0).
//!
//! Each matrix carries one f32 `scale` = `1 / mean(|W|)` (per tensor, as the model
//! defines it); the dequantized weight is `t / scale`.

pub const BLOCK: usize = 128;
pub const BLOCK_BYTES: usize = BLOCK / 4;

/// Padded column count (multiple of [`BLOCK`]).
pub fn padded_cols(cols: usize) -> usize {
    cols.div_ceil(BLOCK) * BLOCK
}

/// Borrowed view of a packed ternary matrix.
#[derive(Clone, Copy, Debug)]
pub struct TernaryView<'a> {
    pub rows: usize,
    /// Logical (unpadded) number of columns.
    pub cols: usize,
    /// `rows * padded_cols(cols) / 4` bytes.
    pub data: &'a [u8],
    /// `1 / mean(|W|)` in f32 (the model's `WeightQuant` scale).
    pub scale: f32,
}

impl<'a> TernaryView<'a> {
    pub fn new(rows: usize, cols: usize, data: &'a [u8], scale: f32) -> Self {
        assert_eq!(data.len(), rows * row_bytes(cols), "packed data length mismatch");
        Self { rows, cols, data, scale }
    }
    pub fn row_bytes(&self) -> usize {
        row_bytes(self.cols)
    }
    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.data[r * rb..(r + 1) * rb]
    }
    /// Unpack one row to `{-1,0,1}` (logical columns only).
    pub fn unpack_row(&self, r: usize) -> Vec<i8> {
        let mut out = unpack_row(self.row(r), padded_cols(self.cols));
        out.truncate(self.cols);
        out
    }
}

/// Owned packed ternary matrix.
#[derive(Clone, Debug)]
pub struct PackedTernary {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
    pub scale: f32,
}

impl PackedTernary {
    /// Pack an already-ternary row-major matrix (`values` in `{-1,0,1}`).
    pub fn from_ternary(rows: usize, cols: usize, values: &[i8], scale: f32) -> Self {
        assert_eq!(values.len(), rows * cols);
        let rb = row_bytes(cols);
        let mut data = vec![0u8; rows * rb];
        for r in 0..rows {
            pack_row(&values[r * cols..(r + 1) * cols], &mut data[r * rb..(r + 1) * rb]);
        }
        Self { rows, cols, data, scale }
    }
    pub fn view(&self) -> TernaryView<'_> {
        TernaryView { rows: self.rows, cols: self.cols, data: &self.data, scale: self.scale }
    }
}

pub fn row_bytes(cols: usize) -> usize {
    padded_cols(cols) / 4
}

/// Pack one row of ternary values into `out` (`row_bytes(values.len())` bytes).
pub fn pack_row(values: &[i8], out: &mut [u8]) {
    let pc = padded_cols(values.len());
    assert_eq!(out.len(), pc / 4);
    for (b, blk) in out.chunks_exact_mut(BLOCK_BYTES).enumerate() {
        for (j, byte) in blk.iter_mut().enumerate() {
            let mut v = 0u8;
            for k in 0..4 {
                let idx = b * BLOCK + k * 32 + j;
                let w = if idx < values.len() { values[idx] } else { 0 };
                debug_assert!((-1..=1).contains(&w), "non-ternary value {w}");
                v |= ((w + 1) as u8 & 3) << (2 * k);
            }
            *byte = v;
        }
    }
}

/// Unpack a packed row (padded length) back to ternary values.
pub fn unpack_row(row: &[u8], padded: usize) -> Vec<i8> {
    let mut out = vec![0i8; padded];
    for (b, blk) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        for (j, &byte) in blk.iter().enumerate() {
            for k in 0..4 {
                out[b * BLOCK + k * 32 + j] = ((byte >> (2 * k)) & 3) as i8 - 1;
            }
        }
    }
    out
}

/// Absmean ternarization, reproducing `transformers.integrations.bitnet.WeightQuant`:
///
/// ```text
/// w = weight.float()
/// scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
/// t = (w * scale).round().clamp(-1, 1)        # torch.round = round-half-to-even
/// ```
///
/// The mean is accumulated in f64 and rounded once to f32 (torch accumulates f32 with a
/// cascade sum; the two can differ by 1 ulp, which is checked against torch per tensor in
/// the converter's golden test). Returns `(ternary values, scale)`.
pub fn ternarize_absmean(w: &[f32]) -> (Vec<i8>, f32) {
    let mean = absmean_f32(w);
    let scale = 1.0f32 / mean.max(1e-5);
    (ternarize_with_scale(w, scale), scale)
}

pub fn absmean_f32(w: &[f32]) -> f32 {
    let s: f64 = w.iter().map(|&v| (v as f64).abs()).sum();
    (s / w.len() as f64) as f32
}

pub fn ternarize_with_scale(w: &[f32], scale: f32) -> Vec<i8> {
    w.iter().map(|&v| (v * scale).round_ties_even().clamp(-1.0, 1.0) as i8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_with_padding() {
        for cols in [1usize, 31, 128, 129, 300, 512] {
            let rows = 3;
            let vals: Vec<i8> = (0..rows * cols).map(|i| ((i * 7 + i / 3) % 3) as i8 - 1).collect();
            let p = PackedTernary::from_ternary(rows, cols, &vals, 1.0);
            for r in 0..rows {
                assert_eq!(p.view().unpack_row(r), &vals[r * cols..(r + 1) * cols]);
            }
            // padding decodes to zero weights
            let full = unpack_row(p.view().row(0), padded_cols(cols));
            assert!(full[cols..].iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn ternarize_matches_torch_semantics() {
        // mean(|w|) = 1.0 -> scale = 1; 0.5 rounds to 0 (half-even), 1.5 -> 2 -> clamp 1, -0.5 -> -0
        let w = [0.5f32, 1.5, -0.5, -2.0, 0.0, 1.5];
        let (t, s) = ternarize_absmean(&w);
        assert_eq!(s, 1.0);
        assert_eq!(t, vec![0, 1, 0, -1, 0, 1]);
    }
}
