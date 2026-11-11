//! X7b: transcode on residency. A matrix whose storage type the device kernels do not read
//! (IQ4_XS, IQ4_NL, Q4_K, Q5_K, Q8_0, ...: anything `lb_gguf` can dequantize) is re-encoded to
//! Q4_0, the format the per-card math variants run on, **when it becomes resident**. Weights stay
//! compact in RAM, on disk and on the wire; only the resident copy changes format.
//!
//! This is the host form: dequantize a row to f32, re-quantize it to Q4_0 blocks, then the
//! normal upload (repack + DMA). The device form (raw compact bytes DMA'd and re-encoded by a
//! kernel, like `q4_0_repack_kernel` does for Q4_0) is not built: see docs/WORK_TO_WEIGHTS.md X7.
//!
//! Re-quantizing is lossy (a 4-bit grid onto a different 4-bit grid). The Q4_0 encoder here
//! searches the block scale ([`quantize_q4_0_search`]) instead of taking `max / -8`, which is
//! what keeps the compounded error small; `LB_TRANSCODE_SIMPLE=1` takes the plain encoder for
//! the A/B. `LB_TRANSCODE=0` refuses non-Q4_0 uploads (the old behaviour).

use half::f16;
use lb_gguf::{GgmlType, QuantMat};
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

/// Nanoseconds spent transcoding and source bytes transcoded, process-wide (for reports).
pub static TRANSCODE_NS: AtomicU64 = AtomicU64::new(0);
pub static TRANSCODE_BYTES: AtomicU64 = AtomicU64::new(0);

fn enabled() -> bool {
    std::env::var("LB_TRANSCODE").map(|v| v != "0").unwrap_or(true)
}

/// Encode one block of 32 values as Q4_0 (18 bytes into `out`) with the scale that minimises the
/// block's squared error over a small search around `max / -8`. Q4_0 decodes `d * (q - 8)`,
/// q in 0..15, so the largest-magnitude value sets the sign of `d`; the search scales `|d|`
/// down by up to ~20 % (clipping the extreme value a little is often cheaper than coarsening all
/// the others) and keeps the best. Ties keep the earlier (plain) candidate, so a block the plain
/// encoder already gets exactly right is unchanged.
pub fn quantize_q4_0_block_search(blk: &[f32], out: &mut [u8]) {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in blk {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let base = max / -8.0;
    let mut best: (f32, f32) = (f32::INFINITY, 0.0);
    for k in 0..12 {
        // the stored scale is f16: evaluate the error with the scale as it will decode
        let d = f16::from_f32(base * (1.0 - 0.018 * k as f32)).to_f32();
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut err = 0f32;
        for &v in blk {
            let q = ((v * id + 8.5).floor() as i32).clamp(0, 15);
            let e = v - d * (q - 8) as f32;
            err += e * e;
        }
        if err < best.0 {
            best = (err, d);
        }
    }
    let d = best.1;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    for j in 0..16 {
        let q0 = ((blk[j] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
        let q1 = ((blk[j + 16] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
        out[2 + j] = q0 | (q1 << 4);
    }
}

/// `x` (a multiple of 32 values) as Q4_0 bytes, scale-searched per block.
pub fn quantize_q4_0_search(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % 32, 0);
    let mut out = vec![0u8; x.len() / 32 * 18];
    for (blk, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(18)) {
        quantize_q4_0_block_search(blk, o);
    }
    out
}

/// Whether `ty` can be made resident through a transcode (dequantizable, not already Q4_0).
pub fn transcodable(ty: GgmlType) -> bool {
    ty != GgmlType::Q4_0 && ty.block().is_some() && enabled()
}

/// The matrix's bytes as Q4_0: borrowed when it already is, else transcoded (rows in parallel).
pub fn q4_0_bytes<'a>(m: &QuantMat<'a>) -> Result<Cow<'a, [u8]>, String> {
    if m.ty == GgmlType::Q4_0 {
        return Ok(Cow::Borrowed(m.bytes));
    }
    if !transcodable(m.ty) {
        return Err(format!("needs Q4_0, got {} (LB_TRANSCODE=0 or not dequantizable)", m.ty.name()));
    }
    if m.cols % 32 != 0 {
        return Err(format!("transcode to Q4_0 needs cols % 32 == 0, got {}", m.cols));
    }
    use rayon::prelude::*;
    let t = std::time::Instant::now();
    let simple = std::env::var_os("LB_TRANSCODE_SIMPLE").is_some();
    let rb = m.cols / 32 * 18;
    let mut out = vec![0u8; m.rows * rb];
    out.par_chunks_mut(rb).enumerate().for_each_init(
        || vec![0f32; m.cols],
        |buf, (r, o)| {
            m.dequant_row(r, buf);
            if simple {
                o.copy_from_slice(&lb_gguf::types::quantize_q4_0(buf));
            } else {
                for (blk, ob) in buf.chunks_exact(32).zip(o.chunks_exact_mut(18)) {
                    quantize_q4_0_block_search(blk, ob);
                }
            }
        },
    );
    TRANSCODE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    TRANSCODE_BYTES.fetch_add(m.bytes.len() as u64, Ordering::Relaxed);
    Ok(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn sse(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
    }

    #[test]
    fn search_never_worse_than_plain_and_q4_0_passes_through() {
        let x = rng(32 * 4096, 7);
        let plain = lb_gguf::types::quantize_q4_0(&x);
        let srch = quantize_q4_0_search(&x);
        let deq = |b: &[u8]| {
            let mut o = vec![0f32; x.len()];
            GgmlType::Q4_0.dequant_row(b, &mut o);
            o
        };
        let (ep, es) = (sse(&deq(&plain), &x), sse(&deq(&srch), &x));
        assert!(es <= ep, "search {es} > plain {ep}");
        eprintln!("q4_0 encode SSE: plain {ep:.4}, search {es:.4} ({:.1} % lower)", 100.0 * (1.0 - es / ep));
        let m = QuantMat::new(GgmlType::Q4_0, 64, 2048, &plain[..64 * 64 * 18]);
        assert!(matches!(q4_0_bytes(&m).unwrap(), Cow::Borrowed(_)));
    }

    #[test]
    fn iq4_xs_transcodes_close_to_its_values() {
        // a synthetic IQ4_XS super-block row: d = 0.01, all sub-block scales 40 (ls - 32 = 8)
        let cols = 512usize;
        let mut row = Vec::new();
        for sb in 0..cols / 256 {
            row.extend_from_slice(&f16::from_f32(0.01).to_le_bytes());
            // ls = 40 = 0b10_1000: low nibble 8, high 2 bits 2
            row.extend_from_slice(&(0xAAAAu16).to_le_bytes());
            row.extend_from_slice(&[0x88u8; 4]);
            for i in 0..128 {
                row.push(((i * 7 + sb * 3) % 256) as u8);
            }
        }
        let m = QuantMat::new(GgmlType::IQ4XS, 1, cols, &row);
        let mut want = vec![0f32; cols];
        m.dequant_row(0, &mut want);
        let q = q4_0_bytes(&m).unwrap();
        let mut got = vec![0f32; cols];
        GgmlType::Q4_0.dequant_row(&q, &mut got);
        let rel = (sse(&got, &want) / want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
        eprintln!("IQ4_XS -> Q4_0 relative RMS error {rel:.4}");
        assert!(rel < 0.12, "relative RMS error {rel}");
    }
}
