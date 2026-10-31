//! Q4_0 matvec on the GPU (CubeCL, wgpu runtime): `out[t, r] = sum_c W[r, c] * x[t, c]`
//! with `W` in GGUF Q4_0 blocks (32 values: f16 scale `d`, 16 nibble bytes; low nibble =
//! value j, high nibble = value j + 16, each minus 8). Same numbers as
//! `lb_gguf::matvec_f32` up to f32 summation order.
//!
//! Layout on the device: the row bytes are uploaded unchanged, viewed as little-endian
//! u32 words (a row of `cols` values is `cols / 32 * 18` bytes; 18 is not a multiple of
//! 4, so blocks are addressed at byte granularity through word/shift arithmetic; the
//! whole matrix is padded to a multiple of 4 bytes). The f16 scale is decoded in the
//! kernel from its 16 bits (sign, 5-bit exponent, 10-bit mantissa; subnormals included).

use cubecl::prelude::*;
use lb_gguf::QuantMat;
use lb_ternary_gpu::pick_device;

/// Little-endian byte `i` of the byte stream held in `w` (u32 words).
#[cube]
fn byte_at(w: &[u32], i: usize) -> u32 {
    (w[i / 4] >> (((i % 4) * 8) as u32)) & 255u32
}

/// f16 bits -> f32, exactly: normal numbers are rebuilt bit for bit (exponent rebased
/// 15 -> 127, mantissa widened 10 -> 23 bits); subnormals are `man * 2^-24`.
#[cube]
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u32) & 1u32;
    let exp = (bits >> 10u32) & 31u32;
    let man = bits & 1023u32;
    let mut v = 0.0f32;
    if exp == 0u32 {
        v = man as f32 * 5.9604645e-8f32;
        if sign == 1u32 {
            v = -v;
        }
    } else {
        let b = (sign << 31u32) | ((exp + 112u32) << 23u32) | (man << 13u32);
        v = f32::reinterpret(b);
    }
    v
}

/// One unit per output element (token t, row r). `dims` = [rows, cols, row_bytes].
#[cube(launch)]
fn q4_0_matvec_kernel(w: &[u32], x: &[f32], dims: &[u32], out: &mut [f32]) {
    let pos = ABSOLUTE_POS;
    if pos < out.len() {
        let rows = dims[0] as usize;
        let cols = dims[1] as usize;
        let row_bytes = dims[2] as usize;
        let t = pos / rows;
        let r = pos % rows;
        let blocks = cols / 32;
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let base = r * row_bytes + b * 18;
            let d = f16_to_f32(byte_at(w, base) | (byte_at(w, base + 1) << 8u32));
            let mut s = 0.0f32;
            for j in 0..16usize {
                let byte = byte_at(w, base + 2 + j);
                let lo = (byte & 15u32) as i32 - 8;
                let hi = (byte >> 4u32) as i32 - 8;
                s += lo as f32 * x[t * cols + b * 32 + j];
                s += hi as f32 * x[t * cols + b * 32 + j + 16];
            }
            acc += d * s;
        }
        out[pos] = acc;
    }
}

/// `x`: `m` rows of `w.cols` f32. Returns `m * w.rows` values, token-major like the CPU kernel.
pub fn q4_0_matvec_gpu(w: QuantMat<'_>, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    if w.ty != lb_gguf::GgmlType::Q4_0 {
        return Err(format!("expected Q4_0, got {}", w.ty.name()));
    }
    if x.len() != m * w.cols {
        return Err(format!("x has {} values, expected {} x {}", x.len(), m, w.cols));
    }
    let n_out = m * w.rows;
    if n_out == 0 {
        return Ok(Vec::new());
    }
    let device = pick_device()?;
    let client = std::panic::catch_unwind(|| cubecl::Device::Wgpu(device.clone()).client())
        .map_err(|_| "no wgpu adapter available (WgpuRuntime::client panicked)".to_string())?;

    let mut bytes = w.bytes.to_vec();
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let dims = [w.rows as u32, w.cols as u32, w.row_bytes() as u32];

    let w_h = client.create_from_slice(u32::as_bytes(&words));
    let x_h = client.create_from_slice(f32::as_bytes(x));
    let d_h = client.create_from_slice(u32::as_bytes(&dims));
    let o_h = client.empty(n_out * std::mem::size_of::<f32>());

    let threads = 64usize;
    let cubes = n_out.div_ceil(threads) as u32;
    unsafe {
        q4_0_matvec_kernel::launch(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(threads as u32),
            BufferArg::from_raw_parts(w_h, words.len()),
            BufferArg::from_raw_parts(x_h, x.len()),
            BufferArg::from_raw_parts(d_h, 3),
            BufferArg::from_raw_parts(o_h.clone(), n_out),
        );
    }
    let out = client.read_one(o_h).map_err(|e| format!("GPU read failed: {e:?}"))?;
    Ok(f32::from_bytes(&out).to_vec())
}
