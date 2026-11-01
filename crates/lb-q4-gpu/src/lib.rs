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

pub mod store;
pub use store::{Gpu, GpuExpert, GpuQ4Mat};
pub use cubecl::server as cubecl_server;

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

/// Partial dot of one Q4_0 row with `x[x_off..]`: this unit sums the blocks `b` with
/// `b % CUBE_DIM_X == UNIT_POS` (`n_iter` = ceil(blocks / CUBE_DIM_X), passed in).
#[cube]
fn q4_row_partial(w: &[u32], base: usize, x: &[f32], x_off: usize, blocks: usize, n_iter: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n_iter {
        let b = UNIT_POS as usize + i * CUBE_DIM_X as usize;
        if b < blocks {
            let bb = base + b * 18;
            let d = f16_to_f32(byte_at(w, bb) | (byte_at(w, bb + 1) << 8u32));
            let mut s = 0.0f32;
            for j in 0..16usize {
                let byte = byte_at(w, bb + 2 + j);
                let lo = (byte & 15u32) as i32 - 8;
                let hi = (byte >> 4u32) as i32 - 8;
                s += lo as f32 * x[x_off + b * 32 + j];
                s += hi as f32 * x[x_off + b * 32 + j + 16];
            }
            acc += d * s;
        }
    }
    acc
}

/// Sum the 64 partials of this cube (CUBE_DIM_X must be 64); valid in unit 0 after return.
#[cube]
fn reduce64(sm: &mut Shared<[f32]>, v: f32) -> f32 {
    sm[UNIT_POS as usize] = v;
    sync_cube();
    if UNIT_POS < 32u32 {
        sm[UNIT_POS as usize] += sm[UNIT_POS as usize + 32];
    }
    sync_cube();
    if UNIT_POS < 16u32 {
        sm[UNIT_POS as usize] += sm[UNIT_POS as usize + 16];
    }
    sync_cube();
    if UNIT_POS < 8u32 {
        sm[UNIT_POS as usize] += sm[UNIT_POS as usize + 8];
    }
    sync_cube();
    if UNIT_POS < 4u32 {
        sm[UNIT_POS as usize] += sm[UNIT_POS as usize + 4];
    }
    sync_cube();
    if UNIT_POS < 2u32 {
        sm[UNIT_POS as usize] += sm[UNIT_POS as usize + 2];
    }
    sync_cube();
    let r = sm[0] + sm[1];
    sync_cube();
    r
}

/// One cube of 64 units per output element (token t, row r): the units split the row's
/// blocks and reduce through shared memory. `dims` = [rows, cols, row_bytes, n_iter].
#[cube(launch)]
fn q4_0_matvec_kernel(w: &[u32], x: &[f32], dims: &[u32], out: &mut [f32]) {
    let pos = CUBE_POS as usize;
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let row_bytes = dims[2] as usize;
    let n_iter = dims[3] as usize;
    let mut sm = Shared::<[f32]>::new_slice(64usize);
    let mut part = 0.0f32;
    if pos < out.len() {
        let t = pos / rows;
        let r = pos % rows;
        part = q4_row_partial(w, r * row_bytes, x, t * cols, cols / 32, n_iter);
    }
    let total = reduce64(&mut sm, part);
    if pos < out.len() && UNIT_POS == 0u32 {
        out[pos] = total;
    }
}

/// Cube size used by every kernel here (the reduction assumes it).
pub const THREADS: usize = 64;

/// Cube count for `n` outputs: wgpu caps each dispatch dimension at 65535, so the cubes
/// are laid out in two dimensions and the kernel reads the linear `CUBE_POS`.
pub fn cube_count(n: usize) -> CubeCount {
    let x = n.clamp(1, 65535);
    let y = n.div_ceil(x).max(1);
    CubeCount::Static(x as u32, y as u32, 1)
}

/// `dims` for [`q4_0_matvec_kernel`].
pub fn matvec_dims(rows: usize, cols: usize, row_bytes: usize) -> [u32; 4] {
    [rows as u32, cols as u32, row_bytes as u32, (cols / 32).div_ceil(THREADS) as u32]
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
    let dims = matvec_dims(w.rows, w.cols, w.row_bytes());

    let w_h = client.create_from_slice(u32::as_bytes(&words));
    let x_h = client.create_from_slice(f32::as_bytes(x));
    let d_h = client.create_from_slice(u32::as_bytes(&dims));
    let o_h = client.empty(n_out * std::mem::size_of::<f32>());

    unsafe {
        q4_0_matvec_kernel::launch(
            &client,
            cube_count(n_out),
            CubeDim::new_1d(THREADS as u32),
            BufferArg::from_raw_parts(w_h, words.len()),
            BufferArg::from_raw_parts(x_h, x.len()),
            BufferArg::from_raw_parts(d_h, 4),
            BufferArg::from_raw_parts(o_h.clone(), n_out),
        );
    }
    let out = client.read_one(o_h).map_err(|e| format!("GPU read failed: {e:?}"))?;
    Ok(f32::from_bytes(&out).to_vec())
}

// ---- Seeds for forge lanes (bodies to be written; tests in tests/q6k_matches_scalar.rs and
// tests/q8_0_matches_scalar.rs are `#[ignore]` until then: run with `--include-ignored`
// on a GPU host, never on t440). ----

/// Q8_0 matvec on the GPU: blocks of 32 values, f16 scale `d` then 32 signed bytes
/// (`value = d * q`), row bytes `cols / 32 * 34`. Same contract as [`q4_0_matvec_gpu`].
pub fn q8_0_matvec_gpu(w: QuantMat<'_>, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    if w.ty != lb_gguf::GgmlType::Q8_0 {
        return Err(format!("expected Q8_0, got {}", w.ty.name()));
    }
    let _ = (x, m);
    Err("q8_0_matvec_gpu: not implemented yet (forge seed)".to_string())
}

/// Q6_K matvec on the GPU: super-blocks of 256 values, 210 bytes (`ql[128]`, `qh[64]`,
/// `scales[16]` i8, `d` f16); see `lb_gguf::types::dequant_q6_k_block` for the exact
/// bit layout. Row bytes `cols / 256 * 210`. Same contract as [`q4_0_matvec_gpu`].
pub fn q6_k_matvec_gpu(w: QuantMat<'_>, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    if w.ty != lb_gguf::GgmlType::Q6K {
        return Err(format!("expected Q6_K, got {}", w.ty.name()));
    }
    let _ = (x, m);
    Err("q6_k_matvec_gpu: not implemented yet (forge seed)".to_string())
}
