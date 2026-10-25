//! Ternary GEMM on the GPU (CubeCL, wgpu runtime). Same layout and exact integers as
//! `lb_ternary::ternary_gemm_i32(w, x, Kernel::Scalar)`: out[t * rows + r] = sum_k w[r,k] * x_q[t,k].

use cubecl::prelude::*;
use cubecl::wgpu::WgpuDevice;
use lb_ternary::{QuantActs, TernaryView};

/// One unit per output element (token t, row r).
/// `w`: packed rows as little-endian u32 words (kp / 16 words per row).
/// `x`: activations as i32, m rows of kp. `dims` = [rows, kp].
#[cube(launch)]
fn ternary_gemm_kernel(w: &[u32], x: &[i32], dims: &[u32], out: &mut [i32]) {
    let pos = ABSOLUTE_POS;
    if pos < out.len() {
        let rows = dims[0] as usize;
        let kp = dims[1] as usize;
        let t = pos / rows;
        let r = pos % rows;
        let words_per_row = kp / 16;
        let blocks = kp / 128;
        let mut acc = 0i32;
        // FORGE TODO (LB1): loop over the `blocks` blocks of row r. Block b of row r is the 8 words
        // w[r * words_per_row + b * 8 .. r * words_per_row + b * 8 + 8]; byte j (0..32) of the block is
        // bits 8*(j%4) .. 8*(j%4)+7 of word j/4. Bits 2k..2k+1 (k in 0..4) of byte j hold the code of
        // column b*128 + k*32 + j; weight = code - 1. Add weight * x[t * kp + column] to acc.
        out[pos] = acc;
    }
}

pub fn ternary_gemm_i32_gpu(w: TernaryView<'_>, x: &QuantActs) -> Result<Vec<i32>, String> {
    let kp = lb_ternary::pack::padded_cols(w.cols);
    if x.kp != kp {
        return Err(format!("activation row length {} does not match padded weight cols {}", x.kp, kp));
    }
    let n_out = x.m * w.rows;
    if n_out == 0 {
        return Ok(Vec::new());
    }
    let client = std::panic::catch_unwind(|| cubecl::Device::Wgpu(WgpuDevice::default()).client())
        .map_err(|_| "no wgpu adapter available (WgpuRuntime::client panicked)".to_string())?;

    let words: Vec<u32> = w.data[..w.rows * w.row_bytes()]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let acts: Vec<i32> = x.q.iter().map(|&v| v as i32).collect();
    let dims = [w.rows as u32, kp as u32];

    let w_h = client.create_from_slice(u32::as_bytes(&words));
    let x_h = client.create_from_slice(i32::as_bytes(&acts));
    let d_h = client.create_from_slice(u32::as_bytes(&dims));
    let o_h = client.empty(n_out * std::mem::size_of::<i32>());

    let threads = 64usize;
    let cubes = n_out.div_ceil(threads) as u32;
    unsafe {
        ternary_gemm_kernel::launch(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(threads as u32),
            BufferArg::from_raw_parts(w_h, words.len()),
            BufferArg::from_raw_parts(x_h, acts.len()),
            BufferArg::from_raw_parts(d_h, 2),
            BufferArg::from_raw_parts(o_h.clone(), n_out),
        );
    }
    let bytes = client.read_one(o_h).map_err(|e| format!("GPU read failed: {e:?}"))?;
    Ok(i32::from_bytes(&bytes).to_vec())
}
