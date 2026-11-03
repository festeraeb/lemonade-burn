//! The device repack (`q4_0_repack_kernel`, the zero-copy miss path) must write exactly the
//! words the host repack writes, on whichever runtime the env selects.
mod common;
use common::Rng;

#[test]
fn device_repack_matches_host() {
    let gpu = match lb_q4_gpu::Gpu::open() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    // even counts only (the device path's precondition): tiny, one cube, many cubes, a real matrix
    for &n in &[2usize, 64, 510, 4096, 1536 * 64] {
        let bytes: Vec<u8> = (0..n * 18).map(|_| rng.byte()).collect();
        let (hq, hs) = lb_q4_gpu::repack_q4_0(&bytes, n);
        let ((qs, n_qs), (sc, n_sc)) = gpu.upload_q4_0_raw(&bytes, n);
        let dq = gpu.read_u32(&qs, n_qs).expect("read qs");
        let ds = gpu.read_u32(&sc, n_sc).expect("read sc");
        assert_eq!(dq, hq, "qs differ at n_blocks={n}");
        assert_eq!(ds, hs, "sc differ at n_blocks={n}");
        eprintln!("n_blocks={n}: device repack == host repack ({n_qs} qs words, {n_sc} sc words)");
    }
}
