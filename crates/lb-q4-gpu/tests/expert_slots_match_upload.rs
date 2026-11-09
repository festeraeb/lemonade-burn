//! The cache's slot bank must compute exactly what a plain upload of the same expert computes,
//! whichever slot it was filled into and whether it runs through the per-slot views (the host-routed
//! prefill path) or through device routes holding slot numbers (engine step E2's decode path).
//! Bit-identical: the same words reach the same kernels, only the base offset differs.
//! Run on the card chosen by LB_CUDA_DEVICE / LB_WGPU_DEVICE.
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{ExpertSlots, Gpu, GpuExpert};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn q4(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

#[test]
fn slots_compute_what_an_upload_computes_in_any_slot_and_after_a_refill() {
    let gpu = Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}"));
    // 256 x 512: 4096 blocks a matrix, so a slot's views are 256-byte aligned (as LFM2.5's are)
    let (n_ff, n_embd) = (256usize, 512usize);
    let mut rng = Rng(0x5eed_e2e2);
    let raw: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..5).map(|_| (q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_embd, n_ff))).collect();
    let x: Vec<f32> = (0..n_embd).map(|_| rng.unit() * 3.0).collect();
    let xh = gpu.upload_f32(&x);
    let plain: Vec<Vec<f32>> = raw
        .iter()
        .map(|(g, u, d)| {
            let ex = GpuExpert::upload_per_matrix(
                &gpu,
                QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, g),
                QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, u),
                QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, d),
            )
            .expect("upload");
            ex.ffn(&gpu, &x, 1).expect("plain expert")
        })
        .collect();

    let nb = n_ff * n_embd / 32;
    let slots = ExpertSlots::new(&gpu, 6, [(n_ff, n_embd), (n_ff, n_embd), (n_embd, n_ff)], 2).expect("bank");
    assert_eq!(slots.bytes, 6 * ExpertSlots::slot_bytes(nb) + ExpertSlots::staging_bytes(nb, 2));
    let fill = |e: usize, s: usize| slots.fill(&gpu, s, [&raw[e].0, &raw[e].1, &raw[e].2]).expect("fill");

    // experts into scattered slots, more fills than staging buffers
    let map = [(0usize, 3usize), (1, 0), (2, 5), (3, 1)];
    for &(e, s) in &map {
        fill(e, s);
    }
    for &(e, s) in &map {
        let got = slots.expert(s).ffn(&gpu, &x, 1).expect("slot view");
        assert_eq!(got, plain[e], "view of slot {s} differs from expert {e}");
    }

    // device routes holding slot numbers, at a non-zero base like a later MoE layer's
    let (base, k, n_route) = (4usize, 4usize, 12usize);
    let mut words = vec![0u32; 2 * n_route];
    for (i, &(_, s)) in map.iter().enumerate() {
        words[2 * (base + i)] = s as u32;
    }
    let routes = gpu.upload_u32(words);
    let (out, n) = slots.bank().launch_device(&gpu, &xh, 1, &routes, n_route, base, k);
    assert_eq!(n, k * n_embd);
    let got = gpu.read_all(vec![(out, n)]).expect("read")[0].clone();
    for (i, &(e, s)) in map.iter().enumerate() {
        assert_eq!(&got[i * n_embd..(i + 1) * n_embd], &plain[e][..], "routed slot {s} differs from expert {e}");
    }

    // a refill replaces the slot's contents completely
    fill(4, 5);
    assert_eq!(slots.expert(5).ffn(&gpu, &x, 1).expect("refilled"), plain[4]);
    assert_eq!(slots.expert(3).ffn(&gpu, &x, 1).expect("neighbour"), plain[0], "a refill must not touch other slots");
}
