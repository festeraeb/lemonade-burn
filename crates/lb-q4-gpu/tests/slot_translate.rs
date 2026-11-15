//! `slot_translate` (E2 speculative layers) must give every routed expert its table slot, the dummy slot
//! for a non-resident expert, copy the token word, and leave other layers' words alone.
//! Run on the card chosen by LB_CUDA_DEVICE / LB_WGPU_DEVICE.
use lb_q4_gpu::Gpu;

#[test]
fn routes_become_slots_through_the_table() {
    let gpu = Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}"));
    let (n_layer, n_exp, n_slots, k) = (5usize, 64usize, 20usize, 4usize);
    // table: expert e of layer l -> slot (l * 7 + e) % 31 unless (l + e) % 3 == 0 (not resident)
    let table: Vec<u32> = (0..n_layer * n_exp).map(|i| { let (l, e) = (i / n_exp, i % n_exp); if (l + e) % 3 == 0 { u32::MAX } else { ((l * 7 + e) % 31) as u32 } }).collect();
    let th = gpu.upload_u32(table.clone());
    let mut routes = vec![0u32; 2 * n_slots];
    for (i, w) in routes.chunks_exact_mut(2).enumerate() {
        w[0] = ((i * 11 + 3) % n_exp) as u32;
        w[1] = i as u32 + 100;
    }
    let rh = gpu.upload_u32(routes.clone());
    let sentinel = 0xdead_beefu32;
    let sh = gpu.upload_u32(vec![sentinel; 2 * n_slots]);
    for (layer, base) in [(2usize, 4usize), (4, 12), (0, 0)] {
        gpu.slot_translate(&rh, &th, n_layer * n_exp, &sh, (layer * n_exp, k, base, 9, n_slots));
        let got = gpu.read_u32(&sh, 2 * n_slots).expect("read");
        for i in 0..k {
            let e = routes[2 * (base + i)] as usize;
            let t = table[layer * n_exp + e];
            let want = if t == u32::MAX { 9 } else { t };
            assert_eq!(got[2 * (base + i)], want, "layer {layer} slot {i} expert {e}");
            assert_eq!(got[2 * (base + i) + 1], routes[2 * (base + i) + 1], "token word");
        }
    }
    let got = gpu.read_u32(&sh, 2 * n_slots).expect("read");
    // a slot never translated keeps the sentinel
    assert_eq!(got[2 * 8], sentinel);
    assert_eq!(got[2 * 19 + 1], sentinel);
}
