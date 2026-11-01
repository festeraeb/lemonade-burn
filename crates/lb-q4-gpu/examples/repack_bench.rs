// Cost of repack_q4_0 for one whole expert (gate 1536x2048, up 1536x2048, down 2048x1536).
// On-demand upload would pay this on the host for every miss, so it decides the design.
fn main() {
    let mats = [(1536usize, 2048usize), (1536, 2048), (2048, 1536)];
    let mut total_blocks = 0usize;
    let bufs: Vec<Vec<u8>> = mats
        .iter()
        .map(|(r, c)| {
            let nb = r * (c / 32);
            total_blocks += nb;
            (0..nb * 18).map(|i| (i % 251) as u8).collect()
        })
        .collect();
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = std::time::Instant::now();
        for (b, (r, c)) in bufs.iter().zip(mats.iter()) {
            let nb = r * (c / 32);
            let out = lb_q4_gpu::repack_q4_0(b, nb);
            std::hint::black_box(&out);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("{{\"metric\":\"repack_one_expert_ms\",\"value\":{best:.3},\"blocks\":{total_blocks}}}");
}
