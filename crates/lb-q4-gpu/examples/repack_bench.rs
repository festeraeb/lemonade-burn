// Cost of the Q4_0 host repack for one whole expert (gate 1536x2048, up 1536x2048, down
// 2048x1536): what an on-demand miss pays on the host before the copy, and what every
// resident pays at load. Two forms: `repack_q4_0_into` into reused buffers (the pinned-pool
// path the engine takes on a miss) and `repack_q4_0` with fresh `Vec`s (the pageable path).
// Medians of `LB_BENCH_REPS` reps after `LB_BENCH_WARM`; thread count from rayon
// (`RAYON_NUM_THREADS`); `LB_REPACK_SCALAR=1` forces the scalar inner loop. The FNV hash of
// the output is printed so two builds can be checked to agree on identical input.
fn fnv(words: &[u32]) -> u64 {
    words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &w| (h ^ w as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let reps: usize = std::env::var("LB_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let warm: usize = std::env::var("LB_BENCH_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let mats = [(1536usize, 2048usize), (1536, 2048), (2048, 1536)];
    let mut total_blocks = 0usize;
    let bufs: Vec<Vec<u8>> = mats
        .iter()
        .map(|(r, c)| {
            let nb = r * (c / 32);
            total_blocks += nb;
            let mut x = 0x9E37_79B9u32 ^ nb as u32;
            (0..nb * 18)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    (x >> 24) as u8
                })
                .collect()
        })
        .collect();
    let blocks: Vec<usize> = mats.iter().map(|(r, c)| r * (c / 32)).collect();
    let mut qs: Vec<Vec<u32>> = blocks.iter().map(|nb| vec![0u32; nb * 4]).collect();
    let mut sc: Vec<Vec<u32>> = blocks.iter().map(|nb| vec![0u32; nb.div_ceil(2)]).collect();
    let threads = rayon::current_num_threads();
    let scalar = std::env::var_os("LB_REPACK_SCALAR").is_some();

    // the miss path: into reused buffers, per expert and for the first matrix alone
    let mut per_expert = Vec::with_capacity(reps);
    let mut per_mat = Vec::with_capacity(reps);
    for i in 0..warm + reps {
        let t = std::time::Instant::now();
        for m in 0..3 {
            lb_q4_gpu::repack_q4_0_into(&bufs[m], blocks[m], &mut qs[m], &mut sc[m]);
        }
        let dt = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        lb_q4_gpu::repack_q4_0_into(&bufs[0], blocks[0], &mut qs[0], &mut sc[0]);
        let dm = t.elapsed().as_secs_f64() * 1e3;
        if i >= warm {
            per_expert.push(dt);
            per_mat.push(dm);
        }
    }
    let hash = fnv(&qs.concat()) ^ fnv(&sc.concat()).rotate_left(17);
    // the pageable path: fresh Vecs each call
    let mut alloc_form = Vec::with_capacity(reps);
    for i in 0..warm + reps {
        let t = std::time::Instant::now();
        for m in 0..3 {
            let out = lb_q4_gpu::repack_q4_0(&bufs[m], blocks[m]);
            std::hint::black_box(&out);
        }
        if i >= warm {
            alloc_form.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let (lo, hi) = (per_expert.iter().cloned().fold(f64::MAX, f64::min), per_expert.iter().cloned().fold(0.0, f64::max));
    println!(
        "{{\"metric\":\"repack_into_one_expert_ms\",\"median\":{:.3},\"min\":{lo:.3},\"max\":{hi:.3},\"one_matrix_ms\":{:.3},\"alloc_form_ms\":{:.3},\"threads\":{threads},\"scalar\":{scalar},\"reps\":{reps},\"blocks\":{total_blocks},\"out_fnv\":\"{hash:016x}\"}}",
        median(&mut per_expert),
        median(&mut per_mat),
        median(&mut alloc_form)
    );
}
