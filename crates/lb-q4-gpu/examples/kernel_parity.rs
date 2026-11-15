//! Backend parity of the engine's unit kernels on identical seeded inputs.
//!
//!   kernel_parity dump OUT.bin   run every kernel on the device named by LB_CUDA_DEVICE /
//!                                LB_WGPU_DEVICE (and on the CPU reference) and write the outputs
//!   kernel_parity cmp A.bin B.bin  compare two dumps section by section (max |diff|, bit-identical count)
//!
//! A dump is a list of sections: name length (u8), name, count (u32 LE), `count` u32 words (f32 bits
//! or route ids). Sections named `cpu_*` are the lb-gguf reference; the rest come from the device.
use lb_gguf::types::{quantize_q4_0, quantize_q8_0};
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert, GpuQ4Mat, GpuQ6KMat, GpuQ8Mat};

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

fn q6k_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(rows * cols / 256 * 210);
    for _ in 0..rows * cols / 256 {
        for _ in 0..192 {
            v.push((rng.next() >> 33) as u8);
        }
        for _ in 0..16 {
            v.push(((rng.next() >> 33) % 33) as u8 as i8 as u8); // scales 0..32
        }
        v.extend_from_slice(&0x2400u16.to_le_bytes()); // d = 0.0625
    }
    v
}

struct Out(Vec<(String, Vec<u32>)>);
impl Out {
    fn f(&mut self, n: &str, v: &[f32]) {
        self.0.push((n.into(), v.iter().map(|x| x.to_bits()).collect()));
    }
    fn u(&mut self, n: &str, v: Vec<u32>) {
        self.0.push((n.into(), v));
    }
}

fn dump(path: &str) -> Result<(), String> {
    let gpu = Gpu::open()?;
    eprintln!("device {} plane {:?} use_plane {} q4_math {:?}", gpu.name, gpu.plane_size, gpu.use_plane, gpu.q4_math_kind());
    let mut rng = Rng(7);
    let mut out = Out(Vec::new());
    let x: Vec<f32> = (0..4 * 2048).map(|_| rng.unit()).collect();
    // Q4_0 matvec, m = 1 and 4, dense-sized and expert-sized
    for (rows, cols, m) in [(6144usize, 2048usize, 1usize), (6144, 2048, 4), (1536, 2048, 1), (2048, 1536, 1)] {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
        let b: Vec<u8> = w.chunks_exact(cols).flat_map(quantize_q4_0).collect();
        let q = QuantMat::new(GgmlType::Q4_0, rows, cols, &b);
        let xi: Vec<f32> = (0..m * cols).map(|i| x[i % x.len()]).collect();
        let g = GpuQ4Mat::upload(&gpu, q)?;
        out.f(&format!("q4_{rows}x{cols}_m{m}"), &g.matvec(&gpu, &xi, m)?);
        let mut cpu = vec![0f32; m * rows];
        matvec_f32(q, &xi, m, &mut cpu);
        out.f(&format!("cpu_q4_{rows}x{cols}_m{m}"), &cpu);
    }
    // Q8_0
    {
        let (rows, cols) = (2048usize, 2048usize);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
        let b: Vec<u8> = w.chunks_exact(cols).flat_map(quantize_q8_0).collect();
        let q = QuantMat::new(GgmlType::Q8_0, rows, cols, &b);
        let g = GpuQ8Mat::upload(&gpu, q)?;
        out.f("q8_2048x2048_m1", &g.matvec(&gpu, &x[..cols], 1)?);
        let mut cpu = vec![0f32; rows];
        matvec_f32(q, &x[..cols], 1, &mut cpu);
        out.f("cpu_q8_2048x2048_m1", &cpu);
    }
    // Q6_K (the head shape is 65536 x 2048)
    {
        let (rows, cols) = (8192usize, 2048usize);
        let b = q6k_bytes(&mut rng, rows, cols);
        let q = QuantMat::new(GgmlType::Q6K, rows, cols, &b);
        let g = GpuQ6KMat::upload(&gpu, q)?;
        out.f("q6k_8192x2048_m1", &g.matvec(&gpu, &x[..cols], 1)?);
        let mut cpu = vec![0f32; rows];
        matvec_f32(q, &x[..cols], 1, &mut cpu);
        out.f("cpu_q6k_8192x2048_m1", &cpu);
    }
    // expert FFN (gate, up, silu, down) m = 1 and 4
    {
        let (e, ff) = (2048usize, 1536usize);
        let mk = |rng: &mut Rng, r: usize, c: usize| -> Vec<u8> { (0..r * c).map(|_| rng.unit()).collect::<Vec<f32>>().chunks_exact(c).flat_map(quantize_q4_0).collect() };
        let (gb, ub, db) = (mk(&mut rng, ff, e), mk(&mut rng, ff, e), mk(&mut rng, e, ff));
        let (g, u, d) = (QuantMat::new(GgmlType::Q4_0, ff, e, &gb), QuantMat::new(GgmlType::Q4_0, ff, e, &ub), QuantMat::new(GgmlType::Q4_0, e, ff, &db));
        let ex = GpuExpert::upload(&gpu, g, u, d)?;
        for m in [1usize, 4] {
            out.f(&format!("expert_ffn_m{m}"), &ex.ffn(&gpu, &x[..m * e], m)?);
            // CPU reference: silu(gate x) * (up x) then down
            let (mut hg, mut hu, mut y) = (vec![0f32; m * ff], vec![0f32; m * ff], vec![0f32; m * e]);
            matvec_f32(g, &x[..m * e], m, &mut hg);
            matvec_f32(u, &x[..m * e], m, &mut hu);
            let h: Vec<f32> = hg.iter().zip(&hu).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
            matvec_f32(d, &h, m, &mut y);
            out.f(&format!("cpu_expert_ffn_m{m}"), &y);
        }
    }
    // add + rms norm, then router top-4 of 64 on the normed row
    {
        let n = 2048usize;
        let xr: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
        let yr: Vec<f32> = (0..n).map(|_| rng.unit() * 0.5).collect();
        let w: Vec<f32> = (0..n).map(|_| 1.0 + rng.unit() * 0.1).collect();
        let eps = 1e-5f32;
        let (xh, yh, wh) = (gpu.upload_f32(&xr), gpu.upload_f32(&yr), gpu.upload_f32(&w));
        let h = gpu.add_rms_norm(&xh, Some(&yh), &wh, n, eps);
        let r = gpu.read_all(vec![(h.clone(), n), (xh.clone(), n)])?;
        out.f("norm_h", &r[0]);
        out.f("norm_x_after_add", &r[1]);
        let s: Vec<f32> = xr.iter().zip(&yr).map(|(a, b)| a + b).collect();
        let ms = s.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        let cpu: Vec<f32> = s.iter().zip(&w).map(|(a, b)| (*a as f64 * inv) as f32 * b).collect();
        out.f("cpu_norm_h", &cpu);
        let (ne, k, nslots) = (64usize, 4usize, 4usize);
        let router: Vec<f32> = (0..ne * n).map(|_| rng.unit() * 0.05).collect();
        let bias: Vec<f32> = (0..ne).map(|_| rng.unit() * 0.1).collect();
        let (rh, bh) = (gpu.upload_f32(&router), gpu.upload_f32(&bias));
        let (routes, weights) = (gpu.upload_u32(vec![0u32; 2 * nslots]), gpu.upload_f32(&vec![0f32; nslots]));
        for seq in [false, true] {
            gpu.router_topk(&h, &rh, &bh, &routes, &weights, (ne, k, n, 0, nslots), seq);
            let ids = gpu.read_u32(&routes, 2 * nslots)?;
            let wv = gpu.read_all(vec![(weights.clone(), nslots)])?;
            out.u(&format!("router_routes_seq{}", seq as u8), ids);
            out.f(&format!("router_weights_seq{}", seq as u8), &wv[0]);
        }
        // CPU: logits = router . h, sigmoid, + bias for selection, normalised weights
        let logit: Vec<f32> = (0..ne).map(|e| (0..n).map(|i| router[e * n + i] * cpu[i]).sum()).collect();
        let prob: Vec<f32> = logit.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|a, b| (prob[*b] + bias[*b]).partial_cmp(&(prob[*a] + bias[*a])).unwrap());
        out.u("cpu_router_top4_ids_sigmoid_bias_guess", idx[..k].iter().map(|i| *i as u32).collect());
    }
    let mut f = Vec::new();
    for (n, v) in &out.0 {
        f.push(n.len() as u8);
        f.extend_from_slice(n.as_bytes());
        f.extend_from_slice(&(v.len() as u32).to_le_bytes());
        for w in v {
            f.extend_from_slice(&w.to_le_bytes());
        }
    }
    std::fs::write(path, f).map_err(|e| e.to_string())?;
    println!("wrote {} sections to {path}", out.0.len());
    Ok(())
}

fn load(path: &str) -> Vec<(String, Vec<u32>)> {
    let b = std::fs::read(path).expect("read dump");
    let (mut i, mut v) = (0usize, Vec::new());
    while i < b.len() {
        let l = b[i] as usize;
        let name = String::from_utf8_lossy(&b[i + 1..i + 1 + l]).to_string();
        i += 1 + l;
        let n = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        v.push((name, b[i..i + 4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()));
        i += 4 * n;
    }
    v
}

/// Each device section against its `cpu_` twin in the same dump.
fn cmp_cpu(a: &str) {
    let a = load(a);
    println!("{:<34} {:>8} {:>10} {:>12} {:>12}", "device vs cpu reference", "n", "bit-equal", "max|diff|", "max|ref|");
    for (n, va) in a.iter().filter(|(n, _)| !n.starts_with("cpu_")) {
        let base = n.replace("_seq0", "").replace("_seq1", "");
        let Some((_, vb)) = a.iter().find(|(m, _)| *m == format!("cpu_{base}")) else { continue };
        if va.len() != vb.len() {
            continue;
        }
        let eq = va.iter().zip(vb).filter(|(x, y)| x == y).count();
        let (mut md, mut mr) = (0f32, 0f32);
        for (x, y) in va.iter().zip(vb) {
            let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
            md = md.max((x - y).abs());
            mr = mr.max(y.abs());
        }
        println!("{:<34} {:>8} {:>9.4}% {:>12.3e} {:>12.3e}", n, va.len(), 100.0 * eq as f64 / va.len() as f64, md, mr);
    }
}

fn cmp(a: &str, b: &str) {
    let (a, b) = (load(a), load(b));
    println!("{:<34} {:>8} {:>10} {:>12} {:>12}", "section", "n", "bit-equal", "max|diff|", "max|ref|");
    for (n, va) in &a {
        if n.starts_with("cpu_") {
            continue;
        }
        let Some((_, vb)) = b.iter().find(|(m, _)| m == n) else { continue };
        let is_ids = n.contains("routes");
        let eq = va.iter().zip(vb).filter(|(x, y)| x == y).count();
        let (mut md, mut mr) = (0f32, 0f32);
        if !is_ids {
            for (x, y) in va.iter().zip(vb) {
                let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                md = md.max((x - y).abs());
                mr = mr.max(y.abs());
            }
        }
        println!("{:<34} {:>8} {:>9.4}% {:>12.3e} {:>12.3e}", n, va.len(), 100.0 * eq as f64 / va.len() as f64, md, mr);
    }
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("dump") => dump(&a[2]),
        Some("cpu") => {
            cmp_cpu(&a[2]);
            Ok(())
        }
        Some("cmp") => {
            cmp(&a[2], &a[3]);
            Ok(())
        }
        _ => Err("usage: kernel_parity dump OUT | cmp A B".into()),
    }
}
