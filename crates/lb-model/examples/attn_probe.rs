//! Scratch probe: time the decode attention ops on Flex at a given context length.
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Distribution, Tensor};
use std::time::Instant;

fn main() {
    let dev = Device::flex();
    let total: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(520);
    let cap = total.next_power_of_two().max(256);
    let (kvh, group, hd, m) = (5usize, 4usize, 128usize, 1usize);
    let kc = Tensor::<3>::random([kvh, cap, hd], Distribution::Default, &dev);
    let vc = Tensor::<3>::random([kvh, cap, hd], Distribution::Default, &dev);
    let qg = Tensor::<3>::random([kvh, group * m, hd], Distribution::Default, &dev);
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let full = |q: &Tensor<3>| -> Vec<f32> {
        let keys = kc.clone().slice([0..kvh, 0..total, 0..hd]);
        let vals = vc.clone().slice([0..kvh, 0..total, 0..hd]);
        let p = softmax(q.clone().matmul(keys.swap_dims(1, 2)) * 0.088, 2);
        p.matmul(vals).into_data().to_vec::<f32>().unwrap()
    };
    let a = full(&qg);
    let b = one.install(|| full(&qg));
    let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
    let reps = 100;
    let t0 = Instant::now();
    for _ in 0..reps { std::hint::black_box(full(&qg)); }
    let tp = t0.elapsed().as_secs_f64() / reps as f64;
    let t0 = Instant::now();
    for _ in 0..reps { std::hint::black_box(one.install(|| full(&qg))); }
    let t1 = t0.elapsed().as_secs_f64() / reps as f64;
    println!("total {total}: whole attention: global pool {:.1} us, 1-thread pool {:.1} us, bit-identical {same}", tp * 1e6, t1 * 1e6);
    {
        let keys = kc.clone().slice([0..kvh, 0..total, 0..hd]);
        let a = qg.clone().matmul(keys.clone().swap_dims(1, 2)).into_data().to_vec::<f32>().unwrap();
        let bt = keys.clone().matmul(qg.clone().swap_dims(1, 2)).swap_dims(1, 2).into_data().to_vec::<f32>().unwrap();
        let same = a.iter().zip(&bt).all(|(x, y)| x.to_bits() == y.to_bits());
        let maxd = a.iter().zip(&bt).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let t0 = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(keys.clone().matmul(qg.clone().swap_dims(1, 2)).swap_dims(1, 2).into_data());
        }
        println!("total {total}: K@q^T orientation {:.1} us, bit-identical to q@K^T: {same} (max diff {maxd:e})", t0.elapsed().as_secs_f64() / 100.0 * 1e6);
    }
    let n = 200;
    let mut t = [0f64; 6];
    for _ in 0..n {
        let t0 = Instant::now();
        let keys = kc.clone().slice([0..kvh, 0..total, 0..hd]);
        let vals = vc.clone().slice([0..kvh, 0..total, 0..hd]);
        let kt = keys.swap_dims(1, 2);
        let t1 = Instant::now();
        let scores = qg.clone().matmul(kt) * 0.088;
        let _ = scores.clone().into_data();
        let t2 = Instant::now();
        let probs = softmax(scores, 2);
        let _ = probs.clone().into_data();
        let t3 = Instant::now();
        let attn = probs.matmul(vals);
        let _ = attn.clone().into_data();
        let t4 = Instant::now();
        let kc2 = kc.clone().slice([0..kvh, 0..total, 0..hd]).swap_dims(1, 2);
        let kcont = Tensor::<3>::from_data(kc2.into_data(), &dev);
        let t5 = Instant::now();
        let s2 = qg.clone().matmul(kcont);
        let _ = s2.into_data();
        let t6 = Instant::now();
        for (i, d) in [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5].iter().enumerate() {
            t[i] += d.as_secs_f64();
        }
    }
    let names = ["slice+swap", "q@kT", "softmax", "p@v", "make kT contiguous", "q@kT(contig)"];
    for (i, nm) in names.iter().enumerate() {
        println!("total {total}: {nm:<20} {:8.1} us", t[i] / n as f64 * 1e6);
    }
}
