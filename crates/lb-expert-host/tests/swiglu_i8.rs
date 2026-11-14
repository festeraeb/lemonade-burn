//! `swiglu_i8`: the int8-activation expert must stay close to the f32 reference on Q4_0
//! weights and be the f32 reference itself for any other weight type.
use lb_expert_host::expert::{swiglu, swiglu_f32x, swiglu_i8};
use lb_gguf::types::{quantize_q4_0, quantize_q8_0};
use lb_gguf::{GgmlType, QuantMat};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn mats(ty: GgmlType, n_ff: usize, n_embd: usize, s: &mut u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let q = |rows: usize, cols: usize, s: &mut u64| -> Vec<u8> {
        (0..rows)
            .flat_map(|_| {
                let r: Vec<f32> = (0..cols).map(|_| lcg(s) * 0.05).collect();
                if ty == GgmlType::Q4_0 { quantize_q4_0(&r) } else { quantize_q8_0(&r) }
            })
            .collect()
    };
    (q(n_ff, n_embd, s), q(n_ff, n_embd, s), q(n_embd, n_ff, s))
}

#[test]
fn q4_0_expert_is_close_to_the_f32_reference() {
    let (n_ff, n_embd) = (1024usize, 512usize);
    let mut s = 31u64;
    let (g, u, d) = mats(GgmlType::Q4_0, n_ff, n_embd, &mut s);
    let (gate, up, down) = (
        QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &g),
        QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &u),
        QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &d),
    );
    for m in [1usize, 3] {
        let x: Vec<f32> = (0..m * n_embd).map(|_| lcg(&mut s) * 4.0).collect();
        let want = swiglu(gate, up, down, &x, m);
        let got = swiglu_i8(gate, up, down, &x, m);
        assert_eq!(got.len(), want.len());
        let err: f64 = want.iter().zip(&got).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        assert!(norm > 0.0);
        assert!(err / norm < 0.03, "m {m}: relative error {}", err / norm);
        assert!(err > 0.0, "m {m}: identical to the f32 path, so the int8 kernel did not run");
    }
}

#[test]
fn other_weight_types_take_the_f32_reference() {
    let (n_ff, n_embd) = (256usize, 128usize);
    let mut s = 77u64;
    let (g, u, d) = mats(GgmlType::Q8_0, n_ff, n_embd, &mut s);
    let (gate, up, down) = (
        QuantMat::new(GgmlType::Q8_0, n_ff, n_embd, &g),
        QuantMat::new(GgmlType::Q8_0, n_ff, n_embd, &u),
        QuantMat::new(GgmlType::Q8_0, n_embd, n_ff, &d),
    );
    let x: Vec<f32> = (0..2 * n_embd).map(|_| lcg(&mut s)).collect();
    assert_eq!(swiglu_i8(gate, up, down, &x, 2), swiglu(gate, up, down, &x, 2));
}

#[test]
fn f32x_expert_is_the_f32_reference_to_rounding() {
    let (n_ff, n_embd) = (1024usize, 512usize);
    let mut s = 91u64;
    let (g, u, d) = mats(GgmlType::Q4_0, n_ff, n_embd, &mut s);
    let (gate, up, down) = (
        QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &g),
        QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &u),
        QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &d),
    );
    for m in [1usize, 3] {
        let x: Vec<f32> = (0..m * n_embd).map(|_| lcg(&mut s) * 4.0).collect();
        let want = swiglu(gate, up, down, &x, m);
        let got = swiglu_f32x(gate, up, down, &x, m);
        let err: f64 = want.iter().zip(&got).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        assert!(norm > 0.0 && err / norm < 1e-5, "m {m}: relative error {}", err / norm);
    }
}

#[test]
fn spin_pool_batch_matches_the_single_expert_kernels() {
    use lb_expert_host::spin::{swiglu_batch, SpinPool};
    let (n_ff, n_embd) = (1024usize, 512usize);
    let mut s = 123u64;
    let sets: Vec<_> = (0..3).map(|_| mats(GgmlType::Q4_0, n_ff, n_embd, &mut s)).collect();
    let q: Vec<_> = sets
        .iter()
        .map(|(g, u, d)| (QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, g), QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, u), QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, d)))
        .collect();
    let x: Vec<f32> = (0..n_embd).map(|_| lcg(&mut s) * 4.0).collect();
    let pool = SpinPool::new(5, std::time::Duration::from_millis(1));
    let exact = swiglu_batch(&pool, &q, &x, false);
    let i8s = swiglu_batch(&pool, &q, &x, true);
    for (j, (g, u, d)) in q.iter().enumerate() {
        let want = swiglu(*g, *u, *d, &x, 1);
        let norm: f64 = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let rel = |v: &[f32]| v.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt() / norm;
        assert!(rel(&exact[j]) < 1e-5, "expert {j} exact: {}", rel(&exact[j]));
        assert!(rel(&i8s[j]) < 0.03, "expert {j} int8: {}", rel(&i8s[j]));
        assert_eq!(i8s[j], swiglu_i8(*g, *u, *d, &x, 1), "expert {j}: same int8 arithmetic as swiglu_i8");
    }
}
