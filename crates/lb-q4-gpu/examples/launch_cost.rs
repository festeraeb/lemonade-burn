//! What one kernel launch costs on the host: N launches of a trivial kernel (4 values) queued, one
//! sync at the end. Also the same with a sync after each. Run with a device named.
use lb_q4_gpu::Gpu;
use std::time::Instant;

fn main() {
    let gpu = Gpu::open().expect("device");
    let x = gpu.upload_f32(&[0.0f32; 256]);
    let y = gpu.upload_f32(&[1.0f32; 256]);
    for _ in 0..200 {
        let _ = gpu.add_rms_norm(&x, None, &y, 256, 1e-5);
    }
    gpu.sync_recycle().unwrap();
    for &n in &[400usize, 2000] {
        let t = Instant::now();
        for i in 0..n {
            let _ = gpu.add_rms_norm(&x, None, &y, 256, 1e-5);
            if i % 350 == 349 {
                gpu.sync_recycle().unwrap();
            }
        }
        let queued = t.elapsed().as_secs_f64();
        gpu.sync_recycle().unwrap();
        let total = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..n {
            let _ = gpu.add_rms_norm(&x, None, &y, 256, 1e-5);
            gpu.sync().unwrap();
        }
        let synced = t.elapsed().as_secs_f64();
        println!("{}: {n} launches: {:.1} us each to queue, {:.1} us each incl. the final sync; {:.1} us each with a sync per launch", gpu.name, queued / n as f64 * 1e6, total / n as f64 * 1e6, synced / n as f64 * 1e6);
    }
}
