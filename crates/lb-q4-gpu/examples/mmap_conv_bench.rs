//! Why a wgpu expert miss was ~20 ms on the host: the raw-bytes -> `Vec<u32>` conversion of
//! `ExpertSlots::fill` over a file mapping vs a heap copy. `mmap_conv_bench FILE [populate]`.
use std::time::Instant;
fn conv(raw: &[u8]) -> Vec<u32> {
    raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let f = std::fs::File::open(&a[1]).unwrap();
    let len = f.metadata().unwrap().len() as usize;
    use std::os::fd::AsRawFd;
    let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0) };
    assert!(p != libc::MAP_FAILED);
    let map = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
    if a.get(2).map(|s| s == "populate").unwrap_or(false) {
        let t = Instant::now();
        let rc = unsafe { libc::madvise(p, len, libc::MADV_POPULATE_READ) };
        println!("populate rc {rc} in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    let n = 5_308_416usize / 3;
    let mut v = Vec::new();
    for i in 0..16 {
        let off = (i * 7919 * 4096 * 7) % (len - 3 * n);
        let raw = &map[off..off + 3 * n];
        let t = Instant::now();
        let w = conv(raw);
        v.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&w);
    }
    println!("mmap conv of {} MB: first {:.2} ms, then median {:.2} ms", 3 * n / 1_000_000, v[0], { let mut s = v.clone(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[8] });
    let heap: Vec<u8> = map[..3 * n].to_vec();
    let t = Instant::now();
    let w = conv(&heap);
    println!("heap conv: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    std::hint::black_box(&w);
    let t = Instant::now();
    let c: Vec<u8> = map[1_000_000_000..1_000_000_000 + 3 * n].to_vec();
    println!("mmap memcpy: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    std::hint::black_box(&c);
}
