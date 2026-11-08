//! Host DRAM traffic from the uncore memory-controller counters (Intel `uncore_imc_*` PMUs,
//! `cas_count_read` / `cas_count_write`, 64 bytes a CAS), sampled every `LB_IMC_MS` (default
//! 1000) and printed as one JSON line per sample: `{"t":..,"read_mb_s":..,"write_mb_s":..}`.
//! For measuring what expert prefetch copies cost the host's RAM bandwidth (Avalon's PREGATE
//! lesson, `docs/LESSONS_FROM_AVALON.md` 2.5) on a host without `perf`. System-wide counters:
//! run as root (or with `kernel.perf_event_paranoid` <= 0). Stops after `LB_IMC_SECS` (default
//! 3600) or when killed.
use std::time::{Duration, Instant};

fn read_file(p: &str) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

/// `perf_event_open` for one uncore event on one CPU: returns the fd.
fn open(ty: u32, config: u64, cpu: i32) -> Result<i32, String> {
    // struct perf_event_attr, PERF_ATTR_SIZE_VER5 (112 bytes) is enough: type, size, config
    let mut attr = [0u8; 112];
    attr[0..4].copy_from_slice(&ty.to_ne_bytes());
    attr[4..8].copy_from_slice(&112u32.to_ne_bytes());
    attr[8..16].copy_from_slice(&config.to_ne_bytes());
    let fd = unsafe { libc::syscall(libc::SYS_perf_event_open, attr.as_ptr(), -1i32, cpu, -1i32, 0u64) };
    if fd < 0 {
        return Err(format!("perf_event_open(type {ty}, config {config:#x}, cpu {cpu}): {}", std::io::Error::last_os_error()));
    }
    Ok(fd as i32)
}

fn count(fd: i32) -> u64 {
    let mut b = [0u8; 8];
    let n = unsafe { libc::read(fd, b.as_mut_ptr() as *mut libc::c_void, 8) };
    if n == 8 { u64::from_ne_bytes(b) } else { 0 }
}

fn main() -> Result<(), String> {
    let ms: u64 = std::env::var("LB_IMC_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let secs: u64 = std::env::var("LB_IMC_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(3600);
    let mut fds: Vec<(bool, i32)> = Vec::new(); // (is_write, fd)
    for i in 0..64 {
        let dir = format!("/sys/bus/event_source/devices/uncore_imc_{i}");
        let Some(ty) = read_file(&format!("{dir}/type")).and_then(|s| s.parse::<u32>().ok()) else { continue };
        let cpus = read_file(&format!("{dir}/cpumask")).unwrap_or_else(|| "0".into());
        for cpu in cpus.split(',').filter_map(|c| c.parse::<i32>().ok()) {
            // event 0x04: CAS_COUNT; umask 0x03 = RD, 0x0c = WR (Sandy/Ivy/Haswell-EP)
            fds.push((false, open(ty, 0x04 | (0x03 << 8), cpu)?));
            fds.push((true, open(ty, 0x04 | (0x0c << 8), cpu)?));
        }
    }
    if fds.is_empty() {
        return Err("no uncore_imc PMU".into());
    }
    eprintln!("imc_bw: {} counters", fds.len());
    let t0 = Instant::now();
    let snap = |fds: &[(bool, i32)]| fds.iter().fold((0u64, 0u64), |(r, w), (is_w, fd)| if *is_w { (r, w + count(*fd)) } else { (r + count(*fd), w) });
    let mut prev = snap(&fds);
    let mut tp = Instant::now();
    while t0.elapsed().as_secs() < secs {
        std::thread::sleep(Duration::from_millis(ms));
        let cur = snap(&fds);
        let dt = tp.elapsed().as_secs_f64();
        tp = Instant::now();
        let mb = |a: u64, b: u64| (a.wrapping_sub(b)) as f64 * 64.0 / 1e6 / dt;
        let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        println!("{{\"t\":{unix:.3},\"read_mb_s\":{:.1},\"write_mb_s\":{:.1}}}", mb(cur.0, prev.0), mb(cur.1, prev.1));
        prev = cur;
    }
    Ok(())
}
