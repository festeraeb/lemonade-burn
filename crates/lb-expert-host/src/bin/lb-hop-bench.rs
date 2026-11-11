//! W0: what one expert call costs on the wire. Sends the real request (a `Run` for `n` experts
//! with one activation row of `d` f32) and gets the real reply (`n` x `d` f32) through the real
//! encode / transport / decode, with a host that does no arithmetic (it copies the activation
//! into each output), so the number is the hop alone.
//!
//! ```text
//! lb-hop-bench serve --listen tcp://0.0.0.0:7400 --wait block
//! lb-hop-bench run --connect tcp://10.0.10.204:7400 --wait block --server-wait block \
//!        --label "t440->c4 25G tcp" --out /mnt/fast/w2w/results/w0-hop.jsonl
//! lb-hop-bench local --out ...      # in-process channel, loopback TCP, unix socket; block and busy
//! ```

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use lb_expert_host::placement::Wait;
use lb_expert_host::transport::{self, Conn, Listener};
use lb_expert_host::wire::{self, Kind, Run, RunReply};
use std::io::Write;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Echo host: answers every `Run` with `n` copies of its activation row.
    Serve {
        #[arg(long)]
        listen: String,
        #[arg(long, default_value = "block")]
        wait: String,
        /// Exit after this many connections have ended (0 = serve for ever).
        #[arg(long, default_value_t = 0)]
        conns: usize,
    },
    /// Measure round trips against a `serve`.
    Run {
        #[arg(long)]
        connect: String,
        #[arg(long, default_value = "block")]
        wait: String,
        /// The server's wait mode (recorded in the row; the server was started with it).
        #[arg(long, default_value = "block")]
        server_wait: String,
        #[command(flatten)]
        shape: Shape,
        #[arg(long)]
        label: String,
        /// The remote host's 1-minute load average, as read by the caller (recorded).
        #[arg(long)]
        remote_load: Option<f64>,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// The same-host transports, client and echo host as two threads of this process.
    Local {
        #[command(flatten)]
        shape: Shape,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
}

#[derive(clap::Args, Clone)]
struct Shape {
    #[arg(long, default_value_t = 10_000)]
    trips: usize,
    #[arg(long, default_value_t = 1000)]
    warm: usize,
    /// Experts per request (top-k).
    #[arg(long, default_value_t = 4)]
    n: usize,
    /// Activation width.
    #[arg(long, default_value_t = 2048)]
    d: usize,
}

fn serve_conn(mut c: Box<dyn Conn>) -> std::io::Result<()> {
    let (mut buf, mut out, mut run) = (Vec::new(), Vec::new(), Run::default());
    loop {
        c.recv(&mut buf, None)?;
        match wire::peek(&buf) {
            Ok((Kind::Run, seq)) => match wire::decode_run(&buf, &mut run) {
                Ok(()) => {
                    let t = Instant::now();
                    wire::begin_run_ok(&mut out, seq, 0, run.ids.len() as u16, run.d);
                    let d = run.d as usize;
                    for g in &run.groups {
                        for &r in g {
                            wire::put_f32s(&mut out, &run.x[r as usize * d..(r as usize + 1) * d]);
                        }
                    }
                    wire::set_host_ns(&mut out, t.elapsed().as_nanos() as u64);
                }
                Err(e) => wire::encode_error(&mut out, seq, &e.to_string()),
            },
            Ok((_, seq)) => wire::encode_simple(&mut out, Kind::Pong, seq),
            Err(e) => wire::encode_error(&mut out, 0, &e.to_string()),
        }
        c.send(&out)?;
    }
}

fn serve(mut l: Box<dyn Listener>, conns: usize) -> Result<()> {
    let mut done = 0;
    loop {
        let c = l.accept()?;
        // one connection at a time: the bench has one client, and this keeps `--conns` exact
        if let Err(e) = serve_conn(c) {
            if e.kind() != std::io::ErrorKind::UnexpectedEof {
                eprintln!("lb-hop-bench serve: connection ended: {e}");
            }
        }
        done += 1;
        if conns > 0 && done >= conns {
            return Ok(());
        }
    }
}

struct Stats {
    p50: f64,
    p90: f64,
    p99: f64,
    p999: f64,
    min: f64,
    mean: f64,
    max: f64,
    host_p50: f64,
}

fn measure(c: &mut dyn Conn, s: &Shape) -> Result<Stats> {
    let x: Vec<f32> = (0..s.d).map(|i| (i as f32 * 0.37).sin()).collect();
    let ids: Vec<u16> = (0..s.n as u16).map(|i| i * 13 % 64).collect();
    let row = [0u32];
    let groups: Vec<&[u32]> = (0..s.n).map(|_| &row[..]).collect();
    let (mut out, mut buf) = (Vec::new(), Vec::new());
    let mut us = Vec::with_capacity(s.trips);
    let mut host_us = Vec::with_capacity(s.trips);
    for i in 0..s.warm + s.trips {
        let t = Instant::now();
        wire::encode_run(&mut out, i as u64, 7, &ids, &groups, &x, 1, s.d);
        c.send(&out)?;
        c.recv(&mut buf, Some(Instant::now() + Duration::from_secs(10))).with_context(|| format!("trip {i}"))?;
        let reply = wire::decode_run_reply(&buf).map_err(|e| anyhow!("{e}"))?;
        let dt = t.elapsed();
        let RunReply::Ok { y, host_ns, .. } = reply else { bail!("trip {i}: {reply:?}") };
        if y.len() != s.n * s.d || y[s.d - 1].to_bits() != x[s.d - 1].to_bits() || y[(s.n - 1) * s.d].to_bits() != x[0].to_bits() {
            bail!("trip {i}: the echo is not the activation");
        }
        if i >= s.warm {
            us.push(dt.as_nanos() as f64 / 1e3);
            host_us.push(host_ns as f64 / 1e3);
        }
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    host_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |v: &[f64], p: f64| v[((v.len() as f64 * p) as usize).min(v.len() - 1)];
    Ok(Stats { p50: q(&us, 0.5), p90: q(&us, 0.9), p99: q(&us, 0.99), p999: q(&us, 0.999), min: us[0], mean: us.iter().sum::<f64>() / us.len() as f64, max: us[us.len() - 1], host_p50: q(&host_us, 0.5) })
}

fn load1() -> f64 {
    std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok())).unwrap_or(-1.0)
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string()).unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn report(label: &str, transport: &str, wait: &str, server_wait: &str, s: &Shape, st: &Stats, load_before: f64, remote_load: Option<f64>, out: Option<&std::path::Path>) -> Result<()> {
    let r1 = |x: f64| (x * 10.0).round() / 10.0;
    let row = serde_json::json!({
        "ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "step": "W0", "label": label, "transport": transport, "client_wait": wait, "server_wait": server_wait,
        "trips": s.trips, "warm": s.warm, "n": s.n, "d": s.d,
        "request_bytes": wire::HEADER + 12 + 2 * s.n + (s.n % 2) * 2 + 4 * s.n + 4 * s.n + 4 * s.d,
        "reply_bytes": wire::HEADER + 16 + 4 * s.n * s.d,
        "p50_us": r1(st.p50), "p90_us": r1(st.p90), "p99_us": r1(st.p99), "p999_us": r1(st.p999), "min_us": r1(st.min), "mean_us": r1(st.mean), "max_us": r1(st.max),
        "echo_host_p50_us": r1(st.host_p50),
        "client_host": hostname(), "client_load1_before": load_before, "client_load1_after": load1(), "remote_load1": remote_load,
    });
    println!("{row}");
    if let Some(p) = out {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).with_context(|| format!("open {}", p.display()))?;
        writeln!(f, "{row}")?;
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve { listen, wait, conns } => {
            let wait = Wait::parse(&wait).map_err(|e| anyhow!(e))?;
            let l = transport::listen(&listen, wait)?;
            eprintln!("lb-hop-bench: echo host on {} (wait {})", l.local(), wait.name());
            serve(l, conns)
        }
        Cmd::Run { connect, wait, server_wait, shape, label, remote_load, out } => {
            let w = Wait::parse(&wait).map_err(|e| anyhow!(e))?;
            let load = load1();
            let mut c = transport::connect(&connect, w, Duration::from_secs(10)).with_context(|| format!("connect {connect}"))?;
            let st = measure(&mut *c, &shape)?;
            report(&label, c.kind(), &wait, &server_wait, &shape, &st, load, remote_load, out.as_deref())
        }
        Cmd::Local { shape, out } => {
            let sock = std::env::temp_dir().join(format!("w2w-hop-{}.sock", std::process::id()));
            for (name, addr) in [("same process, channel", "inproc://hop".to_string()), ("same host, loopback TCP_NODELAY", "tcp://127.0.0.1:0".to_string()), ("same host, unix socket", format!("unix://{}", sock.display()))] {
                for wait in ["block", "busy"] {
                    let w = Wait::parse(wait).map_err(|e| anyhow!(e))?;
                    let l = transport::listen(&addr, w)?;
                    let at = l.local();
                    let srv = std::thread::spawn(move || serve(l, 1));
                    let load = load1();
                    let mut c = transport::connect(&at, w, Duration::from_secs(10))?;
                    let st = measure(&mut *c, &shape)?;
                    report(name, c.kind(), wait, wait, &shape, &st, load, None, out.as_deref())?;
                    drop(c);
                    let _ = srv.join();
                }
            }
            Ok(())
        }
    }
}
