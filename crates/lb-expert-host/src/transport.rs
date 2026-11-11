//! Transports: how a frame gets from the trunk to a host and back. A [`Conn`] carries whole
//! frames in order; the four here are in-process channels, TCP (`TCP_NODELAY`), unix sockets
//! and (feature `rdma`) a RoCE / InfiniBand reliable-connected queue pair.
//!
//! Addresses: `inproc://name`, `tcp://host:port`, `unix:///path`, `rdma://host:port` (the
//! host:port is a TCP rendezvous that carries the queue pair numbers; the frames go by verbs).
//!
//! A receive that times out leaves a stream transport mid-frame: **a `Conn` whose `recv`
//! returned `TimedOut` must be dropped**, not reused. The trunk does exactly that when it marks
//! a host down.

use crate::placement::Wait;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Frames larger than this are refused (a corrupt length must not allocate the machine away).
pub const MAX_FRAME: usize = 1 << 30;

pub trait Conn: Send {
    /// Send one frame.
    fn send(&mut self, frame: &[u8]) -> io::Result<()>;
    /// Receive one frame into `buf` (replacing its contents). `ErrorKind::TimedOut` when
    /// `deadline` passes first; the connection is then unusable.
    fn recv(&mut self, buf: &mut Vec<u8>, deadline: Option<Instant>) -> io::Result<()>;
    fn kind(&self) -> &'static str;
}

pub trait Listener: Send {
    fn accept(&mut self) -> io::Result<Box<dyn Conn>>;
    /// The address a client can connect to (with the real port when 0 was asked for).
    fn local(&self) -> String;
}

fn timed_out() -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, "no reply inside the deadline")
}

/// When the spin phase of `wait` ends for a receive that starts now (`None` = it never does).
pub(crate) fn spin_until(wait: Wait) -> Option<Option<Instant>> {
    match wait {
        Wait::Block => None,
        Wait::Busy => Some(None),
        Wait::Spin(us) => Some(Some(Instant::now() + Duration::from_micros(us as u64))),
    }
}

// ---------------------------------------------------------------- stream sockets (TCP, unix)

trait Sock: Read + Write + Send {
    fn nonblocking(&self, on: bool) -> io::Result<()>;
    fn read_timeout(&self, t: Option<Duration>) -> io::Result<()>;
}

impl Sock for TcpStream {
    fn nonblocking(&self, on: bool) -> io::Result<()> {
        self.set_nonblocking(on)
    }
    fn read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.set_read_timeout(t)
    }
}

impl Sock for UnixStream {
    fn nonblocking(&self, on: bool) -> io::Result<()> {
        self.set_nonblocking(on)
    }
    fn read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.set_read_timeout(t)
    }
}

struct StreamConn<S: Sock> {
    s: S,
    wait: Wait,
    kind: &'static str,
    out: Vec<u8>,
    nb: bool,
}

impl<S: Sock> StreamConn<S> {
    fn new(s: S, wait: Wait, kind: &'static str) -> Self {
        Self { s, wait, kind, out: Vec::new(), nb: false }
    }

    fn set_nb(&mut self, on: bool) -> io::Result<()> {
        if self.nb != on {
            self.s.nonblocking(on)?;
            self.nb = on;
        }
        Ok(())
    }

    /// Fill `dst` exactly. `spin`: `Some(until)` while spinning (`Some(None)` = forever).
    fn fill(&mut self, dst: &mut [u8], deadline: Option<Instant>, spin: &mut Option<Option<Instant>>) -> io::Result<()> {
        let mut got = 0;
        while got < dst.len() {
            if let Some(until) = *spin {
                self.set_nb(true)?;
                match self.s.read(&mut dst[got..]) {
                    Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                    Ok(n) => got += n,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        let now = Instant::now();
                        if deadline.is_some_and(|d| now >= d) {
                            return Err(timed_out());
                        }
                        if until.is_some_and(|u| now >= u) {
                            *spin = None;
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            } else {
                self.set_nb(false)?;
                let t = match deadline {
                    Some(d) => {
                        let left = d.saturating_duration_since(Instant::now());
                        if left.is_zero() {
                            return Err(timed_out());
                        }
                        Some(left)
                    }
                    None => None,
                };
                self.s.read_timeout(t)?;
                match self.s.read(&mut dst[got..]) {
                    Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                    Ok(n) => got += n,
                    Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => return Err(timed_out()),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(())
    }
}

impl<S: Sock> Conn for StreamConn<S> {
    fn send(&mut self, frame: &[u8]) -> io::Result<()> {
        // one buffer, one write: the length must not go out as its own segment
        self.out.clear();
        self.out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        self.out.extend_from_slice(frame);
        self.set_nb(false)?;
        let out = std::mem::take(&mut self.out);
        let r = self.s.write_all(&out);
        self.out = out;
        r
    }

    fn recv(&mut self, buf: &mut Vec<u8>, deadline: Option<Instant>) -> io::Result<()> {
        let mut spin = spin_until(self.wait);
        let mut len = [0u8; 4];
        self.fill(&mut len, deadline, &mut spin)?;
        let n = u32::from_le_bytes(len) as usize;
        if n > MAX_FRAME {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame of {n} bytes")));
        }
        buf.clear();
        buf.resize(n, 0);
        self.fill(buf, deadline, &mut spin)
    }

    fn kind(&self) -> &'static str {
        self.kind
    }
}

struct TcpL {
    l: TcpListener,
    wait: Wait,
}

impl Listener for TcpL {
    fn accept(&mut self) -> io::Result<Box<dyn Conn>> {
        let (s, _) = self.l.accept()?;
        s.set_nodelay(true)?;
        Ok(Box::new(StreamConn::new(s, self.wait, "tcp")))
    }
    fn local(&self) -> String {
        self.l.local_addr().map(|a| format!("tcp://{a}")).unwrap_or_default()
    }
}

struct UnixL {
    l: UnixListener,
    path: String,
    wait: Wait,
}

impl Listener for UnixL {
    fn accept(&mut self) -> io::Result<Box<dyn Conn>> {
        let (s, _) = self.l.accept()?;
        Ok(Box::new(StreamConn::new(s, self.wait, "unix")))
    }
    fn local(&self) -> String {
        format!("unix://{}", self.path)
    }
}

impl Drop for UnixL {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub(crate) fn tcp_connect(hostport: &str, timeout: Duration) -> io::Result<TcpStream> {
    let mut last = io::Error::new(io::ErrorKind::NotFound, format!("{hostport}: no address"));
    for a in hostport.to_socket_addrs()? {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(s) => {
                s.set_nodelay(true)?;
                return Ok(s);
            }
            Err(e) => last = e,
        }
    }
    Err(last)
}

// ---------------------------------------------------------------- in-process

struct InProc {
    tx: Sender<Vec<u8>>,
    rx: Receiver<Vec<u8>>,
    wait: Wait,
}

impl Conn for InProc {
    fn send(&mut self, frame: &[u8]) -> io::Result<()> {
        self.tx.send(frame.to_vec()).map_err(|_| io::ErrorKind::BrokenPipe.into())
    }

    fn recv(&mut self, buf: &mut Vec<u8>, deadline: Option<Instant>) -> io::Result<()> {
        let mut spin = spin_until(self.wait);
        loop {
            if let Some(until) = spin {
                match self.rx.try_recv() {
                    Ok(v) => {
                        *buf = v;
                        return Ok(());
                    }
                    Err(TryRecvError::Disconnected) => return Err(io::ErrorKind::UnexpectedEof.into()),
                    Err(TryRecvError::Empty) => {
                        let now = Instant::now();
                        if deadline.is_some_and(|d| now >= d) {
                            return Err(timed_out());
                        }
                        if until.is_some_and(|u| now >= u) {
                            spin = None;
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                }
            } else {
                let r = match deadline {
                    Some(d) => self.rx.recv_timeout(d.saturating_duration_since(Instant::now())),
                    None => self.rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
                };
                return match r {
                    Ok(v) => {
                        *buf = v;
                        Ok(())
                    }
                    Err(RecvTimeoutError::Timeout) => Err(timed_out()),
                    Err(RecvTimeoutError::Disconnected) => Err(io::ErrorKind::UnexpectedEof.into()),
                };
            }
        }
    }

    fn kind(&self) -> &'static str {
        "inproc"
    }
}

type Registry = Mutex<HashMap<String, Sender<(Sender<Vec<u8>>, Receiver<Vec<u8>>)>>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

struct InProcL {
    name: String,
    rx: Receiver<(Sender<Vec<u8>>, Receiver<Vec<u8>>)>,
    wait: Wait,
}

impl Listener for InProcL {
    fn accept(&mut self) -> io::Result<Box<dyn Conn>> {
        let (tx, rx) = self.rx.recv().map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))?;
        Ok(Box::new(InProc { tx, rx, wait: self.wait }))
    }
    fn local(&self) -> String {
        format!("inproc://{}", self.name)
    }
}

impl Drop for InProcL {
    fn drop(&mut self) {
        registry().lock().unwrap().remove(&self.name);
    }
}

// ---------------------------------------------------------------- by address

fn split(addr: &str) -> io::Result<(&str, &str)> {
    addr.split_once("://").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("address {addr:?}: expected scheme://...")))
}

/// Listen on `addr`. `wait` is how accepted connections wait for a request.
pub fn listen(addr: &str, wait: Wait) -> io::Result<Box<dyn Listener>> {
    let (scheme, rest) = split(addr)?;
    match scheme {
        "tcp" => Ok(Box::new(TcpL { l: TcpListener::bind(rest)?, wait })),
        "unix" => {
            let _ = std::fs::remove_file(rest);
            Ok(Box::new(UnixL { l: UnixListener::bind(rest)?, path: rest.to_string(), wait }))
        }
        "inproc" => {
            let (tx, rx) = mpsc::channel();
            let mut reg = registry().lock().unwrap();
            if reg.contains_key(rest) {
                return Err(io::Error::new(io::ErrorKind::AddrInUse, format!("inproc://{rest} is taken")));
            }
            reg.insert(rest.to_string(), tx);
            Ok(Box::new(InProcL { name: rest.to_string(), rx, wait }))
        }
        #[cfg(feature = "rdma")]
        "rdma" => crate::rdma::listen(rest, wait),
        #[cfg(not(feature = "rdma"))]
        "rdma" => Err(io::Error::new(io::ErrorKind::Unsupported, "built without the `rdma` feature")),
        other => Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unknown transport {other:?}"))),
    }
}

/// Connect to `addr`. `wait` is how this side waits for a reply.
pub fn connect(addr: &str, wait: Wait, timeout: Duration) -> io::Result<Box<dyn Conn>> {
    let (scheme, rest) = split(addr)?;
    match scheme {
        "tcp" => Ok(Box::new(StreamConn::new(tcp_connect(rest, timeout)?, wait, "tcp"))),
        "unix" => Ok(Box::new(StreamConn::new(UnixStream::connect(rest)?, wait, "unix"))),
        "inproc" => {
            let reg = registry().lock().unwrap();
            let l = reg.get(rest).ok_or_else(|| io::Error::new(io::ErrorKind::ConnectionRefused, format!("nothing listens on inproc://{rest}")))?;
            let (to_srv, srv_rx) = mpsc::channel();
            let (to_cli, cli_rx) = mpsc::channel();
            l.send((to_cli, srv_rx)).map_err(|_| io::Error::from(io::ErrorKind::ConnectionRefused))?;
            Ok(Box::new(InProc { tx: to_srv, rx: cli_rx, wait }))
        }
        #[cfg(feature = "rdma")]
        "rdma" => crate::rdma::connect(rest, wait, timeout),
        #[cfg(not(feature = "rdma"))]
        "rdma" => Err(io::Error::new(io::ErrorKind::Unsupported, "built without the `rdma` feature")),
        other => Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unknown transport {other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo(mut l: Box<dyn Listener>, n: usize) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut c = l.accept().unwrap();
            let mut buf = Vec::new();
            for _ in 0..n {
                c.recv(&mut buf, None).unwrap();
                buf.reverse();
                c.send(&buf).unwrap();
            }
        })
    }

    fn round_trips(addr: &str, srv_wait: Wait, cli_wait: Wait) {
        let l = listen(addr, srv_wait).unwrap();
        let at = l.local();
        let h = echo(l, 3);
        let mut c = connect(&at, cli_wait, Duration::from_secs(5)).unwrap();
        let mut buf = Vec::new();
        for n in [0usize, 5, 40_000] {
            let msg: Vec<u8> = (0..n).map(|i| (i * 7) as u8).collect();
            c.send(&msg).unwrap();
            c.recv(&mut buf, Some(Instant::now() + Duration::from_secs(10))).unwrap();
            let mut want = msg.clone();
            want.reverse();
            assert_eq!(buf, want, "{addr} {n}");
        }
        h.join().unwrap();
    }

    #[test]
    fn every_local_transport_carries_frames_in_every_wait_mode() {
        let sock = std::env::temp_dir().join(format!("w2w-test-{}.sock", std::process::id()));
        for (i, (sw, cw)) in [(Wait::Block, Wait::Block), (Wait::Busy, Wait::Busy), (Wait::Spin(50), Wait::Spin(50)), (Wait::Block, Wait::Busy)].into_iter().enumerate() {
            round_trips("tcp://127.0.0.1:0", sw, cw);
            round_trips(&format!("unix://{}", sock.display()), sw, cw);
            round_trips(&format!("inproc://rt-{i}"), sw, cw);
        }
    }

    #[test]
    fn a_silent_peer_times_out_and_a_dead_one_is_an_error() {
        for addr in ["tcp://127.0.0.1:0", "inproc://silent"] {
            for wait in [Wait::Block, Wait::Busy, Wait::Spin(100)] {
                let mut l = listen(addr, Wait::Block).unwrap();
                let at = l.local();
                let h = std::thread::spawn(move || {
                    let c = l.accept().unwrap();
                    std::thread::sleep(Duration::from_millis(150));
                    drop(c);
                });
                let mut c = connect(&at, wait, Duration::from_secs(5)).unwrap();
                let mut buf = Vec::new();
                let t = Instant::now();
                let e = c.recv(&mut buf, Some(Instant::now() + Duration::from_millis(30))).unwrap_err();
                assert_eq!(e.kind(), io::ErrorKind::TimedOut, "{addr} {wait:?}");
                assert!(t.elapsed() < Duration::from_millis(140), "{addr} {wait:?}: {:?}", t.elapsed());
                h.join().unwrap();
                // the peer is gone now: not a timeout, an error, and promptly
                let e = c.recv(&mut buf, Some(Instant::now() + Duration::from_secs(5))).unwrap_err();
                assert_ne!(e.kind(), io::ErrorKind::TimedOut, "{addr} {wait:?}");
            }
        }
        assert!(connect("inproc://nobody", Wait::Block, Duration::from_secs(1)).is_err());
        assert!(connect("carrier-pigeon://x", Wait::Block, Duration::from_secs(1)).is_err());
    }
}
