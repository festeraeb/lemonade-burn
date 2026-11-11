//! The trunk's end of one host connection: hello, then `Run` requests. Sending and receiving
//! are separate calls so a trunk can send to several hosts, do its own work, and then collect.

use crate::placement::Wait;
use crate::transport::{self, Conn};
use crate::wire::{self, HostInfo, Kind, RunReply};
use std::io;
use std::time::{Duration, Instant};

pub struct HostClient {
    conn: Box<dyn Conn>,
    seq: u64,
    pub info: HostInfo,
    out: Vec<u8>,
    buf: Vec<u8>,
}

fn invalid(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

impl HostClient {
    /// Connect and say hello. `timeout` bounds the connect and the hello reply.
    pub fn connect(addr: &str, wait: Wait, timeout: Duration) -> io::Result<Self> {
        let mut conn = transport::connect(addr, wait, timeout)?;
        let (mut out, mut buf) = (Vec::new(), Vec::new());
        wire::encode_simple(&mut out, Kind::Hello, 0);
        conn.send(&out)?;
        conn.recv(&mut buf, Some(Instant::now() + timeout))?;
        let info = match wire::peek(&buf).map_err(invalid)? {
            (Kind::HelloOk, _) => wire::decode_hello_ok(&buf).map_err(invalid)?,
            (Kind::Error, _) => return Err(invalid(wire::decode_error(&buf))),
            (k, _) => return Err(invalid(format!("{k:?} in reply to Hello"))),
        };
        Ok(Self { conn, seq: 0, info, out, buf })
    }

    pub fn transport(&self) -> &'static str {
        self.conn.kind()
    }

    /// Ask the host again what it holds (after a placement reload).
    pub fn refresh(&mut self, timeout: Duration) -> io::Result<()> {
        self.seq += 1;
        wire::encode_simple(&mut self.out, Kind::Hello, self.seq);
        self.conn.send(&self.out)?;
        self.conn.recv(&mut self.buf, Some(Instant::now() + timeout))?;
        match wire::peek(&self.buf).map_err(invalid)? {
            (Kind::HelloOk, s) if s == self.seq => {
                self.info = wire::decode_hello_ok(&self.buf).map_err(invalid)?;
                Ok(())
            }
            (k, s) => Err(invalid(format!("{k:?} seq {s} in reply to Hello seq {}", self.seq))),
        }
    }

    /// Send a `Run`; the reply is collected with [`HostClient::recv_run`].
    pub fn send_run(&mut self, layer: u16, ids: &[u16], groups: &[&[u32]], x: &[f32], m: usize, d: usize) -> io::Result<()> {
        self.seq += 1;
        wire::encode_run(&mut self.out, self.seq, layer, ids, groups, x, m, d);
        self.conn.send(&self.out)
    }

    /// The reply to the last `send_run`. `TimedOut` when `deadline` passes: the connection is
    /// then unusable and must be dropped.
    pub fn recv_run(&mut self, deadline: Option<Instant>) -> io::Result<RunReply> {
        self.conn.recv(&mut self.buf, deadline)?;
        let (_, seq) = wire::peek(&self.buf).map_err(invalid)?;
        if seq != self.seq {
            return Err(invalid(format!("reply seq {seq}, expected {}", self.seq)));
        }
        wire::decode_run_reply(&self.buf).map_err(invalid)
    }

    /// One request, one reply.
    pub fn run(&mut self, layer: u16, ids: &[u16], groups: &[&[u32]], x: &[f32], m: usize, d: usize, deadline: Option<Instant>) -> io::Result<RunReply> {
        self.send_run(layer, ids, groups, x, m, d)?;
        self.recv_run(deadline)
    }
}
