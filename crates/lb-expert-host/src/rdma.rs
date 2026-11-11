//! RDMA transport (feature `rdma`): frames by SEND/RECV on one reliable-connected queue pair,
//! kernel bypassed on both sides. `rdma://host:port` names a TCP rendezvous: the client
//! connects, each side sends one line with its queue pair numbers, both connect, and the TCP
//! socket then only stays open as the pair's lifetime. libibverbs is reached through
//! `csrc/w2w_rdma.c`.
//!
//! The device and RoCE v2 GID are found from the rendezvous socket's own IPv4 address (the GID
//! table entry `::ffff:a.b.c.d` of type "RoCE v2" under /sys/class/infiniband), so nothing is
//! configured by index. `LB_RDMA_DEV` + `LB_RDMA_GID` override the search.
//!
//! A frame larger than one registered buffer goes as several messages; the first carries the
//! total length. The receive ring plus the queue pair's receiver-not-ready retry is the flow
//! control.

use crate::placement::Wait;
use crate::transport::{tcp_connect, Conn, Listener, MAX_FRAME};
use std::ffi::{c_char, c_int, c_void, CString};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

#[repr(C)]
struct w2w_rdma {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn w2w_rdma_open(dev: *const c_char, port: c_int, gid_index: c_int, max_msg: usize, n_recv: c_int, err: *mut c_char, errlen: usize) -> *mut w2w_rdma;
    fn w2w_rdma_local(h: *mut w2w_rdma, qpn: *mut u32, psn: *mut u32, gid: *mut u8, lid: *mut u16, mtu: *mut c_int);
    fn w2w_rdma_connect(h: *mut w2w_rdma, qpn: u32, psn: u32, gid: *const u8, lid: u16, peer_mtu: c_int, err: *mut c_char, errlen: usize) -> c_int;
    fn w2w_rdma_send(h: *mut w2w_rdma, data: *const c_void, len: usize, timeout_ms: c_int, err: *mut c_char, errlen: usize) -> c_int;
    fn w2w_rdma_recv(h: *mut w2w_rdma, data: *mut *const u8, len: *mut u32, idx: *mut c_int, timeout_us: i64, spin_us: i64, watch_fd: c_int, err: *mut c_char, errlen: usize) -> c_int;
    fn w2w_rdma_repost(h: *mut w2w_rdma, idx: c_int, err: *mut c_char, errlen: usize) -> c_int;
    fn w2w_rdma_close(h: *mut w2w_rdma);
}

/// One registered buffer: a decode reply (4 x 8 KB) fits many times over; prefill batches
/// fragment.
const MAX_MSG: usize = 1 << 20;
const N_RECV: i32 = 8;
const FIRST: usize = 8;

fn errstr(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The queue pair numbers one side tells the other.
#[derive(Clone, Debug, PartialEq, Eq)]
struct QpInfo {
    qpn: u32,
    psn: u32,
    gid: [u8; 16],
    lid: u16,
    mtu: i32,
}

impl QpInfo {
    fn line(&self) -> String {
        let gid: String = self.gid.iter().map(|b| format!("{b:02x}")).collect();
        format!("W2WRDMA {} {} {} {} {}\n", self.qpn, self.psn, gid, self.lid, self.mtu)
    }

    fn parse(line: &str) -> io::Result<Self> {
        let bad = || io::Error::new(io::ErrorKind::InvalidData, format!("rendezvous line {line:?}"));
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() != 6 || p[0] != "W2WRDMA" || p[3].len() != 32 {
            return Err(bad());
        }
        let mut gid = [0u8; 16];
        for (i, g) in gid.iter_mut().enumerate() {
            *g = u8::from_str_radix(&p[3][2 * i..2 * i + 2], 16).map_err(|_| bad())?;
        }
        Ok(Self { qpn: p[1].parse().map_err(|_| bad())?, psn: p[2].parse().map_err(|_| bad())?, gid, lid: p[4].parse().map_err(|_| bad())?, mtu: p[5].parse().map_err(|_| bad())? })
    }
}

/// The RDMA device and RoCE v2 GID index that carry `ip` (see the module doc).
pub fn find_device(ip: Ipv4Addr) -> io::Result<(String, i32)> {
    if let (Ok(dev), Ok(gid)) = (std::env::var("LB_RDMA_DEV"), std::env::var("LB_RDMA_GID")) {
        let gid = gid.trim().parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "LB_RDMA_GID is not a number"))?;
        return Ok((dev, gid));
    }
    let o = ip.octets();
    let want = format!("0000:0000:0000:0000:0000:ffff:{:02x}{:02x}:{:02x}{:02x}", o[0], o[1], o[2], o[3]);
    for dev in std::fs::read_dir("/sys/class/infiniband")? {
        let dev = dev?;
        let port = dev.path().join("ports/1");
        for i in 0..256 {
            let Ok(gid) = std::fs::read_to_string(port.join(format!("gids/{i}"))) else { break };
            if gid.trim() != want {
                continue;
            }
            let ty = std::fs::read_to_string(port.join(format!("gid_attrs/types/{i}"))).unwrap_or_default();
            if ty.contains("v2") {
                return Ok((dev.file_name().to_string_lossy().into_owned(), i));
            }
        }
    }
    Err(io::Error::new(io::ErrorKind::NotFound, format!("no RoCE v2 GID for {ip} under /sys/class/infiniband")))
}

struct RdmaConn {
    h: *mut w2w_rdma,
    wait: Wait,
    /// Held so the pair's lifetime is the rendezvous connection's.
    _tcp: TcpStream,
}

// SAFETY: the handle is owned by this one value and used by one thread at a time; libibverbs
// objects are not tied to the thread that made them.
unsafe impl Send for RdmaConn {}

impl Drop for RdmaConn {
    fn drop(&mut self) {
        // SAFETY: the handle came from w2w_rdma_open and is closed once.
        unsafe { w2w_rdma_close(self.h) };
    }
}

impl RdmaConn {
    fn open(tcp: TcpStream, wait: Wait) -> io::Result<(Self, QpInfo)> {
        let ip = match tcp.local_addr()?.ip() {
            IpAddr::V4(v4) => v4,
            IpAddr::V6(v6) => v6.to_ipv4_mapped().ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported, "the RDMA rendezvous needs an IPv4 address"))?,
        };
        let (dev, gid_index) = find_device(ip)?;
        let cdev = CString::new(dev.clone()).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let mut err = [0 as c_char; 256];
        // SAFETY: a valid C string and error buffer; the shim writes within errlen.
        let h = unsafe { w2w_rdma_open(cdev.as_ptr(), 1, gid_index, MAX_MSG, N_RECV, err.as_mut_ptr(), err.len()) };
        if h.is_null() {
            return Err(io::Error::other(format!("{dev} gid {gid_index}: {}", errstr(&err))));
        }
        let (mut qpn, mut psn, mut lid, mut mtu) = (0u32, 0u32, 0u16, 0 as c_int);
        let mut gid = [0u8; 16];
        // SAFETY: valid out-pointers for the handle just opened.
        unsafe { w2w_rdma_local(h, &mut qpn, &mut psn, gid.as_mut_ptr(), &mut lid, &mut mtu) };
        Ok((Self { h, wait, _tcp: tcp }, QpInfo { qpn, psn, gid, lid, mtu }))
    }

    fn connect_qp(&self, peer: &QpInfo) -> io::Result<()> {
        let mut err = [0 as c_char; 256];
        // SAFETY: valid pointers; the shim reads 16 GID bytes.
        let rc = unsafe { w2w_rdma_connect(self.h, peer.qpn, peer.psn, peer.gid.as_ptr(), peer.lid, peer.mtu, err.as_mut_ptr(), err.len()) };
        if rc != 0 {
            return Err(io::Error::other(format!("connect to qp {}: {}", peer.qpn, errstr(&err))));
        }
        Ok(())
    }

    fn send_one(&mut self, b: &[u8]) -> io::Result<()> {
        let mut err = [0 as c_char; 256];
        // SAFETY: `b` is valid for `b.len()` bytes; the shim copies it into its own buffer.
        let rc = unsafe { w2w_rdma_send(self.h, b.as_ptr() as *const c_void, b.len(), 5000, err.as_mut_ptr(), err.len()) };
        if rc != 0 {
            return Err(io::Error::other(errstr(&err)));
        }
        Ok(())
    }

    /// One message appended to `buf`, minus `skip` leading bytes; returns the message's bytes.
    fn recv_one(&mut self, deadline: Option<Instant>, started: Instant) -> io::Result<(*const u8, usize, c_int)> {
        let timeout_us = match deadline {
            Some(d) => d.saturating_duration_since(Instant::now()).as_micros() as i64,
            None => -1,
        };
        let spin_us = match self.wait {
            Wait::Block => 0,
            Wait::Busy => -1,
            // the spin window is counted from the start of the frame, not of each fragment
            Wait::Spin(us) => (us as i64 - started.elapsed().as_micros() as i64).max(0),
        };
        let (mut data, mut len, mut idx) = (std::ptr::null(), 0u32, 0 as c_int);
        let mut err = [0 as c_char; 256];
        // SAFETY: valid out-pointers; on 0 the shim returns a buffer that stays valid until
        // the repost.
        let rc = unsafe { w2w_rdma_recv(self.h, &mut data, &mut len, &mut idx, timeout_us, spin_us, self._tcp.as_raw_fd(), err.as_mut_ptr(), err.len()) };
        match rc {
            0 => Ok((data, len as usize, idx)),
            1 => Err(io::Error::new(io::ErrorKind::TimedOut, "no reply inside the deadline")),
            2 => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the peer closed the rendezvous connection")),
            _ => Err(io::Error::other(errstr(&err))),
        }
    }

    fn repost(&mut self, idx: c_int) -> io::Result<()> {
        let mut err = [0 as c_char; 256];
        // SAFETY: `idx` came from w2w_rdma_recv and its buffer is no longer read.
        let rc = unsafe { w2w_rdma_repost(self.h, idx, err.as_mut_ptr(), err.len()) };
        if rc != 0 {
            return Err(io::Error::other(errstr(&err)));
        }
        Ok(())
    }
}

impl Conn for RdmaConn {
    fn send(&mut self, frame: &[u8]) -> io::Result<()> {
        // first message: total length, then as much of the frame as fits
        let mut first = Vec::with_capacity(FIRST + frame.len().min(MAX_MSG - FIRST));
        first.extend_from_slice(&(frame.len() as u64).to_le_bytes());
        let head = frame.len().min(MAX_MSG - FIRST);
        first.extend_from_slice(&frame[..head]);
        self.send_one(&first)?;
        for chunk in frame[head..].chunks(MAX_MSG) {
            self.send_one(chunk)?;
        }
        Ok(())
    }

    fn recv(&mut self, buf: &mut Vec<u8>, deadline: Option<Instant>) -> io::Result<()> {
        let started = Instant::now();
        let (data, len, idx) = self.recv_one(deadline, started)?;
        if len < FIRST {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "short first message"));
        }
        // SAFETY: the shim's buffer holds `len` valid bytes until the repost below.
        let first = unsafe { std::slice::from_raw_parts(data, len) };
        let total = u64::from_le_bytes(first[..FIRST].try_into().unwrap()) as usize;
        if total > MAX_FRAME || len - FIRST > total {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame of {total} bytes")));
        }
        buf.clear();
        buf.reserve(total);
        buf.extend_from_slice(&first[FIRST..]);
        self.repost(idx)?;
        while buf.len() < total {
            let (data, len, idx) = self.recv_one(deadline, started)?;
            if buf.len() + len > total {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "fragment runs past the frame"));
            }
            // SAFETY: as above.
            buf.extend_from_slice(unsafe { std::slice::from_raw_parts(data, len) });
            self.repost(idx)?;
        }
        Ok(())
    }

    fn kind(&self) -> &'static str {
        "rdma"
    }
}

fn read_line(tcp: &TcpStream) -> io::Result<String> {
    tcp.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut line = String::new();
    BufReader::new(tcp).read_line(&mut line)?;
    Ok(line)
}

struct RdmaL {
    l: TcpListener,
    wait: Wait,
}

impl Listener for RdmaL {
    fn accept(&mut self) -> io::Result<Box<dyn Conn>> {
        let (tcp, _) = self.l.accept()?;
        tcp.set_nodelay(true)?;
        let peer = QpInfo::parse(&read_line(&tcp)?)?;
        let (conn, mine) = RdmaConn::open(tcp, self.wait)?;
        // ready to receive before the client learns our numbers
        conn.connect_qp(&peer)?;
        (&conn._tcp).write_all(mine.line().as_bytes())?;
        Ok(Box::new(conn))
    }

    fn local(&self) -> String {
        self.l.local_addr().map(|a| format!("rdma://{a}")).unwrap_or_default()
    }
}

pub fn listen(hostport: &str, wait: Wait) -> io::Result<Box<dyn Listener>> {
    Ok(Box::new(RdmaL { l: TcpListener::bind(hostport)?, wait }))
}

pub fn connect(hostport: &str, wait: Wait, timeout: Duration) -> io::Result<Box<dyn Conn>> {
    let tcp = tcp_connect(hostport, timeout)?;
    let (conn, mine) = RdmaConn::open(tcp, wait)?;
    (&conn._tcp).write_all(mine.line().as_bytes())?;
    let peer = QpInfo::parse(&read_line(&conn._tcp)?)?;
    conn.connect_qp(&peer)?;
    Ok(Box::new(conn))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qp_info_line_round_trips() {
        let q = QpInfo { qpn: 4242, psn: 0xabcdef, gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 0, 10, 61], lid: 0, mtu: 5 };
        assert_eq!(QpInfo::parse(&q.line()).unwrap(), q);
        assert!(QpInfo::parse("W2WRDMA 1 2 zz 0 5").is_err());
        assert!(QpInfo::parse("hello").is_err());
    }
}
