//! The host end to end on the synthetic model: every transport returns exactly what the shared
//! CPU kernel returns, and an expert the host does not hold is refused by name.

use lb_expert_host::expert::swiglu;
use lb_expert_host::host::{self, CpuBackend, Host};
use lb_expert_host::placement::{HeldSet, PlacementMap, Wait};
use lb_expert_host::wire::RunReply;
use lb_expert_host::{testgguf, transport, HostClient, ModelExperts};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn model(tag: &str) -> (ModelExperts, std::path::PathBuf) {
    let p = testgguf::temp_model(tag, 7).unwrap();
    (ModelExperts::open(&p).unwrap(), p)
}

fn x_rows(m: usize, d: usize) -> Vec<f32> {
    (0..m * d).map(|i| ((i * 37 % 101) as f32 / 50.0 - 1.0) * 0.7).collect()
}

fn local(model: &ModelExperts, layer: usize, e: usize, x: &[f32], rows: &[u32]) -> Vec<f32> {
    let le = model.layers[layer].as_ref().unwrap();
    let d = model.d;
    let xs: Vec<f32> = rows.iter().flat_map(|&r| x[r as usize * d..(r as usize + 1) * d].to_vec()).collect();
    swiglu(model.mat(&le.gate, e), model.mat(&le.up, e), model.mat(&le.down, e), &xs, rows.len())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_model_view_finds_the_expert_stacks() {
    let (m, p) = model("view");
    assert_eq!((m.n_layer(), m.n_expert, m.d, m.n_ff), (5, 8, 64, 32));
    assert!(m.layers[0].is_none() && m.layers[1].is_some() && m.layers[4].is_some());
    assert_eq!(m.expert_bytes(0), 0);
    assert_eq!(m.expert_bytes(1), 3 * 32 * 64 / 32 * 18);
    // the fingerprint is the file's: same bytes, same print; other weights, another
    let again = ModelExperts::open(&p).unwrap();
    assert_eq!(m.fingerprint, again.fingerprint);
    let other = testgguf::temp_model("view-other", 8).unwrap();
    assert_ne!(m.fingerprint, ModelExperts::open(&other).unwrap().fingerprint);
    let _ = std::fs::remove_file(p);
    let _ = std::fs::remove_file(other);
}

#[test]
fn every_transport_returns_the_shared_kernel_bit_for_bit() {
    let (reference, p) = model("transports");
    let sock = std::env::temp_dir().join(format!("w2w-host-{}.sock", std::process::id()));
    for (i, addr) in ["inproc://host-bits", "tcp://127.0.0.1:0", &format!("unix://{}", sock.display())].into_iter().enumerate() {
        let wait = [Wait::Block, Wait::Busy, Wait::Spin(100)][i];
        let m = ModelExperts::open(&p).unwrap();
        let held = HeldSet::all(m.n_layer(), m.n_expert);
        let mut held = held;
        for e in 0..m.n_expert {
            held.set(0, e, false); // layer 0 is dense
        }
        let h = Arc::new(Host::new("t", m, held, Box::new(CpuBackend)).unwrap());
        let l = transport::listen(addr, wait).unwrap();
        let at = l.local();
        host::spawn(h.clone(), l);
        let mut c = HostClient::connect(&at, wait, Duration::from_secs(5)).unwrap();
        assert_eq!(c.info.fingerprint, reference.fingerprint);
        assert_eq!((c.info.d, c.info.n_ff, c.info.backend.as_str(), c.info.name.as_str()), (64, 32, "cpu", "t"));
        assert!(c.info.holds(1, 0) && !c.info.holds(0, 0));
        assert_eq!(c.info.n_held(), 4 * 8);
        let d = reference.d;
        // decode shape: one row, every expert on it
        let x1 = x_rows(1, d);
        let ids = [1u16, 4, 6];
        let g: Vec<&[u32]> = vec![&[0], &[0], &[0]];
        let RunReply::Ok { y, n, d: rd, .. } = c.run(2, &ids, &g, &x1, 1, d, Some(Instant::now() + Duration::from_secs(20))).unwrap() else { panic!("{addr}: not ok") };
        assert_eq!((n as usize, rd as usize, y.len()), (3, d, 3 * d));
        for (k, &e) in ids.iter().enumerate() {
            assert_eq!(bits(&y[k * d..(k + 1) * d]), bits(&local(&reference, 2, e as usize, &x1, &[0])), "{addr} decode expert {e}");
        }
        // prefill shape: five rows, uneven groups (one of them a single row, which takes the
        // kernel's fused path: the group shape is part of the answer)
        let x5 = x_rows(5, d);
        let ids = [0u16, 3, 7];
        let groups: Vec<Vec<u32>> = vec![vec![0, 1, 4], vec![2], vec![4, 3, 2, 1, 0]];
        let g: Vec<&[u32]> = groups.iter().map(|v| v.as_slice()).collect();
        let RunReply::Ok { y, .. } = c.run(4, &ids, &g, &x5, 5, d, Some(Instant::now() + Duration::from_secs(20))).unwrap() else { panic!("{addr}: not ok") };
        let mut at_row = 0;
        for (k, &e) in ids.iter().enumerate() {
            let want = local(&reference, 4, e as usize, &x5, &groups[k]);
            assert_eq!(bits(&y[at_row..at_row + want.len()]), bits(&want), "{addr} prefill expert {e}");
            at_row += want.len();
        }
        assert_eq!(at_row, y.len());
        assert_eq!(h.stats.runs.load(std::sync::atomic::Ordering::Relaxed), 2);
    }
    let _ = std::fs::remove_file(p);
    // the listener thread outlives the test: take its socket file away here
    let _ = std::fs::remove_file(sock);
}

#[test]
fn a_host_refuses_an_expert_it_does_not_hold() {
    let (m, p) = model("refuse");
    let d = m.d;
    let plan = PlacementMap::parse(r#"{"hosts":{"a":{"addr":"inproc://refuse-a"},"b":{"addr":"inproc://refuse-b"}},"layers":{"1-4":{"0-3":"a","4-7":"b"}}}"#, m.n_layer(), m.n_expert).unwrap();
    let h = Arc::new(Host::new("a", m, plan.held_by("a").unwrap(), Box::new(CpuBackend)).unwrap());
    let l = transport::listen("inproc://refuse-a", Wait::Block).unwrap();
    host::spawn(h.clone(), l);
    let mut c = HostClient::connect("inproc://refuse-a", Wait::Block, Duration::from_secs(5)).unwrap();
    let x = x_rows(1, d);
    let dl = || Some(Instant::now() + Duration::from_secs(20));
    // held: answered
    let g: Vec<&[u32]> = vec![&[0], &[0]];
    assert!(matches!(c.run(1, &[0, 3], &g, &x, 1, d, dl()).unwrap(), RunReply::Ok { .. }));
    // one of three not held: the whole request is refused and the missing expert is named;
    // no partial answer, no zeros
    let g: Vec<&[u32]> = vec![&[0], &[0], &[0]];
    assert_eq!(c.run(1, &[0, 5, 3], &g, &x, 1, d, dl()).unwrap(), RunReply::Refused { layer: 1, ids: vec![5] });
    // a layer with no experts, and a layer past the model: refused the same way
    let g: Vec<&[u32]> = vec![&[0]];
    assert_eq!(c.run(0, &[0], &g, &x, 1, d, dl()).unwrap(), RunReply::Refused { layer: 0, ids: vec![0] });
    assert_eq!(c.run(40, &[2], &g, &x, 1, d, dl()).unwrap(), RunReply::Refused { layer: 40, ids: vec![2] });
    // an expert id past the model
    assert_eq!(c.run(1, &[200], &g, &x, 1, d, dl()).unwrap(), RunReply::Refused { layer: 1, ids: vec![200] });
    // the wrong activation width is an error, not a computation
    assert!(matches!(c.run(1, &[0], &g, &x[..32], 1, 32, dl()).unwrap(), RunReply::Error(_)));
    assert_eq!(h.stats.refused.load(std::sync::atomic::Ordering::Relaxed), 4);
    // the placement changes under the running host: it now holds 5 and no longer 0
    let mut held = h.held();
    held.set(1, 5, true);
    held.set(1, 0, false);
    h.set_held(held).unwrap();
    assert!(matches!(c.run(1, &[5], &g, &x, 1, d, dl()).unwrap(), RunReply::Ok { .. }));
    assert_eq!(c.run(1, &[0], &g, &x, 1, d, dl()).unwrap(), RunReply::Refused { layer: 1, ids: vec![0] });
    c.refresh(Duration::from_secs(5)).unwrap();
    assert!(c.info.holds(1, 5) && !c.info.holds(1, 0));
    // a held set that names a layer without experts cannot be loaded at all
    let mut bad = h.held();
    bad.set(0, 1, true);
    assert!(h.set_held(bad).is_err());
    let _ = std::fs::remove_file(p);
}

/// A listener whose first accepts fail the way a bad RDMA rendezvous does.
struct FlakyListener {
    inner: Box<dyn lb_expert_host::transport::Listener>,
    fail: u32,
}

impl lb_expert_host::transport::Listener for FlakyListener {
    fn accept(&mut self) -> std::io::Result<Box<dyn lb_expert_host::transport::Conn>> {
        if self.fail > 0 {
            self.fail -= 1;
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "rendezvous line \"GET / HTTP/1.1\""));
        }
        self.inner.accept()
    }
    fn local(&self) -> String {
        self.inner.local()
    }
}

#[test]
fn a_bad_client_does_not_take_the_listener_down() {
    let (m, p) = model("flaky");
    let held = HeldSet::parse_spec("1-4:0-7", m.n_layer(), m.n_expert).unwrap();
    let h = Arc::new(Host::new("f", m, held, Box::new(CpuBackend)).unwrap());
    let inner = transport::listen("inproc://flaky-f", Wait::Block).unwrap();
    host::spawn(h, Box::new(FlakyListener { inner, fail: 3 }));
    // the three failed accepts are behind the listener; the next client is served
    let c = HostClient::connect("inproc://flaky-f", Wait::Block, Duration::from_secs(5)).unwrap();
    assert_eq!(c.info.name, "f");
    let _ = std::fs::remove_file(p);
}
