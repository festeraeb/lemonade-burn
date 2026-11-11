//! Work-to-weights on the synthetic model (`lb_expert_host::testgguf`): the trunk with experts
//! on hosts gives the one-process path's logits **bit for bit** — all on one host, split over
//! two, with a host that dies or hangs mid-decode, with a host that refuses, with the placement
//! rewritten under it — and counts its hops.

use burn::tensor::Device;
use lb_expert_host::host::{self, CpuBackend, Host};
use lb_expert_host::placement::{HeldSet, PlacementMap, Wait};
use lb_expert_host::{testgguf, transport, ModelExperts};
use lb_lfm2::model::Lfm2;
use lb_lfm2::remote::PassRecord;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

const PROMPT: [u32; 9] = [1, 17, 40, 3, 88, 21, 60, 5, 33];
const STEPS: usize = 12;
/// MoE layers of the tiny model (layers 1..=4).
const MOE: u64 = 4;

fn tiny(tag: &str) -> PathBuf {
    testgguf::temp_model(&format!("lfm2-{tag}"), 11).unwrap()
}

fn bits(t: burn::tensor::Tensor<2>) -> Vec<u32> {
    t.into_data().try_to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect()
}

/// A prefill over the prompt (all logits) and `STEPS` teacher-forced decode tokens (the tiny
/// random model's greedy text is one token for ever, which would route the same experts every
/// step): every logit of every pass as bits, and each pass's top-1. `between(step)` runs before
/// each decode step.
fn run(model: &Lfm2, mut between: impl FnMut(usize)) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut cache = model.new_cache();
    let mut all = Vec::new();
    let logits = model.forward(&PROMPT, &mut cache, true);
    let mut top = vec![*Lfm2::argmax_rows(&logits).last().unwrap()];
    all.push(bits(logits));
    for step in 0..STEPS {
        between(step);
        let next = (7 + 13 * step as u32) % 96;
        let l = model.forward(&[next], &mut cache, false);
        top.push(Lfm2::argmax_rows(&l)[0]);
        all.push(bits(l));
    }
    (all, top)
}

fn reference(path: &Path) -> (Vec<Vec<u32>>, Vec<u32>) {
    let model = Lfm2::load(path, &Device::default()).unwrap();
    let r = run(&model, |_| {});
    // the decode steps must not all be the same computation
    assert!(r.0[1..].iter().collect::<std::collections::HashSet<_>>().len() == STEPS, "decode steps repeat");
    r
}

/// A CPU host for `path` holding what `placement` gives `name`, listening on `addr`.
fn start_host(path: &Path, name: &str, placement: &str, addr: &str) -> (Arc<Host>, String) {
    let m = ModelExperts::open(path).unwrap();
    let held = PlacementMap::parse(placement, m.n_layer(), m.n_expert).unwrap().held_by(name).unwrap();
    let h = Arc::new(Host::new(name, m, held, Box::new(CpuBackend)).unwrap());
    let l = transport::listen(addr, Wait::Block).unwrap();
    let at = l.local();
    host::spawn(h.clone(), l);
    (h, at)
}

fn decode_records(recs: &[PassRecord]) -> Vec<&PassRecord> {
    recs.iter().filter(|r| r.m == 1).collect()
}

#[test]
fn all_experts_on_one_host_is_bit_identical_and_one_hop_a_layer() {
    let path = tiny("one");
    let want = reference(&path);
    let placement = r#"{"hosts":{"a":{"addr":"inproc://w2w-one","deadline_ms":20000}},"layers":{"1-4":"a"}}"#;
    let (h, _) = start_host(&path, "a", placement, "inproc://w2w-one");
    let mut model = Lfm2::load(&path, &Device::default()).unwrap();
    let rt = model.attach_remote_json(placement).unwrap();
    assert_eq!(rt.n_placed(), 4 * 8);
    let got = run(&model, |_| {});
    assert_eq!(got.1, want.1, "greedy tokens");
    assert!(got.0 == want.0, "logits differ from the one-process path");
    let recs = rt.take_records();
    assert_eq!(recs.len(), 1 + STEPS);
    // prefill: one message per MoE layer for the whole prompt
    assert_eq!((recs[0].m, recs[0].hops), (PROMPT.len(), MOE));
    for r in decode_records(&recs) {
        assert_eq!((r.hops, r.experts_remote, r.experts_fallback), (MOE, MOE * 3, 0), "{r:?}");
        assert!(r.hop_wall_ms > 0.0 && r.host_ms > 0.0 && r.down.is_empty());
    }
    // nothing ran locally: the trunk's own CPU expert counter never moved
    assert_eq!(model.stats.cpu_calls.load(Ordering::Relaxed), 0);
    assert_eq!(h.stats.runs.load(Ordering::Relaxed), MOE * (1 + STEPS as u64));
    assert_eq!(rt.stats.fallback_experts.load(Ordering::Relaxed), 0);
    let _ = std::fs::remove_file(path);
}

#[test]
fn split_over_two_hosts_and_the_trunk_is_bit_identical() {
    let path = tiny("split");
    let want = reference(&path);
    let sock = std::env::temp_dir().join(format!("w2w-split-{}.sock", std::process::id()));
    // a over TCP, b over a unix socket, expert 7 of every layer and all of layer 3 stay local
    let (ha, at_a) = start_host(&path, "a", r#"{"hosts":{"a":{"addr":"x"}},"layers":{"1-4":{"0-3":"a"},"3":"local"}}"#, "tcp://127.0.0.1:0");
    let (hb, at_b) = start_host(&path, "b", r#"{"hosts":{"b":{"addr":"x"}},"layers":{"1-4":{"4-6":"b"},"3":"local"}}"#, &format!("unix://{}", sock.display()));
    let placement = format!(
        r#"{{"hosts":{{"a":{{"addr":"{at_a}","deadline_ms":20000,"wait":"spin:50"}},"b":{{"addr":"{at_b}","deadline_ms":20000}}}},
            "layers":{{"1-4":{{"0-3":"a","4-6":"b","7":"local"}},"3":"local"}}}}"#
    );
    let mut model = Lfm2::load(&path, &Device::default()).unwrap();
    let rt = model.attach_remote_json(&placement).unwrap();
    let got = run(&model, |_| {});
    assert_eq!(got.1, want.1, "greedy tokens");
    assert!(got.0 == want.0, "logits differ from the one-process path");
    let (ra, rb) = (ha.stats.experts.load(Ordering::Relaxed), hb.stats.experts.load(Ordering::Relaxed));
    let local = model.stats.cpu_calls.load(Ordering::Relaxed);
    assert!(ra > 0 && rb > 0 && local > 0, "a {ra} b {rb} local {local}");
    let recs = rt.take_records();
    for r in decode_records(&recs) {
        // three remote layers, at most two hosts each; top-3 over the three homes
        assert!(r.hops >= 1 && r.hops <= 6 && r.experts_fallback == 0, "{r:?}");
    }
    assert_eq!(recs.iter().map(|r| r.experts_remote).sum::<u64>(), ra + rb);
    assert!(rt.hosts().iter().all(|h| h.up));
    let _ = std::fs::remove_file(path);
}

#[test]
fn a_host_that_dies_or_hangs_mid_decode_is_marked_down_and_the_answer_does_not_change() {
    let path = tiny("kill");
    let want = reference(&path);
    for (case, hang) in [("dies", false), ("hangs", true)] {
        let place_a = r#"{"hosts":{"a":{"addr":"x"}},"layers":{"1-4":{"0-3":"a"}}}"#;
        let place_b = r#"{"hosts":{"b":{"addr":"x"}},"layers":{"1-4":{"4-7":"b"}}}"#;
        let (ha, at_a) = start_host(&path, "a", place_a, "tcp://127.0.0.1:0");
        let (hb, at_b) = start_host(&path, "b", place_b, "tcp://127.0.0.1:0");
        // a hung host is found by the deadline: keep it short for b, generous for a
        let placement = format!(
            r#"{{"hosts":{{"a":{{"addr":"{at_a}","deadline_ms":20000}},"b":{{"addr":"{at_b}","deadline_ms":300}}}},
                "layers":{{"1-4":{{"0-3":"a","4-7":"b"}}}}}}"#
        );
        let mut model = Lfm2::load(&path, &Device::default()).unwrap();
        let rt = model.attach_remote_json(&placement).unwrap();
        let kill_at = 5;
        let t = std::time::Instant::now();
        let got = run(&model, |step| {
            if step == kill_at {
                if hang {
                    hb.mute.store(true, Ordering::Relaxed);
                } else {
                    hb.die.store(true, Ordering::Relaxed);
                }
            }
        });
        assert_eq!(got.1, want.1, "{case}: greedy tokens");
        assert!(got.0 == want.0, "{case}: logits differ from the one-process path");
        // never a hang: the hung host costs one deadline, not one per layer or per token
        assert!(t.elapsed() < std::time::Duration::from_secs(10), "{case}: {:?}", t.elapsed());
        let hosts = rt.hosts();
        let (a, b) = (hosts.iter().find(|h| h.name == "a").unwrap(), hosts.iter().find(|h| h.name == "b").unwrap());
        assert!(a.up, "{case}: a must stay up");
        assert!(!b.up, "{case}: b must be marked down");
        let why = b.reason.clone().unwrap();
        if hang {
            assert!(why.contains("no reply"), "{case}: {why}");
            assert_eq!(rt.stats.timeouts.load(Ordering::Relaxed), 1);
        } else {
            assert!(why.contains("connection lost"), "{case}: {why}");
        }
        assert_eq!(rt.stats.mark_downs.load(Ordering::Relaxed), 1, "{case}");
        let recs = rt.take_records();
        let dec = decode_records(&recs);
        // before the kill b answers and nothing falls back; from the kill on b's experts run here
        assert!(dec[..kill_at].iter().all(|r| r.experts_fallback == 0 && r.down.is_empty()), "{case}");
        assert!(dec[kill_at].down == vec!["b".to_string()], "{case}: {:?}", dec[kill_at]);
        let fell: u64 = dec[kill_at..].iter().map(|r| r.experts_fallback).sum();
        assert!(fell > 0, "{case}");
        assert_eq!(model.stats.cpu_calls.load(Ordering::Relaxed), fell, "{case}: every fallback ran on the trunk's own path");
        // after the mark-down the dead host is not asked again
        let asked_b_after = hb.stats.runs.load(Ordering::Relaxed);
        assert!(dec[kill_at + 1..].iter().all(|r| r.hops <= MOE), "{case}");
        let _ = (ha, asked_b_after);
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn a_refusing_host_a_wrong_model_and_an_absent_host_all_fall_back_to_the_right_answer() {
    let path = tiny("refuse");
    let want = reference(&path);
    // the host holds less than the trunk's placement says: experts 4-7 are refused
    let (h, at) = start_host(&path, "a", r#"{"hosts":{"a":{"addr":"x"}},"layers":{"1-4":{"0-3":"a"}}}"#, "tcp://127.0.0.1:0");
    let placement = format!(r#"{{"hosts":{{"a":{{"addr":"{at}","deadline_ms":20000}}}},"layers":{{"1-4":"a"}}}}"#);
    let mut model = Lfm2::load(&path, &Device::default()).unwrap();
    let rt = model.attach_remote_json(&placement).unwrap();
    let got = run(&model, |_| {});
    assert!(got.0 == want.0 && got.1 == want.1, "refusing host: output differs");
    assert!(rt.stats.refusals.load(Ordering::Relaxed) > 0 && h.stats.refused.load(Ordering::Relaxed) > 0);
    assert!(rt.stats.fallback_experts.load(Ordering::Relaxed) > 0);
    assert!(rt.hosts()[0].up, "a refusal is not a reason to mark a host down");
    assert_eq!(rt.stats.mark_downs.load(Ordering::Relaxed), 0);

    // a host that serves another model file is never used
    let other = testgguf::temp_model("lfm2-refuse-other", 12).unwrap();
    let (ho, at_o) = start_host(&other, "a", r#"{"hosts":{"a":{"addr":"x"}},"layers":{"1-4":"a"}}"#, "tcp://127.0.0.1:0");
    // and a host nobody started
    let placement = format!(r#"{{"hosts":{{"a":{{"addr":"{at_o}"}},"ghost":{{"addr":"tcp://127.0.0.1:9"}}}},"layers":{{"1-2":"a","3-4":"ghost"}}}}"#);
    let mut model = Lfm2::load(&path, &Device::default()).unwrap();
    let rt = model.attach_remote_json(&placement).unwrap();
    let hosts = rt.hosts();
    assert!(hosts.iter().all(|h| !h.up), "{hosts:?}");
    assert!(hosts.iter().find(|h| h.name == "a").unwrap().reason.as_ref().unwrap().contains("different model"));
    let got = run(&model, |_| {});
    assert!(got.0 == want.0 && got.1 == want.1, "down hosts: output differs");
    assert_eq!(ho.stats.runs.load(Ordering::Relaxed), 0);
    assert_eq!(rt.stats.hops.load(Ordering::Relaxed), 0);
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(other);
}

#[test]
fn the_placement_file_is_reloaded_while_the_model_runs() {
    let path = tiny("reload");
    let want = reference(&path);
    let m = ModelExperts::open(&path).unwrap();
    let h = Arc::new(Host::new("a", m, HeldSet::parse_spec("1-4:*", 5, 8).unwrap(), Box::new(CpuBackend)).unwrap());
    let l = transport::listen("tcp://127.0.0.1:0", Wait::Block).unwrap();
    let at = l.local();
    host::spawn(h.clone(), l);
    let file = std::env::temp_dir().join(format!("w2w-placement-{}.json", std::process::id()));
    let hosts = format!(r#""hosts":{{"a":{{"addr":"{at}","deadline_ms":20000}}}}"#);
    std::fs::write(&file, format!("{{{hosts},\"layers\":{{}}}}")).unwrap();
    let mut model = Lfm2::load(&path, &Device::default()).unwrap();
    let rt = model.attach_remote(&file).unwrap();
    assert_eq!(rt.n_placed(), 0);
    let got = run(&model, |step| {
        // the file's mtime must move and the tier looks at it every 250 ms
        if step == 3 {
            std::thread::sleep(std::time::Duration::from_millis(1100));
            std::fs::write(&file, format!("{{{hosts},\"layers\":{{\"1-4\":\"a\"}}}}")).unwrap();
        }
        if step == 8 {
            std::thread::sleep(std::time::Duration::from_millis(1100));
            std::fs::write(&file, "{ this is not json").unwrap();
        }
    });
    assert!(got.0 == want.0 && got.1 == want.1, "reload: output differs");
    let recs = rt.take_records();
    let dec = decode_records(&recs);
    assert!(dec[..3].iter().all(|r| r.hops == 0), "before the rewrite everything is local");
    assert!(dec[3..].iter().all(|r| r.hops == MOE), "after it every MoE layer is one hop: {:?}", dec.iter().map(|r| r.hops).collect::<Vec<_>>());
    // the broken file was noticed and refused; the good map stayed in force
    assert_eq!(rt.stats.reloads.load(Ordering::Relaxed), 1);
    assert_eq!(rt.n_placed(), 32);
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(file);
}
