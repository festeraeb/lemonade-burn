//! Multi-token-prediction speculative decoding on the CPU reference: the model's own nextn head drafts `k` tokens, the trunk verifies them
//! in one round (here token by token, so the trunk arithmetic is exactly plain decode's), the longest matching prefix is kept and the state
//! rolls back to the last accepted token. Greedy only. The device loop (`pipeline.rs`) follows the same round structure.

use crate::{LayerState, Model, Mtp, MtpState, State};

pub fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

#[derive(Default, Debug, Clone)]
pub struct SpecStats {
    pub rounds: usize,
    pub drafted: usize,
    /// accepted_at[i] = rounds in which draft i+1 (and all before it) was accepted
    pub accepted_at: Vec<usize>,
    pub emitted: usize,
}

/// Plain greedy decode of `n` tokens after `prompt`.
pub fn plain_greedy(m: &Model<'_>, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut st = m.new_state();
    let mut l = None;
    for &t in prompt {
        l = m.forward(&mut st, t, true);
    }
    let mut out = Vec::new();
    let mut next = argmax(&l.unwrap());
    for _ in 0..n {
        out.push(next);
        let l = m.forward(&mut st, next, true).unwrap();
        next = argmax(&l);
    }
    out
}

fn lin_snapshot(st: &State) -> Vec<Option<LayerState>> {
    st.layers.iter().map(|l| if let LayerState::Lin { .. } = l { Some(l.clone()) } else { None }).collect()
}

/// MTP-speculative greedy decode of `n` tokens after `prompt` with draft depth `k` (>= 1). Returns the tokens and the round statistics.
pub fn mtp_greedy(m: &Model<'_>, mtp: &Mtp<'_>, prompt: &[u32], n: usize, k: usize) -> (Vec<u32>, SpecStats) {
    let c = &m.cfg;
    let kv = c.n_head_kv * c.head_dim;
    let mut st = m.new_state();
    let mut hiddens: Vec<Vec<f32>> = Vec::new(); // trunk hidden of every consumed position
    let mut l = None;
    for &t in prompt {
        l = m.forward(&mut st, t, true);
        hiddens.push(st.last_hidden.clone());
    }
    let mut ms = MtpState::default();
    // the head's cache: entry j = (H_j, token j+1); the prompt's entries are built now, the last one with the first generated token as `x`
    let mut x = argmax(&l.unwrap());
    for j in 0..prompt.len() - 1 {
        m.mtp_step(mtp, &mut ms, &hiddens[j], prompt[j + 1], j, true);
    }
    let mut out: Vec<u32> = Vec::new();
    let mut stats = SpecStats { accepted_at: vec![0; k], ..Default::default() };
    while out.len() < n {
        let p = st.pos; // x is the token at position p; the trunk has consumed 0..p
        // draft
        let h = st.last_hidden.clone();
        let (lg, mut hid) = m.mtp_step(mtp, &mut ms, &h, x, p - 1, true); // true entry p-1
        let keep_len = ms.k.len();
        let mut drafts = vec![argmax(&lg)];
        for d in 1..k {
            let (lg, hn) = m.mtp_step(mtp, &mut ms, &hid, drafts[d - 1], p - 1 + d, true);
            hid = hn;
            drafts.push(argmax(&lg));
        }
        ms.k.truncate(keep_len);
        ms.v.truncate(keep_len);
        // verify: x, d1..dk through the trunk, one by one, keeping the recurrent state after each
        let attn_len: Vec<usize> = st.layers.iter().map(|l| if let LayerState::Attn { k, .. } = l { k.len() } else { 0 }).collect();
        let mut snaps = Vec::new();
        let mut hs = Vec::new();
        let mut g = Vec::new();
        let toks: Vec<u32> = std::iter::once(x).chain(drafts.iter().cloned()).collect();
        for &t in &toks {
            let lg = m.forward(&mut st, t, true).unwrap();
            g.push(argmax(&lg));
            hs.push(st.last_hidden.clone());
            snaps.push(lin_snapshot(&st));
        }
        let mut a = 0;
        while a < k && drafts[a] == g[a] {
            a += 1;
        }
        stats.rounds += 1;
        stats.drafted += k;
        for i in 0..a {
            stats.accepted_at[i] += 1;
        }
        // emit g[0..=a]: g[i] is the token after toks[i]; toks[0] = x itself was already known
        out.push(x);
        out.extend_from_slice(&g[..a]);
        // roll back to the state after consuming toks[..=a] (positions p..=p+a)
        let sn = &snaps[a];
        for (il, l) in st.layers.iter_mut().enumerate() {
            match l {
                LayerState::Lin { .. } => *l = sn[il].clone().unwrap(),
                LayerState::Attn { k, v } => {
                    k.truncate(attn_len[il] + (a + 1) * kv);
                    v.truncate(attn_len[il] + (a + 1) * kv);
                }
            }
        }
        st.pos = p + a + 1;
        st.last_hidden = hs[a].clone();
        // the head's true entries for positions p..p+a-1: (H_{p+j}, token p+j+1 = toks[j+1])
        for j in 0..a {
            m.mtp_step(mtp, &mut ms, &hs[j], toks[j + 1], p + j, true);
        }
        x = g[a];
        stats.emitted = out.len();
    }
    out.truncate(n);
    (out, stats)
}
