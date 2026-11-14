//! Prompt-prefix reuse: checkpoint a sequence's state and rewind to it.
//!
//! The KV cache is positional (`[kv_heads, cap, head_dim]`, valid for `0..len`), so going back
//! to an earlier position needs no copy: the rows past it are simply overwritten by the next
//! pass. The conv layers are different: each keeps a rolling history of the last
//! `l_cache - 1` inputs, which only describes the *current* position. A [`Checkpoint`] is that
//! history at one position (a few hundred kB for LFM2.5), which is all a later
//! [`Lfm2::rewind`] needs to continue from there.
//!
//! A server keeps the last conversation's `Cache`, its tokens and a checkpoint at each prompt
//! end; a new request that shares a prefix rewinds to the latest checkpoint inside the shared
//! part and runs only the rest of its prompt.

use crate::model::{Cache, Lfm2};
use crate::weights::Operator;

/// The conv history of every conv layer at sequence position `pos`.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub pos: usize,
    /// Per layer: `(l_cache - 1) * n_embd` values for a conv layer that has run, else `None`.
    conv: Vec<Option<Vec<f32>>>,
}

impl Checkpoint {
    /// Bytes this checkpoint holds.
    pub fn bytes(&self) -> usize {
        self.conv.iter().flatten().map(|v| v.len() * 4).sum()
    }
}

impl Lfm2 {
    /// Record the conv history at the cache's current position. Reads only the conv state
    /// (from the device when the device is ahead of the host), never the KV cache.
    pub fn checkpoint(&self, cache: &mut Cache) -> Result<Checkpoint, String> {
        let cfg = self.cfg();
        let n_layer = cfg.n_layer;
        let mut conv: Vec<Option<Vec<f32>>> = vec![None; n_layer];
        #[cfg(feature = "gpu")]
        if cache.host_len < cache.len {
            let (e, l) = (cfg.n_embd, cfg.l_cache);
            let gpu = self.store.gpu.as_ref().ok_or("device state without a device")?;
            let dev = cache.dev.as_ref().ok_or("host behind with no device copy")?;
            let want: Vec<_> = (0..n_layer).filter_map(|li| dev.conv[li].clone().map(|h| (li, h))).collect();
            let bytes = gpu.read_raw(want.iter().map(|(_, h)| (h.clone(), (l - 1) * e * 4)).collect())?;
            for ((li, _), b) in want.iter().zip(bytes) {
                conv[*li] = Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
            }
            return Ok(Checkpoint { pos: cache.len, conv });
        }
        for (li, layer) in self.w.layers.iter().enumerate() {
            if let Operator::Conv { in_proj, .. } = &layer.op {
                #[cfg(feature = "gpu")]
                if let (Some(gpu), Some(block)) = (self.store.gpu.as_ref(), self.dense.conv.get(&in_proj.start)) {
                    if self.dense.is_conv_dirty(in_proj.start) {
                        conv[li] = Some(block.read_state(gpu)?);
                        continue;
                    }
                }
                let _ = in_proj;
                conv[li] = cache.conv[li].as_ref().map(|t| Self::to_vec(t.clone()));
            }
        }
        Ok(Checkpoint { pos: cache.len, conv })
    }

    /// Put `cache` back at `cp.pos` (which must not be past the cache's position). The KV rows
    /// of `0..cp.pos` are kept; the conv history is restored from the checkpoint. The device
    /// copy, if any, is dropped and re-uploaded from the host by the next device pass.
    pub fn rewind(&self, cache: &mut Cache, cp: &Checkpoint) -> Result<(), String> {
        if cp.pos > cache.len {
            return Err(format!("checkpoint at {} is past the sequence's {}", cp.pos, cache.len));
        }
        let cfg = self.cfg();
        let (e, l) = (cfg.n_embd, cfg.l_cache);
        // the host must hold the KV rows before any device copy is dropped
        #[cfg(feature = "gpu")]
        crate::device_decode::pull_if_stale(self, cache);
        for (li, layer) in self.w.layers.iter().enumerate() {
            if let Operator::Conv { in_proj, .. } = &layer.op {
                cache.conv[li] = cp.conv[li].as_ref().map(|v| self.from_vec(v.clone(), l - 1, e));
                #[cfg(feature = "gpu")]
                if let (Some(gpu), Some(block)) = (self.store.gpu.as_ref(), self.dense.conv.get(&in_proj.start)) {
                    match &cp.conv[li] {
                        Some(v) => block.set_state(gpu, v)?,
                        None => block.reset(gpu),
                    }
                    self.dense.mark_conv_clean(in_proj.start);
                }
                let _ = in_proj;
            }
        }
        cache.len = cp.pos;
        cache.host_len = cp.pos;
        #[cfg(feature = "gpu")]
        {
            cache.dev = None;
        }
        Ok(())
    }
}

/// Length of the longest common prefix of two token sequences.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Which reusable position to continue from, given the positions of the kept checkpoints
/// (ascending), the length of the kept sequence and of the new prompt, and their common prefix.
/// At least one prompt token must still be run (its logits start the answer), so the position is
/// at most `prompt_len - 1`. Returns `(position, Some(index of the checkpoint to rewind to))`, or
/// `(seq_len, None)` when the whole kept sequence is a prefix of the prompt (continue as is), or
/// `None` when nothing can be reused.
pub fn reuse_point(cp_pos: &[usize], seq_len: usize, prompt_len: usize, common: usize) -> Option<(usize, Option<usize>)> {
    let limit = common.min(prompt_len.saturating_sub(1));
    if seq_len > 0 && seq_len <= limit {
        return Some((seq_len, None));
    }
    cp_pos.iter().rposition(|&p| p > 0 && p <= limit).map(|i| (cp_pos[i], Some(i)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_prefix_counts_equal_leading_tokens() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4, 5]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[7, 8], &[7, 8]), 2);
    }

    #[test]
    fn reuse_point_picks_the_latest_usable_state() {
        // kept: prompt of 10, answer to 25; checkpoints at 10 (prompt end)
        // next turn repeats all 25 and adds more: continue as is
        assert_eq!(reuse_point(&[10], 25, 40, 25), Some((25, None)));
        // the client dropped the reasoning: only the first prompt is shared
        assert_eq!(reuse_point(&[10], 25, 40, 12), Some((10, Some(0))));
        // the same prompt again: one token must still run, so the prompt-end checkpoint is out
        assert_eq!(reuse_point(&[10], 25, 10, 10), None);
        // two turns kept, the edit is inside the second
        assert_eq!(reuse_point(&[10, 30], 45, 50, 33), Some((30, Some(1))));
        assert_eq!(reuse_point(&[10, 30], 45, 50, 29), Some((10, Some(0))));
        // nothing shared
        assert_eq!(reuse_point(&[10, 30], 45, 50, 3), None);
        assert_eq!(reuse_point(&[], 0, 5, 0), None);
    }
}
