//! Token sampling (greedy, temperature, top-k, top-p) on host logits.

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0x5eed }
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

const CUTOFF_NATS: f32 = 20.0;

pub struct Sampler {
    p: SamplingParams,
    state: u64,
}

impl Sampler {
    pub fn new(p: SamplingParams) -> Self {
        let state = p.seed.max(1);
        Self { p, state }
    }

    fn uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.p.temperature <= 0.0 {
            return argmax(logits);
        }
        // Only logits within CUTOFF_NATS * t of the maximum can carry probability above e^-CUTOFF_NATS
        // (2e-9 of the top token): drop the rest before sorting. The old code sorted and exponentiated all
        // vocabulary entries (65k for LFM2.5) on every token, ~13 ms, which halved tok/s at temperature > 0.
        let t = self.p.temperature;
        let mut mx = f32::NEG_INFINITY;
        for &v in logits {
            if v > mx {
                mx = v;
            }
        }
        let floor = mx - CUTOFF_NATS * t;
        let mut idx: Vec<usize> = (0..logits.len()).filter(|&i| logits[i] >= floor).collect();
        if self.p.top_k > 0 && idx.len() > self.p.top_k {
            idx.select_nth_unstable_by(self.p.top_k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
            idx.truncate(self.p.top_k);
        }
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let mx = logits[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - mx) / t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        if self.p.top_p < 1.0 {
            let mut c = 0.0;
            let mut keep = probs.len();
            for (i, p) in probs.iter().enumerate() {
                c += p;
                if c >= self.p.top_p {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            let s: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= s);
        }
        let r = self.uniform();
        let mut c = 0.0;
        for (i, p) in probs.iter().enumerate() {
            c += p;
            if r < c {
                return idx[i] as u32;
            }
        }
        idx[probs.len() - 1] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(p: &SamplingParams, logits: &[f32], r: f32) -> u32 {
        // the previous full-vocabulary implementation, with the random draw supplied
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if p.top_k > 0 {
            idx.truncate(p.top_k);
        }
        let mx = logits[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - mx) / p.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= sum);
        if p.top_p < 1.0 {
            let (mut c, mut keep) = (0.0, probs.len());
            for (i, x) in probs.iter().enumerate() {
                c += x;
                if c >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            let s: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|x| *x /= s);
        }
        let mut c = 0.0;
        for (i, x) in probs.iter().enumerate() {
            c += x;
            if r < c {
                return idx[i] as u32;
            }
        }
        idx[probs.len() - 1] as u32
    }

    #[test]
    fn matches_full_vocab_sampler_for_the_same_draw() {
        let mut g = 12345u64;
        let mut next = || {
            g ^= g << 13;
            g ^= g >> 7;
            g ^= g << 17;
            ((g >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for (temp, top_k, top_p) in [(0.2, 0, 1.0), (0.8, 40, 1.0), (1.0, 0, 0.9), (0.7, 20, 0.95)] {
            let p = SamplingParams { temperature: temp, top_p, top_k, seed: 7 };
            for _ in 0..200 {
                let logits: Vec<f32> = (0..4096).map(|_| next() * 6.0).collect();
                let mut s = Sampler::new(p.clone());
                let mut probe = Sampler::new(p.clone());
                let r = probe.uniform();
                assert_eq!(s.sample(&logits), reference(&p, &logits, r));
            }
        }
    }
}
