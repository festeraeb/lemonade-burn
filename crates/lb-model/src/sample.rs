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
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if self.p.top_k > 0 {
            idx.truncate(self.p.top_k);
        }
        let t = self.p.temperature;
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
