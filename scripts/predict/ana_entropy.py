import os
import numpy as np, json
R = os.environ.get("PRED_RES", "predict-results")
def ld(n): return dict(np.load(f"{R}/feat-{n}.npz"))
out = {}
F = {n: ld(n) for n in ("TRAIN", "TEST_A", "TEST_B")}
H = np.concatenate([F[n]["hit"] for n in F]); T, L, E = H.shape
cnt = H.sum(0).astype(np.float64); p = cnt / cnt.sum(1, keepdims=True)
ent = -(p * np.log2(np.where(p > 0, p, 1))).sum(1)
srt = -np.sort(-p, 1)
top = {m: srt[:, :m].sum(1) for m in (4, 8, 16, 32)}
# token-to-token overlap, same layer
Hs = F["TRAIN"]["hit"]; pr = F["TRAIN"]["prompt"]
same = (pr[1:] == pr[:-1])
ov = ((Hs[1:] & Hs[:-1]).sum(2) / 4.0)[same].mean(0)
# prompt-specific distributions: JS divergence of per-prompt expert distribution from global, per layer
def js(a, b):
    m = (a + b) / 2
    f = lambda x, y: (x * np.log2(np.where(x > 0, x / np.where(y > 0, y, 1), 1))).sum(-1)
    return 0.5 * f(a, m) + 0.5 * f(b, m)
pj = np.zeros(L); n = 0
for q in range(pr.max() + 1):
    c = Hs[pr == q].sum(0).astype(np.float64); c = c / c.sum(1, keepdims=True)
    g = (cnt - Hs[pr == q].sum(0)); g = g / g.sum(1, keepdims=True)
    pj += js(c, g); n += 1
pj /= n
# per-prompt top-8 overlap with the rest
ovp = np.zeros(L)
for q in range(pr.max() + 1):
    c = Hs[pr == q].sum(0); g = cnt - c
    a = np.argsort(-c, 1)[:, :8]; b = np.argsort(-g, 1)[:, :8]
    ovp += np.array([len(set(a[j]) & set(b[j])) / 8 for j in range(L)])
ovp /= (pr.max() + 1)
# entropy of a prompt's own distribution (how concentrated within one topic)
pe = np.zeros(L)
for q in range(pr.max() + 1):
    c = Hs[pr == q].sum(0).astype(np.float64); c = c / c.sum(1, keepdims=True)
    pe += -(c * np.log2(np.where(c > 0, c, 1))).sum(1)
pe /= (pr.max() + 1)
# the cache's misses: which experts miss? frequency class of the missed expert (share of tokens it was used in, per layer)
Hc = F["TEST_A"]["hit"]; Cc = F["TEST_A"]["cached"]
freq = cnt / T                                    # P(expert used in a token), per layer
miss = Hc & ~Cc
fr = np.broadcast_to(freq[None], Hc.shape)[miss]
print("layer  H(bits)  eff.n  top4  top8  top16  tokenOverlap  JS(prompt|rest)  top8overlap(prompt,rest)  H(prompt)")
for j in range(L):
    print(f"{j+2:4d}  {ent[j]:.2f}  {2**ent[j]:5.1f}  {top[4][j]:.2f}  {top[8][j]:.2f}  {top[16][j]:.2f}   {ov[j]:.2f}   {pj[j]:.3f}   {ovp[j]:.2f}   {pe[j]:.2f}")
print("mean   %.2f %5.1f %.2f %.2f %.2f  %.2f  %.3f %.2f %.2f" % (ent.mean(), (2**ent).mean(), top[4].mean(), top[8].mean(), top[16].mean(), ov.mean(), pj.mean(), ovp.mean(), pe.mean()))
print("max entropy log2(64) = 6.00; uniform top-4 mass 0.0625")
print("TEST_A misses: n/token %.1f; expert use freq of a missed expert: median %.3f mean %.3f; share of misses on experts used in <2%% of tokens: %.2f, <5%%: %.2f, <10%%: %.2f" % (miss.sum() / Hc.shape[0], np.median(fr), fr.mean(), (fr < .02).mean(), (fr < .05).mean(), (fr < .10).mean()))
json.dump({"ent": ent.tolist(), "top4": top[4].tolist(), "top8": top[8].tolist(), "top16": top[16].tolist(), "overlap_prev_token": ov.tolist(), "js": pj.tolist(), "top8_overlap_prompt_rest": ovp.tolist(), "ent_prompt": pe.tolist()}, open(f"{R}/entropy.json", "w"))
