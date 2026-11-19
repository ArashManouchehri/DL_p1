"""
Part 2 (localiser): compare input designs / training data / a non-neural baseline.
Same time split as the detector (train: L01-L12, validation: L13-L20).
The localiser is evaluated at the steps where the (chosen) detector raises an alarm.
Run: python run_part2_localiser.py
"""
import time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import leaklib as ll
import pipeline as pl

ctx = pl.Ctx()
idx, y, K = ctx.idx, ctx.y, ctx.K
TRAIN_END = pd.Timestamp('2018-09-09')
tr = np.asarray(idx < TRAIN_END); va = ~tr
free = (y == 0)
leaks_va = ctx.leaks[ctx.leaks['start'] >= TRAIN_END].reset_index(drop=True)
base = ll.Baseline().fit(ctx.data, free & tr)
runs_tr = pl.free_runs(free & tr, 3 * 24 * ll.STEP)

# ---- detector alarm on the validation period (chosen design N3b, trained on the training period only)
DET = dict(mode='resid', channels='P', windows=[1, 3, 6], whiten=True, norms=True, time=True)
Xd, _, _ = pl.feats_full(ctx, base, DET)
rows_tr = np.flatnonzero(tr)[::6]
w0 = np.ones(len(rows_tr)); w0[(y[rows_tr] == 1) & (ctx.dom_frac[rows_tr] < 0.10)] = 0
Xs, ys, ws, _, _ = pl.synth_segments(ctx, base, DET, runs_tr, 300, np.random.default_rng(0))
Xt = np.vstack([Xd[rows_tr], Xs]); yt = np.concatenate([y[rows_tr], ys]); wt = np.concatenate([w0, ws])
sc, clf = pl.train_mlp(Xt, yt, pl.balance_weights(yt, wt))
p = pl.predict_proba_binary(sc, clf, Xd)
thr, n, m = pl.choose_rule(p[va], y[va], leaks_va, idx[va])
alarm = ll.alarm_rule(p, thr, n) * va
print('detector rule p>=%.2f for %d steps -> detected %d/%d, alarm steps in val: %d (true-leak alarm steps: %d)' % (
    thr, n, m['detected'], m['n_leaks'], alarm.sum(), (alarm & (y == 1)).sum()))

# ---- real training rows for the localiser: leak running at >=30% of its peak flow, label = cluster of the dominant leak
real_rows = np.flatnonzero(tr & (y == 1) & (ctx.dom_frac >= 0.3))[::6]
print('real training rows', len(real_rows), 'classes present', sorted(set(ctx.dom_cluster[real_rows])))


def run(name, cfg, use_real=True, n_synth=0, hidden=(128, 64), alpha=1e-2, seed=0):
    t0 = time.time()
    X, _, _ = pl.feats_full(ctx, base, cfg)
    Xl, yl = [], []
    if use_real:
        Xl.append(X[real_rows]); yl.append(ctx.dom_cluster[real_rows])
    if n_synth:
        Xs, ys, ws, cs, fs = pl.synth_segments(ctx, base, cfg, runs_tr, n_synth, np.random.default_rng(seed), stride=12)
        k = fs >= 0.3
        Xl.append(Xs[k]); yl.append(cs[k])
    Xl, yl = np.vstack(Xl), np.concatenate(yl)
    sc, clf = pl.train_mlp(Xl, yl, None, hidden=hidden, alpha=alpha, seed=seed)
    prob = pl.predict_proba_cluster(sc, clf, X, K)
    ev, per = pl.eval_localiser(prob, alarm, ctx, leaks_va, idx)
    rec = dict(design=name, top1=ev['top1'], search_effort=ev['effort'], n_train=len(Xl), n_in=X.shape[1],
               params=pl.count_params(clf), train_s=time.time() - t0)
    print(rec, flush=True)
    return rec, prob, per


res = []
L = dict(mode='resid', channels='P', whiten=True, time=True)
res.append(run('L1 whitened resid 6h, real only', dict(L, windows=[6]))[0])
res.append(run('L2 unit-vector + log-norm of 6h resid, real only', dict(L, windows=[], unit=True, unit_window=6))[0])
res.append(run('L3 = L2 + synthetic leaks', dict(L, windows=[], unit=True, unit_window=6), n_synth=600)[0])
res.append(run('L4 whitened resid 6h+24h + synthetic', dict(L, windows=[6, 24]), n_synth=600)[0])
res.append(run('L5 unit 6h + 24h unit-free windows + synthetic', dict(L, windows=[24], unit=True, unit_window=6), n_synth=600)[0])
res.append(run('L6 = L3 synthetic ONLY (no real)', dict(L, windows=[], unit=True, unit_window=6), use_real=False, n_synth=600)[0])
res.append(run('L7 = L3 with no whitening', dict(mode='resid', channels='P', time=True, windows=[], unit=True, unit_window=6), n_synth=600)[0])

# ---- non-neural baseline: nearest centroid (cosine, whitened) using centroids from the hydraulic model only
C = np.load(pl.ART + '/cluster_centroids.npy')
# centroids were built for the clusters at clustering time; rebuild them from the CURRENT clusters.csv
sg = np.load(pl.ART + '/signatures.npz')
FLOW_SHARE = pl.FLOW_SHARE
sig_mean = sg['pipe_sig'].mean(1)
sig_res = np.stack([base.delta_residual(sig_mean[i:i + 1], sg['pipe_dq'][i] * FLOW_SHARE[None])[0, :ll.NS] for i in range(len(sig_mean))])
Zs = sig_res @ base.whiten
Zs /= np.linalg.norm(Zs, axis=1, keepdims=True)
cents = np.stack([Zs[ctx.sg_cluster == k].mean(0) for k in range(K)]); cents /= np.linalg.norm(cents, axis=1, keepdims=True)
R, _ = base.residuals(ctx.data, whiten=True)
M6 = ll.causal_mean(R[:, :ll.NS], 6 * ll.STEP)
sc_ = (M6 / (np.linalg.norm(M6, axis=1, keepdims=True) + 1e-9)) @ cents.T
prob = np.exp(10 * sc_); prob /= prob.sum(1, keepdims=True)
ev, per = pl.eval_localiser(prob, alarm, ctx, leaks_va, idx)
res.append(dict(design='B3 non-neural: nearest cluster centroid (cosine, whitened, from model only)', top1=ev['top1'], search_effort=ev['effort']))
print(res[-1])
out = pd.DataFrame(res)
out.to_csv(pl.ART + '/part2_localiser_designs.csv', index=False)
print(out.round(3).to_string(index=False))
