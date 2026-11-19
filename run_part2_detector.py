"""
Part 2 (detector): compare input designs / window lengths / training data / non-neural baselines
with ONE fixed time split:  train = leaks L01-L12 (up to 2018-09-09),  validation = L13-L20 (after).
Run: python run_part2_detector.py
"""
import time, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import leaklib as ll
import pipeline as pl

ctx = pl.Ctx()
idx, y = ctx.idx, ctx.y
TRAIN_END = pd.Timestamp('2018-09-09')
tr = np.asarray(idx < TRAIN_END); va = ~tr
free = (y == 0)
leaks_va = ctx.leaks[ctx.leaks['start'] >= TRAIN_END].reset_index(drop=True)
print('train steps', tr.sum(), 'val steps', va.sum(), 'val leaks', list(leaks_va['leak_id']))

bases = {True: ll.Baseline(use_amr=True).fit(ctx.data, free & tr), False: ll.Baseline(use_amr=False).fit(ctx.data, free & tr)}
runs_tr = pl.free_runs(free & tr, 3 * 24 * ll.STEP)
print('train free runs (days):', [round((b - a) / 288, 1) for a, b in runs_tr])
rows_tr = np.flatnonzero(tr)[::6]
w_prior = np.ones(len(rows_tr))
w_prior[(y[rows_tr] == 1) & (ctx.dom_frac[rows_tr] < 0.10)] = 0.0   # ignore first hours of incipient leaks (no signal)


from sklearn.metrics import roc_auc_score
post = np.zeros(len(idx), bool)
for e in ctx.leaks['end']:
    post |= (idx > e) & (idx <= e + pd.Timedelta(hours=24))
vis = va & ~post & ((y == 0) | (ctx.dom_frac >= 0.3))        # rows where a leak of >=30% of its peak is running


def evaluate(p, name, t_train=None, nparams=None, extra=None):
    thr, n, m = pl.choose_rule(p[va], y[va], leaks_va, idx[va])
    rec = dict(design=name, AUROC=m['AUROC'], AUROC_visible=roc_auc_score(y[vis], p[vis]), detected=f"{m['detected']}/{m['n_leaks']}", delay_h=m['median_delay_h'],
               FA_per30=m['FA_days_per_30'], rule=f'p>={thr} for {n} steps', train_s=t_train, params=nparams)
    if extra: rec.update(extra)
    return rec


def mlp_design(name, cfg, synth=0, base_key=True, seed=0, hidden=(128, 64)):
    t0 = time.time()
    base = bases[base_key]
    X, R, Yr = pl.feats_full(ctx, base, cfg)
    Xtr, ytr = X[rows_tr], y[rows_tr]
    w = w_prior.copy()
    if synth:
        rng = np.random.default_rng(seed)
        Xs, ys, ws, cs, fs = pl.synth_segments(ctx, base, cfg, runs_tr, synth, rng)
        print(f'   [{name}] synthetic rows {len(Xs)} (positives)')
        Xtr = np.vstack([Xtr, Xs]); ytr = np.concatenate([ytr, ys]); w = np.concatenate([w, ws])
    wb = pl.balance_weights(ytr, w)
    sc, clf = pl.train_mlp(Xtr, ytr, wb, hidden=hidden, seed=seed)
    p = pl.predict_proba_binary(sc, clf, X)
    rec = evaluate(p, name, time.time() - t0, pl.count_params(clf), dict(n_in=X.shape[1], n_train=len(Xtr)))
    print(rec, flush=True)
    return rec, p


C = dict(time=True)
results = []
which = sys.argv[1] if len(sys.argv) > 1 else 'all'

if which in ('all', 'A'):
    results.append(mlp_design('A1 raw P, 1h mean', dict(mode='raw', channels='P', windows=[1], **C))[0])
    results.append(mlp_design('A2 raw P, 1h+6h+24h', dict(mode='raw', channels='P', windows=[1, 6, 24], **C))[0])
    results.append(mlp_design('A3 baseline-resid P, 1h+6h+24h', dict(mode='resid', channels='P', windows=[1, 6, 24], **C))[0])
    results.append(mlp_design('A4 + noise whitening', dict(mode='resid', channels='P', windows=[1, 6, 24], whiten=True, **C))[0])
    r5, p5 = mlp_design('A5 + flows & tank level', dict(mode='resid', channels='PF', windows=[1, 6, 24], whiten=True, **C))
    results.append(r5)
    results.append(mlp_design('A6 A5 baseline without AMR', dict(mode='resid', channels='PF', windows=[1, 6, 24], whiten=True, **C), base_key=False)[0])
    results.append(mlp_design('A7 A5 + synthetic leaks (300)', dict(mode='resid', channels='PF', windows=[1, 6, 24], whiten=True, **C), synth=300)[0])
    results.append(mlp_design('A9 A5 + norm features', dict(mode='resid', channels='PF', windows=[1, 6, 24], whiten=True, norms=True, **C))[0])
    results.append(mlp_design('A10 A5 + norm features + synthetic (300)', dict(mode='resid', channels='PF', windows=[1, 6, 24], whiten=True, norms=True, **C), synth=300)[0])
    results.append(mlp_design('A8 A5 smaller net (32)', dict(mode='resid', channels='PF', windows=[1, 6, 24], whiten=True, **C), hidden=(32,))[0])
    pd.DataFrame(results).to_csv(pl.ART + '/part2_detector_designs.csv', index=False)

if which == 'N3':
    res = []
    for nm, cfg in [('N3a resid P 1/3/6h + synth', dict(mode='resid', channels='P', windows=[1, 3, 6])),
                    ('N3b resid P 1/3/6h whitened + norms + synth', dict(mode='resid', channels='P', windows=[1, 3, 6], whiten=True, norms=True)),
                    ('N3c resid P 3h + synth, 3 hidden layers', dict(mode='resid', channels='P', windows=[3]))]:
        res.append(mlp_design(nm, dict(cfg, time=True), synth=300, hidden=(128, 64, 32) if 'N3c' in nm else (128, 64))[0])
    pd.DataFrame(res).to_csv(pl.ART + '/part2_detector_designs3.csv', index=False)

if which == 'N2':
    res = []
    for nm, cfg, sy in [('N2a raw P 1/6/24 + synth', dict(mode='raw', channels='P', windows=[1, 6, 24]), 300),
                        ('N2b raw P+F+L 1/6/24', dict(mode='raw', channels='PF', windows=[1, 6, 24]), 0),
                        ('N2c raw P+F+L 1/6/24 + synth', dict(mode='raw', channels='PF', windows=[1, 6, 24]), 300),
                        ('N2d resid P (no whitening) + synth', dict(mode='resid', channels='P', windows=[1, 6, 24]), 300),
                        ('N2e resid P+F+L (no whitening) + synth', dict(mode='resid', channels='PF', windows=[1, 6, 24]), 300)]:
        res.append(mlp_design(nm, dict(cfg, time=True), synth=sy)[0])
    pd.DataFrame(res).to_csv(pl.ART + '/part2_detector_designs2.csv', index=False)

if which == 'N':
    for n in (0, 300):
        mlp_design(f'N norm feats synth={n}', dict(mode='resid', channels='PF', windows=[1, 6, 24], whiten=True, norms=True, **C), synth=n)

BEST = dict(mode='resid', channels='P', windows=[1, 6, 24], time=True)      # chosen design (see report)
if which in ('all', 'W'):
    res = []
    for w in (1, 3, 6, 12, 24):
        res.append(mlp_design(f'W window={w}h (resid P, +synth)', dict(BEST, windows=[w]), synth=300)[0])
    res.append(mlp_design('W windows 1h+6h+24h (resid P, +synth)', BEST, synth=300)[0])
    pd.DataFrame(res).to_csv(pl.ART + '/part2_detector_windows.csv', index=False)

if which in ('all', 'B'):
    res = []
    base = bases[True]
    # B1: norm of the 6 h whitened residual (no learning besides baseline + noise covariance)
    R, _ = base.residuals(ctx.data, whiten=True)
    M6 = ll.causal_mean(R[:, :ll.NS], 6 * ll.STEP)
    sn = np.linalg.norm(M6, axis=1)
    res.append(evaluate(sn / (sn + ctx.noise_thr), 'B1 non-neural: whitened-residual norm (6h) threshold'))
    # B2: logistic regression on the SAME inputs and the SAME (real + synthetic) training data as the chosen MLP
    t0 = time.time()
    X, _, _ = pl.feats_full(ctx, base, BEST)
    Xs, ys, ws, cs, fs = pl.synth_segments(ctx, base, BEST, runs_tr, 300, np.random.default_rng(0))
    Xtr = np.vstack([X[rows_tr], Xs]); ytr = np.concatenate([y[rows_tr], ys]); w = np.concatenate([w_prior, ws])
    wb = pl.balance_weights(ytr, w); keep = wb > 0
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.1, max_iter=1000).fit(sc.transform(Xtr[keep]), ytr[keep], sample_weight=wb[keep])
    res.append(evaluate(lr.predict_proba(sc.transform(X))[:, 1], 'B2 non-neural: logistic regression, same inputs/data', time.time() - t0))
    for r in res: print(r)
    pd.DataFrame(res).to_csv(pl.ART + '/part2_detector_baselines.csv', index=False)
