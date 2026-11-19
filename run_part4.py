"""
Part 4 -- test of the riskiest assumption: "the hydraulic-model signature of a leak matches the real one".

Test A (direct): for every real leak, measured mean residual during the full-size phase vs the model signature of the
                 SAME pipe at the SAME diameter and hours of day (cosine similarity + magnitude ratio).
Test B (end-to-end): localiser trained ONLY on synthetic leaks, evaluated on all 20 real leaks.
Success criterion fixed before looking at the results:
   A: cosine similarity >= 0.7 (whitened) for >= 12 of 20 leaks      B: first-choice accuracy >= 0.4 and
   mean search effort <= 0.25 (random ranking would give ~0.5; centroid baseline 0.26 in Part 2).
Run: python run_part4.py
"""
import numpy as np, pandas as pd
import leaklib as ll
import pipeline as pl

ctx = pl.Ctx()
idx, y, K = ctx.idx, ctx.y, ctx.K
free = (y == 0)
base = ll.Baseline().fit(ctx.data, free)          # uses only the 'no leak' mask, never a leak signature
hour_f = idx.hour.values + idx.minute.values / 60.0
R, _ = base.residuals(ctx.data)
Rw, _ = base.residuals(ctx.data, whiten=True)

rows = []
for _, r in ctx.leaks.iterrows():
    pi = ctx.sg_pipes.index(r['pipe_id'])
    # full-size phase: abrupt -> from start (skip first 6h of transient), incipient -> from peak to end
    a = r['start'] + pd.Timedelta(hours=6) if r['type'] == 'abrupt' else r['peak']
    m = np.flatnonzero((idx >= a) & (idx <= r['end']))
    scale = (r['diameter_m'] / 0.015) ** 2
    dP = pl.interp_sig(ctx.sg_sig[pi], hour_f[m]) * scale
    dF = (ctx.sg_dq[pi] * scale) * pl.FLOW_SHARE[None] * np.ones((len(m), 1))
    pred = base.delta_residual(dP, dF).mean(0)
    meas = R[m].mean(0)
    # use only steps with ONE leak running, so the measured signature is not a mixture
    only = np.flatnonzero(((idx >= a) & (idx <= r['end'])) & (ctx.dom_flow > 0))
    cos_raw = float(meas[:ll.NS] @ pred[:ll.NS] / (np.linalg.norm(meas[:ll.NS]) * np.linalg.norm(pred[:ll.NS]) + 1e-12))
    mw, pw = meas[:ll.NS] @ base.whiten, pred[:ll.NS] @ base.whiten
    cos_w = float(mw @ pw / (np.linalg.norm(mw) * np.linalg.norm(pw) + 1e-12))
    ratio = float(np.linalg.norm(meas[:ll.NS]) / np.linalg.norm(pred[:ll.NS]))
    # does the model identify the pipe's cluster by itself? rank of the true cluster by cosine to centroid
    rows.append(dict(leak_id=r['leak_id'], pipe=r['pipe_id'], type=r['type'], D_cm=100 * r['diameter_m'], cluster=r['cluster'],
                     cos_raw=cos_raw, cos_whitened=cos_w, magnitude_ratio=ratio,
                     measured_extra_inflow=float(meas[ll.NS:ll.NS + 3].sum()), predicted_extra_inflow=float(pred[ll.NS:ll.NS + 3].sum()),
                     leak_flow_m3h=float(r['peak_flow_m3h'])))
A = pd.DataFrame(rows)
A.to_csv(pl.ART + '/part4_cosine_test.csv', index=False)
print(A.round(2).to_string(index=False))
print('median cos (whitened) %.2f ; #leaks with cos_w>=0.7: %d / 20 ; median magnitude ratio %.2f' % (
    A.cos_whitened.median(), (A.cos_whitened >= 0.7).sum(), A.magnitude_ratio.median()))

# ---- Test B: synthetic-only localiser on all 20 real leaks
cfg = dict(mode='resid', channels='P', whiten=True, time=True, windows=[], unit=True, unit_window=6)
runs = pl.free_runs(free, 3 * 24 * ll.STEP)
X, _, _ = pl.feats_full(ctx, base, cfg)
out = []
for seed in range(3):
    Xs, ys, ws, cs, fs = pl.synth_segments(ctx, base, cfg, runs, 800, np.random.default_rng(seed), stride=12)
    k = fs >= 0.3
    sc, clf = pl.train_mlp(Xs[k], cs[k], None, hidden=(128, 64), alpha=1e-2, seed=seed)
    prob = pl.predict_proba_cluster(sc, clf, X, K)
    rows_ok = (y == 1) & (ctx.dom_frac >= 0.3)
    ev, per = pl.eval_localiser(prob, None, ctx, ctx.leaks, idx, rows=rows_ok)
    print('seed', seed, ev)
    out.append((ev, per))
ev_m = {k: float(np.mean([o[0][k] for o in out])) for k in ('top1', 'effort')}
print('synthetic-only localiser on 20 real leaks (mean of 3 seeds):', ev_m)
per = pd.DataFrame([(a[0], a[1], np.mean([o[1][i][2] for o in out]), np.mean([o[1][i][3] for o in out])) for i, a in enumerate(out[0][1])],
                   columns=['leak_id', 'n_rows', 'effort', 'top1'])
per = per.merge(A[['leak_id', 'cos_whitened', 'magnitude_ratio', 'D_cm', 'type']], on='leak_id')
per.to_csv(pl.ART + '/part4_synth_only_per_leak.csv', index=False)
print(per.round(2).to_string(index=False))
pd.DataFrame([dict(top1=ev_m['top1'], effort=ev_m['effort'])]).to_csv(pl.ART + '/part4_synth_only_summary.csv', index=False)
