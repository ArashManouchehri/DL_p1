"""
Part 5 -- cross-validated performance estimate + final model.

  python train_final.py cv      4 time-blocks, out-of-fold predictions -> alarm rule + expected performance
  python train_final.py final   fit on all of 2018, save model/ (weights, preprocessing, thresholds)
"""
import os, sys, json, time
import numpy as np, pandas as pd, joblib
import leaklib as ll
import pipeline as pl

DET_CFG = dict(mode='resid', channels='P', windows=[1, 3, 6], whiten=True, norms=True, time=True)
LOC_CFG = dict(mode='resid', channels='P', windows=[], whiten=True, time=True, unit=True, unit_window=6)
N_SYN_DET, N_SYN_LOC, SEEDS = 300, 600, (0, 1)
MODEL_DIR = os.path.join(pl.HERE, 'model')


def fit_models(ctx, train_mask, seeds=SEEDS, verbose=True):
    """Fit baseline + detector + localiser on the rows where train_mask is True (nothing else is touched)."""
    idx, y = ctx.idx, ctx.y
    free = (y == 0)
    base = ll.Baseline().fit(ctx.data, free & train_mask)
    runs = pl.free_runs(free & train_mask, 3 * 24 * ll.STEP)
    models = dict(base=base, det_cfg=DET_CFG, loc_cfg=LOC_CFG, K=ctx.K, det=[], loc=[])
    Xd, _, _ = pl.feats_full(ctx, base, DET_CFG)
    Xl, _, _ = pl.feats_full(ctx, base, LOC_CFG)
    rows = np.flatnonzero(train_mask)[::6]
    w0 = np.ones(len(rows)); w0[(y[rows] == 1) & (ctx.dom_frac[rows] < 0.10)] = 0.0
    real_loc = np.flatnonzero(train_mask & (y == 1) & (ctx.dom_frac >= 0.3))[::6]
    for s in seeds:
        t0 = time.time()
        rng = np.random.default_rng(100 + s)
        Xs, ys, ws, _, _ = pl.synth_segments(ctx, base, DET_CFG, runs, N_SYN_DET, rng)
        Xt = np.vstack([Xd[rows], Xs]); yt = np.concatenate([y[rows], ys]); wt = np.concatenate([w0, ws])
        sc, clf = pl.train_mlp(Xt, yt, pl.balance_weights(yt, wt), seed=s)
        models['det'].append((sc, clf))
        d_params, d_n = pl.count_params(clf), len(Xt)
        Xs, ys, ws, cs, fs = pl.synth_segments(ctx, base, LOC_CFG, runs, N_SYN_LOC, rng, stride=12)
        k = fs >= 0.3
        Xt = np.vstack([Xl[real_loc], Xs[k]]); ct = np.concatenate([ctx.dom_cluster[real_loc], cs[k]])
        sc, clf = pl.train_mlp(Xt, ct, None, alpha=1e-2, seed=s)
        models['loc'].append((sc, clf))
        if verbose:
            print(f'   seed {s}: detector {d_n} rows/{d_params} params, localiser {len(Xt)} rows/{pl.count_params(clf)} params, {time.time()-t0:.0f}s', flush=True)
    return models


def summarize(ctx, p, prob, rule, mask=None, leaks=None):
    idx, y = ctx.idx, ctx.y
    leaks = ctx.leaks if leaks is None else leaks
    thr, n = rule
    alarm = ll.alarm_rule(p, thr, n)
    m = ll.evaluate_detection(p, alarm, y, leaks, idx)
    ev, per = pl.eval_localiser(prob, alarm, ctx, leaks, idx)
    m.update(search_effort=ev['effort'], top1=ev['top1'])
    return m, per, alarm


BLOCKS = ['2018-01-01', '2018-03-14', '2018-07-15', '2018-10-05', '2019-01-01']

if __name__ == '__main__':
    mode = sys.argv[1]
    ctx = pl.Ctx()
    idx, y = ctx.idx, ctx.y
    if mode == 'cv':
        p_oof = np.full(ctx.n, np.nan); prob_oof = np.zeros((ctx.n, ctx.K))
        for f in range(4):
            lo, hi = pd.Timestamp(BLOCKS[f]), pd.Timestamp(BLOCKS[f + 1])
            blk = np.asarray((idx >= lo) & (idx < hi))
            margin = np.asarray((idx >= lo - pd.Timedelta(hours=48)) & (idx < hi + pd.Timedelta(hours=48)))
            print(f'fold {f}: test {lo.date()}..{hi.date()}  train rows {(~margin).sum()}', flush=True)
            models = fit_models(ctx, ~margin)
            p, prob = ll.predict_all(models, ctx.data)
            p_oof[blk] = p[blk]; prob_oof[blk] = prob[blk]
        np.save(pl.ART + '/oof_p.npy', p_oof); np.save(pl.ART + '/oof_prob.npy', prob_oof)
        thr, n, m = pl.choose_rule(p_oof, y, ctx.leaks, idx, fa_max=1.0)
        print('alarm rule from OOF predictions: p >= %.2f for %d steps' % (thr, n))
        m, per, alarm = summarize(ctx, p_oof, prob_oof, (thr, n))
        print('OOF overall:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
        pd.DataFrame(per, columns=['leak_id', 'n_alarm_rows', 'effort', 'top1']).to_csv(pl.ART + '/cv_per_leak.csv', index=False)
        json.dump(dict(thr=thr, n=n, overall=m), open(pl.ART + '/cv_summary.json', 'w'), default=float, indent=1)
        # per-fold and bootstrap intervals
        rows = []
        for f in range(4):
            lo, hi = pd.Timestamp(BLOCKS[f]), pd.Timestamp(BLOCKS[f + 1])
            sel = np.asarray((idx >= lo) & (idx < hi))
            lk = ctx.leaks[(ctx.leaks['start'] >= lo) & (ctx.leaks['start'] < hi)].reset_index(drop=True)
            al = alarm * sel
            mm = ll.evaluate_detection(np.where(sel, p_oof, 0.0)[sel], al[sel], y[sel], lk, idx[sel])
            ev, _ = pl.eval_localiser(prob_oof, al, ctx, lk, idx)
            rows.append(dict(fold=f, period=f'{lo.date()}..{hi.date()}', n_leaks=len(lk), AUROC=mm['AUROC'], detected=mm['detected'],
                             median_delay_h=mm['median_delay_h'], FA_days_per_30=mm['FA_days_per_30'],
                             search_effort=ev['effort'], top1=ev['top1']))
        F = pd.DataFrame(rows); F.to_csv(pl.ART + '/cv_folds.csv', index=False)
        print(F.round(3).to_string(index=False))
        # bootstrap over leaks (detection, delay, effort, top1) and over days (AUROC, false alarms)
        rng = np.random.default_rng(0)
        perdf = pd.DataFrame(per, columns=['leak_id', 'n', 'effort', 'top1']).set_index('leak_id')
        det_ok, delays = {}, {}
        for _, r in ctx.leaks.iterrows():
            mk = np.asarray((idx >= r['start']) & (idx <= r['end']))
            h = np.flatnonzero(alarm[mk])
            det_ok[r['leak_id']] = len(h) > 0
            delays[r['leak_id']] = (idx[mk][h[0]] - r['start']).total_seconds() / 3600 if len(h) else np.nan
        ids = list(ctx.leaks['leak_id']); B = []
        from sklearn.metrics import roc_auc_score
        days = idx.floor('D'); udays = np.unique(days)
        day_rows = {d: np.flatnonzero(days == d) for d in udays}
        elig = (y == 0)
        for e in ctx.leaks['end']: elig &= ~((idx > e) & (idx <= e + pd.Timedelta(hours=24)))
        for b in range(300):
            s = rng.choice(ids, len(ids))
            dd = rng.choice(udays, len(udays))
            rr = np.concatenate([day_rows[d] for d in dd])
            fa = pd.Series(alarm[rr][elig[rr]], index=days[rr][elig[rr]])
            fa_rate = fa.groupby(level=0).max().mean() * 30 if len(fa) else np.nan
            dl = [delays[i] for i in s if det_ok[i]]
            B.append(dict(AUROC=roc_auc_score(y[rr], p_oof[rr]), detected=np.sum([det_ok[i] for i in s]),
                          median_delay_h=np.median(dl) if dl else np.nan, FA=fa_rate,
                          effort=perdf.loc[s, 'effort'].mean(), top1=perdf.loc[s, 'top1'].mean()))
        B = pd.DataFrame(B)
        ci = pd.DataFrame({'mean': B.mean(), 'lo5': B.quantile(0.05), 'hi95': B.quantile(0.95)})
        ci.to_csv(pl.ART + '/cv_bootstrap_ci.csv'); print(ci.round(3))
    elif mode == 'final':
        os.makedirs(MODEL_DIR, exist_ok=True)
        cvs = json.load(open(pl.ART + '/cv_summary.json'))
        models = fit_models(ctx, np.ones(ctx.n, bool))
        models['rule'] = dict(thr=cvs['thr'], n_persist=int(cvs['n']))
        joblib.dump(models, os.path.join(MODEL_DIR, 'model.joblib'), compress=3)
        pl_p, pl_prob = ll.predict_all(models, ctx.data)
        m, per, _ = summarize(ctx, pl_p, pl_prob, (cvs['thr'], cvs['n']))
        print('in-sample (optimistic!) metrics on 2018:', m)
        params = {'detector_params_per_member': pl.count_params(models['det'][0][1]), 'localiser_params_per_member': pl.count_params(models['loc'][0][1]),
                  'members': len(models['det'])}
        json.dump(params, open(os.path.join(MODEL_DIR, 'model_info.json'), 'w'))
        print(params)
