"""
pipeline.py -- training-side library: data context, synthetic leaks, MLP training, evaluation.
(predict.py only needs leaklib.py + the files in model/)
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import leaklib as ll

HERE, ART = ll.HERE, os.path.join(ll.HERE, 'artifacts')
NS = ll.NS


# =============================================================== context
class Ctx:
    def __init__(self, folder='historical_2018'):
        f = os.path.join(HERE, folder)
        self.data = ll.load_folder(f)
        self.idx = self.data['P'].index
        self.n = len(self.idx)
        self.y = pd.read_csv(os.path.join(f, 'labels.csv'), index_col=0, parse_dates=True)['leak_active'].values
        self.leaks = pd.read_csv(os.path.join(f, 'leaks.csv'), parse_dates=['start', 'peak', 'end'])
        cl = pd.read_csv(os.path.join(HERE, 'clusters.csv'))
        self.clusters = cl
        self.K = int(cl['cluster'].max()) + 1
        self.pipe_cluster = dict(zip(cl['pipe_id'], cl['cluster']))
        self.sizes = np.bincount(cl['cluster'].values, minlength=self.K)
        self.leaks['cluster'] = self.leaks['pipe_id'].map(self.pipe_cluster)
        # dominant leak per step (largest current leak flow) -> localiser label
        lf = pd.read_csv(os.path.join(f, 'leak_flows_m3h.csv'), index_col=0, parse_dates=True)
        arr = lf.values
        self.dom_flow = arr.max(1)
        dom_pipe = np.array(lf.columns)[arr.argmax(1)]
        self.dom_cluster = np.array([self.pipe_cluster[p] for p in dom_pipe])
        peak = dict(zip(self.leaks['pipe_id'], self.leaks['peak_flow_m3h']))
        self.dom_frac = self.dom_flow / np.array([peak[p] for p in dom_pipe])
        self.dom_frac[self.dom_flow <= 0] = 0
        # synthetic-leak library
        sg = np.load(os.path.join(ART, 'signatures.npz'))
        self.sg_pipes = list(sg['pipes'])
        self.sg_sig = sg['pipe_sig']                       # (905,24,33)
        self.sg_dq = sg['pipe_dq']
        det = pd.read_csv(os.path.join(ART, 'pipe_details.csv')).set_index('pipe_id')
        self.sg_z = det.loc[self.sg_pipes, 'z_norm_1p5cm'].values
        self.sg_cluster = np.array([self.pipe_cluster[p] for p in self.sg_pipes])
        self.noise_thr = json.load(open(os.path.join(ART, 'cluster_info.json')))['noise_thr_95']


def free_runs(mask_free, min_steps):
    """list of (a,b) half-open index ranges where mask_free is True for >= min_steps."""
    ch = np.flatnonzero(np.diff(mask_free.astype(int), prepend=0, append=0))
    return [(a, b) for a, b in zip(ch[::2], ch[1::2]) if b - a >= min_steps]


# =============================================================== features
def make_R(ctx, base, cfg):
    return base.residuals(ctx.data, whiten=cfg.get('whiten', False))


def feats_full(ctx, base, cfg, raw_mean=None):
    R, Y = make_R(ctx, base, cfg)
    c = dict(cfg)
    if c.get('mode') == 'raw':
        c['raw_mean'] = raw_mean if raw_mean is not None else base.y_mean
    return ll.build_features(R, ctx.idx, c, Yraw=Y), R, Y


def interp_sig(sig24, hour_f):
    """sig24 (24,33) hourly table -> (n,33) at fractional hours (circular linear interpolation)."""
    i0 = np.floor(hour_f).astype(int) % 24
    w = (hour_f - np.floor(hour_f))[:, None]
    return sig24[i0] * (1 - w) + sig24[(i0 + 1) % 24] * w


FLOW_SHARE = np.array([0.05, 0.475, 0.475])


def synth_segments(ctx, base, cfg, runs, n_leaks, rng, stride=6, min_hours=24, raw_mean=None,
                   min_scale_z=0.8, max_days=10, pools=None):
    """Create synthetic leaks inside leak-free runs.  Returns X, label_detect, weight_detect, cluster, frac."""
    R0, Y0 = make_R(ctx, base, cfg)
    c = dict(cfg)
    if c.get('mode') == 'raw':
        c['raw_mean'] = raw_mean if raw_mean is not None else base.y_mean
    Wh = base.whiten if cfg.get('whiten', False) else np.eye(NS)
    cand = np.flatnonzero(ctx.sg_z >= 0.0) if pools is None else pools
    Xs, lab, wts, cls, frs = [], [], [], [], []
    hour_f = ctx.idx.hour.values + ctx.idx.minute.values / 60.0
    runs = [(a, b) for a, b in runs if (b - a) >= 3 * 24 * ll.STEP]
    for _ in range(n_leaks):
        a, b = runs[rng.integers(len(runs))]
        for _try in range(50):
            pi = rng.choice(cand)
            D = rng.uniform(0.009, 0.024)
            if ctx.sg_z[pi] * (D / 0.015) ** 2 >= min_scale_z * ctx.noise_thr:
                break
        scale = (D / 0.015) ** 2
        st = a + 24 * ll.STEP + rng.integers(0, max(1, (b - a) - 48 * ll.STEP))
        ln = min(b - st, int(rng.uniform(1, max_days) * 24 * ll.STEP))
        if ln < 12 * ll.STEP:
            continue
        kind = rng.random() < 0.5                      # True: incipient
        ramp = np.ones(ln)
        if kind:
            rl = int(rng.uniform(2, 10) * 24 * ll.STEP)
            ramp = np.minimum(1.0, np.arange(ln) / rl)
        # segment: from run start (expanding windows, as at the start of a file) to the end of the leak
        lo, hi = a, st + ln
        dP = np.zeros((hi - lo, NS)); dF = np.zeros((hi - lo, 3))
        sg = interp_sig(ctx.sg_sig[pi], hour_f[st:st + ln]) * scale * ramp[:, None]
        dP[st - lo:] = sg
        dF[st - lo:] = (ctx.sg_dq[pi] * scale * ramp)[:, None] * FLOW_SHARE[None]
        R = R0[lo:hi].copy(); Y = Y0[lo:hi].copy()
        dR = base.delta_residual(dP, dF)
        dR[:, :NS] = dR[:, :NS] @ Wh
        R += dR
        Y[:, :NS] += dP; Y[:, NS:NS + 3] += dF
        X = ll.build_features(R, ctx.idx[lo:hi], c, Yraw=Y)
        sel = np.arange(st - lo, hi - lo, stride)
        Xs.append(X[sel]); r = ramp[sel - (st - lo)]
        lab.append(np.ones(len(sel))); wts.append((r >= 0.1).astype(float))
        cls.append(np.full(len(sel), ctx.sg_cluster[pi])); frs.append(r)
    return (np.vstack(Xs), np.concatenate(lab), np.concatenate(wts), np.concatenate(cls), np.concatenate(frs))


# =============================================================== MLPs
def train_mlp(X, y, w=None, hidden=(128, 64), alpha=1e-3, seed=0, max_iter=150, lr=1e-3):
    sc = StandardScaler().fit(X)
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', alpha=alpha, batch_size=256,
                        learning_rate_init=lr, max_iter=max_iter, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=8, random_state=seed)
    if w is not None:
        keep = w > 0
        clf.fit(sc.transform(X[keep]), y[keep], sample_weight=w[keep])
    else:
        clf.fit(sc.transform(X), y)
    return sc, clf


def predict_proba_binary(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, list(clf.classes_).index(1)]


def predict_proba_cluster(sc, clf, X, K):
    pr = clf.predict_proba(sc.transform(X))
    out = np.zeros((len(X), K))
    out[:, clf.classes_] = pr
    return out


def balance_weights(y, w=None):
    """class-balanced sample weights (both classes carry equal total weight); w is a prior weight (0 = ignore)."""
    w = np.ones(len(y)) if w is None else w.astype(float).copy()
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    w[y == 1] *= 0.5 / max(pos, 1e-9)
    w[y == 0] *= 0.5 / max(neg, 1e-9)
    return w * (w > 0).sum() / w.sum()


def count_params(clf):
    return int(sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_))


# =============================================================== alarm rule + evaluation
RULE_GRID = [(t, n) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95) for n in (1, 6, 12, 36, 72, 144)]


def choose_rule(p, y, leaks, idx, fa_max=2.0):
    best = None
    rows = []
    for thr, n in RULE_GRID:
        al = ll.alarm_rule(p, thr, n)
        m = ll.evaluate_detection(p, al, y, leaks, idx)
        rows.append((thr, n, m['detected'], m['median_delay_h'], m['FA_days_per_30']))
        ok = m['FA_days_per_30'] <= fa_max
        key = (ok, m['detected'] if ok else -m['FA_days_per_30'], -(m['median_delay_h'] if not np.isnan(m['median_delay_h']) else 1e9))
        if best is None or key > best[0]:
            best = (key, thr, n, m)
    return best[1], best[2], best[3]


def eval_localiser(prob, alarm, ctx, leaks, idx, rows=None):
    """prob (n,K); per leak average over its alarm steps; undetected -> effort 1, top1 0 (project metric)."""
    efforts, top1, per = [], [], []
    for _, r in leaks.iterrows():
        m = (idx >= r['start']) & (idx <= r['end'])
        sel = np.flatnonzero(m & (alarm == 1)) if rows is None else np.flatnonzero(m & rows)
        if len(sel) == 0:
            efforts.append(1.0); top1.append(0.0); per.append((r['leak_id'], np.nan, 1.0, 0.0)); continue
        e = np.mean([ll.search_effort(prob[i], r['cluster'], ctx.sizes) for i in sel])
        t = np.mean(prob[sel].argmax(1) == r['cluster'])
        efforts.append(e); top1.append(t); per.append((r['leak_id'], len(sel), e, t))
    return dict(effort=float(np.mean(efforts)), top1=float(np.mean(top1))), per
