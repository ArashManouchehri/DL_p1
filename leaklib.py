"""
leaklib.py -- shared code for training and for predict.py.

Everything here is CAUSAL: the feature vector at time t only uses data at times <= t.
All fitted parameters (baseline regression, scalers) are fitted on training data only and
are stored in model/ ; predict.py never refits anything.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SENSORS = ['n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188', 'n215', 'n229', 'n288', 'n296', 'n332', 'n342',
           'n410', 'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516', 'n519', 'n549', 'n613', 'n636', 'n644',
           'n679', 'n722', 'n726', 'n740', 'n752', 'n769']
FLOWS = ['PUMP_1', 'p227', 'p235']
STEP = 12                      # 12 five-minute steps = 1 hour
CH = SENSORS + FLOWS + ['T1']  # the 37 "channels" we build residuals for
NS = len(SENSORS)


# ------------------------------------------------------------------ loading
def _read(folder, name):
    f = os.path.join(folder, name)
    if not os.path.exists(f):
        return None
    return pd.read_csv(f, index_col=0, parse_dates=True)


def load_folder(folder):
    """Return dict of DataFrames P,F,L,D on the index of the pressure file (missing files -> NaN)."""
    P = _read(folder, 'pressures_m.csv')
    idx = P.index
    out = {'P': P.reindex(columns=SENSORS)}
    F = _read(folder, 'flows_m3h.csv')
    L = _read(folder, 'levels_m.csv')
    D = _read(folder, 'demands_Lh.csv')
    out['F'] = (F.reindex(idx).reindex(columns=FLOWS) if F is not None else pd.DataFrame(np.nan, index=idx, columns=FLOWS))
    out['L'] = (L.reindex(idx).reindex(columns=['T1']) if L is not None else pd.DataFrame(np.nan, index=idx, columns=['T1']))
    out['D'] = D.reindex(idx) if D is not None else pd.DataFrame(np.nan, index=idx, columns=['none'])
    return out


def clean_pressure(P):
    """Sensor dropouts show up as 0 / NaN / absurd values; mark them missing (-> baseline later)."""
    P = P.copy()
    P[(P <= 1.0)] = np.nan
    return P


# ------------------------------------------------------------------ baseline
def time_regressors(idx, D, use_amr=True):
    """Hour-of-day harmonics, weekend flag and the (causal) 30-min mean of the total AMR demand
    (NaN where the AMR file is missing; Baseline replaces NaN by the training mean)."""
    h = idx.hour.values + idx.minute.values / 60.0
    cols = [np.sin(2 * np.pi * k * h / 24) for k in range(1, 5)] + [np.cos(2 * np.pi * k * h / 24) for k in range(1, 5)]
    cols.append((idx.day_of_week.values >= 5).astype(float))
    if use_amr:
        tot = (D.sum(axis=1, min_count=1).ffill().rolling(6, min_periods=1).mean() / 1000.0).values
        cols.append(tot)
    return np.stack(cols, 1)


class Baseline:
    """'Normal behaviour' of the sensors, fitted by ridge regression on leak-free steps of the TRAINING period.

      pressures (33): regressors = time of day (4 harmonics), weekend, AMR demand, pump/pipe inflows, tank level
                      -> the residual is the pressure change NOT explained by the current hydraulic state,
                         i.e. the local signature of a leak (extra inflow is removed from the pressure).
      inflows(3)+level(1): regressors = time of day, weekend, AMR demand only
                      -> the residual is 'extra inflow / tank drawdown' relative to expected demand.
    """

    def __init__(self, alpha=1.0, use_amr=True):
        self.alpha = alpha
        self.use_amr = use_amr

    # -- design matrices ------------------------------------------------------------------
    def _T(self, data):
        T = time_regressors(data['P'].index, data['D'], self.use_amr)
        if hasattr(self, 'mu_T'):
            T = np.where(np.isnan(T), self.mu_T, T)
        return T

    def _fl(self, data):
        FL = np.hstack([data['F'].values, data['L'].values])
        if hasattr(self, 'mu_FL'):
            FL = np.where(np.isnan(FL), self.mu_FL, FL)
        return FL

    def _design(self, data):
        T, FL = self._T(data), self._fl(data)
        Xq = (T - self.mu_T) / self.sd_T
        Xp = np.hstack([Xq, (FL - self.mu_FL) / self.sd_FL])
        one = np.ones((len(T), 1))
        return np.hstack([Xp, one]), np.hstack([Xq, one])

    def fit(self, data, mask):
        P = data['P'].pipe(clean_pressure).values
        FL = self._fl(data); T = self._T(data)
        Y = np.hstack([P, FL])
        m = mask & ~np.isnan(Y).any(1) & ~np.isnan(T).any(1)
        self.mu_T, self.sd_T = T[m].mean(0), T[m].std(0) + 1e-9
        self.mu_FL, self.sd_FL = FL[m].mean(0), FL[m].std(0) + 1e-9
        self.y_mean = np.nanmean(Y[m], axis=0)
        Xp, Xq = self._design(data)
        def ridge(X, Yt):
            reg = self.alpha * np.eye(X.shape[1]); reg[-1, -1] = 0
            return np.linalg.solve(X[m].T @ X[m] + reg, X[m].T @ Yt[m])
        self.Wp = ridge(Xp, P)                       # (nT+4+1, 33)
        self.Wq = ridge(Xq, FL)                      # (nT+1, 4)
        Rm, _ = self.residuals(data)
        self.noise_sd = Rm[m].std(0)
        # --- noise whitening of the 33 pressure residuals.  The day-to-day noise is ~99% two common-mode
        # directions, so shape information lives in the small directions.  Sigma = covariance of the
        # 6-hour-mean residual over leak-free windows; a 0.03 m floor keeps the transform bounded.
        W6 = 6 * STEP
        M6 = causal_mean(Rm[:, :NS], W6)
        fw = pd.Series(mask.astype(float)).rolling(W6, min_periods=W6).min().values == 1
        S = np.cov(M6[fw][::STEP].T)
        ev, V = np.linalg.eigh(S)
        self.whiten = (V / np.sqrt(np.maximum(ev, 0) + 0.03 ** 2)) @ V.T
        self.noise6_cov = S
        return self

    def residuals(self, data, whiten=False):
        """-> R (n,37): [33 pressure residuals | 3 inflow residuals | tank-level residual], Y raw channels (NaN filled)."""
        P = data['P'].pipe(clean_pressure).values
        FL = np.hstack([data['F'].values, data['L'].values])
        Xp, Xq = self._design(data)
        Rp = np.nan_to_num(P - Xp @ self.Wp, nan=0.0)      # missing sensor -> 'as expected' (no information)
        Rq = np.nan_to_num(FL - Xq @ self.Wq, nan=0.0)
        R = np.hstack([Rp, Rq])
        if hasattr(self, 'noise_sd'):
            R = np.clip(R, -10 * self.noise_sd - 5, 10 * self.noise_sd + 5)
        Y = np.where(np.isnan(np.hstack([P, FL])), self.y_mean, np.hstack([P, FL]))
        if whiten:
            R = R.copy(); R[:, :NS] = R[:, :NS] @ self.whiten
        return R, Y

    def delta_residual(self, dP, dF):
        """Change of the residual channels (unwhitened) caused by adding a leak effect dP (n,33), dF (n,3)."""
        nT = len(self.mu_T)
        Wf = self.Wp[nT:nT + 3]                            # pressure-regression weights of the 3 inflows
        dRp = dP - (dF / self.sd_FL[:3]) @ Wf
        return np.hstack([dRp, dF, np.zeros((len(dF), 1))])


# ------------------------------------------------------------------ features
def causal_mean(A, w):
    """Mean of the last w rows (or all rows so far, at the start of the series: no future used)."""
    C = np.cumsum(np.vstack([np.zeros((1, A.shape[1])), A]), axis=0)
    n = len(A)
    t = np.arange(1, n + 1)
    lo = np.maximum(t - w, 0)
    return (C[t] - C[lo]) / (t - lo)[:, None]


def build_features(R, idx, cfg, Yraw=None):
    """R: (n,37) residual channels. cfg keys:
         windows  : list of window lengths in HOURS (causal means)
         channels : 'P' (33 pressures), 'PF' (+3 flows +level)
         mode     : 'resid' (baseline-subtracted) or 'raw' (pressures minus a fixed mean; needs Yraw)
         norms    : add log-norm of the pressure deviation (and of the inflow deviation) per window
         time     : add sin/cos hour & weekend
         unit     : also return pressure part of the longest window as unit vector + log-norm (localiser)"""
    nch = NS if cfg.get('channels', 'PF') == 'P' else len(CH)
    A = R[:, :nch] if cfg.get('mode', 'resid') == 'resid' else (np.nan_to_num(Yraw[:, :nch]) - cfg['raw_mean'][:nch])
    parts = []
    for wh in cfg['windows']:
        M = causal_mean(A, int(round(wh * STEP)))
        parts.append(M)
        if cfg.get('norms', False):          # scale-free 'how big is the deviation' features (log of the norms)
            parts.append(np.log(np.linalg.norm(M[:, :NS], axis=1, keepdims=True) + 0.1))
            if nch > NS:
                parts.append(np.log(np.abs(M[:, NS:NS + 3]).sum(1, keepdims=True) + 1.0))
    if cfg.get('unit', False):
        M = causal_mean(R[:, :NS], int(round(cfg['unit_window'] * STEP)))
        nrm = np.linalg.norm(M, axis=1, keepdims=True)
        parts.append(M / (nrm + 0.05))
        parts.append(np.log(nrm + 0.05))
    if cfg.get('time', True):
        h = idx.hour.values + idx.minute.values / 60.0
        parts.append(np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                               (idx.day_of_week.values >= 5).astype(float)], 1))
    return np.hstack(parts).astype(np.float32)


# ------------------------------------------------------------------ alarm rule
def alarm_rule(p, thr, n_persist):
    """alarm(t)=1 iff p>=thr for the last n_persist consecutive steps (causal)."""
    above = (np.asarray(p) >= thr).astype(np.int32)
    c = np.cumsum(np.concatenate([[0], above]))
    t = np.arange(1, len(above) + 1)
    lo = np.maximum(t - n_persist, 0)
    full = (c[t] - c[lo]) == n_persist
    return (full & (t >= n_persist)).astype(int)


# ------------------------------------------------------------------ metrics
def cluster_sizes(clusters_df, K):
    return np.bincount(clusters_df['cluster'].values, minlength=K)


def search_effort(prob, true_c, sizes, total=905):
    """# pipes in the clusters ranked up to and including the true one / 905."""
    order = np.argsort(-prob)
    cs = np.cumsum(sizes[order])
    pos = int(np.where(order == true_c)[0][0])
    return cs[pos] / total


def evaluate_detection(p, alarm, y, leaks, idx):
    """Returns dict with the five project metrics for the detector."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    out = {}
    out['AUROC'] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float('nan')
    det, delays = 0, []
    for _, r in leaks.iterrows():
        m = (idx >= r['start']) & (idx <= r['end'])
        if m.sum() == 0:
            continue
        hits = np.flatnonzero(alarm[m])
        if len(hits):
            det += 1
            delays.append((idx[m][hits[0]] - r['start']).total_seconds() / 3600)
    out['detected'] = det
    out['n_leaks'] = int(len(leaks))
    out['median_delay_h'] = float(np.median(delays)) if delays else float('nan')
    # false alarms: eligible = no leak and not within 24h after a repair
    elig = (y == 0)
    for e in leaks['end']:
        elig &= ~((idx > e) & (idx <= e + pd.Timedelta(hours=24)))
    if elig.sum() > 0:
        days = pd.Series(alarm[elig], index=idx[elig]).groupby(idx[elig].floor('D')).max()
        out['FA_days_per_30'] = float(days.mean() * 30)
        out['n_eligible_days'] = int(len(days))
    else:
        out['FA_days_per_30'] = float('nan'); out['n_eligible_days'] = 0
    return out


# ------------------------------------------------------------------ full inference (used by training CV and predict.py)
def features_for(models, data, cfg):
    R, Y = models['base'].residuals(data, whiten=cfg.get('whiten', False))
    return build_features(R, data['P'].index, cfg, Yraw=Y)


def predict_all(models, data):
    """Returns p_leak (n,) and cluster probabilities (n,K), averaged over the ensemble members.
    Every feature at row t uses rows <= t only."""
    Xd = features_for(models, data, models['det_cfg'])
    p = np.zeros(len(Xd))
    for sc, clf in models['det']:
        p += clf.predict_proba(sc.transform(Xd))[:, list(clf.classes_).index(1)]
    p /= len(models['det'])
    Xl = features_for(models, data, models['loc_cfg'])
    K = models['K']
    prob = np.zeros((len(Xl), K))
    for sc, clf in models['loc']:
        pr = clf.predict_proba(sc.transform(Xl))
        prob[:, clf.classes_] += pr
    prob /= len(models['loc'])
    return p, prob
