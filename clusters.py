"""
Part 1b -- noise comparison, detectability, clustering of the 905 pipes -> clusters.csv

Signatures are compared in the NOISE-WHITENED space (see leaklib.Baseline): the day-to-day noise of the
33 sensors is almost entirely two common-mode directions, so a plain Euclidean/cosine comparison would
be dominated by noise directions.  Whitened distance = Mahalanobis distance w.r.t. the real 6-hour noise.

needs artifacts/signatures.npz (python signatures.py)
Run:  python clusters.py
"""
import os, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import leaklib as ll

HERE, ART = ll.HERE, os.path.join(ll.HERE, 'artifacts')
sg = np.load(os.path.join(ART, 'signatures.npz'))
assert list(sg['sensors']) == ll.SENSORS
pipes = list(sg['pipes'])
psig = sg['pipe_sig']                      # (905,24,33) change for a 1.5 cm leak (m)
sig_mean = psig.mean(1)                    # (905,33)
all_pipes = list(sg['all_pipes'])

# ---------------------------------------------------------------- noise of the real sensors (leak-free hours)
data = ll.load_folder(os.path.join(HERE, 'historical_2018'))
y = pd.read_csv(os.path.join(HERE, 'historical_2018', 'labels.csv'), index_col=0, parse_dates=True)
free = (y['leak_active'].values == 0)
base = ll.Baseline().fit(data, free)
Wh = base.whiten
noise6 = np.sqrt(np.diag(base.noise6_cov))
print('raw noise std of 6h-mean residual per sensor: median %.3f max %.3f' % (np.median(noise6), noise6.max()))
ev = np.linalg.eigvalsh(base.noise6_cov)[::-1]
print('share of noise variance in top-2 principal directions: %.3f' % (ev[:2].sum() / ev.sum()))

# empirical distribution of the whitened noise norm (leak-free 6 h means, one per hour)
R, _ = base.residuals(data)
M6 = ll.causal_mean(R[:, :ll.NS], 6 * ll.STEP)
fw = pd.Series(free.astype(float)).rolling(6 * ll.STEP, min_periods=6 * ll.STEP).min().values == 1
noise_z = np.linalg.norm(M6[fw][::ll.STEP] @ Wh, axis=1)
thr_noise = float(np.quantile(noise_z, 0.95))
print('whitened noise norm: median %.2f  95%% %.2f' % (np.median(noise_z), thr_noise))

# what the localiser sees: the model's pressure change minus the part the flow-conditioned baseline
# explains by the extra inflow, then whitened.  inflow share per source: see pipeline.FLOW_SHARE
FLOW_SHARE = np.array([0.05, 0.475, 0.475])
dq = sg['pipe_dq']
sig_res = np.stack([base.delta_residual(sig_mean[i:i + 1], dq[i] * FLOW_SHARE[None])[0, :ll.NS] for i in range(len(pipes))])
Z = sig_res @ Wh                                    # whitened signature of a 1.5 cm leak
z_norm = np.linalg.norm(Z, axis=1)
rows = []
for D in (0.0085, 0.015, 0.024):
    s = z_norm * (D / 0.015) ** 2
    rows.append((D, int((s < thr_noise).sum())))
print('pipes whose signature is below the noise 95 percent level for D=0.85,1.5,2.4 cm:', rows)
detectable = z_norm * (0.012 / 0.015) ** 2 >= thr_noise     # clearly visible for a 1.2 cm leak
print('low-visibility pipes (below noise for a 1.2 cm leak):', int((~detectable).sum()), '(of', len(pipes), ')')

# ---------------------------------------------------------------- clustering on shape (cosine, whitened)
U = Z / (z_norm[:, None] + 1e-12)
rng = np.random.default_rng(0)
Zn = base.noise6_cov


def expected_effort(C, assign, reps=10, D=0.012):
    """Simulate noisy 6h observations of every detectable pipe and rank clusters by cosine (whitened)."""
    sizes = np.bincount(assign, minlength=len(C))
    eff, top1 = [], []
    L = np.linalg.cholesky(Zn + 1e-8 * np.eye(ll.NS))
    for i in np.flatnonzero(detectable):
        s = sig_res[i] * (D / 0.015) ** 2
        obs = s + rng.normal(size=(reps, ll.NS)) @ L.T
        z = obs @ Wh
        sc = (z / np.linalg.norm(z, axis=1, keepdims=True)) @ C.T
        for r in range(reps):
            order = np.argsort(-sc[r])
            pos = int(np.where(order == assign[i])[0][0])
            eff.append(sizes[order][:pos + 1].sum() / len(pipes)); top1.append(pos == 0)
    return float(np.mean(eff)), float(np.mean(top1))


table, models = [], {}
for K in range(10, 21):
    km = KMeans(K, n_init=20, random_state=0).fit(U[detectable])
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    assign = (U @ C.T).argmax(1)
    sil = silhouette_score(U[detectable], assign[detectable], metric='cosine')
    eff, t1 = expected_effort(C, assign)
    table.append(dict(K=K, silhouette=sil, exp_search_effort=eff, exp_top1=t1, max_cluster=int(np.bincount(assign).max())))
    models[K] = (C, assign.copy())
tab = pd.DataFrame(table)
print(tab.round(3).to_string(index=False))
tab.to_csv(os.path.join(ART, 'k_selection.csv'), index=False)
# Expected effort falls (almost) monotonically with K, so 'argmin' would just pick the largest K.  Every extra
# cluster is also one more class for a localiser that has only 20 real leaks.  We take the first K whose
# expected effort is below 0.10; larger K buys at most ~0.02 more.
K = int(tab.loc[tab['exp_search_effort'] < 0.10, 'K'].min())
print('chosen K =', K)
C, assign = models[K]

# renumber clusters by size (0 = biggest), build clusters.csv for ALL 905 pipes
order = np.argsort(-np.bincount(assign, minlength=K))
remap = {old: new for new, old in enumerate(order)}
cl = np.array([remap[a] for a in assign]); C = C[order]
assert len(all_pipes) == len(pipes) == 905
cdf = pd.DataFrame({'pipe_id': pipes, 'cluster': cl}).set_index('pipe_id').loc[all_pipes].reset_index()
cdf.to_csv(os.path.join(HERE, 'clusters.csv'), index=False)
np.save(os.path.join(ART, 'cluster_centroids.npy'), C)

pd.DataFrame({'pipe_id': pipes, 'cluster': cl, 'z_norm_1p5cm': z_norm,
              'z_norm_0p85cm': z_norm * (0.0085 / 0.015) ** 2, 'detectable_1p2cm': detectable}).to_csv(
    os.path.join(ART, 'pipe_details.csv'), index=False)

# ---------------------------------------------------------------- map the 20 real leaks
leaks = pd.read_csv(os.path.join(HERE, 'historical_2018', 'leaks.csv'), parse_dates=['start', 'peak', 'end'])
leaks['cluster'] = leaks['pipe_id'].map(cdf.set_index('pipe_id')['cluster'])
leaks['z_at_D'] = [z_norm[pipes.index(p)] * (d / 0.015) ** 2 for p, d in zip(leaks['pipe_id'], leaks['diameter_m'])]
print(leaks[['leak_id', 'pipe_id', 'type', 'diameter_m', 'cluster', 'z_at_D']].round(2).to_string(index=False))
sizes = np.bincount(cl, minlength=K)
print('cluster sizes', sizes.tolist())
print('clusters without any real leak:', sorted(set(range(K)) - set(leaks['cluster'])))
print('clusters with real leak counts:', leaks['cluster'].value_counts().sort_index().to_dict())
leaks.to_csv(os.path.join(ART, 'leaks_with_clusters.csv'), index=False)
json.dump({'K': K, 'noise_thr_95': thr_noise, 'n_undetectable_1p5cm': int((~detectable).sum())},
          open(os.path.join(ART, 'cluster_info.json'), 'w'))
