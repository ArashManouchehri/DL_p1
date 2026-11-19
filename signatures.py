"""
Part 1a -- leak signatures from the hydraulic model.

For every junction of L-TOWN_Model.inp we simulate a 1.5 cm orifice leak (EPANET emitter),
keep the 2nd simulated day (the tank needs ~1 day to settle) and store

  node_sig[j] : (24, 33)  hourly-mean pressure change at the 33 sensors (m)
  node_dq[j]  : extra pump inflow (m3/h)

A pipe's signature is the mean of the signatures of its two end junctions
(ends that are reservoirs / tanks have no signature and are skipped).

Outputs (in artifacts/):  signatures.npz
Run:  python signatures.py
"""
import os, sys, json
import numpy as np, pandas as pd, wntr
import multiprocessing as mp
Pool = mp.get_context('fork').Pool

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, 'network', 'L-TOWN_Model.inp')
ART = os.path.join(HERE, 'artifacts')
D_REF = 0.015
INFLOWS = ['PUMP_1', 'p227', 'p235']


def sensors():
    return list(pd.read_csv(os.path.join(HERE, 'historical_2018', 'pressures_m.csv'), nrows=2, index_col=0).columns)


def simulate(leak_node=None, diameter_m=D_REF, tag='x'):
    wn = wntr.network.WaterNetworkModel(INP)
    wn.options.time.duration = 48 * 3600
    wn.options.time.hydraulic_timestep = 900
    wn.options.time.report_timestep = 900
    if leak_node is not None:
        area = np.pi * (diameter_m / 2) ** 2
        wn.get_node(leak_node).emitter_coefficient = 0.75 * area * np.sqrt(2 * 9.81)
    prefix = os.path.join('/tmp', f'sim_{tag}_{os.getpid()}')
    r = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)
    p = r.node['pressure'][SENS]
    q = r.link['flowrate'][INFLOWS].sum(axis=1) * 3600
    return p.iloc[96:192], q.iloc[96:192]


SENS = sensors()
_BASE = {}


def base():
    if 'p0' not in _BASE:
        _BASE['p0'], _BASE['q0'] = simulate(None, tag='base')
    return _BASE['p0'], _BASE['q0']


def node_job(args):
    j, d = args
    p0, q0 = base()
    try:
        p1, q1 = simulate(j, d, tag='n')
    except Exception as e:  # pragma: no cover
        return j, d, None, None, str(e)
    hour = (p1.index // 3600) % 24
    sig = (p1 - p0).groupby(hour).mean().values          # (24, 33)
    return j, d, sig, float((q1 - q0).mean()), None


if __name__ == '__main__':
    os.makedirs(ART, exist_ok=True)
    wn = wntr.network.WaterNetworkModel(INP)
    juncs = wn.junction_name_list
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        res = pool.map(node_job, [(j, D_REF) for j in juncs], chunksize=8)
        # size-scaling check (guide step 3) on a few random nodes
        rng = np.random.default_rng(0)
        chk_nodes = list(rng.choice(juncs, 12, replace=False))
        chk = pool.map(node_job, [(j, d) for j in chk_nodes for d in (0.008, 0.025)])
    node_sig, node_dq, bad = {}, {}, []
    for j, d, s, q, err in res:
        if s is None or not np.isfinite(s).all():
            bad.append(j); continue
        node_sig[j], node_dq[j] = s, q
    print('junctions simulated:', len(node_sig), 'failed:', bad)

    pipe_sig, pipe_dq, pipe_ends = {}, {}, {}
    for name, pipe in wn.pipes():
        ends = [n for n in (pipe.start_node_name, pipe.end_node_name) if n in node_sig]
        pipe_ends[name] = ends
        if ends:
            pipe_sig[name] = np.mean([node_sig[n] for n in ends], axis=0)
            pipe_dq[name] = float(np.mean([node_dq[n] for n in ends]))
    print('pipes with a signature:', len(pipe_sig), 'of', len(list(wn.pipes())))

    # scaling check: signature(D) / signature(1.5cm) vs (D/1.5cm)^2
    rows = []
    for j, d, s, q, err in chk:
        if s is None: continue
        ref = node_sig[j].mean(0)
        got = s.mean(0)
        k = float(np.dot(got, ref) / max(np.dot(ref, ref), 1e-12))
        rows.append(dict(node=j, D=d, measured_ratio=k, area_rule=(d / D_REF) ** 2,
                         ref_norm=float(np.linalg.norm(ref))))
    chk_df = pd.DataFrame(rows)
    chk_df.to_csv(os.path.join(ART, 'scaling_check.csv'), index=False)
    print(chk_df.round(3).to_string())

    np.savez_compressed(
        os.path.join(ART, 'signatures.npz'),
        sensors=np.array(SENS),
        pipes=np.array(list(pipe_sig)), pipe_sig=np.stack(list(pipe_sig.values())),
        pipe_dq=np.array(list(pipe_dq.values())),
        nodes=np.array(list(node_sig)), node_sig=np.stack(list(node_sig.values())),
        all_pipes=np.array([n for n, _ in wn.pipes()]))
    json.dump({k: v for k, v in pipe_ends.items() if not v}, open(os.path.join(ART, 'pipes_without_signature.json'), 'w'))
