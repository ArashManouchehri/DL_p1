"""
predict.py -- run the final detector + cluster localiser on an evaluation folder.

    python predict.py <evaluation_folder> <output_csv>

The folder must contain pressures_m.csv and (optionally, they improve accuracy) flows_m3h.csv, levels_m.csv,
demands_Lh.csv with the same names/columns as historical_2018/.  labels.csv / leaks.csv are NOT read.
Output columns: Timestamp, p_leak, alarm, c0 ... c{K-1}   (cluster probabilities only while alarm == 1).

Nothing is fitted here: baseline regression, noise whitening, scalers, MLP weights and the alarm rule are loaded
from model/model.joblib (fitted on 2018 training data).  Every output row uses only data up to its own time.
"""
import os, sys, warnings
import numpy as np, pandas as pd, joblib

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leaklib as ll


def main(folder, out_csv):
    models = joblib.load(os.path.join(HERE, 'model', 'model.joblib'))
    data = ll.load_folder(folder)
    idx = data['P'].index
    p, prob = ll.predict_all(models, data)
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
    rule = models['rule']
    alarm = ll.alarm_rule(p, rule['thr'], rule['n_persist'])
    K = models['K']
    out = pd.DataFrame({'Timestamp': idx, 'p_leak': p, 'alarm': alarm})
    cl = pd.DataFrame(prob, columns=[f'c{k}' for k in range(K)])
    cl = cl.div(cl.sum(axis=1).replace(0, 1), axis=0)
    cl[alarm == 0] = np.nan
    out = pd.concat([out, cl.reset_index(drop=True)], axis=1)
    out.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}: {len(out)} rows, alarm rows: {int(alarm.sum())}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
