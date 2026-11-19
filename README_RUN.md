# Installation and running

```
pip install numpy pandas scikit-learn wntr matplotlib joblib six
python predict.py <evaluation_folder> <output_csv>
```
The evaluation folder needs `pressures_m.csv` and, optionally, `flows_m3h.csv`, `levels_m.csv` and `demands_Lh.csv` (same names and columns as `historical_2018/`).
The output has the columns `Timestamp, p_leak, alarm, c0..c13`. Cluster probabilities are filled only while `alarm == 1`.

`predict.py` loads `model/model.joblib` (weights, baseline regression, noise whitening, scalers and alarm rule) and `clusters.csv`. It does not refit anything. It needs `leaklib.py` next to it.

## Reproducing training (from this folder)
```
python signatures.py          # 782 EPANET runs, about 4 min -> artifacts/signatures.npz
python clusters.py            # -> clusters.csv (K=14), artifacts/k_selection.csv
python run_part2_detector.py all ; python run_part2_detector.py N2 ; python run_part2_detector.py N3
python run_part2_localiser.py
python run_part4.py
python train_final.py cv      # out-of-fold performance estimate
python train_final.py final   # -> model/model.joblib
```
Notes:
- `signatures.py` uses `fork` multiprocessing and writes temporary EPANET files to `/tmp`.
- In this environment `wntr` needed the `six` package.
- Expected performance is in `REPORT.md`, Part 5.
