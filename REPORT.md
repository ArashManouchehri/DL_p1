# L-Town leak detection and cluster localisation with MLPs: report (parts 1-5)

All numbers come from the scripts in this folder. The CSVs are in `artifacts/`. Convert this file to PDF for submission (e.g. `pandoc REPORT.md -o report.pdf`). Slides are not included.

## Validation protocol (used everywhere)
- **Part 2 comparisons:** one fixed time split. Train is L01-L12 (up to 2018-09-09). Validation is L13-L20 (after it). The baseline, noise covariance, scalers and synthetic leaks use training data only.
- **Part 5 estimate:** 4 time-blocked folds with 48 h margins around the test block. The baseline, whitening and all MLPs are refit inside each fold. Alarm rule and thresholds are chosen on out-of-fold predictions.
- **Synthetic leaks:** used for training only, never for validation.
- **Causality:** every feature at time t uses rows <= t only. At the start of a file the windows are expanding (all rows so far).

## Part 1: problem, signatures, clusters (`signatures.py`, `clusters.py`)
**Signatures.** I simulated a 1.5 cm emitter at each of the 782 junctions (second simulated day only). A pipe's signature is the mean of its two end junctions, giving 905 pipes with a 24-hour x 33 table each. Every pipe has at least one junction end, so none had to be dropped.
- **Area rule check** (`scaling_check.csv`): for 11 of 12 random nodes the measured/predicted ratio is 0.26-0.27 vs 0.28 at 0.8 cm, and 3.0-3.4 vs 2.8 at 2.5 cm. Trust is within about +/-15% for most nodes.
- **Exception:** nodes n13 and n32 deviate strongly (ratio -0.36 at 2.5 cm for n13). This is a non-linear regime, probably near the PRV.
- **Not removed:** I did not remove any low-signal pipes from `clusters.csv`.

**Noise.** Day-to-day noise of the 6 h mean residual is small per sensor, but 96% of its variance sits in two common-mode directions. I therefore compare shapes in a noise-whitened space (Mahalanobis distance with the real 6 h noise covariance, floor 0.03 m).
- **Baseline:** ridge regression on hour harmonics, weekend, AMR demand, pump/pipe inflows and tank level.
- **Inflow correction:** a leak's own extra inflow is part of what the baseline "explains", so the signature is corrected by the same regression (`Baseline.delta_residual`). The localiser therefore sees the pressure change not explained by inflow.
- **Detectability:** the 95th percentile of the whitened noise norm is 2.62. Signature norms for a 1.5 cm leak range 3.1-19.5, so no pipe is below noise at 1.5 cm. At 0.85 cm, 529 of 905 pipes are below noise. At 1.2 cm, 93 pipes (mainly clusters around the PRV/mains, e.g. p57, p683, p779) are low-visibility. They stay in `clusters.csv`.

**Choice of K** (`k_selection.csv`). I used k-means on unit-normalised whitened signatures, K = 10..20. I scored each K by expected search effort when noisy 1.2 cm leaks on all pipes are ranked by cosine. Effort falls from 0.145 (K=10) to about 0.08 (K=18-20), and top-1 accuracy stays near 0.96. Each extra class is one more output for a localiser with only 20 real leaks, so I took the first K with effort < 0.10, **K = 14**. Clusters are numbered by size, from 0 (160 pipes) to 13 (21 pipes).

**Real leaks mapped to clusters.** Cluster 3 has 4 leaks, cluster 0 has 3, and clusters 5, 7, 8 and 9 have 2 each. **Clusters 2, 6 and 13 have no real leak**, so they rely on synthetic data only (confirmed by `leaks_with_clusters.csv`).

## Part 2: method comparison (`run_part2_detector.py`, `run_part2_localiser.py`)
### Detector. Validation on L13-L20 (8 leaks), rule chosen on validation
| design | AUROC | detected | median delay h | FA/30d |
|---|---|---|---|---|
| A1 raw pressure, 1 h | 0.80 | 8/8 | 3.7 | 0 |
| A2 raw pressure 1+6+24 h | 0.86 | 7/8 | 5.2 | 0 |
| A3 baseline residual (flow-conditioned) | 0.72 | 6/8 | 73 | 0 |
| A5 + whitening + flows/level | 0.67 | 5/8 | 4.8 | 0 |
| A7 A5 + synthetic leaks | 0.80 | 8/8 | 6.0 | 0 |
| N2a raw P + synthetic | 0.84 | 8/8 | 1.0 | 0 |
| N2d residual P + synthetic | 0.87 | 8/8 | 12.2 | 0 |
| window 1 h / 3 h / 6 h / 12 h / 24 h (resid P + synth) | 0.90 / 0.92 / 0.89 / 0.90 / 0.87 | 8/8 each | 1.3 / 2.3 / 7.2 / 13.6 / 4.7 | 0 / 0 / 0 / 0 / 2.1 |
| **N3b windows 1+3+6 h, whitened, log-norm features + synthetic** | **0.906** | 8/8 | 1.0 | 0 |
| B1 non-neural: whitened-residual norm | 0.906 | 8/8 | 2.0 | 0 |
| B2 logistic regression, same inputs/data | 0.59 | 8/8 | 8.0 | 0 |

**Findings**
- **Windows:** 3 h is the best single window. Adding 24 h hurts (false alarms at 24 h).
- **Flows and tank level as inputs:** hurt the detector on the unseen period. The 2018 leaks run in different seasons, and the demand/inflow relation drifts. Using them only inside the baseline regression is fine.
- **Synthetic leaks:** help consistently (e.g. A5 0.67 -> A7 0.80). Training on only 12 real leaks over-fits.
- **Whitening:** helps only together with synthetic data.
- **MLP vs simple baseline:** the MLP does **not** clearly beat the non-neural whitened-norm rule (0.906 vs 0.906 AUROC). The MLP's gain is a softer, learned output and a tunable rule. I report this honestly; the project still requires the MLP.
- **Caveat:** validation has only 8 leaks and the variance is large (differences below about 0.03 AUROC are not meaningful).

### Localiser. Evaluated at detector alarms on L13-L20
| design | top-1 | search effort |
|---|---|---|
| L1 whitened residual 6 h, real data only | 0.15 | 0.71 |
| L2 unit vector + log-norm, real only | 0.20 | 0.70 |
| **L3 = L2 + synthetic leaks** | **0.63** | **0.105** |
| L4 whitened 6+24 h + synthetic | 0.59 | 0.112 |
| L6 L3 with synthetic only | 0.49 | 0.117 |
| L7 L3 without whitening | 0.51 | 0.132 |
| B3 non-neural nearest centroid (model only) | 0.35 | 0.26 |

**Findings.** Real data alone cannot cover the classes. Only 7 of 14 clusters have a real leak in the training period. Synthetic data is decisive (0.70 -> 0.105 effort). Whitening helps (0.132 -> 0.105). The MLP clearly beats nearest centroid. Random ranking gives about 0.5 effort.

## Part 3: final choice
- **Detector:** an MLP with inputs = whitened baseline residuals of the 33 pressures, causal means over 1/3/6 h, log-norm features and time of day/weekend, trained on real + 300 synthetic leaks per member.
- **Localiser:** an MLP with inputs = unit vector of the 6 h whitened residual + log-norm + time, trained on real leaks (>= 30% of peak) + 600 synthetic leaks.
- **Ensemble:** both models average 2 seeds.
- **Alarm rule:** p >= 0.60 for 144 consecutive steps (12 h), chosen on out-of-fold predictions so that false alarms are at most 1 per 30 days. The 12 h persistence is the main reason the median delay is about 12 h.
- **Reasoning:** based on the Part 2 evidence above.
- **Limits:** the detector sees only pressure, so leaks with a signature below noise (small leaks far from sensors, PRV area) are invisible. The localiser's label for overlapping leaks is the cluster of the largest current leak flow. The evaluator scores each leak separately, so a single top-1 cluster cannot be right for two simultaneous leaks in different clusters.

## Part 4: riskiest assumption: model signature = real leak (`run_part4.py`)
**Test, success criterion fixed before running:**
- **(A)** Cosine between the measured and the model signature (same pipe, diameter and hours of day) >= 0.7 (whitened) for at least 12 of 20 leaks.
- **(B)** A localiser trained on synthetic leaks only, scored on all 20 real leaks, reaches top-1 >= 0.4 and effort <= 0.25.

**Results** (`part4_cosine_test.csv`, `part4_synth_only_per_leak.csv`)
- **A:** the median whitened cosine is 0.76 and 12 of 20 leaks pass (borderline). Leaks L04, L05, L07, L10, L15, L19 match almost perfectly (cos >= 0.95). L03 (cos 0.13), L01 (0.46), L17 (0.43), L11 (0.56) and L14 (**-0.66**) do not.
  - L14 is a special case: the measured inflow is 64 m3/h against 27 m3/h leak flow, and the tank level drops 12 sigma. This is probably a tank/pump event or a second effect in the real network.
  - The model also under- or over-predicts size by 0.5-1.9x for several leaks.
- **B:** top-1 = 0.67 and effort = 0.154 (mean of 3 seeds). Both criteria pass.

**Impact.** The assumption holds well enough in practice. This justifies the synthetic data, which gives the localiser most of its skill. The failures show where it breaks: pipes near the PRV and tank/inflow-affected leaks (L03, L14). This is a limitation to state, and it caps real performance below the synthetic estimate. I kept the final models mixed (real + synthetic) because real leaks add 6 to 7 clusters of true behaviour.

## Part 5: expected performance on the hidden dataset (`train_final.py cv`)
Out-of-fold results with 4 blocked folds (`cv_summary.json`, `cv_folds.csv`, `cv_bootstrap_ci.csv`):

| metric | out-of-fold value | 90% bootstrap interval |
|---|---|---|
| AUROC | 0.909 | 0.89-0.93 |
| leaks detected | 20/20 | 20-20 |
| median detection delay (h) | 12 | 8-16 |
| false-alarm days per 30 | 0.9 | 0-1.5 |
| search effort | 0.132 | 0.10-0.17 |
| top-1 cluster accuracy | 0.745 | 0.64-0.83 |

- **Per fold:** AUROC 0.82-0.98. Delay ranges from 11 h to 107 h (fold 1 contains the slow incipient leaks L06-L08). False alarms are 0-2.1 per 30 days.
- **Wider uncertainty than the intervals suggest:** the bootstrap resamples only 20 leaks. The hidden set uses different leaks, so treat these intervals as optimistic. I would expect AUROC 0.85-0.92, 17-20 of 20 leaks detected, effort 0.12-0.25 and delay 8-40 h (depending on the share of incipient leaks).
- **In-sample metrics of the final model** (AUROC 0.96, top-1 0.90) are optimistic and should not be quoted.
- **Runtime:** `predict.py` on a full year runs in well under 10 minutes on CPU.
- **Final model:** `model/model.joblib` (detector 21,889 params, localiser 14,030 params per member; 2 members each). Training time is about 40-60 s per seed per block.

## Assumptions and failure actions
| item | assumption | action if violated |
|---|---|---|
| noise | day-to-day noise is stationary | the baseline drifts and false alarms rise; refit the baseline on recent leak-free days |
| model error | emitter signatures resemble the real network | see Part 4; drop synthetic weight for affected clusters |
| class balance | about 20% leak-free time | class-balanced weights and persistence rule |
| missing sensor | pressure <= 1 m or NaN is treated as "as expected" (residual 0); missing flows/AMR files fall back to the training mean | check the log; accuracy degrades gracefully |
| start of a file | expanding windows, no future data | the first hours are less reliable |

External code: scikit-learn `MLPClassifier`, WNTR/EPANET. No pretrained components. No data from the confidential period was used.
