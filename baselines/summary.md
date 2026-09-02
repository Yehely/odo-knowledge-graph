# Baseline results — temporal split(s)

Two different temporal splits are in play; they are **not** the same split, and their numbers are not directly comparable to each other. See `baselines/run_baselines.py` module docstring for the full explanation of why.


## Hetero split (train: year < 2015, test: year >= 2015, no cap)

This is the split described in the baseline request, and the split the heterogeneous-GNN reference rows below were evaluated on.

| Baseline | RMSE | MAE | Pearson r | R² | Notes |
|---|---|---|---|---|---|
| Global mean | 1.43 | 1.19 | nan | -0.00 | n_test=2854.  |
| Per-target mean | 1.42 | 1.18 | 0.14 | 0.01 | n_test=2854. fallback_to_global_mean=34 rows (target unseen in the 19-target training set) |
| Fingerprint-only RF (ECFP4 + target one-hot) | 1.25 | 1.01 | 0.49 | 0.23 | n_test=2854 (±0.00 RMSE sd, ±0.00 R² sd across seeds). RandomForestRegressor({'n_estimators': 500, 'n_jobs': 4}, random_state=seed), seeds=[42, 123, 456, 789, 1024], ECFP4(2048-bit)+target-one-hot (73 targets), 4 test targets unseen in train |
| Heterogeneous GNN (previously reported, not re-run here) | 1.23 | 0.98 | 0.54 | 0.26 | source: hetero_gnn/test_results.txt (on-disk, verified) |
| "Attention" GNN variant (previously reported, not re-run here, **unverified** — see below) | 1.19 | 0.92 | 0.59 | 0.31 | source: reports/generate_final_report.py hardcoded literal — UNVERIFIED, no surviving results file for this run (see results/seed_variance/summary.md) |

## Bipartite split (train: year <= 2015, test: year 2016-2020, capped — NOT the split described in the baseline request)

Included only because the 0.22 bipartite-ensemble reference number was produced on this split. Do not compare these rows to the hetero table above as if they were on the same test set.

| Baseline | RMSE | MAE | Pearson r | R² | Notes |
|---|---|---|---|---|---|
| Global mean | 1.40 | 1.16 | nan | -0.01 | n_test=2628.  |
| Per-target mean | 1.35 | 1.13 | 0.27 | 0.06 | n_test=2628. fallback_to_global_mean=20 rows (target unseen in the 31-target training set) |
| Fingerprint-only RF (ECFP4 + target one-hot) | 1.26 | 0.98 | 0.47 | 0.18 | n_test=2628 (±0.00 RMSE sd, ±0.00 R² sd across seeds). RandomForestRegressor({'n_estimators': 500, 'n_jobs': 4}, random_state=seed), seeds=[42, 123, 456, 789, 1024], ECFP4(2048-bit)+target-one-hot (43 targets), 4 test targets unseen in train |
| Bipartite GNN ensemble (previously reported, not re-run here) | 1.23 | 0.98 | 0.53 | 0.22 | source: gnn/ensemble_test_results.txt (on-disk, verified) |
