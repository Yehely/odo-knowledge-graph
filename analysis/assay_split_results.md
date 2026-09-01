# Assay-stratified scoring of the temporal-split test set

Read-only re-scoring of predictions already saved by `baselines/run_baselines.py` (Part 1). No model was retrained, no hyperparameter or split was changed; rows were only partitioned at scoring time. `binding = {Ki, IC50}`, `functional = {EC50, Emax}`, `other` = everything else, per Section 4.4's own grouping.


## bipartite split

Test set: 2,628 rows. Train set (train+val, as `baselines/run_baselines.py` defines it): 19,751 rows.

### Test-set metrics by assay group (5 seeds, mean ± sample sd, ddof=1)

| Group | n | % of test | label mean | label sd | RMSE | MAE | Pearson r | R² |
|---|---|---|---|---|---|---|---|---|
| binding | 1,854 | 70.5% | 7.30 | 1.36 | 1.222 ± 0.002 | 0.951 ± 0.001 | 0.475 ± 0.002 | 0.188 ± 0.003 |
| functional | 765 | 29.1% | 7.33 | 1.47 | 1.339 ± 0.004 | 1.047 ± 0.003 | 0.454 ± 0.005 | 0.170 ± 0.005 |
| other | 9 | 0.3% | 8.78 | 0.98 | 2.015 ± 0.022 | 1.764 ± 0.025 | -0.811 ± 0.054 | -3.771 ± 0.106 |

`other` distinct endpoint values: ['Kd']


### Train-set binding/functional/other proportions

| Group | n (train) | % (train) |
|---|---|---|
| binding | 17,075 | 86.5% |
| functional | 2,504 | 12.7% |
| other | 172 | 0.9% |

Section 4.4 quotes binding/functional proportions of **67%/33% (train) → 48%/52% (test)**. This split's actual proportions, from the model's own train/test population: **86%/13% (train) → 71%/29% (test)** — does NOT reproduce the reported figures.

## hetero split

Test set: 2,854 rows. Train set (train+val, as `baselines/run_baselines.py` defines it): 10,622 rows.

### Test-set metrics by assay group (5 seeds, mean ± sample sd, ddof=1)

| Group | n | % of test | label mean | label sd | RMSE | MAE | Pearson r | R² |
|---|---|---|---|---|---|---|---|---|
| binding | 2,854 | 100.0% | 7.34 | 1.43 | 1.254 ± 0.004 | 1.014 ± 0.003 | 0.493 ± 0.003 | 0.230 ± 0.005 |
| functional | 0 | 0.0% | nan | nan | n/a (0 rows) | n/a | n/a | n/a |
| other | 0 | 0.0% | nan | nan | n/a (0 rows) | n/a | n/a | n/a |

### Train-set binding/functional/other proportions

| Group | n (train) | % (train) |
|---|---|---|
| binding | 10,622 | 100.0% |
| functional | 0 | 0.0% |
| other | 0 | 0.0% |

Section 4.4 quotes binding/functional proportions of **67%/33% (train) → 48%/52% (test)**. This split's actual proportions, from the model's own train/test population: **100%/0% (train) → 100%/0% (test)** — does NOT reproduce the reported figures.

## Reading the result honestly

**Hetero split: not computable.** `hetero_gnn/preprocess.py`'s loss-eligibility rule (`exact_eligible = is_ki and qualifier=='=' and ...`, `hetero_gnn/preprocess.py:424-434`) restricts every exact-labelled row — train and test alike — to `endpoint == 'Ki'`. The hetero test set is **100.0% binding, 0.0% functional, 0.0% other** by construction, not by measurement. Section 5.3's proposed mechanism cannot be tested on this split at all: there is no functional-endpoint test signal to compare against, and no amount of re-scoring changes that. This is not "inconclusive" — it's a structural property of how the hetero pipeline defines a usable label, worth flagging on its own (the heterogeneous model's reported 0.26 R² is entirely a binding-affinity number; it says nothing about functional-assay generalization).

**Bipartite split: functional endpoints ARE predicted worse than binding ones, and the gap is well outside seed noise, on all four metrics:**

| Metric | binding (mean±sd) | functional (mean±sd) | diff (functional − binding) | diff ÷ larger seed-sd |
|---|---|---|---|---|
| RMSE | 1.2223 ± 0.0022 | 1.3393 ± 0.0038 | +0.1170 (worse) | ≈31× |
| MAE | 0.9512 ± 0.0013 | 1.0474 ± 0.0034 | +0.0962 (worse) | ≈28× |
| Pearson r | 0.4749 ± 0.0018 | 0.4544 ± 0.0046 | −0.0205 (worse) | ≈4.5× |
| R² | 0.1883 ± 0.0029 | 0.1704 ± 0.0047 | −0.0180 (worse) | ≈3.8× |

All four agree in direction. **RMSE/MAE are the comparison to trust as primary evidence** — they're on the model's native output scale, not renormalized by each subset's own label variance, and the gap there is enormous relative to seed noise (~28–31×), not a borderline call. R² and Pearson r agree but with a smaller noise margin (~4×) — real, but the kind of gap that would be worth a second look if it stood alone.

The usual caveat about R² — "a lower R² on a narrower-range subset doesn't mean the rows are harder, it means there's less variance to explain" — **does not apply here and does not explain away the gap**: functional's label sd (1.47) is *higher* than binding's (1.36), not lower. If anything that works against the observed direction (more label variance should make R² easier to earn), yet functional R² is still lower. The R² gap is real signal, not a label-variance artifact, it's just smaller in relative terms than the RMSE/MAE gap.

**So: on the one split where the comparison is possible, functional assays are measurably harder to predict than binding assays, confirming the predictive half of Section 5.3's claim** — this is not just correlation with the temporal-shift story anymore, it's a direct, seed-controlled, re-scored measurement on the same fitted model.

**What does NOT hold up is the compositional premise Section 4.4 uses to explain it.** Neither split's actual train/test population reproduces the quoted 67%/33% → 48%/52% binding/functional shift:

- Bipartite (the split where the shift claim would need to hold for the argument to connect): **86%/13% (train) → 71%/29% (test)** — binding is far more dominant in both train and test than reported, and the test-set functional share (29%) is nowhere near the reported 52%.
- Hetero: **100%/0% → 100%/0%** — not comparable by construction, as above.
- The report's own figure-generation code (`reports/make_figures.py:fig_domain_shift`) traces to `functional = [33, 52]` / `binding = [67, 48]` typed in as bare literals, with no computation and no cited source anywhere in the function — see below.

**Net read:** Section 5.3's mechanism — "functional assays are harder to predict, and that's part of why the temporal test set is harder" — is supported by direct measurement on the bipartite split, cleanly, in RMSE/MAE especially. But the specific 67/33→48/52 compositional numbers used to motivate it in Section 4.4 are not reproducible from the model's actual data and appear to have no computed source at all. The honest fix is to keep the (now-verified) functional-vs-binding performance gap, replace Figure 7's numbers with the measured 86/13%→71/29% bipartite proportions (and note the hetero split can't speak to this question at all), and drop the implication that the reported shift was ever measured.

## Figure 7 / Section 4.4 provenance check

`reports/make_figures.py:fig_domain_shift()` defines `functional = [33, 52]` and `binding = [67, 48]` as bare literals — no computation, no comment pointing at a source script or a results file, anywhere in that function. These are not derived from `Final ODO Dataset_v2026-06-10.xlsx` or from either GNN pipeline's train/test split by any code in this repository.

## Run log

```
Commit: 3bda002813447f43d2e788495cd3be0f2e4f3e37
Runtime: 104.1s
Packages:
  torch: 2.13.0+cpu
  scikit-learn: 1.9.0
  numpy: 2.4.4
  pandas: 3.0.3
  scipy: 1.18.0
  torch-geometric: 2.8.0.post1
  rdkit: 2026.3.5
```
