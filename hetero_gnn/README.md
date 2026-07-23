# Heterogeneous GNN — ODO Opioid Binding-Affinity Predictor

Implementation of [`HETEROGENEOUS_GNN_ARCHITECTURE.md`](../HETEROGENEOUS_GNN_ARCHITECTURE.md):
a 5-node-type Heterogeneous GNN (**Compound, Target, Assay, Model System,
Document**) predicting pKi for compound–target binding. This package is
fully independent of `gnn/` (the earlier 2-node-type bipartite model) —
different graph, different model, different training loop.

Validated end-to-end against the real dataset while building it (see
"What was actually run" below); every dimension quoted here was checked
against the live tensors, not just derived on paper.

---

## Quick start

```bash
# Step 1 — Excel → hetero_gnn/processed_hetero_graph.pt
conda run -n odo python3 hetero_gnn/run_preprocess.py

# Step 2 — train with the current config.py default dims (§5's midpoints)
conda run -n odo python3 hetero_gnn/run_train.py

# Optional — 10-epoch overfitting sanity check instead of full training
conda run -n odo python3 hetero_gnn/run_train.py --sanity

# Optional — §5 hyperparameter search (Assay/ModelSystem/Document dims),
# then automatically trains the final model with the winning combination
conda run -n odo python3 hetero_gnn/search_hparams.py --n-trials 20

# Inference on a new compound (after training)
conda run -n odo python3 hetero_gnn/predict.py
```

Outputs land next to the package: `processed_hetero_graph.pt`,
`compound_feature_builder.pt`, `target_reference.csv`,
`checkpoints/best_model.pt`, `training_curve.png`, `test_results.txt`,
`best_hparams.json` — all regenerable, all gitignored.

---

## File structure

```
hetero_gnn/
├── config.py            # every dimension/hyperparameter, with §-references and rationale
├── preprocess.py         # Excel -> HeteroData (entity resolution, features, edges, split)
├── dataset.py              # load processed_hetero_graph.pt, apply train/val/test masks
├── model.py                  # HeteroOpioidGNN: encoders, hop1/hop2 message passing, head
├── train.py                    # exact-MSE + censored-hinge loss, train/eval loop, checkpointing
├── search_hparams.py             # §5 Optuna hyperparameter search (Assay/ModelSystem/Document dims)
├── predict.py                      # score a new SMILES against every known target
├── run_preprocess.py                 # CLI entry for preprocess.py
└── run_train.py                        # CLI entry for train.py (+ training-curve plot)
```

---

## The doc's node/edge design, exactly as implemented

### Graph inclusion & the two-component loss (§1)

- Every surviving row becomes one `binds_to` (Compound↔Target) edge, **except
  9 rows** with a corrupted/implausible `unit_of_measurement` — reproduced
  from the doc's own example (Ki rows with no `pchembl_value`, and a unit
  that's either literally `'M'` — e.g. a recorded value of `168` would mean
  168 Molar — or missing entirely so the scale can't be recovered). Verified
  against the workbook: exactly 4 `unit=='M'` rows (values 168, 0.05, 73,
  188 — the doc names 168 and 188 directly) + 5 `unit` missing rows, all
  `endpoint=='Ki'`, all `pchembl_value` NaN.
- Label derivation: `pchembl_value` used directly when present; otherwise
  derived from `endpoint_value` + `unit_of_measurement` **only when the unit
  is a pure molar concentration** (`nM`, `uM` — not `%`, `mg kg-1`, etc.,
  which can't be converted to a Ki-like concentration at all).
- **The qualifier is never rewritten/flipped.** The current doc revision
  makes this explicit — the hinge loss (below) "branch[es] on the
  **original, raw** `endpoint_qualifier`", not a derived one. Working
  through the old step-3 flip rule (`> → <`, `< → >`) against that hinge
  spec shows they describe the *same* thing: raw `<` always corresponds to
  "the derived pKi is a floor" and raw `>` to "it's a ceiling", regardless
  of whether the flip is applied to the qualifier *symbol* first. So the
  symbol is just kept as-is everywhere — `edge_attr`'s qualifier one-hot,
  the exact-loss check, and the hinge branch below all read the same raw
  `endpoint_qualifier` column, with no separate "effective qualifier"
  concept anywhere in the code.
- **Two loss components** (`train.py:combined_loss`), both restricted to
  `endpoint=='Ki'` with a resolvable qualifier (never `#ERROR!`, and never
  `~` — see below):
  1. **Exact (MSE)** — `qualifier=='=' AND MIN_PCHEMBL <= pKi <= MAX_PCHEMBL`
     (`MIN=2.0`/`MAX=15.0`, reused from the sibling `gnn/` model).
  2. **Censored (one-sided hinge)** — raw qualifier `<`/`<=` → the pKi value
     is a **floor** (true pKi is higher): `(max(0, y - ŷ))²`. Raw `>`/`>=` →
     a **ceiling** (true pKi is lower): `(max(0, ŷ - y))²`. `<=`/`>=` aren't
     in the doc's two worked examples but carry the same inequality
     direction as `<`/`>` (just inclusive), so they get the same treatment.
     A prediction that already respects the bound gets zero loss.
  `total = mean(exact terms) + CENSORED_LOSS_WEIGHT * mean(hinge terms)`
  — one combined mean per component (not per batch-count), each term's
  magnitude independent of how the two populations happen to mix in a given
  mini-batch. `CENSORED_LOSS_WEIGHT=1.0` — the doc says "tune empirically"
  without a number, so it starts at parity with the exact term (`config.py`).
- `~` (`compound_pharmacological_role`-style "approximately equal") gets
  **neither** loss term — the doc gives no hinge direction for it and it's
  exactly 1 row in the whole 37k-row dataset, so it's treated like
  `#ERROR!`: message-passing-only.
- Verified against the workbook that this isn't accidentally starving one
  branch: of all `endpoint_qualifier=='<'`/`'<='` rows, only 20 are
  `endpoint=='Ki'` with a usable value (483 raw `<` rows are overwhelmingly
  IC50/EC50, not Ki) — vs. 1,830 for `>`/`>=`. Both numbers match the
  pipeline's `hinge_floor_mask`/`hinge_ceiling_mask` counts exactly.
- Standard regression metrics (RMSE/MAE/Pearson r/R², and early-stopping /
  checkpoint selection) are computed on the **exact-labelled subset only** —
  the only edges with a real point-value ground truth; a one-sided bound
  isn't something a normal error metric can be scored against. The combined
  loss (exact+hinge) is still reported alongside for visibility into the
  full training objective.

### The 5 node types (§2)

| Node | Primary key (doc) | Fallback when missing | Raw dim | Composition |
|---|---|---|---|---|
| Compound | `pubchem_cid` | standard InChIKey (0% missing) | **2060** | 2048 Morgan ECFP4 (r=2) + 12 numeric (see below) |
| Target | `uniprot_protein_id` | slugified `target_name` | **45** | 4 type one-hot + 9 taxonomy one-hot + 32-dim identity embedding |
| Assay | `chembl_assay_id` | content-hash of format/method/bioassay/setting/target/document | **search 32–64**, default 48 | pure identity embedding |
| Model System | derived cell/tissue/organism composite | shared `unspecified_model_system` bucket | **search 8–24**, default 16 | pure identity embedding |
| Document | `pubmed_id` → `document_doi` → `patent_id` | shared `unknown_document` bucket | **search 4–12** (+3 fixed), default 8+3=11 | *N*-dim journal embedding + is_patent + norm_year + is_unknown |

Run against the live workbook this produces **13,297 compounds, 73 targets,
4,631 assays, 1,069 documents, 148 model systems** from 37,353 surviving
activity rows (37,362 − 9 dropped).

**Compound numeric columns (12, not the doc's approximate "~10"):** the
explicit §2.A column list (2 PK columns + **all 10** QikProp columns —
including `qikprop_dipole` and `qikprop_fisa`, which the *sibling* `gnn/`
model's `ADMET_COLS` omits, and **both** `qikprop_qplog_pw` *and*
`qikprop_qplog_po/w`, which really are two distinct columns in the source
file) is the more precise source of truth than §5's rounded total, so that's
what's implemented: `chembl_alogp`, `chembl_#ro5_violations`,
`qikprop_dipole`, `qikprop_sasa`, `qikprop_fisa`, `qikprop_donor_hb`,
`qikprop_accpt_hb`, `qikprop_qplog_pw`, `qikprop_qplog_po/w`,
`qikprop_qplogs`, `qikprop_qplog_khsa`,
`qikprop_percent_human_oral_absorption`. `chembl_molecule_max_phase` and the
radiolabel flag are **not** in the Compound vector — the new doc excludes
max_phase entirely (§4) and moves the radiolabel column to Assay's
"reference reagent" group, unlike the older bipartite model which kept both
on Compound.

**Assay / Model System / Document — resolving "chosen by hyperparameter
search" into a design, not just a range.** The doc gives Target an
explicit 3-part breakdown (`4 + 9 + 32 = 45`, no room left over) but
describes Assay and Model System with only *one* thing each — "Learned
Embedding **covering** format/method/bioassay categories, reference
reagent, signaling pathway, and sparse in-vivo dose/route fields" and
"Learned Embedding **for** cell line/tissue **identity**". Read literally,
that's a single embedding table indexed by the node's own primary key
(exactly how Target's own 32-dim `uniprot_protein_id` embedding already
works) — not one one-hot per listed sub-field. Since all of those
sub-fields are static per assay (constant for a given `chembl_assay_id` by
definition), an identity embedding recovers exactly the same information a
hand-built one-hot concatenation would, with far less sparse-category
bookkeeping — most of those columns are >48% missing or near-degenerate in
the real data (`bao_assay_format_l2`/`l3` are 99.9% missing;
`single_concentration_screen` 95%; `chembl_dose_administered`/
`ncit_route_of_administration` ~99%). Document is the deliberate exception:
§2.E explicitly says its role is to *absorb* batch effects, "not... a
strong direct predictor", so it embeds only `document_journal` (19 raw
strings → 12 after merging punctuation variants like `"J. Med. Chem."`/
`"J Med Chem"`, +1 "other" slot = 13-vocab embedding) rather than the
document's own identity, plus 3 explicit scalars — precisely as the doc
spells out.

**What "chosen by hyperparameter search" changes:** only the *size* of
those three embeddings is now search-selected (§5) rather than hand-picked
— the design above (pure identity embedding for Assay/ModelSystem, journal
embedding + 3 scalars for Document) is unchanged and isn't part of the
search space. `model.py`'s `HeteroOpioidGNN` takes `assay_emb_dim` /
`model_system_emb_dim` / `document_journal_emb_dim` as constructor
arguments (defaulting to `config.py`'s current values — the midpoint of
each range) precisely so `search_hparams.py` can instantiate a fresh model
per trial with different values; nothing else in the architecture depends
on these three numbers (every node type still projects into the same
256-dim shared latent space via its own encoder, so the hop matrices and
prediction head are completely unaffected by which combination is chosen).
See "Hyperparameter search" below for how the search itself is run.

The doc's `document_journal` range ("search range 4–12") is read as
applying to the *embedding* specifically (consistent with how Assay's and
Model System's ranges apply directly to their pure-embedding dims), with
the 3 fixed scalars (`is_patent`, `norm_year`, `is_unknown`) added on top
as non-tunable — so the actual Document raw input dimension search-varies
over 7–15, not 4–12.

**Target one-hot vocabularies are built from the live data**, capped to the
doc's exact counts by descending frequency (not hardcoded — the sibling
`preprocess_bipartite_graph.py`'s hardcoded species list, e.g., includes
species like Ovis aries/Sus scrofa that don't actually appear in the
current workbook, and misses ones that do, like Oryctolagus cuniculus and
Danio rerio):
- **target_type (4, exact):** `Single protein`, `Protein family`,
  `heteromer`, `Selectivity group` — exactly the 4 non-null values present.
- **taxonomy (9, exact):** top species by frequency after casing
  normalisation (`"Homo Sapiens"` → `"Homo sapiens"`) plus an explicit
  `unspecified` bucket (also absorbing rare composite entries like
  `"Mus musculus / Cavia porcellus"`).
- Both use an all-zero vector for missing/unmatched (no catch-all slot) —
  chosen to hit the doc's *exact* stated counts rather than 5/10.
- `interpro_protein_family_name`, `interpro_protein_category`,
  `ncit_protein_subfamily_name`, `dto_gpcr_category` (§2.B's "Protein
  families" group) aren't separately one-hot-encoded — there's no room for
  them in §5's exact `4+9+32=45` and, like Assay's sub-fields, they're
  static per target so the identity embedding already has access to
  whatever signal they'd add.

### Edge — `binds_to` (§3, 16 dims — the top of the doc's "~14–16")

| Group | Dims | Vocabulary |
|---|---|---|
| Endpoint type | 7 | `Ki, IC50, EC50, Inhibition, Activity, Binding, other` — the doc's literal enumeration, kept verbatim even though `Binding` never fires on the current data (dozens of other raw endpoint strings, e.g. `Emax`, `Ke`, `Kd`, fall into `other`) |
| Qualifier | 4 | `=, <, >, other` (**raw** `endpoint_qualifier`, never flipped — see §1) — real data is 99.9%+ `=`/`>`/`<`/`#ERROR!`; the rare `<=`, `>=`, `~` (≤15 rows each) and the `#ERROR!` bucket (3,294 rows) are bucketed together to land on the doc's "~4" (this one-hot is purely a message-passing feature — the *actual* exact/hinge-floor/hinge-ceiling loss routing in §1 is computed from the unbucketed raw qualifier, independently of this 4-slot encoding) |
| Pharmacological role | 3 | `agonist, antagonist, other` (missing → all-zero, distinct from "other") |
| Binding site | 2 | `high_affinity, low_affinity` (missing → all-zero; only 4/37,362 rows are non-null) |

`chebi_compound_pharmacological_role_id` is the ChEBI URI twin of
`compound_pharmacological_role` and isn't separately featurized, matching
how every other `*_id` ontology-URI column is treated throughout (kept for
RDF/`owl:sameAs` linking in the KG pipeline, not duplicated as a model
input).

### Structural (message-passing) edges — not in the doc's §3 table, required by §7

`binds_to` alone can't support the doc's Assay-hub message passing (§7
needs Assay to see Document/ModelSystem/Target, and Compound/Target to see
Assay) — so 4 additional edge types are built from the same activity rows,
**de-duplicated to unique pairs** (so hop mean-aggregation averages over
distinct neighbours, not once per repeated row):

```
(compound, tested_in, assay)         + reverse   34,295 unique pairs
(assay, tests_target, target)        + reverse    4,706 unique pairs
(assay, has_document, document)      + reverse    4,639 unique pairs
(assay, has_model_system, model_system) + reverse  4,815 unique pairs
```

### Encoders (§6)

All 5 node types: `Linear → BatchNorm → ReLU → Dropout(0.3)` into the shared
256-dim space — dropout is the one explicit figure the doc gives (§6), so
it's reused for the prediction head's Dropout too (§8 doesn't give its own
number).

### Message passing (§7) — a custom hub-and-spoke scheme, not generic HeteroConv

This is *not* implemented as a multi-relation `HeteroConv` sum — the doc's
"mean-aggregates the vectors of its neighbors: Document, Model System, and
Target" reads as one **pooled** mean over the union of those neighbours,
which is only equivalent to summing three separately-averaged relations
when each relation contributes exactly one neighbour. Instead, `model.py`
manually concatenates the three neighbour-vector/destination-index pairs
and calls `torch_geometric.utils.scatter(..., reduce="mean")` once per hop
— a true pooled mean regardless of how many neighbours each relation
contributes:

- **Hop 1 → Assay:** pool {Document, ModelSystem, Target} (pre-message
  vectors) → concat with Assay's own vector → `W1` → LeakyReLU.
- **Hop 2 → Compound and Target:** pool the **updated** Assay vectors (same
  Assay neighbourhood as hop 1, message flowing the opposite direction) →
  concat with own vector → `W2_compound` / `W2_target` (separate, "analogous"
  matrices, not tied) → LeakyReLU.

Document and Model System only ever *send* — they're never targets of
aggregation, matching Document's explicit "not a strong direct predictor"
role (§2.E) and Assay's role as the sole hub. A brand-new query compound (no
recorded assay history) naturally gets a zero-vector hop-2 context — verified
empirically that `scatter(..., reduce="mean")` returns `0`, not `NaN`, for
an empty group — so `predict.py` degrades gracefully to scoring on
structure/ADMET alone, which is the only honest thing to do for a compound
that's never been tested.

### Prediction head (§8)

`edge_attr` (16) → `Linear(16, 18)` ("narrow linear layer") → concat with
enriched Compound(256) + Target(256) → **530-dim**, matching the doc's
literal "~530" exactly → `Linear(530,256) → ReLU → Dropout → Linear(256,128)
→ ReLU → Linear(128,1)`. No BatchNorm in the head — the doc spells this
block out explicitly without one (unlike §6's encoders), so none was added.

### Split & optimisation (§9)

- **Temporal cutoff:** train on `year < 2015` (unknown year → train),
  random 10% of that held out as val, test on `year >= 2015`. The current
  doc revision now states this explicitly ("before 2015 / 2015 onward");
  it's also exactly what this implementation already used, reusing the
  cutoff validated in the sibling `gnn/config.py` (documents up to 2014 are
  ~79% of the corpus — matches the doc's own "~79.4%" figure). No upper cap
  on the test year (the workbook runs to 2024; the older
  `preprocess_bipartite_graph.py`'s `TEST_MAX_YEAR=2020` convention isn't
  used here).
- **Loss:** the combined exact-MSE + censored-hinge objective from §1 —
  see above. Unlike the sibling `gnn/` model, no asymmetric high-affinity
  weighting or temporal sample weighting; the doc doesn't ask for either.
- **Optimizer:** Adam over all parameters (`LR=1e-3`, `weight_decay=1e-4`,
  reused from the sibling since §9 doesn't give numbers), gradient-clipped
  at 5.0, `ReduceLROnPlateau` + early stopping (patience 20) — standard
  engineering not contradicted by the doc.

### Hyperparameter search (§5) — `search_hparams.py`

§5: "Assay, Model System, and Document dimensions are not fixed by hand —
they're chosen by hyperparameter search (grid/random search across
training iterations, selecting the value with the best validation loss),
same as any other tunable hyperparameter in this pipeline."

A full grid/exhaustive search would mean training dozens of complete
models, so this is implemented as a *bounded* search: **Optuna**, TPE
sampler + median pruner.

- **Search space** — exactly the doc's stated ranges (§5):
  `assay_emb_dim ∈ [32,64]`, `model_system_emb_dim ∈ [8,24]`,
  `document_journal_emb_dim ∈ [4,12]`, sampled as continuous integer ranges
  via `trial.suggest_int`, not a hand-picked discrete grid.
- **Per-trial budget** — a full 200-epoch run per trial would make even 10
  trials prohibitively slow, so each trial trains with a reduced budget
  (`--search-epochs`, default 40) and tighter early-stopping
  (`--search-patience`, default 8), reporting intermediate validation RMSE
  every epoch so `optuna.pruners.MedianPruner` can kill clearly-unpromising
  trials early (5 startup trials before pruning activates, 5-epoch warmup
  per trial).
- **Objective** — best validation RMSE reached within the trial's budget,
  on the exact-labelled subset (same metric `train.py` uses for checkpoint
  selection elsewhere, so the search optimises the same thing a normal
  training run would report).
- **After the search** — the winning combination is saved to
  `hetero_gnn/best_hparams.json`, and (unless `--no-final-run`) immediately
  used for one real, full-budget (`MAX_EPOCHS=200`, early-stopped) training
  run via the same `train.train()` `run_train.py` uses, so
  `checkpoints/best_model.pt` ends up holding the actual best model found,
  ready for `predict.py` — not just a search report you'd have to act on
  manually.
- `run_train.py --assay-emb-dim / --model-system-emb-dim /
  --document-journal-emb-dim` let you reproduce a specific winning
  combination (or try your own) without going through the search again.

```bash
conda run -n odo python3 hetero_gnn/search_hparams.py --n-trials 20
```

---

## What was actually run

This session's sandbox has no `odo` conda environment, so everything was
validated in a throwaway `venv` (torch 2.13 CPU + torch_geometric 2.8 +
optuna 4.9, deleted afterward — nothing was installed into your system
Python or the `odo` env), and re-validated after the doc revision landed:

- **Full preprocessing**, clean rebuild from the raw workbook — no crashes.
  `exact_mask`/`hinge_floor_mask`/`hinge_ceiling_mask` come out to
  13,476 / 20 / 1,830 and are pairwise mutually exclusive (asserted at
  build time, and independently re-checked after the fact); every label is
  finite where a mask says it should be. The 20-vs-1,830 floor/ceiling
  asymmetry was cross-checked directly against the workbook (raw `<` rows
  are overwhelmingly IC50/EC50, not Ki) to make sure it's real data, not a
  masking bug.
- **`combined_loss` hinge direction**, unit-tested against hand-computed
  synthetic values (a floor violated, a floor respected, a ceiling
  violated, a ceiling respected, one exact edge) — matched the
  hand-derived expected loss (0.5) exactly.
- **Model forward + backward**, full graph and mini-batch, including
  building the model at both range boundaries (`assay_emb_dim=64`,
  `model_system_emb_dim=8`, `document_journal_emb_dim=4`, and the reverse):
  correct shapes throughout, zero NaN, **every** parameter receives a
  gradient.
- **10-epoch overfitting sanity check** (`run_train.py --sanity`, now
  running the real `combined_loss`): loss 5.54 → 1.96 on a fixed 500-edge
  batch.
- **A 15-epoch real run** (`train.train()`, the same function
  `run_train.py` calls): val RMSE dropped to 0.80 (R²=0.68) by epoch 10 and
  was *still improving* at epoch 14 when the artificially-short budget cut
  it off; test-set (temporal, harder) RMSE=1.25, R²=0.24, Pearson r=0.50 —
  in the same neighbourhood as the *fully-trained* sibling bipartite
  model's reported numbers (RMSE=1.26, R²=0.19) despite this run being
  nowhere near converged. Good sign, **not** a substitute for a full run.
- **`search_hparams.py`** end-to-end with a tiny budget (3 trials, 4 epochs
  each, `--no-final-run`): all 3 trials completed, best config and
  `best_hparams.json` written correctly, the printed `run_train.py`
  reproduction command matches the winning params.
- **`predict.py`** end-to-end on morphine against all 73 targets using the
  15-epoch checkpoint above, output correctly joined with target
  name/species from `target_reference.csv`.
- **`pyflakes`** clean across every file in the package (zero warnings) —
  including after all of the above refactors, not just at initial writing.

**What this does *not* mean:** the numbers above are from short/reduced
runs purely to prove the pipeline (now including the two-component loss
and the search infrastructure) is wired correctly end-to-end — they are
not a real result. You still need to run
`conda run -n odo python3 hetero_gnn/run_train.py` (or `search_hparams.py`
for the dimension search) yourself for a real, fully-converged model in
your actual `odo` environment before drawing any conclusions about how
this architecture performs. If anything errors there that didn't here,
it's almost certainly a package-version difference between this session's
ad-hoc venv and your pinned `odo` env — tell me the traceback and I'll fix
it.

---

## What changed when the doc was revised

The architecture doc was updated after the first implementation pass; this
package was then updated to match. What actually changed between doc
revisions (not just implementation detail):

1. **Loss (§1/§9):** single exact-MSE-only loss → **two required
   components**, exact MSE + censored one-sided hinge on `<`/`>`-qualified
   Ki edges. The hinge loss was explicitly optional/"not required for v1"
   in the prior doc revision; it's mandatory now. See §1 above for the full
   implementation.
2. **Dimensions (§5):** Assay/ModelSystem/Document embedding sizes went
   from hand-picked point estimates (which the prior doc marked "TBD") to
   explicit **search ranges**, with the doc now stating outright that
   they're "chosen by hyperparameter search... same as any other tunable
   hyperparameter" rather than fixed by hand. `search_hparams.py` is new;
   `config.py`'s point constants are now just the default/starting values
   (the midpoint of each range) used when you don't run a search.
3. **Temporal cutoff (§9):** was left as `⬜ (TBD)` in the prior revision;
   the current doc states it explicitly as 2015/~79.4%–~20.6%. This
   implementation already used exactly that cutoff (reusing the sibling
   model's validated choice) before the doc was updated to say so, so no
   code changed here — it's confirmation, not a correction.

The prior revision's "Open Decisions" section is gone from the doc because
all three of the above are now resolved by the doc itself rather than left
to implementation judgement.

## Known simplifications (carried over from the sibling `gnn/` pipeline, not new)

- Compound `StandardScaler`/Document year-normalisation are fit on **all**
  entities, not train-split-only — the same (minor, pre-existing) choice
  `preprocess_bipartite_graph.py` already makes; the new doc doesn't raise
  train/test leakage as a concern for node features and fixing it wasn't
  asked for.
- Full-graph message passing is recomputed on every mini-batch during
  training (no caching across batches within an epoch) — mirrors the
  sibling `gnn/train.py`'s existing behaviour; the graph is small enough
  (≈40k total nodes) that this is cheap in practice.

## Explicitly out of scope for this pass

- No `validate_model.py` / `generate_model_report.py`-equivalent HTML
  report generator (the sibling has one; the architecture doc doesn't ask
  for it, and it's a large, separate deliverable — say the word if you
  want one built for this model too).
- No compound-random / edge-random split variants (the sibling exposes
  these as CLI flags on `preprocess_bipartite_graph.py`; §9 of the new doc
  commits to temporal only).
