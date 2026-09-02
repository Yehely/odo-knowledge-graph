# Target/compound entity-count discrepancy — investigation

Read-only analysis. No pipeline, model, or config file was modified. New code:
`analysis/reconcile_counts.py`, `analysis/target_inventory.py`, `analysis/target_ablation.py`.

Commit: `3bda002813447f43d2e788495cd3be0f2e4f3e37`
Package versions: torch 2.13.0+cpu, scikit-learn 1.9.0, numpy 2.4.4, pandas 3.0.3,
scipy 1.18.0, torch-geometric 2.8.0.post1, rdkit 2026.3.5, openpyxl 3.1.5

## Task 1 — identity rules, side by side

**Compounds**

| Pipeline | Primary key | Fallback | Quote |
|---|---|---|---|
| `kg/build_kg.py` (`build_compound`) | `chembl_compound_id` | `inchikey_<rdkit_library_standard_inchi_key>` | `kg/build_kg.py:90-101` — `cid = safe(row.get("chembl_compound_id")) ... if cid: key = cid ... elif ikey: key = f"inchikey_{ikey}"` |
| `gnn/preprocess_bipartite_graph.py` (`build_compound_table`) | `chembl_compound_id` | `inchikey_<rdkit_library_standard_inchi_key>` | `gnn/preprocess_bipartite_graph.py:202-204` — `cid = safe(row.get("chembl_compound_id")) ... key = cid or (f"inchikey_{ikey}" if ikey else None)` |
| `hetero_gnn/preprocess.py` (`compound_key_of`) | **`pubchem_cid`** | `inchikey_<rdkit_library_standard_inchi_key>` | `hetero_gnn/preprocess.py:198-202` — `cid = safe(row.get("pubchem_cid")) ... if cid is not None: return f"cid_{cid}" ... ikey = ...; return f"inchikey_{ikey}"` |

**Targets**

| Pipeline | Primary key | Fallback | Quote |
|---|---|---|---|
| `kg/build_kg.py` (`build_target`) | `chembl_target_id` | `slug(target_name)` | `kg/build_kg.py:263-266` — `tcid = safe(row.get("chembl_target_id")); key = tcid or slugify(tname) if tname else None` |
| `gnn/preprocess_bipartite_graph.py` (`build_target_table`) | `chembl_target_id` | `slug(target_name)` | `gnn/preprocess_bipartite_graph.py:243-247` — `tcid = safe(row.get("chembl_target_id")); ... key = tcid or (slugify(tname) if tname else None)` |
| `hetero_gnn/preprocess.py` (`target_key_of`) | **`uniprot_protein_id`** | `tname_<slug(target_name)>` | `hetero_gnn/preprocess.py:208-212` — `uid = safe(row.get("uniprot_protein_id")); if uid is not None: return f"uniprot_{uid}" ... return f"tname_{slugify(tname)}"` |

**Do the three agree? No.** `kg/build_kg.py` and `gnn/preprocess_bipartite_graph.py` use **byte-identical** logic for both compounds and targets — same primary key, same fallback, same fallback trigger condition. `hetero_gnn/preprocess.py` uses a **different primary key for both entity types** (PubChem CID instead of ChEMBL compound ID; UniProt accession instead of ChEMBL target ID), plus it additionally drops 9 rows with a corrupted/implausible `unit_of_measurement` (`hetero_gnn/preprocess.py:167-185`, `drop_corrupted_unit_rows`) *before* any deduplication runs — a filter neither `kg/build_kg.py` nor `gnn/preprocess_bipartite_graph.py` performs (both process every one of the 37,362 raw rows unconditionally).

Note also: `kg/build_kg.py`'s ontology models `Target` (`build_target`, keyed by `chembl_target_id`) and `Protein` (`build_protein`, keyed by `uniprot_protein_id`) as **two separate RDF classes**, linked by `odo:encodedBy` (`kg/build_kg.py:715-716`). So "how many distinct UniProt proteins" is not even the same *concept* as "how many `odo:Target` entities" in the actual knowledge graph — they're different node types by design.

## Task 2 — pandas reproduction from the source spreadsheet

`analysis/reconcile_counts.py`, run directly against `Final ODO Dataset_v2026-06-10.xlsx` with plain pandas (no GNN code, no GraphDB). Extended beyond the two entities Task 1 asked about to cover **every** count in Table 1 (assays, documents, model systems too), since the same two identity-rule pairs extend cleanly to those entity types and the extra check turned out to matter — see below.

```
Raw sheet: 37,362 rows
hetero-rule corrupt-row filter drops 9 rows (37362 -> 37353)

pubchem_cid missing        : 2.9% of rows
uniprot_protein_id missing : 4.1% of rows
chembl_compound_id missing : 9.0% of rows
chembl_target_id missing   : 7.5% of rows

==========================================================================================================================
Rule                                       compounds       targets        assays     documents model_systems    activities
==========================================================================================================================
chembl-id rule, all rows                      13,396            43         4,515         1,165           115        37,362
hetero rule, post-filter                      13,297            73         4,631         1,069           148        37,353
==========================================================================================================================
Table 1 (reported)                            13,297            73         4,631         1,069           148        37,353
gnn/preprocess_bipartite_graph.py print        13,396            43           n/a           n/a           n/a        37,362
==========================================================================================================================

Cross-check against hetero_gnn/processed_hetero_graph.pt (built graph node counts):
  assays          :    4,631  (== Table 1)
  documents       :    1,069  (== Table 1)
  model_systems   :      148  (== Table 1)
```

**All six of Table 1's reported numbers reproduce exactly, from a single row-population choice, with no invented factor — and it is the same choice for every one of them:**

- The **chembl-id rule** (`kg/build_kg.py` / `gnn/preprocess_bipartite_graph.py` — the pipeline that actually builds the RDF knowledge graph) gives **13,396 compounds / 43 targets / 4,515 assays / 1,165 documents / 115 model systems / 37,362 activities**.
- The **hetero rule** (`hetero_gnn/preprocess.py`) gives **13,297 compounds / 73 targets / 4,631 assays / 1,069 documents / 148 model systems / 37,353 activities** — **Table 1, exactly, on all six numbers**, independently cross-checked against the actual built graph's node counts (`hetero_gnn/processed_hetero_graph.pt`), which match to the integer.

Assay/document/model-system identity turn out to disagree between the two rules for reasons beyond just the primary-key column — worth stating plainly since it generalizes the Task 1 finding:

- **Assay**: the chembl rule (`kg/build_kg.py:build_assay`) requires `chembl_assay_id` and returns no node at all for rows without one. The hetero rule (`assay_key_of`) always assigns a key — falling back to a content hash of the assay's descriptive columns (format/method/bioassay/setting + target + document) for the ~9% of rows missing a ChEMBL assay ID, so those rows still merge into shared nodes instead of being dropped.
- **Document**: fallback chains differ (chembl rule stops at `chembl_document_id`; hetero rule adds a `patent_id` fallback) and null handling differs (chembl rule drops; hetero rule buckets everything unresolvable into one shared `"unknown_document"` node).
- **Model system**: the two don't even key off the same source columns. The chembl rule requires `ncit_model_system`/`_id` as its base identity. The hetero rule never reads that column at all — it keys purely off cell/tissue/organism identity columns (`cell_type_name`, `cellosaurus_cell_line_id`, `tissue_name`, `ncit_animal_model`, `ncit_animal_animal_model_strain`, `ncbi_tissue_taxonomy`), with a shared `"unspecified_model_system"` bucket for rows with none of the six.

**Conclusion for Task 2:** Table 1 does not come from a different *identity-rule choice* applied to `kg/build_kg.py`'s own ontology — that pipeline only ever produces 43 targets / 13,396 compounds / 4,515 assays / 1,165 documents / 115 model systems from this spreadsheet, regardless of which of the two chembl-rule scripts runs it (they're the same rule, cross-checked). **Every single number in Table 1 is `hetero_gnn/preprocess.py`'s node/edge count, not the knowledge graph's.**

## Task 3 — where does 73 actually come from

**GraphDB was checked and is not reachable** (`curl -m 3 http://localhost:7200` → connection failed, no HTTP response). Per instructions, no SPARQL query was run and no count was guessed from it.

Without GraphDB, here is what's true from reading `kg/build_kg.py` directly, and it fully answers the question without needing to guess:

- `build_kg.py` writes exactly one class as `Target` (`build_target`, `kg/build_kg.py:261-297`), keyed by `chembl_target_id` with a `target_name`-slug fallback — same rule as `gnn/preprocess_bipartite_graph.py`. On this spreadsheet that rule yields **43** `odo:Target` entities, not 73 (Task 2, verified independently in two ways).
- `Protein` is a **separate** class (`build_protein`, `kg/build_kg.py:300-344`), keyed by `uniprot_protein_id`, linked to `Target` via `odo:encodedBy` — this is the closer conceptual match to `hetero_gnn/preprocess.py`'s target key, but it is still not identical (Protein has its own fallback to `protein_name`, not `target_name`, and there's no guarantee the two pipelines' fallback buckets line up row-for-row).
- **Given Task 2's exact six-for-six numeric match (compounds, targets, assays, documents, model systems, activities — every entity in Table 1 — all from `hetero_gnn/preprocess.py`'s rule), the direct answer is: Table 1 is not a Knowledge Graph table at all. It is the heterogeneous-GNN preprocessing pipeline's node/edge counts, mislabeled in the report as GraphDB/`build_kg.py` entity counts.** It is not what `kg/build_kg.py` would write as `Target` (that's 43, not 73) or as any of its other five classes either — the chembl-rule numbers for all six entity types are all different from Table 1's, and none of the six is a coincidence: they were independently cross-checked against the actual built graph file's node counts (`hetero_gnn/processed_hetero_graph.pt`), which match to the integer on every one.

## Task 4 — the 43-target inventory (the set the GNN(s) actually trained on)

Full table: `analysis/target_inventory.csv` (43 rows, one per `chembl_target_id`/name-fallback key, sorted by row count descending — see file for the complete listing with UniProt ID(s), organism, row count, exact-qualifier row count, and temporal-test-set row count for every target). Classification method and its one known limitation are documented in `analysis/target_inventory.py` (`OPIOID_RECEPTOR_PATTERNS` / `classify_receptor` — regex on `target_name`; a couple of mixed-pharmacology names, e.g. `"mu/kappa opioid receptor"`, get bucketed under whichever single pattern matches first, which the CSV does not hide).

**Two honest readings of "how many are the four opioid receptors," both reported because the question is genuinely ambiguous once you look at the data — target identity fragments by species and by missing-ID fallback:**

| Reading | Targets | Rows | Row share | Test-set rows | Test share |
|---|---|---|---|---|---|
| **Canonical human single-protein entries only** (`CHEMBL233` MOR, `CHEMBL236` DOR, `CHEMBL237` KOR, `CHEMBL2014` NOP) | 4 | 21,040 | 56.3% | 3,618 | 62.2% |
| **Same 4 receptor types, any species or ID-fallback key** (adds rat/mouse/guinea-pig/macaque/zebrafish orthologs, plus 4 targets that are the *same* human receptor re-fragmented under a `target_name`-slug key because `chembl_target_id` was blank on those specific rows) | 27 | 35,673 | 95.5% | 5,591 | 96.0% |

A genuinely non-obvious finding surfaced while building this table: **the 43-target count is itself partly an artifact of the identity rule, not 43 clean biological entities.** `chembl_target_id` is missing on 7.5% of rows; for four of those cases the rows land on a name-slug fallback key (`Mu_opioid_receptor`, `Delta_opioid_receptor`, `Kappa_opioid_receptor`, `Nociceptin_receptor`) that is **species- and receptor-identical** to an existing ChEMBL-ID-keyed target (`CHEMBL233`, `CHEMBL236`, `Kappa_opioid_receptor`→mouse not human so it doesn't collide with `CHEMBL237`, `CHEMBL2014`) but gets counted as a *distinct* target purely because those particular rows lack a ChEMBL target ID. So even "43" undercounts the true fragmentation and overcounts true biological diversity at the same time.

**What the remaining (non-canonical, non-receptor-family) targets are** — 16 of the 43:

- **12 ambiguous/multi-subtype opioid-system entries**: assays that didn't (or couldn't) distinguish a single receptor subtype — e.g. `"Opioid receptor"` (unspecified, rat/mouse), `"Opioid receptors; mu & delta"`, `"Opioid receptors; mu/kappa/delta"`, `"MOR-DOR"` (heterodimer), `"Opioid receptor (mu and kappa)"`, and two zebrafish-specific paralogs (`Delta1a`/`"delta 1b"`) that don't map 1:1 onto the mammalian MOR/DOR/KOR/NOP nomenclature at all.
- **4 genuine non-opioid off-targets**: two dopamine receptors (`D(2)`, `D(3)`, human, 33 and 76 rows) and two neurotensin receptors (`NTS1`, `NTS2`, human, 40 and 17 rows) — classic selectivity/counter-screen targets. **All four have zero exact-qualifier rows** (`n_exact = 0` for all four in `target_inventory.csv`) — meaning, under the GNN's own loss-eligibility rule (`qualifier=='=' and pChEMBL in [2,15]`), **none of these four ever contributes a real regression label**; they can only ever appear as censored/context edges. (The `NTS1`/`NTS2` UniProt fields also list 17 accessions each rather than one — almost certainly a data artifact, flagged here rather than treated as literal.)

**Targets with fewer than 50 rows: 18. Fewer than 10 rows: 10.** (`analysis/target_inventory.py` output / `target_inventory.csv`.) Fourteen of those 18 small targets are still opioid-receptor-family entries (rare species or rare combined-subtype assays); the smallest, at 1 row each, are `CHEMBL5430` (mouse MOR) and `CHEMBL2095151` ("Opioid receptors; delta & kappa", human).

## Task 5 — do the off-targets help or add noise

Vehicle: `baselines/run_baselines.py`'s fingerprint-only RandomForest, reused via import (`compute_metrics`, `RF_PARAMS`, `SEEDS` — unmodified), on the bipartite pipeline's temporal split (`gnn/processed_temporal_2048fp.pt`, the 43-target graph). No GNN training. No tuning — identical `RandomForestRegressor` config and identical 5 seeds in both configurations.

"Opioid-receptor rows only" = the 27-target set from Task 4's broader reading (any species/ID-fallback key for MOR/DOR/KOR/NOP), i.e. every row on-target for one of the four receptor types; excludes the 12 ambiguous-subtype opioid entries and the 4 non-opioid off-targets. Both configurations use the same temporal cutoff; they differ only in which targets' rows are included in train and test.

**(a) Full target set (43 targets)** — already computed for `baselines/summary.md`, reused unchanged:
RMSE = 1.2610 ± 0.0017, MAE = 0.9820 ± 0.0012, r = 0.4651 ± 0.0019, **R² = 0.1797 ± 0.0023** (n_test = 2,628, 5 seeds)

**(b) Opioid-receptor-only (27 targets)** — `analysis/target_ablation.py` (sanity-checked: `CHEMBL233`/human-MOR graph index carries 7,339 edges, matching Task 4's inventory exactly, confirming the target-index reconstruction lines up with the actual graph):

train n = 18,997, test n = 2,611 (vs. 19,751 / 2,628 for the full set — dropping the 16 non-receptor targets removes very few rows, since Task 4 already showed they're 95.5% of all activity rows)

RMSE = 1.2446 ± 0.0033, MAE = 0.9703 ± 0.0020, r = 0.4804 ± 0.0030, **R² = 0.2031 ± 0.0042** (5 seeds: 0.2050, 0.1979, 0.2025, 0.2090, 0.2009)

| Config | n_train | n_test | R² mean | R² sd |
|---|---|---|---|---|
| Full (43 targets) | 19,751 | 2,628 | 0.1797 | 0.0023 |
| Opioid-receptor-only (27 targets) | 18,997 | 2,611 | **0.2031** | 0.0042 |

**R² difference (receptor-only − full) = +0.0234**, roughly 5–10× the per-config seed-to-seed sd (0.0023 / 0.0042). **Verdict: restricting to the four opioid receptors (any species) helps, and the gap is well outside seed noise, not within it.**

**Caveat, stated plainly rather than glossed over:** this compares R² on two *different* test populations (all test rows vs. the receptor-only subset of test rows — 2,628 vs. 2,611, nearly the same size since the excluded 16 targets are a small minority of rows), not the same rows scored by two differently-trained models. The improvement reflects some mix of (a) whether the 16 excluded off-target/ambiguous-subtype rows dilute the shared fingerprint-based training signal and (b) the receptor-only test subset being a slightly different (and here, apparently easier) population on its own — the two effects aren't separated by this design. It is the most informative comparison available without GNN training, and nothing was tuned to produce either number: same fixed `RandomForestRegressor` config, same 5 seeds, same temporal cutoff, in both configurations.

