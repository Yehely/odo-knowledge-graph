# Heterogeneous GNN Architecture — ODO Opioid Binding-Affinity Predictor

A 5-node-type Heterogeneous GNN (Compound, Target, Assay, Model System,
Document) predicting pKi for compound–target binding.

---

## 1. Data Preprocessing — Graph Inclusion & Loss Masking

**Graph:** every row builds a `binds_to` edge (Compound ↔ Target),
regardless of `endpoint` type (Ki, IC50, EC50, Emax, Inhibition, Activity,
Kd, etc.) — except:

- **Dropped entirely:** rows with a corrupted `unit_of_measurement`
  (e.g. `'M'` with implausible Ki-scale values such as 168, 188, or unit
  missing on scale-ambiguous rows).

**Label (pKi):** target label is `pchembl_value` (= −log10(Ki in
**Molar**)). Where `pchembl_value` is missing but a usable
`endpoint_value` + `unit_of_measurement` exists, derive it:

1. Convert value to Molar (`nM × 1e-9`).
2. `pKi = -log10(value_M)`.
3. Flip the qualifier direction, since `-log10` is decreasing:
   `>` → `<`, `<` → `>`, `>=` → `<=`, `=` stays `=`.

**Loss has two components**, both restricted to `endpoint == 'Ki'` edges
with a resolvable qualifier (i.e. never `#ERROR!`/unrecoverable rows, and
never non-Ki endpoints — those still participate fully in message passing,
just contribute no loss term):

1. **Exact loss (MSE)** — for rows where `endpoint_qualifier == '='` and
   `MIN_PCHEMBL <= pchembl <= MAX_PCHEMBL`:
   ```
   loss_exact = (predicted_pKi - true_pKi)^2
   ```
2. **Censored loss (one-sided hinge)** — for rows with a derived pKi bound
   instead of an exact value. Branch on the **original, raw**
   `endpoint_qualifier` from the source data (`ŷ` = predicted pKi, `y` =
   derived bound pKi, per §1.2):

   - Raw qualifier `<` (Ki < threshold → strong binder → true pKi is
     *above* `y`, i.e. `y` is a floor): penalize only if the prediction
     falls below the floor.
     ```
     Loss = (max(0, y - ŷ))²
     ```
   - Raw qualifier `>` (Ki > threshold → weak binder → true pKi is *below*
     `y`, i.e. `y` is a ceiling): penalize only if the prediction rises
     above the ceiling.
     ```
     Loss = (max(0, ŷ - y))²
     ```

   Either way, a prediction that already respects the known bound gets zero
   loss — the exact true value beyond that bound is unknown, so there's
   nothing to penalize.

Total loss per batch: sum of `loss_exact` over exact-qualifier edges plus
`loss_censored` over censored-qualifier edges (weight the two terms if the
censored term needs down-weighting relative to exact labels — tune
empirically).

Rows with an unrecoverable qualifier (`#ERROR!`) or a non-Ki endpoint
contribute to neither loss term but **remain in the graph** and participate
fully in message passing.

---

## 2. Node Definitions

### A. Compound Node
*Primary key: `pubchem_cid`. Always the tested compound itself, never the parent compound.*

| Group | Columns |
|---|---|
| Structure | `rdkit_canonical_smiles`, `rdkit_library_standard_inchi`, `rdkit_library_standard_inchi_key`, `rdkit_molecular_forula`, `rdkit_molecular_weight` |
| PK (ChEMBL) | `chembl_#ro5_violations`, `chembl_alogp` |
| Computed (QikProp) | `qikprop_dipole`, `qikprop_sasa`, `qikprop_fisa`, `qikprop_donor_hb`, `qikprop_accpt_hb`, `qikprop_qplog_pw`, `qikprop_qplog_po/w`, `qikprop_qplogs`, `qikprop_qplog_khsa`, `qikprop_percent_human_oral_absorption` |

### B. Target Node
*Primary key: `uniprot_protein_id`.*

| Group | Columns |
|---|---|
| Names/classification | `target_name`, `target_type`, `protein_name`, `pro_protein_name_id` |
| Taxonomy | `ncbi_target_taxonomy`, `ncbi_target_taxonomy_id` |
| Protein families | `interpro_protein_family_name(_id)`, `interpro_protein_category(_id)`, `ncit_protein_subfamily_name(_id)`, `dto_gpcr_category(_id)` |

### C. Assay Node
*Primary key: `chembl_assay_id`.*

| Group | Columns |
|---|---|
| Experimental setting | `bao_experimental_setting(_id)` |
| Format/structure | `bao_assay_format(_id)`, `bao_assay_format_l2(_id)`, `bao_assay_format_l3(_id)`, `subcellular_format`, `bao_subcellular_format_id` |
| Method/detection | `bao_assay_method(_id)`, `physical_detection_method(_id)`, `single_concentration_screen` |
| Sub-categories | `bao_bioassay_1`, `bao_bioassay_2`, `bao_bioassay_type` |
| Reference reagent | `reference_radiolabeled_molecular_entity` |
| Signaling | `embl_ebi_gpcr_signaling_pathway(_id)` |
| In-vivo dosing | `chembl_dose_administered`, `ncit_route_of_administration(_id)` |

Use the text-label column for the model feature wherever a `*_id` URI twin
exists (e.g. `bao_bioassay_1` over `bao_bioasssay_1_id`); keep the URI only
for `owl:sameAs` linking in the RDF graph, not as a duplicate model input.

### D. Model System Node
*Primary key: derived from cell line / tissue / organism identity.*

| Group | Columns |
|---|---|
| Cell lines | `cell_line_name`, `cellosaurus_cell_line_id`, `clo_cell_line_id` |
| Tissue | `tissue_name`, `mba_tissue_id`, `pato_tissue_id`, `bto_tissue_id`, `ncbi_tissue_taxonomy(_id)` |
| Animal models | `ncit_animal_model`, `ncit_animal_animal_model_strain(_id)`, `mgi_animal_strain_id`, `ncit_vertebrate_taxonomy(_id)`, `ncit_model_system(_id)` |

### E. Document Node
*Primary key: `pubmed_id`, fallback `document_doi`, fallback `patent_id`,
fallback an explicit `unknown_document` bucket.*

| Group | Columns |
|---|---|
| Source | `document_journal`, `document_year`, `patent_id` |

This node's role is to absorb per-publication batch effects (shared lab
protocol/equipment across all rows from the same paper) so they don't leak
into the Compound/Target embeddings — not to be a strong direct predictor.
Keep its dimensionality small and `document_year`'s weight low.

---

## 3. Edge Definition (`binds_to`)

| Group | Columns |
|---|---|
| Endpoint type | `endpoint` (one-hot: Ki, IC50, EC50, Inhibition, Activity, Binding, other) |
| Label | `endpoint_value` + `unit_of_measurement`(+`uo_id`), or `pchembl_value` (canonical, see §1) |
| Qualifier | `endpoint_qualifier` (`=`, `<`, `>`, `>=`, `~`, plus an explicit bucket for unrecoverable/corrupted values) |
| Pharmacological role | `compound_pharmacological_role`, `chebi_compound_pharmacological_role_id`, `chembl_binding_site_description` |

The edge vector carries only what varies *per measurement* — everything
constant per assay lives on the Assay node instead (§2.C).

---

## 4. Excluded Data

Removed at ETL time, never entering the graph:

- Invalidated ChEMBL IDs: `chembl_compound_id`, `chembl_target_id`,
  `chembl_document_id`, `chembl_source_id`, `chembl_chemical_entity_key`.
- Non-learning names: `pubchem_iupac_name`, `chembl_chemical_entity_name`.
- Parent-structure duplicates: `rdkit_parent_structure_smiles`,
  `rdkit_parent_structure_inchi`, `rdkit_parent_structure_inchi_key`,
  `rdkit_parent_structure_molecular_formula`,
  `rdkit_parent_structure_molecular_weight`.
- Reference (non-tested) compounds: `bao_reference_compound`,
  `dose_reference_compound`.
- Error margins (predicting absolute value, not uncertainty):
  `sem_endpoint_qualifier`, `sem_value`, `cl_lower_95%`, `cl_upper_95%`.
- QC "unmapped" flags: `compound_pharmacological_role_unmapped`,
  `subcellular_format_unmapped`, `assay_kit_unmapped`,
  `cell_line_unmapped`, `tissue_unmapped`.
- Free text/system metadata: `assay_description`, `odo_assay_property`,
  `chembl_assay_property`, `odo_functional_bias_assay_1`,
  `odo_functional_bias_assay_2`, `chembl_assay_parameters`,
  `bias_assay_2_cell_line`, `mi_database_citation(_id)`,
  `source_description`, `odo_supplementary_reference`,
  `odo_assay_endpoint_description`.
- Compound clinical/labeling metadata: `chembl_molecule_max_phase`.
- Assay kit brand names: `assay_kit`, `bao_assay_kit_id`.
- 9 rows with corrupted/implausible `unit_of_measurement` (§1).

---

## 5. Vector Dimensions per Node/Edge

Assay, Model System, and Document dimensions are not fixed by hand — they're
chosen by hyperparameter search (grid/random search across training
iterations, selecting the value with the best validation loss), same as any
other tunable hyperparameter in this pipeline.

| Entity | Dimension | Composition |
|---|---|---|
| **Compound** | ~2,058 | 2,048 (Morgan ECFP4) + ~10 (ADMET, `StandardScaler`-normalized) |
| **Target** | 45 | 4 (target_type one-hot) + 9 (taxonomy one-hot) + 32 (Learned Embedding for `uniprot_protein_id`) |
| **Assay** | search range 32–64 | Learned Embedding covering format/method/bioassay categories, reference reagent, signaling pathway, and sparse in-vivo dose/route fields |
| **Model System** | search range 8–24 | Compact Learned Embedding for cell line/tissue identity |
| **Document** | search range 4–12 | Tiny Learned Embedding for journal (19 categories) + `is_patent` flag + normalized year + `unknown_document` bucket |
| **Edge** (`binds_to`) | ~14–16 | 7 (endpoint-type one-hot) + ~4 (qualifier one-hot) + ~3 (pharmacological role/binding-site encoding) |

---

## 6. Encoders (shared latent space)

Every node type's raw vector passes through its own encoder block into a
common **256-dimensional latent space**:

```
Linear → BatchNorm → ReLU → Dropout(0.4)
```

- Compound: 2,058 → 256 (compression)
- Target: 45 → 256 (expansion)
- Assay: ~48 → 256 (expansion)
- Model System: ~16 → 256 (expansion)
- Document: ~8 → 256 (expansion)

---

## 7. Message Passing — 2-Hop HeteroSAGE

Capped at 2 hops to avoid oversmoothing across 5 node types.

**Hop 1 — collect into Assay:**
Each neighbor of an Assay node (Document, Model System, Target) gets a
learned attention score (a small `Linear → Tanh → Linear` head over its
256-dim vector), softmax-normalized per Assay, and the neighbors are
summed with those weights (attention-weighted pooling, not a plain mean).
Concatenate the pooled vector with the Assay's own vector, multiply by a
learned weight matrix `W1`, apply `LeakyReLU`.

**Hop 2 — collect into Compound and Target:**
Compound attention-pools the updated 256-dim vectors of every Assay it
connects to (same scoring-head + softmax + weighted-sum scheme as hop 1,
via its own attention head), concatenates with its own vector, multiplies
by `W2`, applies `LeakyReLU`. Target receives the symmetric update from the
same Assay neighborhood, via its own analogous attention head and weight
matrix, so both sides of the edge have an enriched vector for the
prediction head.

Result: Compound (256) encodes structure + experimental history + receptor
tendencies; Target (256) encodes its classification + the experimental/
tissue contexts it's been tested in.

---

## 8. Prediction Head

**Concatenation:**
`Enriched Compound (256) ‖ Enriched Target (256) ‖ Edge vector (~14–16, passed through a narrow linear layer)` → unified vector (~530 dims).

**Edge Head MLP:**
```
~530 → Linear → ReLU → Dropout → 256 → Linear → ReLU → 128 → Linear → 1 (no activation)
```
Output: continuous predicted pKi (regression, no final activation).

---

## 9. Train/Test Split & Optimization

- **Split:** Temporal — train on activities published before **2015**
  (~79.4% of the corpus), test on **2015 onward** (~20.6%). Validation is a
  random 10% split of the pre-cutoff (training) activities.
- **Loss:** combined exact MSE + censored one-sided hinge loss on Ki edges,
  as defined in §1. Non-Ki and unrecoverable-qualifier edges contribute no
  loss term but remain in the graph for message passing.
- **Optimizer:** Adam, backpropagating through the MLP head, `W1`/`W2`
  message-passing matrices, and all node-type Learned Embeddings.
