# Literature Extraction

Downloads and cleans the full-text source papers behind the ODO dataset's
`pubmed_id` column, via NCBI Entrez/PMC. Output is raw material for
LLM-assisted extraction or validation of assay data against the papers
that originally reported it — a preprocessing step, not part of the
KG/GNN pipeline itself.

## Scripts

- **`GetFiles.py`** — reads every unique `pubmed_id` from the ODO dataset
  (`Final ODO Dataset_v2026-06-10.xlsx` at the repo root), looks up each
  PubMed ID's linked PMC full-text record via `Bio.Entrez`, and downloads
  the full-text XML to `Full_Text_Articles/{pmid}.xml`. Rate-limited to
  NCBI's ~3 requests/second policy.
- **`CleanFiles.py`** — parses each `Full_Text_Articles/*.xml`, strips
  reference lists/tables/figures, extracts the abstract + body paragraph
  text, and writes plain text to `Cleaned_Text_Articles/{pmid}.txt`.

## Usage

```bash
# set your own contact email — required by NCBI's Entrez usage policy
export ENTREZ_EMAIL="you@example.com"

conda run -n odo python3 literature/GetFiles.py
conda run -n odo python3 literature/CleanFiles.py
```

Requires `pandas`, `biopython`, `beautifulsoup4`, `lxml`, `openpyxl` in the
`odo` conda environment (not currently pinned in `requirements_gnn.txt`,
which covers only the GNN pipeline — install separately if needed).

## Output

`Full_Text_Articles/` and `Cleaned_Text_Articles/` are gitignored — not
committed, entirely regenerable by re-running the two scripts against the
current dataset (930+ papers, ~127MB combined at time of writing).

Not every `pubmed_id` has a linked PMC full-text record — `GetFiles.py`
prints `"No PMC link"` for those and simply skips them.

## `chembl_extractor/` — automatic ChEMBL data extraction

A separate tool: given a ChEMBL compound ID (or SMILES), fetches and
combines data from **6 external sources** (ChEMBL, PubChem, UniProt,
InterPro, OLS4/BTO, PubMed) plus two trained ML models (assay-format
classifiers, QikProp-style property predictors) into a fully-populated
131-column row matching the ODO database schema — i.e. it auto-generates
new database rows for a given compound, rather than working from existing
rows like `GetFiles.py`/`CleanFiles.py` do. Reached ~81% field-level
accuracy against the real database after several rounds of NLP-assisted
refinement; full write-up (Hebrew) in
[`chembl_extractor/EXPLANATION_HE.md`](chembl_extractor/EXPLANATION_HE.md).

```bash
cd literature/chembl_extractor
pip install -r requirements.txt

# Step 1 — train the two supporting ML models (once)
python3 train_qikprop_models.py
python3 train_nlp_models.py

# Step 2a — CLI: fetch one or more compounds
python3 chembl_fetcher.py --ids CHEMBL101454 --output outputs/result.xlsx

# Step 2b — or the web UI (http://localhost:5050)
python3 app.py

# Step 3 — optional: score output accuracy against the real DB
python3 compare_accuracy.py --output outputs/result.xlsx
# or sample-evaluate the fetcher directly against the DB
python3 evaluate_accuracy.py --fraction 0.05
```

`labeled_data_from_manual_pdfs.json` is manually-labeled ground truth
(~2,126 rows) used to validate the NLP-predicted fields during development.

`nlp_models/`, `qikprop_models/`, and `outputs/` (all model/run artifacts)
are gitignored — regenerate with the training/fetch commands above.
