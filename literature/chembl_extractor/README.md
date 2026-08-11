# chembl_extractor/

Given a ChEMBL compound ID or SMILES, fetches and combines data from **6
external sources** (ChEMBL, PubChem, UniProt, InterPro, OLS4/BTO, PubMed)
plus two trained ML models (an NLP assay-format classifier, QikProp-style
property predictors) into a fully-populated **131-column row** matching
the ODO database schema. Auto-*generates* new candidate database rows for
a compound — a different job from `../GetFiles.py`/`../CleanFiles.py`,
which work from papers behind rows that already exist.

Reached ~81% field-level accuracy against the real database after several
rounds of NLP-assisted refinement. Full write-up (Hebrew):
[`EXPLANATION_HE.md`](EXPLANATION_HE.md).

Ported from the `Calude` branch of `Yehely/LLM-data` — 14 manually
downloaded copyrighted journal PDFs and their dependent comparison
scripts were left out, along with regenerable run output.

## Setup

```bash
pip install -r requirements.txt

# train the two supporting ML models (once)
python3 train_qikprop_models.py
python3 train_nlp_models.py
```

`> If the model directories are empty, the extractor still runs, but the
ML-predicted columns are left blank — make sure training finished first.`

## Usage

```bash
# CLI — single or multiple compounds
python3 chembl_fetcher.py --ids CHEMBL101454 --output outputs/result.xlsx
python3 chembl_fetcher.py --ids CHEMBL101454 CHEMBL2021537 --output outputs/result.xlsx
python3 chembl_fetcher.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --output outputs/result.xlsx
python3 chembl_fetcher.py --file molecules.txt --output outputs/result.xlsx   # one ID per line

# or the web UI
python3 app.py   # -> http://localhost:5050
```

## Accuracy evaluation (optional)

```bash
# score a specific output file against the real DB
python3 compare_accuracy.py --output outputs/result.xlsx
python3 compare_accuracy.py --output outputs/result.xlsx --report outputs/accuracy_report.xlsx

# or sample-evaluate the fetcher directly against the DB (fetches + compares in one go)
python3 evaluate_accuracy.py --fraction 0.05
```

`labeled_data_from_manual_pdfs.json` is Llama-extracted data from
manually-downloaded PDFs (produced by
[`../llama_extractor/LlamaExtractorDownloaded.py`](../llama_extractor/) —
not hand-labeled) used as a reference/validation set during development.

## Output

`nlp_models/`, `qikprop_models/`, and `outputs/` are gitignored — all
regenerable via the commands above.
