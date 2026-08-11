# llama_extractor/

Runs an LLM (Llama 3.3, via the Groq API) over paper text to extract
drug-target binding experiments — a different approach from
[`../chembl_extractor/`](../chembl_extractor/) (which pulls from
structured external databases + trained ML models): this one reads the
actual paper text and asks the model to pull out experiment rows
directly, including for compounds that never made it into any database.

Two entry points depending on where the paper text comes from, plus a
validation script.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-groq-key"   # https://console.groq.com
```

## 1. `LlamaExtractor.py` — from PMC full text

Reads every `.txt` file in `../Cleaned_Text_Articles/` (produced by
`../GetFiles.py` + `../CleanFiles.py`), looks for the `--- TABLE START/END
---` markers `CleanFiles.py` inserts, and asks Llama to extract compound/
target/value rows from just the tables (falling back to the last ~12k
characters of the paper if no tables were found). Saves
`labeled_data_from_llama.json`.

```bash
python3 LlamaExtractor.py              # all cleaned articles
python3 LlamaExtractor.py --limit 10   # first 10 only, for a quick test
```

## 2. `LlamaExtractorDownloaded.py` — from manually downloaded PDFs

For papers with no PMC full text available (paywalled, no PMC link,
etc.), sourced by hand instead. Reads every PDF in
[`manual_pdfs/`](manual_pdfs/) — **see `manual_pdfs/README.md` before
using this** — extracts text with PyMuPDF, and asks Llama to extract
experiment rows. Saves `labeled_data_from_manual_pdfs.json`.

```bash
python3 LlamaExtractorDownloaded.py
```

## 3. Validating the output

- **`Comparison.py`** — for each extracted `(pmid, drug, value)`, checks
  whether a matching row exists in the ODO dataset (allowing for direct
  matches and pKi/nM unit conversions) and reports Verified / New Info /
  Mismatch counts.

  ```bash
  python3 Comparison.py --input labeled_data_from_llama.json
  python3 Comparison.py --input labeled_data_from_manual_pdfs.json
  ```

- **`compare_all_rows.py`** — a per-article, human-readable dump instead
  of aggregate stats: for each PMID, prints the dataset's existing rows
  side-by-side with what Llama extracted, for manual spot-checking.

  ```bash
  python3 compare_all_rows.py --input labeled_data_from_llama.json
  ```

`COL_VALUE`/`COL_DRUG` column names at the top of both scripts should be
double-checked against the dataset's actual current column names before
relying on results — they were carried over from the source repo as-is.

`labeled_data_from_llama.json` in this folder is a small (12KB) sample
output from a real `LlamaExtractor.py` run, kept as a reference for what
the extraction format looks like.

## Security note

The original source of `LlamaExtractorDownloaded.py` had a real Groq API
key hardcoded in it. That key was **not** carried over — both scripts here
read `GROQ_API_KEY` from the environment instead. If you have access to
the original key, treat it as compromised and rotate it.
