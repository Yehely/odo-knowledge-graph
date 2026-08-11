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
