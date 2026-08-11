# Literature Extraction

Tools for acquiring, cleaning, and extracting structured data from the
literature behind the ODO dataset. Not part of the KG/GNN pipeline itself
— a preprocessing/data-enrichment layer that feeds it.

```
GetFiles.py / Fix_GetFiles.py  →  CleanFiles.py  →  sortFullText.py (optional)
                                        │
                                        ├──→ chembl_extractor/  (API + ML based)
                                        └──→ llama_extractor/   (LLM based)
```

## Paper acquisition

- **`GetFiles.py`** — reads every unique `pubmed_id` from the ODO dataset
  (`Final ODO Dataset_v2026-06-10.xlsx` at the repo root), looks up each
  PubMed ID's linked PMC full-text record via `Bio.Entrez`, and downloads
  the full-text XML to `Full_Text_Articles/{pmid}.xml`. Rate-limited to
  NCBI's ~3 requests/second policy.
- **`Fix_GetFiles.py`** — a slower, verified variant: fetches the real
  title from PubMed and confirms it actually appears in the downloaded PMC
  XML before saving (to `Verified_Full_Text/`), catching PMC linking to
  the wrong article. Use this if you suspect `GetFiles.py` pulled in
  mismatched articles.
- **`CleanFiles.py`** — parses each `Full_Text_Articles/*.xml`, strips
  reference lists/figures, converts tables to marked-up plain text
  (`--- TABLE START/END ---`, since binding-affinity values usually live
  in tables), and writes the result to `Cleaned_Text_Articles/{pmid}.txt`.
- **`sortFullText.py`** *(optional)* — splits `Cleaned_Text_Articles/`
  into `Sort_Full_Text_And_Not/{Full_Text,Not_Full_Text}/` using a
  keyword+length heuristic, to filter out abstract-only stubs before
  running an extractor over them.

```bash
export ENTREZ_EMAIL="you@example.com"   # required by NCBI's Entrez usage policy

conda run -n odo python3 literature/GetFiles.py
conda run -n odo python3 literature/CleanFiles.py
```

Requires `pandas`, `biopython`, `beautifulsoup4`, `lxml`, `openpyxl` in the
`odo` conda environment (not currently pinned in `requirements_gnn.txt`,
which covers only the GNN pipeline — install separately if needed).

Not every `pubmed_id` has a linked PMC full-text record — `GetFiles.py`
prints `"No PMC link"` for those and simply skips them.

`Full_Text_Articles/`, `Verified_Full_Text/`, `Cleaned_Text_Articles/`,
and `Sort_Full_Text_And_Not/` are all gitignored — not committed, entirely
regenerable by re-running the scripts above against the current dataset
(930+ papers, ~127MB combined at time of writing).

## `chembl_extractor/` — automatic ChEMBL data extraction

A separate tool: given a ChEMBL compound ID (or SMILES), fetches and
combines data from **6 external sources** (ChEMBL, PubChem, UniProt,
InterPro, OLS4/BTO, PubMed) plus two trained ML models (assay-format
classifiers, QikProp-style property predictors) into a fully-populated
131-column row matching the ODO database schema — i.e. it auto-generates
new database rows for a given compound, rather than working from existing
rows the way the paper-acquisition scripts above do. Reached ~81%
field-level accuracy against the real database after several rounds of
NLP-assisted refinement. See
[`chembl_extractor/README.md`](chembl_extractor/README.md).

## `llama_extractor/` — LLM-based experiment extraction

A third tool: runs Llama 3.3 (via Groq) directly over paper text — from
`Cleaned_Text_Articles/` or from manually-downloaded PDFs — to extract
drug-target binding experiments, including for compounds that never made
it into any structured database. See
[`llama_extractor/README.md`](llama_extractor/README.md).
