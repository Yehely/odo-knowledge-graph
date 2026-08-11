# manual_pdfs/

Place manually downloaded PDF articles here before running
`LlamaExtractorDownloaded.py`. This is for papers that aren't available as
PMC full-text XML via `smart_retrieval/GetFiles.py` (no PMC link, paywalled,
etc.) and were sourced by hand instead.

PDFs are **not** committed to this repo — most are under journal copyright
and can't be redistributed. This folder exists purely as the expected
input location; it's gitignored except for this file.

Name each file `{pubmed_id}.pdf` (e.g. `24107104.pdf`) — the filename
(minus extension) is used as the PMID when extracting.
