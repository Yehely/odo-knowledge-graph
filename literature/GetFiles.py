"""
GetFiles.py – downloads PMC full-text XML for every pubmed_id referenced
in the ODO dataset, via NCBI Entrez. Source paper for each activity row,
kept locally as raw material for literature-backed extraction/validation
of the dataset (see literature/README.md).
"""
import os
import time

import pandas as pd
from Bio import Entrez

# NCBI requires a contact email for Entrez API usage — set your own before running.
Entrez.email = os.environ.get("ENTREZ_EMAIL", "your-email@example.com")

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "Final ODO Dataset_v2026-06-10.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Full_Text_Articles")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def download_pmc_fulltext(pmid):
    try:
        # find the linked PMC id
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        record = Entrez.read(handle)
        handle.close()

        if not record[0]["LinkSetDb"]:
            return False, "No PMC link"

        pmcid = record[0]["LinkSetDb"][0]["Link"][0]["Id"]

        # fetch the full-text XML
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
        content = handle.read()
        handle.close()

        with open(f"{OUTPUT_DIR}/{pmid}.xml", "wb") as f:
            f.write(content)
        return True, "Downloaded"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    df = pd.read_excel(DATASET_PATH)
    unique_pmids = df["pubmed_id"].dropna().unique().astype(int).astype(str).tolist()

    print(f"Starting download for {len(unique_pmids)} PMIDs...")
    for pmid in unique_pmids:
        success, msg = download_pmc_fulltext(pmid)
        print(f"PMID {pmid}: {msg}")
        time.sleep(1 / 3)  # NCBI rate limit (~3 req/sec)
