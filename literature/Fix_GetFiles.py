"""
Fix_GetFiles.py – like GetFiles.py, but verifies each download: fetches the
real title from PubMed and checks it actually appears in the downloaded PMC
XML before saving, to catch PMC linking to the wrong article. Slower than
GetFiles.py (one extra API call per PMID) but safer against silently
corrupting the corpus with mismatched articles.
"""
import os

import pandas as pd
from Bio import Entrez

Entrez.email = os.environ.get("ENTREZ_EMAIL", "your-email@example.com")

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.pardir, "Final ODO Dataset_v2026-06-10.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Verified_Full_Text")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def fetch_pubmed_title(pmid):
    try:
        handle = Entrez.esummary(db="pubmed", id=pmid)
        record = Entrez.read(handle)
        handle.close()
        return record[0]["Title"]
    except Exception:
        return None


def download_correct_xml(pmid):
    print(f"\nChecking PMID: {pmid}...")

    real_title = fetch_pubmed_title(pmid)
    if not real_title:
        print("   Could not find title in PubMed.")
        return
    print(f"   Real title: {real_title[:50]}...")

    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        record = Entrez.read(handle)
        handle.close()
    except Exception:
        print("   Error connecting to PMC link.")
        return

    if not record or not record[0]["LinkSetDb"]:
        print("   No full text available in PMC (not open access).")
        return

    pmcid = record[0]["LinkSetDb"][0]["Link"][0]["Id"]

    try:
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
        xml_content = handle.read()
        handle.close()
    except Exception:
        print("   Error fetching XML content.")
        return

    # crude but effective: check the real title's first few words actually
    # appear in the downloaded XML, to filter out completely different articles
    first_words = " ".join(real_title.split()[:5]).lower()
    try:
        xml_str = xml_content.decode("utf-8").lower()
    except Exception:
        xml_str = str(xml_content).lower()

    if first_words not in xml_str:
        print("   MISMATCH DETECTED! Downloaded article seems different.")
        print(f"      Expected: {first_words}...")
        print("      Skipping this file to avoid data corruption.")
        return

    filename = f"{OUTPUT_DIR}/{pmid}.xml"
    with open(filename, "wb") as f:
        f.write(xml_content)
    print(f"   Success! Verified and saved to {filename}")


if __name__ == "__main__":
    df = pd.read_excel(DATASET_PATH)
    unique_pmids = df["pubmed_id"].dropna().astype(str).unique().tolist()

    print("Starting verified download process...")
    for pmid in unique_pmids:
        download_correct_xml(pmid)
