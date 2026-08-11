"""
CleanFiles.py – strips PMC full-text XML (from GetFiles.py) down to plain
abstract + body text, ready for downstream LLM-based extraction/validation.
Tables are converted to marked-up plain text (not dropped) since binding-
affinity values usually live in tables — llama_extractor/LlamaExtractor.py
looks for the "--- TABLE START/END ---" markers this produces.
"""
import os

from bs4 import BeautifulSoup

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Full_Text_Articles")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cleaned_Text_Articles")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def clean_xml_robust(xml_file):
    with open(xml_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml-xml")

    # strip references and figures — noise for text extraction (tables are kept, see below)
    for tag in soup.find_all(["ref-list", "fig"]):
        tag.decompose()

    # convert tables to marked-up plain text instead of dropping them
    for table in soup.find_all("table-wrap"):
        table_text = "\n--- TABLE START ---\n"
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            row_text = " | ".join(col.get_text(separator=" ").strip() for col in cols)
            table_text += row_text + "\n"
        table_text += "--- TABLE END ---\n"
        table.replace_with(table_text)

    # extract abstract
    abstract = soup.find("abstract").get_text(separator=" ") if soup.find("abstract") else ""

    # extract body — keep paragraph/table breaks
    body_text = ""
    body_tag = soup.find("body")
    if body_tag:
        body_text = body_tag.get_text(separator="\n\n")

    return f"ABSTRACT:\n{abstract}\n\nBODY:\n{body_text}"


if __name__ == "__main__":
    print("Starting cleaning (tables preserved)...")
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".xml"):
            pmid = filename.replace(".xml", "")
            try:
                full_text = clean_xml_robust(os.path.join(INPUT_DIR, filename))

                if len(full_text) < 500:
                    print(f"Warning: PMID {pmid} has very short text.")

                with open(f"{OUTPUT_DIR}/{pmid}.txt", "w", encoding="utf-8") as f:
                    f.write(full_text)

                print(f"PMID {pmid}: Cleaned.")
            except Exception as e:
                print(f"PMID {pmid}: Error - {e}")
