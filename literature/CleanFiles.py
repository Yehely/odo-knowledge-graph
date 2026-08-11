"""
CleanFiles.py – strips PMC full-text XML (from GetFiles.py) down to plain
abstract + body text, ready for downstream LLM-based extraction/validation.
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

    # strip references and complex tables/figures — noise for text extraction
    for tag in soup.find_all(["ref-list", "table-wrap", "fig"]):
        tag.decompose()

    # extract abstract
    abstract = soup.find("abstract").get_text(separator=" ") if soup.find("abstract") else ""

    # extract body — walk paragraphs to preserve order
    body_text = ""
    body_tag = soup.find("body")
    if body_tag:
        paragraphs = body_tag.find_all("p")
        body_text = "\n".join([p.get_text(separator=" ") for p in paragraphs])

    return abstract + "\n\n" + body_text


if __name__ == "__main__":
    print("Starting cleaning...")
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
