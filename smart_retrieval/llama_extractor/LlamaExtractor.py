"""
LlamaExtractor.py – runs an LLM (Llama 3.3 via Groq) over
smart_retrieval/Cleaned_Text_Articles/*.txt (from CleanFiles.py) to extract
drug-target binding experiments from tables, saved as
labeled_data_from_llama.json.
"""
import json
import os
import re
import time
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# requires a Groq API key — https://console.groq.com
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.environ["GROQ_API_KEY"],
)


class SingleExperiment(BaseModel):
    drug_name: str = Field(description="The compound identifier found in the table (e.g., '7a', 'Compound 12', 'Morphine').")
    target_name: str = Field(description="The receptor name (e.g., 'Mu opioid receptor', 'Delta opioid receptor', 'Kappa opioid receptor'). Avoid abbreviations like 'DOP' if possible.")
    value_numeric: str = Field(description="The numerical value extracted (e.g., 1.2, 50, 10000).")
    units: str = Field(description="Units of measurement (e.g., nM, uM, %).")
    measure_type: str = Field(description="The type of value (e.g., Ki, IC50, EC50).")
    source_sentence: str = Field(description="COPY THE EXACT ROW content from the table text given. Do not invent sentences.")


class ExtractionResult(BaseModel):
    experiments: List[SingleExperiment]


parser = JsonOutputParser(pydantic_object=ExtractionResult)

template = """
You are a strict scientific data extractor.
I will provide you with TEXT extracted from TABLES in a research paper.

Your Goal: Extract experimental results for NEW compounds synthesized in this paper.

RULES:
1. FOCUS ON COMPOUNDS: Look for numbered compounds (e.g., "7a", "4b", "Compound 12").
2. IGNORE CONTROLS: Do not extract well-known drugs (like 'Naltrindole', 'DAMGO') unless they are compared directly in the same table.
3. EXACT VALUES: Extract the numeric value, units, and measure type (Ki/IC50) exactly as shown.
4. TARGET NAMES: Normalize names to: 'Mu opioid receptor', 'Delta opioid receptor', 'Kappa opioid receptor'.
5. SOURCE: For 'source_sentence', strictly copy the raw text of the table row where you found the data.

Format instructions:
{format_instructions}

Table Data / Text to analyze:
{text}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Cleaned_Text_Articles")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "labeled_data_from_llama.json")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (default: all)")
    args = ap.parse_args()

    if not os.path.exists(INPUT_DIR):
        raise SystemExit(f"'{INPUT_DIR}' not found — run GetFiles.py and CleanFiles.py first.")

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))
    files_to_process = files[: args.limit] if args.limit else files

    print(f"Starting extraction on {len(files_to_process)} files using TABLE-ONLY method...")

    final_output_data = []
    for filename in files_to_process:
        pmid = filename.replace(".txt", "")  # PMID comes from the filename

        try:
            with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
                full_text = f.read()

            # look for the table markers CleanFiles.py inserts
            tables = re.findall(r"--- TABLE START ---(.*?)--- TABLE END ---", full_text, re.DOTALL)

            if tables:
                print(f"   {filename}: Found {len(tables)} tables. Extracting data from tables only.")
                text_chunk = "\n\n".join(tables)[:15000]
            else:
                print(f"   {filename}: No table markers found. Using last 12k chars.")
                text_chunk = full_text[-12000:]

            result = chain.invoke({"text": text_chunk})

            if result and "experiments" in result:
                count = 0
                for exp in result["experiments"]:
                    exp["pmid"] = pmid
                    final_output_data.append(exp)
                    count += 1
                print(f"      Extracted {count} records.")

            time.sleep(1)  # avoid Groq rate limits

        except Exception as e:
            print(f"   Error processing {filename}: {e}")

    print(f"\nSaving {len(final_output_data)} total records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output_data, f, indent=4, ensure_ascii=False)

    print("Done! Now run Comparison.py to validate against the dataset.")
