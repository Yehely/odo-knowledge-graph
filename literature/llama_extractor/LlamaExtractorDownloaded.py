"""
LlamaExtractorDownloaded.py – runs an LLM (Llama 3.3 via Groq) over manually
downloaded PDF articles (see manual_pdfs/README.md) to extract drug-target
binding experiments, saved as labeled_data_from_manual_pdfs.json.
"""
import json
import os
import time
from typing import List

import fitz  # PyMuPDF
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
    drug_name: str = Field(description="Name of the drug or compound (e.g., Morphine)")
    target_name: str = Field(description="Name of the protein target/receptor (e.g., MOR, Mu Opioid Receptor)")
    value_numeric: str = Field(description="The numerical value only (e.g., 1.2, 50)")
    units: str = Field(description="Units of measurement (e.g., nM, uM)")
    measure_type: str = Field(description="Type of measure (e.g., IC50, Ki, EC50)")
    source_sentence: str = Field(description="The exact sentence or table row from the text where this was found.")


class ExtractionResult(BaseModel):
    experiments: List[SingleExperiment]


parser = JsonOutputParser(pydantic_object=ExtractionResult)

prompt = PromptTemplate(
    template="You are a professional pharmacologist. Extract all drug-target experiments from the text.\n{format_instructions}\n\nText:\n{text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_pdfs")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_data_from_manual_pdfs.json")


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Error: The directory '{INPUT_DIR}' does not exist.")
    else:
        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
        print(f"Found {len(files)} PDF files. Starting extraction...")

        output_data = []
        for filename in files:
            try:
                pdf_path = os.path.join(INPUT_DIR, filename)
                print(f"Processing: {filename}...")

                # PDF -> text (generous char budget: PDFs carry a lot of whitespace/formatting)
                raw_text = extract_text_from_pdf(pdf_path)
                text_chunk = raw_text[:20000]

                result = chain.invoke({"text": text_chunk})

                if result and "experiments" in result:
                    pmid = os.path.splitext(filename)[0]  # PMID taken from the filename
                    for exp in result["experiments"]:
                        exp["pmid"] = pmid
                        output_data.append(exp)
                    print(f"Extracted {len(result['experiments'])} experiments from {pmid}")
                else:
                    print(f"No experiments found in {filename}")

                time.sleep(2)  # avoid Groq rate limits

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nDone! Extracted {len(output_data)} records to {OUTPUT_FILE}")
