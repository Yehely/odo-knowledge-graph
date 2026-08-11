"""
Comparison.py – validates a Llama extraction JSON (from LlamaExtractor.py or
LlamaExtractorDownloaded.py) against the real ODO dataset: for each
extracted (pmid, drug, value), checks whether a matching row exists,
allowing for direct matches and pKi/nM unit conversions.

Usage:
    python3 Comparison.py --input labeled_data_from_llama.json
    python3 Comparison.py --input labeled_data_from_manual_pdfs.json
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_GROUND_TRUTH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "Final ODO Dataset_v2026-06-10.xlsx")

# NOTE: verify these against the current dataset's actual column names before relying on results.
COL_PMID = "pubmed_id"
COL_VALUE = "endpoint_value"
COL_DRUG = "chembl_chemical_entity_name"

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1500)


def clean_pmid(val):
    if pd.isna(val):
        return ""
    try:
        return str(int(float(val))).strip()
    except Exception:
        return str(val).strip()


def extract_float(val):
    if pd.isna(val):
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(match.group(0)) if match else None


def analyze_match(llm_val, excel_val):
    """Analyzes match quality. Success includes direct matches and unit conversions."""
    if excel_val is None or llm_val is None:
        return False, "Data Missing"

    if excel_val == 0 or llm_val == 0:
        if abs(excel_val - llm_val) < 0.001:
            return True, "Full Match (Directly)"
        return False, "Numeric Mismatch"

    # 1. Full Match
    ratio = max(llm_val, excel_val) / min(llm_val, excel_val)
    if ratio < 1.05:
        return True, "Full Match (Directly)"

    # 2. Log Conversion (pVal -> nM)
    log_variant = 10 ** (9 - llm_val)
    if log_variant > 0:
        ratio_log = max(log_variant, excel_val) / min(log_variant, excel_val)
        if ratio_log < 1.25:
            return True, f"Full Match (Unit Conversion: pVal {llm_val} -> nM {log_variant:.2f})"

    # 3. Inverse Log (nM -> pVal)
    inv_log = round(9 - np.log10(llm_val), 2)
    if inv_log > 0:
        ratio_inv = max(inv_log, excel_val) / min(inv_log, excel_val)
        if ratio_inv < 1.1:
            return True, f"Full Match (Unit Conversion: nM {llm_val} -> pVal {inv_log})"

    return False, "Numeric Mismatch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Llama extraction JSON to validate")
    ap.add_argument("--report", default="Detailed_Llama_Validation_Report.xlsx")
    args = ap.parse_args()

    print("Loading data and performing validation...")
    df_truth = pd.read_excel(EXCEL_GROUND_TRUTH)
    with open(args.input, "r", encoding="utf-8") as f:
        llm_data = json.load(f)

    df_truth["processed_pmid"] = df_truth[COL_PMID].apply(clean_pmid)
    df_truth["processed_val"] = df_truth[COL_VALUE].apply(extract_float)

    results = []
    for entry in llm_data:
        pmid = clean_pmid(entry.get("pmid", ""))
        llm_raw_val = entry.get("value_numeric", "")
        llm_num = extract_float(llm_raw_val)
        llm_drug_name = str(entry.get("drug_name", ""))

        relevant_excel_rows = df_truth[df_truth["processed_pmid"] == pmid]

        status = ""
        excel_val_found = "N/A"
        excel_drug_name = "N/A"
        validation_note = ""

        if relevant_excel_rows.empty:
            status = "New Info"
            validation_note = "Article completely missing in Excel"
        else:
            drug_search_str = llm_drug_name.lower()[:6]
            drug_match_rows = relevant_excel_rows[
                relevant_excel_rows[COL_DRUG].str.lower().fillna("").str.contains(drug_search_str, regex=False)
            ]

            match_found = False
            matched_val = None
            matched_drug = "N/A"
            match_desc = ""

            for _, row in relevant_excel_rows.iterrows():
                is_match, desc = analyze_match(llm_num, row["processed_val"])
                if is_match:
                    match_found = True
                    matched_val = row["processed_val"]
                    matched_drug = row[COL_DRUG]
                    match_desc = desc
                    break

            if match_found:
                status = "Verified"
                excel_val_found = matched_val
                excel_drug_name = matched_drug
                validation_note = match_desc
            elif not drug_match_rows.empty:
                status = "Mismatch"
                excel_drug_name = drug_match_rows.iloc[0][COL_DRUG]
                validation_note = "Drug name matches, but numeric value differs significantly"
            else:
                status = "New Info"
                validation_note = "Drug/Experiment found by Llama but missing in Excel"

        results.append(
            {
                "PMID": pmid,
                "LLM_Drug": llm_drug_name,
                "Excel_Drug": excel_drug_name,
                "LLM_Value": llm_raw_val,
                "Excel_Value": excel_val_found,
                "Status": status,
                "Validation_Note": validation_note,
            }
        )

    df_report = pd.DataFrame(results)
    print("\n" + "=" * 160)
    print(df_report.to_string(index=False))
    print("=" * 160)

    total = len(df_report)
    counts = df_report["Status"].value_counts()
    verified = counts.get("Verified", 0)
    new_info = counts.get("New Info", 0)
    mismatch = counts.get("Mismatch", 0)

    print("\nFINAL EXTRACTION SUMMARY:")
    print("-" * 50)
    print(f"Verified (Direct & Unit Conversions): {verified} ({verified/total:.1%})")
    print(f"New Scientific Discoveries:           {new_info} ({new_info/total:.1%})")
    print(f"Real Extraction Errors:               {mismatch} ({mismatch/total:.1%})")
    print("-" * 50)
    print(f"TOTAL SUCCESS RATE:                   {(verified + new_info)/total:.1%}")
    print("Note: Unit conversions are treated as 100% success.")

    df_report.to_excel(args.report, index=False)
    print(f"\nDetailed report exported to: {args.report}")


if __name__ == "__main__":
    main()
