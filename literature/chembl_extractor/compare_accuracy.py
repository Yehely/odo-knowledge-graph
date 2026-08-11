#!/usr/bin/env python3
"""
compare_accuracy.py
====================
משווה בין קובץ הפלט של ChEMBL Fetcher לבין ה-DataBase האמיתי,
ומחשב דיוק של החילוץ לפי שדה.

שימוש:
    python3 compare_accuracy.py
    python3 compare_accuracy.py --output outputs/my_file.xlsx
    python3 compare_accuracy.py --output outputs/my_file.xlsx --report report.xlsx
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
DB_PATH      = SCRIPT_DIR.parent.parent / "Final ODO Dataset_v2026-06-10.xlsx"
OUTPUT_DIR   = SCRIPT_DIR / "outputs"
REPORT_DIR   = SCRIPT_DIR / "accuracy_reports"

# Join keys: rows are identified by compound + assay
JOIN_KEYS = ["chembl_compound_id", "chembl_assay_id"]

# Tolerance for numeric comparison (relative)
NUMERIC_RTOL = 1e-4   # 0.01%
NUMERIC_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_value(val):
    """Convert a cell value to a canonical form for comparison."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, str):
        val = val.strip()
        return val if val != "" else None
    return val


def values_match(a, b) -> bool:
    """Return True if two values are considered equal."""
    a = normalize_value(a)
    b = normalize_value(b)

    # Both missing → match
    if a is None and b is None:
        return True
    # One missing, one not → mismatch
    if a is None or b is None:
        return False

    # Numeric comparison with tolerance
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            af, bf = float(a), float(b)
            if math.isnan(af) and math.isnan(bf):
                return True
            return math.isclose(af, bf, rel_tol=NUMERIC_RTOL, abs_tol=NUMERIC_ATOL)
        except (ValueError, TypeError):
            pass

    # String comparison (case-insensitive strip)
    return str(a).strip().lower() == str(b).strip().lower()


def find_latest_output() -> Path:
    """Return the most recently modified .xlsx in the outputs/ directory."""
    files = sorted(OUTPUT_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        sys.exit(f"[ERROR] No .xlsx files found in {OUTPUT_DIR}")
    return files[0]


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare(db_path: Path, out_path: Path, report_path: Path | None = None):
    print(f"\n{'='*60}")
    print(f"  ChEMBL Extractor – Accuracy Comparison")
    print(f"{'='*60}")
    print(f"  DB:     {db_path.name}")
    print(f"  Output: {out_path.name}")
    print()

    # --- Load files ---
    print("[1/4] Loading DataBase...", end=" ", flush=True)
    db = pd.read_excel(db_path, engine="openpyxl")
    print(f"done  ({db.shape[0]:,} rows × {db.shape[1]} cols)")

    print("[2/4] Loading Extractor output...", end=" ", flush=True)
    out = pd.read_excel(out_path, engine="openpyxl")
    print(f"done  ({out.shape[0]:,} rows × {out.shape[1]} cols)")

    # --- Validate columns ---
    extra_cols = [c for c in out.columns if c not in db.columns]
    missing_cols = [c for c in db.columns if c not in out.columns]
    if extra_cols:
        print(f"  [!] Columns in output but NOT in DB: {extra_cols}")
    if missing_cols:
        print(f"  [!] Columns in DB but NOT in output: {missing_cols}")

    common_cols = [c for c in out.columns if c in db.columns]
    compare_cols = [c for c in common_cols if c not in JOIN_KEYS]

    # --- Match rows ---
    print(f"\n[3/4] Matching rows by: {JOIN_KEYS} ...")

    # Check if join keys exist in both
    for key in JOIN_KEYS:
        if key not in db.columns:
            sys.exit(f"[ERROR] Join key '{key}' not found in DataBase!")
        if key not in out.columns:
            sys.exit(f"[ERROR] Join key '{key}' not found in output!")

    # Build lookup: (compound_id, assay_id) → DB row
    db_indexed = db.set_index(JOIN_KEYS)

    matched_rows = []   # list of (out_row, db_row) pairs
    unmatched    = []   # list of out_rows with no DB match

    for _, out_row in out.iterrows():
        key = tuple(normalize_value(out_row[k]) for k in JOIN_KEYS)
        if key in db_indexed.index:
            db_row = db_indexed.loc[key]
            # If multiple DB rows for same key, take first
            if isinstance(db_row, pd.DataFrame):
                db_row = db_row.iloc[0]
            matched_rows.append((out_row, db_row))
        else:
            unmatched.append(out_row)

    n_matched   = len(matched_rows)
    n_unmatched = len(unmatched)
    print(f"  Matched rows  : {n_matched}")
    print(f"  Unmatched rows: {n_unmatched} (no DB entry found)")

    if n_matched == 0:
        print("\n[!] No rows could be matched – cannot compute accuracy.")
        print("    Make sure the ChEMBL IDs in the output exist in the DataBase.")
        return

    # --- Field-by-field comparison ---
    print(f"\n[4/4] Comparing {len(compare_cols)} fields across {n_matched} matched rows...\n")

    col_stats = {}   # col → {"correct": int, "wrong": int, "both_null": int, "db_null": int, "out_null": int}

    detail_records = []   # for the per-row mismatch report

    for out_row, db_row in matched_rows:
        compound = normalize_value(out_row.get("chembl_compound_id"))
        assay    = normalize_value(out_row.get("chembl_assay_id"))
        row_key  = f"{compound} | {assay}"

        for col in compare_cols:
            out_val = normalize_value(out_row.get(col))
            db_val  = normalize_value(db_row.get(col) if col in db_row.index else None)

            if col not in col_stats:
                col_stats[col] = {"correct": 0, "wrong": 0,
                                  "both_null": 0, "db_null": 0, "out_null": 0}

            if db_val is None and out_val is None:
                col_stats[col]["both_null"] += 1
            elif db_val is None:
                col_stats[col]["db_null"] += 1
                # DB has nothing → treat as N/A (don't penalise)
            elif out_val is None:
                col_stats[col]["out_null"] += 1
                col_stats[col]["wrong"] += 1
                detail_records.append({
                    "Row Key": row_key,
                    "Column": col,
                    "DB Value": db_val,
                    "Output Value": "(missing)",
                    "Match": "MISSING",
                })
            elif values_match(out_val, db_val):
                col_stats[col]["correct"] += 1
            else:
                col_stats[col]["wrong"] += 1
                detail_records.append({
                    "Row Key": row_key,
                    "Column": col,
                    "DB Value": db_val,
                    "Output Value": out_val,
                    "Match": "WRONG",
                })

    # --- Build summary table ---
    summary_rows = []
    for col in compare_cols:
        s        = col_stats[col]
        correct  = s["correct"]
        wrong    = s["wrong"]
        db_null  = s["db_null"]
        out_null = s["out_null"]
        both_null= s["both_null"]

        # Denominator = rows where DB had a value (i.e., there was something to compare)
        denominator = correct + wrong
        if denominator == 0:
            accuracy = None   # column was empty in DB for all matched rows
        else:
            accuracy = correct / denominator * 100

        summary_rows.append({
            "Column":            col,
            "Accuracy (%)":      round(accuracy, 1) if accuracy is not None else "N/A",
            "Correct":           correct,
            "Wrong":             wrong,
            "Missing in Output": out_null,
            "Both Null":         both_null,
            "DB Null (skipped)": db_null,
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df  = pd.DataFrame(detail_records) if detail_records else pd.DataFrame(
        columns=["Row Key", "Column", "DB Value", "Output Value", "Match"])

    # --- Console output ---
    print("  Per-column accuracy (only columns with DB data):")
    print(f"  {'Column':<50} {'Accuracy':>10}  {'Correct':>8}  {'Wrong':>8}")
    print(f"  {'-'*50} {'-'*10}  {'-'*8}  {'-'*8}")

    numeric_accs = []
    for r in summary_rows:
        acc = r["Accuracy (%)"]
        if acc != "N/A":
            numeric_accs.append(acc)
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {r['Column']:<50} {acc:>9.1f}%  {r['Correct']:>8}  {r['Wrong']:>8}  [{bar}]")
        else:
            print(f"  {r['Column']:<50} {'N/A':>10}  (no DB data)")

    # Overall accuracy
    if numeric_accs:
        overall = np.mean(numeric_accs)
        total_correct = sum(r["Correct"] for r in summary_rows)
        total_wrong   = sum(r["Wrong"]   for r in summary_rows)
        print(f"\n  {'─'*70}")
        print(f"  Overall accuracy (mean across columns):  {overall:.1f}%")
        print(f"  Total cells correct:  {total_correct:,}")
        print(f"  Total cells wrong:    {total_wrong:,}")
        print(f"  Rows matched:         {n_matched}")
        print(f"  Rows unmatched:       {n_unmatched}")
        print(f"  {'─'*70}\n")

    # --- Save report ---
    REPORT_DIR.mkdir(exist_ok=True)
    if report_path is None:
        stem = out_path.stem
        report_path = REPORT_DIR / f"accuracy_{stem}.xlsx"

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        # Sheet 1: Summary
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        # Sheet 2: Mismatches
        detail_df.to_excel(writer, sheet_name="Mismatches", index=False)
        # Sheet 3: Unmatched rows
        if unmatched:
            pd.DataFrame(unmatched).to_excel(writer, sheet_name="Unmatched", index=False)

    print(f"  Report saved → {report_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare extractor output against DB")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to the extractor output .xlsx (default: latest file in outputs/)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to the DataBase .xlsx (default: {DB_PATH})",
    )
    parser.add_argument(
        "--report", "-r",
        type=Path,
        default=None,
        help="Where to save the accuracy report .xlsx",
    )
    args = parser.parse_args()

    out_path = args.output or find_latest_output()
    if not out_path.exists():
        sys.exit(f"[ERROR] Output file not found: {out_path}")
    if not args.db.exists():
        sys.exit(f"[ERROR] DataBase file not found: {args.db}")

    compare(args.db, out_path, args.report)
