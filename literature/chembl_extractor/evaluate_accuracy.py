#!/usr/bin/env python3
"""
evaluate_accuracy.py
====================
מעריך את דיוק ה-extractor על מדגם אקראי מה-DATABASE.
מעבד ב-batches ושומר checkpoint אחרי כל batch — ניתן לחדש אם נקטע.

שימוש:
    python3 evaluate_accuracy.py                        # 10% ברירת מחדל
    python3 evaluate_accuracy.py --fraction 0.05        # 5%
    python3 evaluate_accuracy.py --batch-size 30        # batch קטן יותר
    python3 evaluate_accuracy.py --skip-extraction      # רק השוואה (extraction קיים)
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
DB_PATH    = SCRIPT_DIR.parent.parent / "Final ODO Dataset_v2026-06-10.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
REPORT_DIR = SCRIPT_DIR / "accuracy_reports"


def sample_compound_ids(db_path: Path, fraction: float, seed: int) -> list[str]:
    print(f"[1/3] טוען DB מ-{db_path} ...", flush=True)
    df = pd.read_excel(db_path, engine="openpyxl", usecols=["chembl_compound_id"])
    unique_ids = df["chembl_compound_id"].dropna().unique().tolist()
    n_sample = max(1, round(len(unique_ids) * fraction))
    sampled = (
        pd.Series(unique_ids)
        .sample(n=n_sample, random_state=seed)
        .tolist()
    )
    print(
        f"      מולקולות ייחודיות ב-DB: {len(unique_ids):,}  |  "
        f"מדגם ({fraction*100:.0f}%): {n_sample:,} מולקולות",
        flush=True,
    )
    return sampled


def run_fetcher_on_batch(batch_ids: list[str], batch_output: Path) -> bool:
    """מריץ chembl_fetcher על batch אחד. מחזיר True אם הצליח."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="eval_batch_"
    ) as f:
        ids_file = Path(f.name)
        f.write("\n".join(batch_ids))
    try:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "chembl_fetcher.py"),
                "--file", str(ids_file),
                "--output", str(batch_output),
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        ids_file.unlink(missing_ok=True)


def run_extraction_batched(ids: list[str], output_path: Path, batch_size: int) -> None:
    checkpoint_dir = output_path.parent / (output_path.stem + "_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # גלה אילו מולקולות כבר עובדו (checkpoint קיים)
    done_ids: set[str] = set()
    existing_batches = sorted(checkpoint_dir.glob("batch_*.xlsx"))
    if existing_batches:
        print(f"      נמצאו {len(existing_batches)} batches קיימים — ממשיך מנקודת עצירה...", flush=True)
        for bf in existing_batches:
            try:
                tmp = pd.read_excel(bf, engine="openpyxl", usecols=["chembl_compound_id"])
                done_ids.update(tmp["chembl_compound_id"].dropna().unique())
            except Exception:
                pass

    remaining = [cid for cid in ids if cid not in done_ids]
    total_batches = -(-len(remaining) // batch_size)  # ceiling division

    print(
        f"\n[2/3] Extraction ב-batches של {batch_size} מולקולות",
        flush=True,
    )
    print(
        f"      נותרו: {len(remaining):,} מולקולות  |  "
        f"כבר בוצעו: {len(done_ids):,}  |  "
        f"batches: {total_batches}",
        flush=True,
    )

    batch_num_offset = len(existing_batches)
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        batch_idx = batch_num_offset + (i // batch_size) + 1
        batch_file = checkpoint_dir / f"batch_{batch_idx:04d}.xlsx"
        done_so_far = len(done_ids) + i

        print(
            f"  [batch {batch_idx}/{batch_num_offset + total_batches}] "
            f"{len(batch)} מולקולות  "
            f"({done_so_far + 1}–{done_so_far + len(batch)} מתוך {len(ids):,})",
            flush=True,
        )

        success = run_fetcher_on_batch(batch, batch_file)
        if not success:
            print(f"    אזהרה: batch {batch_idx} נכשל — ממשיך לbatch הבא", flush=True)
            continue

        print(f"    batch {batch_idx} הושלם → {batch_file.name}", flush=True)

    # מיזוג כל ה-batches לקובץ אחד
    all_batch_files = sorted(checkpoint_dir.glob("batch_*.xlsx"))
    if not all_batch_files:
        print("[ERROR] אין batches להמזגה — לא נוצר קובץ output", flush=True)
        sys.exit(1)

    print(f"\n      ממזג {len(all_batch_files)} batches → {output_path}", flush=True)
    dfs = []
    for bf in all_batch_files:
        try:
            dfs.append(pd.read_excel(bf, engine="openpyxl"))
        except Exception as e:
            print(f"    אזהרה: לא ניתן לקרוא {bf.name}: {e}", flush=True)

    merged = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(output_path, index=False, engine="openpyxl")
    print(f"      קובץ מוזג: {len(merged):,} שורות", flush=True)


def run_comparison(db_path: Path, output_path: Path, report_path: Path) -> None:
    print(f"\n[3/3] מריץ השוואה → {report_path}", flush=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "compare_accuracy.py"),
            "--output", str(output_path),
            "--db",     str(db_path),
            "--report", str(report_path),
        ],
        check=True,
    )

    print(f"\nהערכה הושלמה.", flush=True)
    print(f"  תוצאות extraction : {output_path}", flush=True)
    print(f"  דוח דיוק          : {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="הערכת דיוק ה-extractor על מדגם מה-DATABASE"
    )
    parser.add_argument(
        "--fraction", type=float, default=0.1,
        help="שבר מה-DB לדגום (ברירת מחדל: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed לאקראיות (ברירת מחדל: 42)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="מולקולות לעיבוד בכל batch (ברירת מחדל: 50)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUT_DIR / "eval_sample.xlsx",
        help="נתיב קובץ output לתוצאות ה-extraction",
    )
    parser.add_argument(
        "--report", type=Path,
        default=REPORT_DIR / "eval_accuracy_report.xlsx",
        help="נתיב קובץ לדוח הדיוק",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="דלג על ה-fetcher, השתמש בקובץ output קיים",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"נתיב ה-DATABASE (ברירת מחדל: {DB_PATH})",
    )
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"[ERROR] DB לא נמצא: {args.db}")

    if args.skip_extraction:
        if not args.output.exists():
            sys.exit(
                f"[ERROR] --skip-extraction הוגדר אך קובץ לא קיים: {args.output}"
            )
        print(f"[דילוג extraction] משתמש בקובץ קיים: {args.output}", flush=True)
    else:
        ids = sample_compound_ids(args.db, args.fraction, args.seed)
        run_extraction_batched(ids, args.output, args.batch_size)

    run_comparison(args.db, args.output, args.report)


if __name__ == "__main__":
    main()
