"""
sortFullText.py – splits smart_retrieval/Cleaned_Text_Articles/ into likely
full-text vs. likely abstract-only files, using a simple keyword+length
heuristic. Useful before feeding files to llama_extractor/, which does
better with genuine full text (tables, methods, results) than with bare
abstracts.
"""
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "Cleaned_Text_Articles")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Sort_Full_Text_And_Not")
FULL_TEXT_DIR = os.path.join(OUTPUT_DIR, "Full_Text")
NOT_FULL_TEXT_DIR = os.path.join(OUTPUT_DIR, "Not_Full_Text")


def is_full_text(content):
    content_lower = content.lower()
    score = 0

    structure_words = ["abstract", "introduction", "discussion", "conclusion",
                        "references", "methods", "experimental", "results", "summary"]
    score += sum(1 for w in structure_words if w in content_lower)

    evidence_words = ["table 1", "figure 1", "scheme 1", "doi:", "et al.", "cite"]
    score += sum(1 for w in evidence_words if w in content_lower)

    length = len(content)
    if length > 12000 and score >= 3:
        return True
    if length > 8000 and score >= 5:
        return True
    return False


if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing existing output directory '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(FULL_TEXT_DIR, exist_ok=True)
    os.makedirs(NOT_FULL_TEXT_DIR, exist_ok=True)

    if not os.path.exists(SOURCE_DIR):
        raise SystemExit(f"Error: source directory '{SOURCE_DIR}' not found — run CleanFiles.py first.")

    print("Sorting files (minimum threshold: 8000 characters)...")
    stats = {"full": 0, "not_full": 0}

    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith((".txt", ".xml")):
            file_path = os.path.join(SOURCE_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if is_full_text(content):
                    shutil.copy(file_path, os.path.join(FULL_TEXT_DIR, filename))
                    stats["full"] += 1
                else:
                    shutil.copy(file_path, os.path.join(NOT_FULL_TEXT_DIR, filename))
                    stats["not_full"] += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("\nSorting complete!")
    print(f"Full-text articles: {stats['full']}")
    print(f"Partial/abstract-only files: {stats['not_full']}")
