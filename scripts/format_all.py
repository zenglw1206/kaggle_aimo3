"""
Format ALL olympiad problems from olympiads_merged.jsonl into the same
XML-tag format used by imosl_cleaned.txt.

Output: data/all_olympiads_cleaned.txt
  ~11,662 problems from HMMT, IMO-SL, IMO, USAMO, APMO, etc.

question_id  : olympiad_NNNNN  (sequential, zero-padded to 5 digits)
category     : Algebra | Combinatorics | Geometry | Number Theory | Unknown
               (from problem_type field, or derived from IMO-SL label prefix)
question     : problem statement (LaTeX)
reasoning    : official solution (LaTeX)
final_answer : integer/expression answer where detectable; empty for proof-only
"""

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "olympiads_merged.jsonl"
OUT_PATH  = Path(__file__).parent.parent / "data" / "all_olympiads_cleaned.txt"

SEPARATOR = "=" * 50

IMOSL_CATEGORY_MAP = {
    "A": "Algebra",
    "C": "Combinatorics",
    "G": "Geometry",
    "N": "Number Theory",
}

VALID_CATEGORIES = {"Algebra", "Combinatorics", "Geometry", "Number Theory"}

ANSWER_PATTERNS = [
    r"[Tt]he answer is\s+(.+?)(?:\.|$)",
    r"[Tt]he only solution[s]? (?:is|are)\s+(.+?)(?:\.|$)",
    r"[Tt]he answer(?:s)? (?:is|are)\s+(.+?)(?:\.|$)",
    r"[Tt]he solution[s]? (?:is|are)\s+(.+?)(?:\.|$)",
]

FIND_KEYWORDS  = ["find all", "find the", "determine all", "determine the",
                  "find every", "what is", "compute", "evaluate", "calculate"]
PROVE_KEYWORDS = ["prove", "show that", "show there", "prove that"]


def get_category(record: dict) -> str:
    # 1. Use problem_type if it's a known category
    pt = (record.get("problem_type") or "").strip()
    if pt in VALID_CATEGORIES:
        return pt

    # 2. For IMO-SL, derive from label prefix (A1 → Algebra, etc.)
    if record.get("exam") == "IMO-SL":
        prefix = (record.get("problem_label") or "")[:1].upper()
        return IMOSL_CATEGORY_MAP.get(prefix, "Unknown")

    return "Unknown"


def extract_final_answer(problem: str, solution: str) -> str:
    problem_lower = (problem or "").lower()
    is_find  = any(kw in problem_lower for kw in FIND_KEYWORDS)
    is_prove = any(kw in problem_lower for kw in PROVE_KEYWORDS)

    if is_prove and not is_find:
        return ""

    first_chunk = (solution or "")[:500]
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, first_chunk)
        if match:
            return match.group(1).strip()[:300]

    return ""


def clean_text(text) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", str(text))
    return text.strip()


def format_record(idx: int, record: dict) -> str:
    qid      = f"olympiad_{idx:05d}"
    question = clean_text(record["problem"])
    reasoning = clean_text(record["solution"])
    final_answer = extract_final_answer(record["problem"], record["solution"])
    category = get_category(record)

    return (
        f"<question_id>{qid}</question_id>\n"
        f"<category>{category}</category>\n"
        f"<question>{question}</question>\n"
        f"<reasoning>{reasoning}</reasoning>\n"
        f"<final_answer>{final_answer}</final_answer>\n"
        f"\n{SEPARATOR}\n"
    )


def main():
    with open(DATA_PATH) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Sort: by exam name, then year, then label for a stable ordering
    records.sort(key=lambda r: (r["exam"], str(r.get("year", "")), str(r.get("problem_label", ""))))

    print(f"Formatting {len(records)} problems from {len({r['exam'] for r in records})} competitions…")

    with open(OUT_PATH, "w") as f:
        for idx, record in enumerate(records, start=1):
            f.write(format_record(idx, record))

    print(f"Saved to {OUT_PATH}")

    # Stats
    from collections import Counter
    exam_counts = Counter(r["exam"] for r in records)
    cat_counts  = Counter(get_category(r) for r in records)
    proof_count = sum(
        1 for r in records
        if not extract_final_answer(r["problem"], r["solution"])
    )

    print(f"\nBy competition (top 10):")
    for exam, cnt in exam_counts.most_common(10):
        print(f"  {cnt:5d}  {exam}")

    print(f"\nBy category:")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cnt:5d}  {cat}")

    print(f"\nProof-based (empty final_answer): {proof_count}")
    print(f"With extracted answer           : {len(records) - proof_count}")


if __name__ == "__main__":
    main()
