"""
Format IMO Shortlist problems following the aops_cleaned.txt format:

<question_id>imosl_NNN</question_id>
<question>...</question>
<reasoning>...</reasoning>
<final_answer>...</final_answer>

==================================================
"""

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "olympiads_merged.jsonl"
OUT_PATH = Path(__file__).parent.parent / "data" / "imosl_cleaned.txt"

SEPARATOR = "=" * 50

CATEGORY_MAP = {
    "A": "Algebra",
    "C": "Combinatorics",
    "G": "Geometry",
    "N": "Number Theory",
}

# Patterns to extract explicit final answers
ANSWER_PATTERNS = [
    r"[Tt]he answer is\s+(.+?)(?:\.|$)",
    r"[Tt]he only solution[s]? (?:is|are)\s+(.+?)(?:\.|$)",
    r"[Tt]he answer(?:s)? (?:is|are)\s+(.+?)(?:\.|$)",
    r"[Tt]he solution[s]? (?:is|are)\s+(.+?)(?:\.|$)",
]

FIND_KEYWORDS = [
    "find all", "find the", "determine all", "determine the",
    "find every", "what is", "compute", "evaluate", "calculate",
]

PROVE_KEYWORDS = ["prove", "show that", "show there", "prove that"]


def extract_final_answer(problem: str, solution: str) -> str:
    problem_lower = (problem or "").lower()
    is_find = any(kw in problem_lower for kw in FIND_KEYWORDS)
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


def get_category(label: str) -> str:
    prefix = label[0].upper() if label else ""
    return CATEGORY_MAP.get(prefix, "Unknown")


def format_record(idx: int, record: dict) -> str:
    qid = f"imosl_{idx:03d}"
    question = clean_text(record["problem"])
    reasoning = clean_text(record["solution"])
    final_answer = extract_final_answer(record["problem"], record["solution"])
    category = get_category(record.get("problem_label", ""))

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

    imosl = [r for r in records if r["exam"] == "IMO-SL"]
    imosl.sort(key=lambda r: (r["year"], r["problem_label"]))

    print(f"Formatting {len(imosl)} IMO Shortlist problems...")

    with open(OUT_PATH, "w") as f:
        for idx, record in enumerate(imosl, start=1):
            f.write(format_record(idx, record))

    print(f"Saved to {OUT_PATH}")

    proof_count = sum(
        1 for r in imosl
        if not extract_final_answer(r["problem"], r["solution"])
    )
    print(f"  Proof-based (empty final_answer): {proof_count}")
    print(f"  With extracted answer: {len(imosl) - proof_count}")


if __name__ == "__main__":
    main()
