"""
Use Claude (via Batches API) to reformulate proof-based IMO-SL problems
into problems with a specific integer answer, without changing the difficulty.

Workflow:
  1. Parse imosl_cleaned.txt
  2. Submit a batch request for every proof-based problem
  3. Poll until the batch finishes
  4. Write imosl_with_answers.txt  (all problems, reformulated where possible)
  5. Print conversion stats
"""

import json
import os
import re
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE  = DATA_DIR / "imosl_cleaned.txt"
OUTPUT_FILE = DATA_DIR / "imosl_with_answers.txt"

SEPARATOR = "=" * 50

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert mathematician specialising in olympiad problems.
Your task is to examine a proof-based competition problem together with its
official solution and decide whether there is a specific non-negative integer
hidden inside the problem — e.g. the maximum/minimum of some quantity, an
exact count, an extremal value — that can be pulled out as an explicit answer
WITHOUT making the problem easier or changing the core mathematical insight.

Rules
-----
1. Only return has_integer_answer=true when the reformulated question still
   requires working through essentially the same proof.
2. The integer answer must appear (or be directly derivable from) the official
   solution — do NOT invent it.
3. The integer must be a non-negative integer in the range 0–999999.
4. "Prove that P holds for all n" with no specific value is NOT convertible.
5. Reformulate to "Find …" / "Determine …" / "What is the …" style.

Return ONLY valid JSON, no markdown fences, no extra text:
{
  "has_integer_answer": true | false,
  "reformulated_question": "<new question text, or null>",
  "integer_answer": <integer, or null>,
  "reasoning": "<one-sentence justification>"
}"""

USER_TEMPLATE = """Problem (LaTeX):
{problem}

Official solution (LaTeX):
{solution}"""

# ---------------------------------------------------------------------------
# Parse the cleaned txt file into a list of dicts
# ---------------------------------------------------------------------------
def parse_cleaned_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    entries = [e.strip() for e in text.split(SEPARATOR) if e.strip()]
    records = []
    for entry in entries:
        def _get(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", entry, re.DOTALL)
            return m.group(1).strip() if m else ""

        records.append({
            "question_id": _get("question_id"),
            "category":    _get("category"),
            "question":    _get("question"),
            "reasoning":   _get("reasoning"),
            "final_answer": _get("final_answer"),
        })
    return records


# ---------------------------------------------------------------------------
# Format one record back to the txt format
# ---------------------------------------------------------------------------
def format_record(r: dict) -> str:
    return (
        f"<question_id>{r['question_id']}</question_id>\n"
        f"<category>{r['category']}</category>\n"
        f"<question>{r['question']}</question>\n"
        f"<reasoning>{r['reasoning']}</reasoning>\n"
        f"<final_answer>{r['final_answer']}</final_answer>\n"
        f"\n{SEPARATOR}\n"
    )


BATCH_SIZE = 200  # split into smaller chunks to avoid connection timeouts

# ---------------------------------------------------------------------------
# Submit Batches (chunked)
# ---------------------------------------------------------------------------
def submit_batches(client: anthropic.Anthropic, proof_records: list[dict]) -> list[str]:
    batch_ids = []
    chunks = [proof_records[i:i+BATCH_SIZE] for i in range(0, len(proof_records), BATCH_SIZE)]

    for idx, chunk in enumerate(chunks):
        requests = []
        for r in chunk:
            requests.append({
                "custom_id": r["question_id"],
                "params": {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 512,
                    "system": SYSTEM_PROMPT,
                    "messages": [{
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            problem=r["question"],
                            solution=r["reasoning"],
                        ),
                    }],
                },
            })

        for attempt in range(3):
            try:
                batch = client.messages.batches.create(requests=requests)
                print(f"Batch {idx+1}/{len(chunks)} submitted: {batch.id}  ({len(requests)} requests)")
                batch_ids.append(batch.id)
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}. Retrying in 10s…")
                time.sleep(10)

    return batch_ids


# ---------------------------------------------------------------------------
# Poll until all batches done
# ---------------------------------------------------------------------------
def wait_for_batch(client: anthropic.Anthropic, batch_id: str) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{batch_id[-6:]}] status={batch.processing_status}  "
            f"processing={counts.processing}  "
            f"succeeded={counts.succeeded}  "
            f"errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(30)


# ---------------------------------------------------------------------------
# Collect results  →  {question_id: parsed_json}
# ---------------------------------------------------------------------------
def collect_results(client: anthropic.Anthropic, batch_id: str) -> dict:
    results = {}
    for result in client.messages.batches.results(batch_id):
        qid = result.custom_id
        if result.result.type != "succeeded":
            results[qid] = None
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            results[qid] = json.loads(text)
        except json.JSONDecodeError:
            results[qid] = None
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    # 1. Parse input
    print(f"Parsing {INPUT_FILE} …")
    records = parse_cleaned_txt(INPUT_FILE)
    proof_records = [r for r in records if not r["final_answer"]]
    print(f"  Total: {len(records)}  |  Proof-based: {len(proof_records)}")

    # 2. Submit batches (chunked)
    batch_ids = submit_batches(client, proof_records)

    # 3. Poll all batches
    print("Waiting for batches to complete …")
    for batch_id in batch_ids:
        wait_for_batch(client, batch_id)

    # 4. Collect from all batches
    print("Collecting results …")
    llm_results = {}
    for batch_id in batch_ids:
        llm_results.update(collect_results(client, batch_id))

    # 5. Merge: update proof records with reformulated questions / answers
    converted = 0
    failed = 0
    record_map = {r["question_id"]: r for r in records}

    for r in proof_records:
        qid = r["question_id"]
        parsed = llm_results.get(qid)
        if not parsed:
            failed += 1
            continue

        if parsed.get("has_integer_answer") and parsed.get("reformulated_question"):
            record_map[qid]["question"]    = parsed["reformulated_question"]
            record_map[qid]["final_answer"] = str(parsed["integer_answer"])
            converted += 1

    # 6. Write output (preserve original order)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(format_record(record_map[r["question_id"]]))

    # 7. Stats
    originally_with_answer = sum(1 for r in records if r["final_answer"])
    total_with_answer = originally_with_answer + converted
    print(f"\nDone. Saved to {OUTPUT_FILE}")
    print(f"  Originally with answer : {originally_with_answer}")
    print(f"  Newly converted        : {converted}")
    print(f"  Total with answer      : {total_with_answer} / {len(records)}")
    print(f"  Batch errors / skipped : {failed}")
    print(f"  Batch IDs              : {batch_ids}")


if __name__ == "__main__":
    main()
