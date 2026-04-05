# Kaggle AIMO3 — Dataset Pipeline

This repository contains data and scripts for the [AI Mathematical Olympiad Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) Kaggle competition.

## Dataset: `data/imosl_cleaned.txt`

Formatted problems from the **IMO Shortlist** (2006–2023), sourced from [AI-MO/olympiads](https://huggingface.co/datasets/AI-MO/olympiads) on HuggingFace.

### Stats

| Field | Value |
|---|---|
| Total problems | 1,288 |
| Years covered | 2006 – 2023 |
| Source | IMO Shortlist official solutions |

### Category breakdown

| Category | Count |
|---|---|
| Algebra | 328 |
| Geometry | 324 |
| Number Theory | 324 |
| Combinatorics | 312 |

### Format

Each entry follows this structure:

```
<question_id>imosl_001</question_id>
<category>Algebra</category>
<question>Problem statement in LaTeX</question>
<reasoning>Full solution / proof in LaTeX</reasoning>
<final_answer>Extracted answer, or empty for proof-only problems</final_answer>

==================================================
```

- **question_id**: sequential ID `imosl_NNN`
- **category**: one of `Algebra`, `Combinatorics`, `Geometry`, `Number Theory` — derived from the problem label prefix (A/C/G/N)
- **question**: original problem statement with LaTeX math
- **reasoning**: full official solution
- **final_answer**: extracted for "find/determine/compute" problems; empty for pure proof problems

### Reproducing

```bash
# 1. Download raw data from HuggingFace
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('AI-MO/olympiads', repo_type='dataset',
    local_dir='data/raw/olympiads',
    allow_patterns=['**/segmented/*.jsonl'])
"

# 2. Merge all JSONL files
python3 -c "
import json, glob
files = glob.glob('data/raw/olympiads/**/segmented/*.jsonl', recursive=True)
with open('data/olympiads_merged.jsonl', 'w') as out:
    for f in files:
        for line in open(f):
            if line.strip(): out.write(line)
"

# 3. Format IMO Shortlist problems
python3 scripts/format_imosl.py
```
