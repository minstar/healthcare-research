#!/usr/bin/env python3
"""Normalize MedQA / PubMedQA / MedMCQA into a common closed-form MC schema.

Output: data/eval_samples/closedform_mc.jsonl, one record per question:
  {id, dataset, prompt, choices: [..], answer: "A"|"B"|.. , answer_text}

Run on the login node (has internet); the eval then reads this JSONL and never
needs HF again, so it can run from anywhere hitting a served endpoint.
"""
import json, os, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset

OUT = "/data/project/private/minstar/workspace/healthcare-research/data/eval_samples/closedform_mc.jsonl"
LETTERS = ["A", "B", "C", "D", "E"]
rows = []

# ---- MedQA (USMLE, 4 options) : test=1273 ----
d = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
for i, x in enumerate(d):
    opts = x["options"]  # dict {"A":..,"B":..,..}
    choices = [opts[l] for l in LETTERS if l in opts]
    rows.append({"id": f"medqa-{i}", "dataset": "MedQA",
                 "prompt": x["question"], "choices": choices,
                 "answer": x["answer_idx"], "answer_text": opts[x["answer_idx"]]})

# ---- PubMedQA (pqa_labeled) : 1000, yes/no/maybe ----
d = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
PM = {"yes": "A", "no": "B", "maybe": "C"}
for i, x in enumerate(d):
    ctx = " ".join(x["context"]["contexts"]) if isinstance(x["context"], dict) else ""
    prompt = f"Context: {ctx}\n\nQuestion: {x['question']}"
    rows.append({"id": f"pubmedqa-{i}", "dataset": "PubMedQA",
                 "prompt": prompt, "choices": ["yes", "no", "maybe"],
                 "answer": PM[x["final_decision"]], "answer_text": x["final_decision"]})

# ---- MedMCQA (validation, has labels) : 4183, 4 options, cop 0-indexed ----
d = load_dataset("openlifescienceai/medmcqa", split="validation")
for i, x in enumerate(d):
    choices = [x["opa"], x["opb"], x["opc"], x["opd"]]
    ans = LETTERS[x["cop"]]
    rows.append({"id": f"medmcqa-{i}", "dataset": "MedMCQA",
                 "prompt": x["question"], "choices": choices,
                 "answer": ans, "answer_text": choices[x["cop"]]})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
c = Counter(r["dataset"] for r in rows)
print(f"wrote {OUT}: {len(rows)} questions  {dict(c)}")
