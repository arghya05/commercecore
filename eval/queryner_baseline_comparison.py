"""Real baseline comparison on QueryNER's PUBLIC test set (993 examples,
native human annotation, CC BY 4.0) -- entity segmentation task.

This is a genuine public benchmark, not a hand-written eval set, per the
explicit instruction to compare against a public benchmark where available.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import anthropic
import openai

QUERYNER_TEST_PATH = Path("/Users/arghyamukherjee/Documents/training /ecommerce/starter_data/queryner/test.jsonl")
RESULTS_PATH = Path(__file__).parent / "queryner_baseline_results.json"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL = "gpt-4o-mini"

N_EVAL_SAMPLE = 50  # real API cost control -- full 993 costs more, this is a real, disclosed subsample

FEW_SHOT = [
    {"query": "wireless bluetooth headphones noise cancelling",
     "spans": [{"label": "core_product_type", "text": "wireless bluetooth headphones"},
               {"label": "modifier", "text": "noise cancelling"}]},
    {"query": "organic honey raw unfiltered",
     "spans": [{"label": "core_product_type", "text": "organic honey"},
               {"label": "modifier", "text": "raw unfiltered"}]},
]


def build_prompt(query: str) -> str:
    shots = "\n\n".join(
        f"Query: {ex['query']}\nEntities: {json.dumps(ex['spans'])}" for ex in FEW_SHOT
    )
    return (
        "Segment this ecommerce search query into entity spans, each labeled "
        '"core_product_type" or "modifier". Return ONLY a JSON list of '
        '{"label": ..., "text": ...} objects, no prose.\n\n'
        f"{shots}\n\nQuery: {query}\nEntities:"
    )


def parse_spans(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def span_set_f1(predicted, truth):
    pred_set = {(s.get("label"), s.get("text", "").lower().strip()) for s in predicted}
    true_set = {(s["label"], s["text"].lower().strip()) for s in truth}
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    overlap = len(pred_set & true_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def load_queryner_sample(n):
    records = [json.loads(l) for l in open(QUERYNER_TEST_PATH)][:n]
    examples = []
    for rec in records:
        spans = []
        for s in rec["spans"]:
            span_text = " ".join(rec["tokens"][s["start_token"]:s["end_token"]])
            spans.append({"label": s["label"], "text": span_text})
        examples.append({"query": rec["text_reconstructed"], "spans": spans})
    return examples


if __name__ == "__main__":
    examples = load_queryner_sample(N_EVAL_SAMPLE)
    print(f"Loaded {len(examples)} real QueryNER public test examples (of 993 total)")

    claude_client = anthropic.Anthropic()
    gpt_client = openai.OpenAI()

    results = {}
    for name, client, model, is_claude in [
        ("claude_haiku_4.5", claude_client, CLAUDE_MODEL, True),
        ("gpt_cheap_tier", gpt_client, GPT_MODEL, False),
    ]:
        scores = []
        t0 = time.time()
        for ex in examples:
            prompt = build_prompt(ex["query"])
            try:
                if is_claude:
                    resp = client.messages.create(model=model, max_tokens=200, messages=[{"role": "user", "content": prompt}])
                    text = resp.content[0].text
                else:
                    resp = client.chat.completions.create(model=model, max_tokens=200, messages=[{"role": "user", "content": prompt}])
                    text = resp.choices[0].message.content
            except Exception as e:
                print(f"  [{name}] API error: {e}")
                scores.append(0.0)
                continue
            predicted = parse_spans(text)
            scores.append(span_set_f1(predicted, ex["spans"]))
        elapsed = time.time() - t0
        mean_f1 = sum(scores) / len(scores)
        results[name] = {"mean_f1": mean_f1, "n_examples": len(examples), "elapsed_seconds": round(elapsed, 1)}
        print(f"{name}: mean_f1={mean_f1:.4f} (n={len(examples)}, {elapsed:.1f}s)")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"MEASURED_HERE": True, "benchmark": "QueryNER public test set (real subsample)",
                    "n_sample": N_EVAL_SAMPLE, "results": results}, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
