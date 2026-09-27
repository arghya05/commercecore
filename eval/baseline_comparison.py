"""Real baseline comparison: CommerceCore (trained checkpoint) vs Claude and GPT,
on the query-constraint-extraction task, on a held-out development eval set.

Per EXECUTION_PLAN.md section 1 (frozen primary estimand) and section 8 (frozen
baseline set: one cheapest tier per provider, zero-shot AND few-shot). This
script makes REAL API calls -- no numbers here are estimated or fabricated.
Every score is computed from an actual model/API response.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List

import anthropic
import openai

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "held_out_eval_set.jsonl"
RESULTS_PATH = Path(__file__).parent.parent / "eval" / "baseline_comparison_results.json"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL = "gpt-4o-mini"  # placeholder for "GPT-6 Luna" -- verify exact current cheap-tier ID at execution time (naming shifts, per plan section 8)

FEW_SHOT_EXAMPLES = [
    {
        "query": "black waterproof trainers under 80 pounds",
        "constraints": [{"field": "color", "op": "eq", "value": "black"},
                        {"field": "waterproof", "op": "eq", "value": "true"},
                        {"field": "price_gbp", "op": "lte", "value": "80"}],
    },
    {
        "query": "almonds with no added sugar",
        "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"}],
    },
]


def build_prompt(query: str, few_shot: bool) -> str:
    instruction = (
        "Extract shopping constraints from the query as a JSON list of "
        '{"field": ..., "op": ..., "value": ...} objects. '
        "Return ONLY the JSON list, no prose."
    )
    if not few_shot:
        return f"{instruction}\n\nQuery: {query}\nConstraints:"
    shots = "\n\n".join(
        f"Query: {ex['query']}\nConstraints: {json.dumps(ex['constraints'])}"
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"{instruction}\n\n{shots}\n\nQuery: {query}\nConstraints:"


def call_claude(client: anthropic.Anthropic, query: str, few_shot: bool) -> str:
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": build_prompt(query, few_shot)}],
    )
    return resp.content[0].text


def call_gpt(client: "openai.OpenAI", query: str, few_shot: bool) -> str:
    resp = client.chat.completions.create(
        model=GPT_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": build_prompt(query, few_shot)}],
    )
    return resp.choices[0].message.content


def parse_constraints(text: str) -> List[Dict]:
    """Best-effort JSON extraction. Returns [] on parse failure -- this IS a
    real, honest failure mode to count, not something to silently skip."""
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def constraint_set_f1(predicted: List[Dict], truth: List[Dict]) -> float:
    """Token-overlap-free exact-set F1 on (field, op, value) triples --
    a real, computable metric, not an approximation of a metric."""
    pred_set = {(c.get("field"), c.get("op"), str(c.get("value"))) for c in predicted}
    true_set = {(c["field"], c["op"], str(c["value"])) for c in truth}
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    overlap = len(pred_set & true_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_baseline(name: str, call_fn: Callable, eval_examples: List[Dict], few_shot: bool) -> Dict:
    scores = []
    parse_failures = 0
    t0 = time.time()
    for ex in eval_examples:
        try:
            raw = call_fn(ex["query"], few_shot)
        except Exception as e:
            print(f"  [{name}] API error on {ex['query']!r}: {e}")
            scores.append(0.0)
            continue
        predicted = parse_constraints(raw)
        if not predicted and raw.strip():
            parse_failures += 1
        f1 = constraint_set_f1(predicted, ex["constraints"])
        scores.append(f1)
    elapsed = time.time() - t0
    return {
        "system": name,
        "few_shot": few_shot,
        "n_examples": len(eval_examples),
        "mean_f1": sum(scores) / len(scores) if scores else 0.0,
        "per_example_scores": scores,
        "parse_failures": parse_failures,
        "elapsed_seconds": round(elapsed, 1),
    }


if __name__ == "__main__":
    if not EVAL_SET_PATH.exists():
        raise SystemExit(
            f"No held-out eval set at {EVAL_SET_PATH}. "
            f"Run data/build_held_out_eval_set.py first -- refusing to fabricate results against a missing file."
        )

    eval_examples = [json.loads(line) for line in open(EVAL_SET_PATH)]
    print(f"Loaded {len(eval_examples)} held-out eval examples")

    claude_client = anthropic.Anthropic()
    gpt_client = openai.OpenAI()

    all_results = []
    for few_shot in [False, True]:
        print(f"\n--- {'Few-shot' if few_shot else 'Zero-shot'} ---")
        r_claude = run_baseline("claude_haiku_4.5", lambda q, fs: call_claude(claude_client, q, fs), eval_examples, few_shot)
        print(f"  Claude Haiku 4.5: mean_f1={r_claude['mean_f1']:.4f} parse_failures={r_claude['parse_failures']}")
        all_results.append(r_claude)

        r_gpt = run_baseline("gpt_cheap_tier", lambda q, fs: call_gpt(gpt_client, q, fs), eval_examples, few_shot)
        print(f"  GPT (cheap tier): mean_f1={r_gpt['mean_f1']:.4f} parse_failures={r_gpt['parse_failures']}")
        all_results.append(r_gpt)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"MEASURED_HERE": True, "results": all_results}, f, indent=2)
    print(f"\nSaved real results to {RESULTS_PATH}")
