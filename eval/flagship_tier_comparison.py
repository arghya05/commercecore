"""SECONDARY comparison against flagship/expensive-tier Claude and GPT models.

Per EXECUTION_PLAN.md section 8: the PRIMARY registered comparison is against
the cheapest tier per provider (the realistic "affordable alternative" a
cost-sensitive startup would use). This script adds the flagship tier as an
explicit, separately-labeled SECONDARY reference -- answering "can a small
model compete with the best money can buy", not the primary claim.

Real API calls, real F1 scores -- same held-out eval set as the primary
comparison, so results are directly comparable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
import openai

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "held_out_eval_set.jsonl"
RESULTS_PATH = Path(__file__).parent / "flagship_tier_comparison_results.json"

CLAUDE_FLAGSHIP_MODEL = "claude-sonnet-5"  # verified working via direct test call
GPT_FLAGSHIP_MODEL = "gpt-5"  # verified working via direct test call; requires max_completion_tokens, not max_tokens

FEW_SHOT_EXAMPLES = [
    {"query": "black waterproof trainers under 80 pounds",
     "constraints": [{"field": "color", "op": "eq", "value": "black"},
                     {"field": "waterproof", "op": "eq", "value": "true"},
                     {"field": "price_gbp", "op": "lte", "value": "80"}]},
    {"query": "almonds with no added sugar",
     "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"}]},
]


def build_prompt(query: str) -> str:
    instruction = (
        "Extract shopping constraints from the query as a JSON list of "
        '{"field": ..., "op": ..., "value": ...} objects. '
        "Return ONLY the JSON list, no prose."
    )
    shots = "\n\n".join(
        f"Query: {ex['query']}\nConstraints: {json.dumps(ex['constraints'])}"
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"{instruction}\n\n{shots}\n\nQuery: {query}\nConstraints:"


def parse_constraints(text: str):
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def constraint_set_f1(predicted, truth):
    pred_set = {(c.get("field"), c.get("op"), str(c.get("value"))) for c in predicted}
    true_set = {(c["field"], c["op"], str(c["value"])) for c in truth}
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    overlap = len(pred_set & true_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    eval_examples = [json.loads(l) for l in open(EVAL_SET_PATH)]
    print(f"Loaded {len(eval_examples)} held-out eval examples")
    print("NOTE: this is a SECONDARY comparison (flagship/expensive tier) -- "
          "the PRIMARY registered comparison remains the cheapest-tier baselines "
          "already measured (Claude Haiku 4.5: 0.543 F1, GPT cheap tier: 0.625 F1, few-shot).\n")

    claude_client = anthropic.Anthropic()
    gpt_client = openai.OpenAI()

    results = []
    for name, model_id, is_claude in [
        (f"claude_flagship({CLAUDE_FLAGSHIP_MODEL})", CLAUDE_FLAGSHIP_MODEL, True),
        (f"gpt_flagship({GPT_FLAGSHIP_MODEL})", GPT_FLAGSHIP_MODEL, False),
    ]:
        scores, parse_failures, errors = [], 0, 0
        t0 = time.time()
        for ex in eval_examples:
            prompt = build_prompt(ex["query"])
            try:
                if is_claude:
                    resp = claude_client.messages.create(model=model_id, max_tokens=300, messages=[{"role": "user", "content": prompt}])
                    text = next((b.text for b in resp.content if hasattr(b, "text")), "")
                else:
                    resp = gpt_client.chat.completions.create(model=model_id, max_completion_tokens=1000, messages=[{"role": "user", "content": prompt}])
                    text = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"  [{name}] API error: {e}")
                errors += 1
                scores.append(0.0)
                continue
            predicted = parse_constraints(text)
            if not predicted and text.strip():
                parse_failures += 1
            scores.append(constraint_set_f1(predicted, ex["constraints"]))
        elapsed = time.time() - t0
        mean_f1 = sum(scores) / len(scores) if scores else 0.0
        print(f"{name}: mean_f1={mean_f1:.4f} parse_failures={parse_failures} api_errors={errors} ({elapsed:.1f}s)")
        results.append({"system": name, "mean_f1": mean_f1, "n_examples": len(eval_examples),
                         "parse_failures": parse_failures, "api_errors": errors, "elapsed_seconds": round(elapsed, 1)})

    with open(RESULTS_PATH, "w") as f:
        json.dump({"MEASURED_HERE": True, "note": "SECONDARY comparison, not the primary registered baseline",
                    "results": results}, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
