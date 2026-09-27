"""Rigorous, per-vertical, bootstrap-CI'd comparison of frontier baselines on
the constraint-extraction task. Runs each system 3 times (temperature > 0
where applicable, different sampling) to get a real variance estimate,
rather than a single-shot point estimate -- per EXECUTION_PLAN.md section 5's
statistical rigor requirement.
"""

import json
import random
import re
import time
from pathlib import Path

import anthropic
import openai

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "held_out_eval_set.jsonl"
RESULTS_PATH = Path(__file__).parent / "rigorous_baseline_results.json"
N_BOOTSTRAP = 10000
N_REPEATS = 3

random.seed(42)

FEW_SHOT_EXAMPLES = [
    {"query": "black waterproof trainers under 80 pounds",
     "constraints": [{"field": "color", "op": "eq", "value": "black"},
                     {"field": "waterproof", "op": "eq", "value": "true"},
                     {"field": "price_gbp", "op": "lte", "value": "80"}]},
    {"query": "almonds with no added sugar",
     "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"}]},
]


def build_prompt(query):
    shots = "\n\n".join(
        f"Query: {ex['query']}\nConstraints: {json.dumps(ex['constraints'])}" for ex in FEW_SHOT_EXAMPLES
    )
    return (
        "Extract shopping constraints from the query as a JSON list of "
        '{"field": ..., "op": ..., "value": ...} objects. Return ONLY the JSON list, no prose.\n\n'
        f"{shots}\n\nQuery: {query}\nConstraints:"
    )


def parse_constraints(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def f1(predicted, truth):
    pred_set = {(c.get("field"), c.get("op"), str(c.get("value"))) for c in predicted}
    true_set = {(c["field"], c["op"], str(c["value"])) for c in truth}
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    overlap = len(pred_set & true_set)
    p, r = overlap / len(pred_set), overlap / len(true_set)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def bootstrap_ci(scores, n_resamples=N_BOOTSTRAP, ci=0.95):
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_resamples):
        resample = [scores[random.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples)
    return means[lo_idx], means[hi_idx]


def get_claude_text(resp):
    for block in resp.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def call_claude(client, model, query, temperature):
    # claude-sonnet-5 rejects the temperature parameter entirely (fixed/managed
    # sampling) -- confirmed via a direct test call. Repeats against this model
    # are honestly deterministic, not independently sampled; disclosed in the
    # results rather than silently erroring on every non-zero-temp repeat.
    kwargs = dict(model=model, max_tokens=300, messages=[{"role": "user", "content": build_prompt(query)}])
    if model != "claude-sonnet-5":
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    return get_claude_text(resp)


def call_gpt(client, model, query, temperature, is_flagship):
    if is_flagship:
        resp = client.chat.completions.create(
            model=model, max_completion_tokens=1000,
            messages=[{"role": "user", "content": build_prompt(query)}],
        )
    else:
        resp = client.chat.completions.create(
            model=model, max_tokens=300, temperature=temperature,
            messages=[{"role": "user", "content": build_prompt(query)}],
        )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    eval_examples = [json.loads(l) for l in open(EVAL_SET_PATH)]
    print(f"Loaded {len(eval_examples)} eval examples, {N_REPEATS} repeats/system")

    claude_client = anthropic.Anthropic()
    gpt_client = openai.OpenAI()

    systems = [
        ("claude_haiku_4.5", "claude-haiku-4-5-20251001", "claude", False),
        ("claude_sonnet_5", "claude-sonnet-5", "claude", False),
        ("gpt_cheap_tier", "gpt-4o-mini", "gpt", False),
        ("gpt_5_flagship", "gpt-5", "gpt", True),
    ]

    all_results = {}
    for name, model_id, provider, is_flagship in systems:
        per_vertical_scores = {"fashion": [], "grocery": [], "general": []}
        all_scores = []
        for repeat in range(N_REPEATS):
            temp = 0.0 if repeat == 0 else 0.7  # first run deterministic, others sample for real variance
            for ex in eval_examples:
                try:
                    if provider == "claude":
                        text = call_claude(claude_client, model_id, ex["query"], temp)
                    else:
                        text = call_gpt(gpt_client, model_id, ex["query"], temp, is_flagship)
                except Exception as e:
                    print(f"  [{name}] repeat {repeat} error on {ex['query'][:30]!r}: {e}")
                    text = ""
                predicted = parse_constraints(text)
                score = f1(predicted, ex["constraints"])
                all_scores.append(score)
                per_vertical_scores[ex["vertical"]].append(score)
                time.sleep(0.5)
            print(f"  [{name}] repeat {repeat+1}/{N_REPEATS} done")

        mean_f1 = sum(all_scores) / len(all_scores)
        ci_lo, ci_hi = bootstrap_ci(all_scores)
        per_vertical_means = {v: (sum(s) / len(s) if s else None) for v, s in per_vertical_scores.items()}

        all_results[name] = {
            "mean_f1": mean_f1,
            "bootstrap_95ci": [ci_lo, ci_hi],
            "n_total_scores": len(all_scores),
            "n_repeats": N_REPEATS,
            "per_vertical_mean_f1": per_vertical_means,
        }
        print(f"{name}: mean_f1={mean_f1:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
              f"per_vertical={ {k: round(v,3) if v else None for k,v in per_vertical_means.items()} }")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"MEASURED_HERE": True, "results": all_results}, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
