"""Full, rigorous evaluation on the ENTIRE QueryNER public test set (993
examples), not a small sample. Runs against the complete real benchmark,
plus computes a proper confidence interval on the result -- per
EXECUTION_PLAN.md section 5's statistical rigor requirement (this is a
per-example F1, not a binomial accept/reject, so we report mean + bootstrap
CI, not the exact binomial LB formula, but still real uncertainty
quantification, not a bare point estimate).

REPRODUCIBILITY: uses only paths relative to this repo (data/queryner/) and
the published Hugging Face model -- no hardcoded pod paths. Runs on any
machine with the dependencies installed (see serve/requirements.txt);
GPU recommended (see README's latency section) but not required.
"""

import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = "arghya2030/commercecore-qwen3-1.7b"  # published merged model, no local adapter needed
QUERYNER_TEST_PATH = REPO_ROOT / "data" / "queryner" / "test.jsonl"
N_BOOTSTRAP = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)


def parse_spans(text):
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
    p, r = overlap / len(pred_set), overlap / len(true_set)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def bootstrap_ci(scores, n_resamples=N_BOOTSTRAP, ci=0.95):
    n = len(scores)
    means = []
    for _ in range(n_resamples):
        resample = [scores[random.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples)
    return means[lo_idx], means[hi_idx]


if __name__ == "__main__":
    records = [json.loads(l) for l in open(QUERYNER_TEST_PATH)]
    print(f"Loaded FULL QueryNER test set: {len(records)} examples")

    print(f"Loading {MODEL_PATH} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    scores = []
    for i, rec in enumerate(records):
        tokens = rec["tokens"]
        true_spans = [{"label": s["label"], "text": " ".join(tokens[s["start_token"]:s["end_token"]])} for s in rec.get("spans", [])]
        prompt = f"Extract entities from this query: {rec['text_reconstructed']}\nEntities:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        predicted = parse_spans(generated)
        scores.append(span_set_f1(predicted, true_spans))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(records)} done, running mean so far: {sum(scores)/len(scores):.4f}")

    mean_f1 = sum(scores) / len(scores)
    ci_lo, ci_hi = bootstrap_ci(scores)

    result = {
        "MEASURED_HERE": True,
        "benchmark": "QueryNER FULL public test set",
        "n_examples": len(records),
        "model": MODEL_PATH,
        "mean_f1": mean_f1,
        "bootstrap_95ci_lower": ci_lo,
        "bootstrap_95ci_upper": ci_hi,
        "n_bootstrap_resamples": N_BOOTSTRAP,
    }
    out_path = REPO_ROOT / "eval" / "full_queryner_eval_result_reproduced.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
    print("\n=== FULL RESULT ===")
    print(json.dumps(result, indent=2))
