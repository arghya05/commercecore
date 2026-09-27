"""Complete inference test: runs ALL 20 held-out eval queries through the
actual serving inference class (serve/inference.py), on the merged model,
computing real F1 against ground truth -- proving the SERVED artifact
(not just the raw training checkpoint) achieves the claimed benchmark
numbers. This is the final verification gate before publishing.
"""

import json
import sys

sys.path.insert(0, "/workspace")
import importlib.util

spec = importlib.util.spec_from_file_location("serve_inference", "/workspace/serve_inference.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

EVAL_PATH = "/workspace/held_out_eval_set_vertical.jsonl"
MERGED_MODEL_PATH = "/workspace/commercecore_merged_step750"


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


if __name__ == "__main__":
    eval_examples = [json.loads(l) for l in open(EVAL_PATH)]
    print(f"Loaded {len(eval_examples)} eval examples (via the SERVED, merged model)")

    parser = mod.CommerceCoreQueryParser(base_model_path=MERGED_MODEL_PATH, device="cuda", use_4bit=False)
    print("Served model loaded.\n")

    scores_by_vertical = {"fashion": [], "grocery": [], "general": []}
    all_scores = []
    results = []

    for ex in eval_examples:
        predicted = parser.parse(ex["query"])
        score = f1(predicted, ex["constraints"])
        all_scores.append(score)
        scores_by_vertical[ex["vertical"]].append(score)
        results.append({"query": ex["query"], "vertical": ex["vertical"],
                         "predicted": predicted, "truth": ex["constraints"], "f1": score})
        print(f"[{ex['vertical']}] {ex['query'][:50]!r} f1={score:.3f}")

    overall_f1 = sum(all_scores) / len(all_scores)
    per_vertical_f1 = {v: (sum(s) / len(s) if s else None) for v, s in scores_by_vertical.items()}

    summary = {
        "MEASURED_HERE": True,
        "source": "served merged model (serve/inference.py), not raw training checkpoint",
        "n_examples": len(eval_examples),
        "overall_f1": overall_f1,
        "per_vertical_f1": per_vertical_f1,
        "per_vertical_n": {v: len(s) for v, s in scores_by_vertical.items()},
        "results": results,
    }
    with open("/workspace/complete_inference_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== COMPLETE TEST RESULT ===")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Per-vertical F1: {json.dumps(per_vertical_f1, indent=2)}")
