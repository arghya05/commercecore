"""Fresh, independent head-to-head comparison: CommerceCore (served, merged
model) vs Claude (cheap + flagship), GPT (cheap + flagship), and unfine-tuned
open models (Qwen3.5-2B, SmolLM3-3B) -- on entirely NEW example queries not
used in any prior eval set in this project, to avoid any appearance of
cherry-picking. Runs the CommerceCore + open-model portion locally on the
pod GPU; the Claude/GPT portion needs API keys sourced separately (run
locally on the Mac, not this pod script, and merge results after).
"""

import json
import sys

sys.path.insert(0, "/workspace")
import importlib.util

spec = importlib.util.spec_from_file_location("serve_inference", "/workspace/serve_inference.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fresh queries -- none of these appear in held_out_eval_set.jsonl or any
# prior eval file in this project.
FRESH_QUERIES = [
    {"query": "green cotton hoodie size L under 40 pounds", "vertical": "fashion",
     "constraints": [{"field": "color", "op": "eq", "value": "green"}, {"field": "material", "op": "eq", "value": "cotton"},
                     {"field": "size", "op": "eq", "value": "L"}, {"field": "price_gbp", "op": "lte", "value": "40"}]},
    {"query": "waterproof boots not black", "vertical": "fashion",
     "constraints": [{"field": "waterproof", "op": "eq", "value": "true"}]},
    {"query": "protein bar no nuts please, sugar is fine", "vertical": "grocery",
     "constraints": [{"field": "contains_nuts", "op": "eq", "value": "false"}]},
    {"query": "vegan chocolate with no added sugar under 5 pounds", "vertical": "grocery",
     "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"}, {"field": "price_gbp", "op": "lte", "value": "5"}]},
    {"query": "portable speaker under 3 year warranty running on batteries", "vertical": "general",
     "constraints": [{"field": "power_source", "op": "eq", "value": "battery"}, {"field": "warranty_months", "op": "lte", "value": "36"}]},
    {"query": "wall powered toaster at least 18 months guarantee", "vertical": "general",
     "constraints": [{"field": "power_source", "op": "eq", "value": "mains"}, {"field": "warranty_months", "op": "gte", "value": "18"}]},
]


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


def run_commercecore():
    parser = mod.CommerceCoreQueryParser(base_model_path="/workspace/commercecore_merged_step750", device="cuda", use_4bit=False)
    scores, details = [], []
    for ex in FRESH_QUERIES:
        predicted = parser.parse(ex["query"])
        score = f1(predicted, ex["constraints"])
        scores.append(score)
        details.append({"query": ex["query"], "predicted": predicted, "truth": ex["constraints"], "f1": score})
    del parser
    torch.cuda.empty_cache()
    return sum(scores) / len(scores), details


def run_open_model(model_path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    few_shot = (
        "Extract shopping constraints from the query as a JSON list of "
        '{"field": ..., "op": ..., "value": ...} objects. Return ONLY the JSON list.\n\n'
        'Query: black waterproof trainers under 80 pounds\n'
        'Constraints: [{"field": "color", "op": "eq", "value": "black"}, {"field": "waterproof", "op": "eq", "value": "true"}, {"field": "price_gbp", "op": "lte", "value": "80"}]\n\n'
    )
    scores, details = [], []
    for ex in FRESH_QUERIES:
        prompt = few_shot + f"Query: {ex['query']}\nConstraints:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        import re
        match = re.search(r"\[.*\]", generated, re.DOTALL)
        try:
            predicted = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            predicted = []
        score = f1(predicted, ex["constraints"])
        scores.append(score)
        details.append({"query": ex["query"], "predicted": predicted, "truth": ex["constraints"], "f1": score})

    del model
    torch.cuda.empty_cache()
    return sum(scores) / len(scores), details


if __name__ == "__main__":
    results = {}

    print("=== CommerceCore (this project's model) ===")
    cc_mean, cc_details = run_commercecore()
    results["commercecore"] = {"mean_f1": cc_mean, "details": cc_details}
    print(f"CommerceCore mean F1 on {len(FRESH_QUERIES)} fresh queries: {cc_mean:.4f}")

    print("\n=== Qwen3.5-2B (unfine-tuned, few-shot) ===")
    q_mean, q_details = run_open_model("/workspace/models/Qwen3.5-2B", "qwen3.5-2b")
    results["qwen3.5-2b_unfinetuned"] = {"mean_f1": q_mean, "details": q_details}
    print(f"Qwen3.5-2B mean F1: {q_mean:.4f}")

    print("\n=== SmolLM3-3B (unfine-tuned, few-shot) ===")
    s_mean, s_details = run_open_model("/workspace/models/SmolLM3-3B", "smollm3-3b")
    results["smollm3-3b_unfinetuned"] = {"mean_f1": s_mean, "details": s_details}
    print(f"SmolLM3-3B mean F1: {s_mean:.4f}")

    with open("/workspace/fresh_headtohead_gpu_results.json", "w") as f:
        json.dump({"MEASURED_HERE": True, "n_queries": len(FRESH_QUERIES),
                    "note": "Fresh queries, not reused from any prior eval set in this project",
                    "results": results}, f, indent=2)
    print("\nSaved GPU-side results to /workspace/fresh_headtohead_gpu_results.json")
    print("Run the Claude/GPT portion separately (needs API keys) and merge.")
