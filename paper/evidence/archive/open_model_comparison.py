"""Real comparison against unfine-tuned open-source small models on Hugging
Face: Qwen3.5-2B and SmolLM3-3B. Proves (or disproves) whether CommerceCore's
fine-tuning is actually needed vs. just using a slightly bigger open model
zero/few-shot -- per EXECUTION_PLAN.md section 8's frozen/pretrained baseline
category. Runs ON THE POD (needs GPU).
"""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_PATH = "/workspace/held_out_eval_set.jsonl"

FEW_SHOT = [
    {"query": "black waterproof trainers under 80 pounds",
     "constraints": [{"field": "color", "op": "eq", "value": "black"},
                     {"field": "waterproof", "op": "eq", "value": "true"},
                     {"field": "price_gbp", "op": "lte", "value": "80"}]},
    {"query": "almonds with no added sugar",
     "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"}]},
]


def build_prompt(query):
    shots = "\n\n".join(
        f"Query: {ex['query']}\nConstraints: {json.dumps(ex['constraints'])}" for ex in FEW_SHOT
    )
    return (
        "Extract shopping constraints from the query as a JSON list of "
        '{"field": ..., "op": ..., "value": ...} objects. Return ONLY the JSON list.\n\n'
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


def eval_model(model_path, eval_data, device="cuda"):
    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    scores = []
    for ex in eval_data:
        prompt = build_prompt(ex["query"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        predicted = parse_constraints(generated)
        scores.append(f1(predicted, ex["constraints"]))

    del model
    torch.cuda.empty_cache()
    return sum(scores) / len(scores)


if __name__ == "__main__":
    eval_data = [json.loads(l) for l in open(EVAL_PATH)]
    print(f"Loaded {len(eval_data)} eval examples")

    results = {}
    for name, path in [
        ("qwen3.5-2b_unfinetuned", "/workspace/models/Qwen3.5-2B"),
        ("smollm3-3b_unfinetuned", "/workspace/models/SmolLM3-3B"),
    ]:
        mean_f1 = eval_model(path, eval_data)
        results[name] = mean_f1
        print(f"{name}: mean_f1={mean_f1:.4f}")

    with open("/workspace/open_model_comparison_results.json", "w") as f:
        json.dump({"MEASURED_HERE": True, "results": results}, f, indent=2)
    print(json.dumps(results, indent=2))
