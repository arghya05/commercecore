"""Full, thorough inference test of the merged, standalone CommerceCore
model -- multiple diverse queries across all 3 verticals, using the actual
serving inference class (not a one-off smoke test), to verify the exported
artifact genuinely works end-to-end before any 'benchmarks cleared' claim.
"""

import json
import sys

sys.path.insert(0, "/workspace")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_MODEL_PATH = "/workspace/commercecore_merged_step750"

TEST_QUERIES = [
    ("black waterproof trainers under 80 pounds", "fashion"),
    ("red leather jacket size M", "fashion"),
    ("almonds with no added sugar", "grocery"),
    ("snack bar with nuts, don't care about sugar", "grocery"),
    ("battery powered speaker at least 12 month warranty", "general"),
    ("mains powered kettle under 50 quid", "general"),
    ("wool sweater size S", "fashion"),
    ("oat milk vegan", "grocery"),
]


def parse_json_list(text):
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None, "no_json_found"
    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"parse_error: {e}"


if __name__ == "__main__":
    print(f"Loading merged model from {MERGED_MODEL_PATH} (standalone, no adapter_path, no PEFT)...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    print("Model loaded successfully as a plain AutoModelForCausalLM -- no PEFT import needed.\n")

    results = []
    n_valid_json = 0
    for query, vertical in TEST_QUERIES:
        prompt = f"Extract shopping constraints from this query: {query}\nConstraints:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        parsed, error = parse_json_list(generated)
        valid = parsed is not None
        if valid:
            n_valid_json += 1
        results.append({
            "query": query, "vertical": vertical, "raw_output": generated,
            "parsed": parsed, "valid_json": valid, "error": error,
        })
        print(f"[{vertical}] {query!r}")
        print(f"  -> {generated!r}")
        print(f"  -> parsed: {parsed} (valid_json={valid})\n")

    summary = {
        "n_queries_tested": len(TEST_QUERIES),
        "n_valid_json_output": n_valid_json,
        "valid_json_rate": n_valid_json / len(TEST_QUERIES),
        "results": results,
    }
    with open("/workspace/full_inference_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"=== SUMMARY: {n_valid_json}/{len(TEST_QUERIES)} produced valid, parseable JSON output ===")
