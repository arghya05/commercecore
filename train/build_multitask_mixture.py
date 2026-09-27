"""Combines all currently available real training data into one diverse
multitask mixture: QueryNER (real, human-annotated entity segmentation),
synthetic query-constraint extraction, and synthetic catalog-attribute
extraction. Runs on the pod, reads from /workspace/.
"""

import json
import random

random.seed(42)

QUERYNER_PATH = "/workspace/train.jsonl"
SYNTHETIC_QUERY_PATH = "/workspace/synthetic_training_text.jsonl"
CATALOG_PATH = "/workspace/catalog_training_text.jsonl"
OUT_PATH = "/workspace/multitask_train_mixture.jsonl"

N_QUERYNER_SAMPLE = 1500  # larger, more diverse sample than the 300-example debug run


def load_queryner(path, n):
    records = [json.loads(l) for l in open(path)]
    random.shuffle(records)
    records = records[:n]
    examples = []
    for rec in records:
        tokens = rec["tokens"]
        spans = [{"label": s["label"], "text": " ".join(tokens[s["start_token"]:s["end_token"]])} for s in rec.get("spans", [])]
        examples.append({
            "task": "entity_segmentation",
            "prompt": f"Extract entities from this query: {rec['text_reconstructed']}",
            "completion": f"\nEntities: {json.dumps(spans)}",
        })
    return examples


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


if __name__ == "__main__":
    queryner_examples = load_queryner(QUERYNER_PATH, N_QUERYNER_SAMPLE)
    synthetic_query = load_jsonl(SYNTHETIC_QUERY_PATH)
    catalog = load_jsonl(CATALOG_PATH)

    all_examples = queryner_examples + synthetic_query + catalog
    random.shuffle(all_examples)

    with open(OUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps({"prompt": ex["prompt"], "completion": ex["completion"]}) + "\n")

    print(f"QueryNER (real, sampled): {len(queryner_examples)}")
    print(f"Synthetic query-constraints: {len(synthetic_query)}")
    print(f"Synthetic catalog-attributes: {len(catalog)}")
    print(f"TOTAL multitask mixture: {len(all_examples)}")
    print(f"Written to {OUT_PATH}")
