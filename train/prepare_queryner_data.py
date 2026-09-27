"""Converts QueryNER JSONL into prompt/completion pairs for causal LM fine-tuning."""

import json

TRAIN_IN = "/workspace/train.jsonl"
TEST_IN = "/workspace/test.jsonl"
TRAIN_OUT = "/workspace/train_prepared.jsonl"
TEST_OUT = "/workspace/test_prepared.jsonl"

PROMPT_PREFIX = "Extract entities from this query: "
COMPLETION_PREFIX = "\nEntities: "


def convert_record(rec):
    text = rec["text_reconstructed"]
    tokens = rec["tokens"]
    spans = []
    for span in rec.get("spans", []):
        span_text = " ".join(tokens[span["start_token"]:span["end_token"]])
        spans.append({"label": span["label"], "text": span_text})
    prompt = PROMPT_PREFIX + text
    completion = COMPLETION_PREFIX + json.dumps(spans)
    return {"prompt": prompt, "completion": completion}


def convert_file(path_in, path_out, limit=None):
    n = 0
    with open(path_in) as fin, open(path_out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            out = convert_record(rec)
            fout.write(json.dumps(out) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


if __name__ == "__main__":
    n_train = convert_file(TRAIN_IN, TRAIN_OUT, limit=300)
    n_test = convert_file(TEST_IN, TEST_OUT, limit=50)
    print(f"Wrote {n_train} training examples to {TRAIN_OUT}")
    print(f"Wrote {n_test} eval examples to {TEST_OUT}")

    with open(TRAIN_OUT) as f:
        first = json.loads(f.readline())
        print("Example prompt:", first["prompt"])
        print("Example completion:", first["completion"])
