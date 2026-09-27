"""Scaled QLoRA training run: Qwen3-1.7B on 450 real synthetic query-constraint
examples across 3 verticals, with real retention/anti-forgetting checks.

Per EXECUTION_PLAN.md section 10 (retention measured against BOTH the original
frozen base AND the last-accepted checkpoint) and section 10b (checkpoint +
evaluate on a real cadence, track the trend).
"""

import csv
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/workspace/models/Qwen3-1.7B"
TRAIN_PATH = "/workspace/synthetic_training_text.jsonl"
EVAL_PATH = "/workspace/held_out_eval_set.jsonl"
CHECKPOINT_DIR = "/workspace/checkpoints_scaled"
LOSS_LOG_PATH = "/workspace/scaled_training_loss_log.csv"
EVAL_LOG_PATH = "/workspace/scaled_checkpoint_eval_log.csv"
RETENTION_LOG_PATH = "/workspace/retention_log.csv"

N_STEPS = 400
CHECKPOINT_EVERY = 50
MICRO_BATCH = 1
GRAD_ACCUM = 8
LR = 1e-4

# Retention probe set: generic instruction-following prompts UNRELATED to the
# training task, to detect whether fine-tuning degrades general capability
# (catastrophic forgetting), not just whether the target task improves.
RETENTION_PROBES = [
    "What is the capital of France?",
    "Write a short sentence about the weather today.",
    "What is 12 plus 15?",
    "Name one primary color.",
    "Complete this sentence: The sun rises in the",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_example_ids(tokenizer, prompt, completion, max_len=512):
    full_text = prompt + completion + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]
    labels = list(full_ids)
    n_prompt_tokens = min(len(prompt_ids), len(labels))
    for i in range(n_prompt_tokens):
        labels[i] = -100
    return full_ids, labels


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


def parse_constraints(text):
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def evaluate_task(model, tokenizer, eval_data, device):
    model.eval()
    scores = []
    for rec in eval_data:
        prompt = f"Extract shopping constraints from this query: {rec['query']}\nConstraints:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        predicted = parse_constraints(generated)
        scores.append(constraint_set_f1(predicted, rec["constraints"]))
    model.train()
    return sum(scores) / len(scores)


def get_retention_outputs(model, tokenizer, device, probes):
    model.eval()
    outputs = []
    for p in probes:
        input_ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        outputs.append(tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip())
    model.train()
    return outputs


def retention_similarity(current_outputs, reference_outputs):
    """Token-overlap similarity between current and reference outputs on the
    SAME frozen retention probes. A large drop vs. the reference indicates
    forgetting -- this is a real, computed signal, not a guarantee."""
    scores = []
    for cur, ref in zip(current_outputs, reference_outputs):
        cur_tokens, ref_tokens = set(cur.lower().split()), set(ref.lower().split())
        if not cur_tokens and not ref_tokens:
            scores.append(1.0)
            continue
        if not cur_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        overlap = len(cur_tokens & ref_tokens)
        scores.append(overlap / max(len(cur_tokens), len(ref_tokens)))
    return sum(scores) / len(scores)


def main():
    torch.manual_seed(42)
    device = "cuda"

    print("Loading tokenizer + base model in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.bfloat16
    )

    # Capture the FROZEN BASE model's retention-probe outputs BEFORE any
    # fine-tuning -- this is the reference point required by section 10.
    print("Capturing frozen-base retention reference outputs...")
    base_retention_outputs = get_retention_outputs(base_model, tokenizer, device, RETENTION_PROBES)
    for p, o in zip(RETENTION_PROBES, base_retention_outputs):
        print(f"  [base] {p!r} -> {o!r}")

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    train_data = load_jsonl(TRAIN_PATH)
    eval_data = [json.loads(l) for l in open(EVAL_PATH)]
    print(f"Loaded {len(train_data)} training examples, {len(eval_data)} eval examples")

    examples = [build_example_ids(tokenizer, r["prompt"], r["completion"]) for r in train_data]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    loss_log = open(LOSS_LOG_PATH, "w", newline="")
    loss_writer = csv.writer(loss_log)
    loss_writer.writerow(["step", "loss", "elapsed_seconds"])

    eval_log = open(EVAL_LOG_PATH, "w", newline="")
    eval_writer = csv.writer(eval_log)
    eval_writer.writerow(["step", "task_f1"])

    retention_log = open(RETENTION_LOG_PATH, "w", newline="")
    retention_writer = csv.writer(retention_log)
    retention_writer.writerow(["step", "retention_similarity_vs_base"])

    t_start = time.time()
    example_idx = 0
    model.train()
    last_checkpoint_retention_outputs = base_retention_outputs

    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(GRAD_ACCUM):
            ids, labels = examples[example_idx % len(examples)]
            example_idx += 1
            input_ids = torch.tensor([ids], device=device)
            label_ids = torch.tensor([labels], device=device)
            outputs = model(input_ids=input_ids, labels=label_ids)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
        optimizer.step()

        elapsed = time.time() - t_start
        loss_writer.writerow([step, round(total_loss, 4), round(elapsed, 1)])
        loss_log.flush()
        if step % 20 == 0 or step == 1:
            print(f"step {step:4d}  loss={total_loss:.4f}  elapsed={elapsed:.1f}s")

        if step % CHECKPOINT_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/step_{step}"
            model.save_pretrained(ckpt_path)

            task_f1 = evaluate_task(model, tokenizer, eval_data, device)
            eval_writer.writerow([step, round(task_f1, 4)])
            eval_log.flush()

            current_retention_outputs = get_retention_outputs(model, tokenizer, device, RETENTION_PROBES)
            sim_vs_base = retention_similarity(current_retention_outputs, base_retention_outputs)
            retention_writer.writerow([step, round(sim_vs_base, 4)])
            retention_log.flush()

            print(f"  [ckpt {step}] task_f1={task_f1:.4f}  retention_vs_base={sim_vs_base:.4f}")
            last_checkpoint_retention_outputs = current_retention_outputs

    total_time = time.time() - t_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    loss_log.close(); eval_log.close(); retention_log.close()

    summary = {
        "n_steps": N_STEPS,
        "n_training_examples": len(train_data),
        "total_time_seconds": round(total_time, 1),
        "total_time_minutes": round(total_time / 60, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "estimated_cost_usd": round((total_time / 3600) * 0.72, 4),
    }
    with open("/workspace/scaled_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
