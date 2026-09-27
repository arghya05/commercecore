"""Debug QLoRA training run: Qwen3-1.7B on prepared QueryNER data.

Proves the training loop mechanics (checkpoint + eval cadence, per EXECUTION_PLAN.md
section 10b) work end-to-end. Not a publication-quality training run.
"""

import csv
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/workspace/models/Qwen3-1.7B"
TRAIN_PATH = "/workspace/train_prepared.jsonl"
EVAL_PATH = "/workspace/test_prepared.jsonl"
CHECKPOINT_DIR = "/workspace/checkpoints"
LOSS_LOG_PATH = "/workspace/training_loss_log.csv"
EVAL_LOG_PATH = "/workspace/checkpoint_eval_log.csv"

N_STEPS = 200
CHECKPOINT_EVERY = 25
MICRO_BATCH = 1
GRAD_ACCUM = 8
LR = 1e-4


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_example_ids(tokenizer, prompt, completion, max_len=512):
    full_text = prompt + completion + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    full_ids = full_ids[:max_len]
    labels = list(full_ids)
    n_prompt_tokens = min(len(prompt_ids), len(labels))
    for i in range(n_prompt_tokens):
        labels[i] = -100
    return full_ids, labels


def compute_token_f1(pred_text, true_text):
    pred_tokens = set(pred_text.lower().split())
    true_tokens = set(true_text.lower().split())
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    overlap = len(pred_tokens & true_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(true_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_checkpoint(model, tokenizer, eval_data, device, n_examples=30):
    model.eval()
    exact_matches = 0
    f1_scores = []
    for rec in eval_data[:n_examples]:
        prompt_ids = tokenizer(rec["prompt"] + "\nEntities:", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        true_completion = rec["completion"].replace("\nEntities: ", "").strip()
        gen_clean = generated.strip()
        if gen_clean == true_completion:
            exact_matches += 1
        f1_scores.append(compute_token_f1(gen_clean, true_completion))
    model.train()
    n = len(eval_data[:n_examples])
    return {
        "n_examples": n,
        "exact_match": exact_matches / n,
        "token_f1": sum(f1_scores) / len(f1_scores),
    }


def main():
    torch.manual_seed(42)
    device = "cuda"

    print("Loading tokenizer + model in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    train_data = load_jsonl(TRAIN_PATH)
    eval_data = load_jsonl(EVAL_PATH)
    print(f"Loaded {len(train_data)} training examples, {len(eval_data)} eval examples")

    examples = [build_example_ids(tokenizer, r["prompt"], r["completion"]) for r in train_data]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    loss_log = open(LOSS_LOG_PATH, "w", newline="")
    loss_writer = csv.writer(loss_log)
    loss_writer.writerow(["step", "loss", "elapsed_seconds"])

    eval_log = open(EVAL_LOG_PATH, "w", newline="")
    eval_writer = csv.writer(eval_log)
    eval_writer.writerow(["step", "n_examples", "exact_match", "token_f1"])

    t_start = time.time()
    example_idx = 0
    model.train()

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

        if step % 10 == 0 or step == 1:
            print(f"step {step:4d}  loss={total_loss:.4f}  elapsed={elapsed:.1f}s")

        if step % CHECKPOINT_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/step_{step}"
            model.save_pretrained(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            metrics = evaluate_checkpoint(model, tokenizer, eval_data, device)
            eval_writer.writerow([step, metrics["n_examples"], round(metrics["exact_match"], 4), round(metrics["token_f1"], 4)])
            eval_log.flush()
            print(f"  Eval @ step {step}: exact_match={metrics['exact_match']:.3f}  token_f1={metrics['token_f1']:.3f}")

    total_time = time.time() - t_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    loss_log.close()
    eval_log.close()

    summary = {
        "n_steps": N_STEPS,
        "total_time_seconds": round(total_time, 1),
        "total_time_minutes": round(total_time / 60, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "estimated_cost_usd": round((total_time / 3600) * 0.72, 4),
    }
    with open("/workspace/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
