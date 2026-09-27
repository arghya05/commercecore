"""Stage 0 hardware profiling smoke test.

Runs a small number of real QLoRA optimizer steps against Qwen3-1.7B on this
machine's actual GPU, to measure peak VRAM, tokens/sec, and seconds/step.
Uses synthetic random-token batches — this is NOT training, it exists only to
characterize hardware behavior before trusting any cost/time estimate.
"""

import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/workspace/models/Qwen3-1.7B"
SEQ_LEN = 2048
MICRO_BATCH = 1
GRAD_ACCUM = 32
N_OPTIMIZER_STEPS = 50
LORA_R = 16
LORA_ALPHA = 32


def main():
    torch.manual_seed(42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading model in 4-bit NF4...")
    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    load_time = time.time() - t_load_start
    print(f"Model loaded in {load_time:.1f}s")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    vocab_size = model.config.vocab_size
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    step_times = []
    tokens_processed = 0

    print(f"Running {N_OPTIMIZER_STEPS} optimizer steps "
          f"(microbatch={MICRO_BATCH}, grad_accum={GRAD_ACCUM}, seq_len={SEQ_LEN})...")

    t_run_start = time.time()
    for opt_step in range(N_OPTIMIZER_STEPS):
        t_step_start = time.time()
        optimizer.zero_grad()

        for _ in range(GRAD_ACCUM):
            input_ids = torch.randint(
                0, vocab_size, (MICRO_BATCH, SEQ_LEN), device="cuda"
            )
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            tokens_processed += MICRO_BATCH * SEQ_LEN

        optimizer.step()
        torch.cuda.synchronize()

        step_time = time.time() - t_step_start
        step_times.append(step_time)

        if opt_step == 0 or (opt_step + 1) % 10 == 0:
            print(f"  step {opt_step + 1}/{N_OPTIMIZER_STEPS}: {step_time:.2f}s, "
                  f"loss={outputs.loss.item():.4f}")

    total_run_time = time.time() - t_run_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    avg_step_time = sum(step_times) / len(step_times)
    tokens_per_sec = tokens_processed / total_run_time

    results = {
        "MEASURED_HERE": True,
        "model": "Qwen/Qwen3-1.7B",
        "gpu": torch.cuda.get_device_name(0),
        "quantization": "NF4 double-quant, BF16 compute",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "seq_len": SEQ_LEN,
        "micro_batch": MICRO_BATCH,
        "grad_accum_steps": GRAD_ACCUM,
        "n_optimizer_steps": N_OPTIMIZER_STEPS,
        "model_load_time_seconds": round(load_time, 2),
        "total_run_time_seconds": round(total_run_time, 2),
        "avg_seconds_per_optimizer_step": round(avg_step_time, 3),
        "tokens_processed": tokens_processed,
        "tokens_per_second": round(tokens_per_sec, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "gpu_total_vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        ),
    }

    print("\n=== RESULTS (MEASURED_HERE) ===")
    print(json.dumps(results, indent=2))

    with open("/workspace/hardware_profile_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/hardware_profile_result.json")


if __name__ == "__main__":
    main()
