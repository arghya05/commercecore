"""Merges the selected LoRA adapter into the base model and saves ONE
self-contained model directory -- no PEFT dependency needed to load it,
no internet access needed at inference time (base weights + adapter deltas
baked into one set of weights). This is what makes the model genuinely
portable: push this directory to Hugging Face, or copy it to any machine
(local, AWS, Azure, GCP, RunPod) and load it with plain
`AutoModelForCausalLM.from_pretrained(path)` -- no adapter_path, no PEFT
import required by the consumer.

Run this ON THE POD (needs the base model + selected checkpoint present).
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_PATH = "/workspace/models/Qwen3-1.7B"
ADAPTER_PATH = "/workspace/checkpoints_multitask/step_750"
MERGED_OUTPUT_PATH = "/workspace/commercecore_merged_step750"

if __name__ == "__main__":
    print("Loading base model (full precision, not quantized -- merge requires this)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0})

    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_OUTPUT_PATH}...")
    merged.save_pretrained(MERGED_OUTPUT_PATH)
    tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

    print("Verifying merged model loads and runs standalone (no PEFT)...")
    del base_model, model, merged
    torch.cuda.empty_cache()

    test_model = AutoModelForCausalLM.from_pretrained(MERGED_OUTPUT_PATH, dtype=torch.bfloat16, device_map={"": 0})
    test_tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_PATH)
    prompt = "Extract shopping constraints from this query: black waterproof trainers under 80 pounds\nConstraints:"
    input_ids = test_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = test_model.generate(input_ids, max_new_tokens=100, do_sample=False, pad_token_id=test_tokenizer.eos_token_id)
    generated = test_tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Standalone merged-model test output: {generated!r}")
    print("\nMerge complete and verified. This directory is fully portable -- "
          f"copy {MERGED_OUTPUT_PATH} anywhere, or push it to Hugging Face.")
