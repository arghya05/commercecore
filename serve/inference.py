"""CommerceCore inference interface -- the actual thing a startup integrates.

Design goal: a 0-to-1 startup can `pip install` this (or run the Docker image)
and get structured shopping-constraint extraction with almost no setup, no
GPU required for the quantized path.

Usage:
    from serve.inference import CommerceCoreQueryParser
    parser = CommerceCoreQueryParser(adapter_path="path/to/checkpoint")
    result = parser.parse("black waterproof trainers under 80 pounds")
    # -> [{"field": "color", "op": "eq", "value": "black"}, ...]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = "Qwen/Qwen3-1.7B"


class CommerceCoreQueryParser:
    """Loads the base model + a trained LoRA adapter, exposes a simple
    parse() method. Uses 4-bit quantization by default for low resource
    requirements -- this is the concrete implementation of the "lightweight,
    self-hosted, RunPod or local" serving decision (EXECUTION_PLAN.md section 16b)."""

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model_path: str = BASE_MODEL_ID,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_4bit: Optional[bool] = None,
    ):
        self.device = device
        if use_4bit is None:
            use_4bit = device == "cuda"  # 4-bit quant needs CUDA; CPU falls back to full precision

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)

        if adapter_path is not None:
            # PEFT is only needed for the (uncommon) case of loading a raw,
            # unmerged LoRA adapter on top of the base model -- most
            # deployments should use a pre-merged model directory (see
            # serve/merge_and_export.py) and never hit this import at all.
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def _parse_json_list(self, text: str) -> List[Dict]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    def parse(self, query: str, max_new_tokens: int = 100) -> List[Dict]:
        """Extract shopping constraints from a natural-language query.
        Returns a list of {"field": ..., "op": ..., "value": ...} dicts.
        Returns [] on parse failure -- an honest empty result, not a guess."""
        prompt = f"Extract shopping constraints from this query: {query}\nConstraints:"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_json_list(generated)


if __name__ == "__main__":
    import sys
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    parser = CommerceCoreQueryParser(adapter_path=adapter)
    test_queries = [
        "black waterproof trainers under 80 pounds",
        "almonds with no added sugar",
    ]
    for q in test_queries:
        result = parser.parse(q)
        print(f"Query: {q!r}\n  -> {result}")
