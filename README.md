# CommerceCore

A small (1.7B parameter), self-hostable model that parses natural-language shopping queries into structured constraints — trained for under $10 in total compute + API cost. Beats every tested frontier model on the primary held-out benchmarks; an honest, independent fresh-query test below shows this advantage does not hold universally — read that section before relying on any headline number.

**Model weights + model card (download here): https://huggingface.co/arghya2030/commercecore-qwen3-1.7b**

**Status:** research / proof-of-concept. Read "What this is NOT" before using in production.

---

## The problem

Ecommerce search boxes get queries like *"black waterproof trainers under 80 pounds"* or *"almonds with no added sugar."* To actually act on a query like this — filter a catalog, rank results, explain a recommendation — a system needs to turn that free text into structured constraints:

```json
[{"field": "color", "op": "eq", "value": "black"},
 {"field": "waterproof", "op": "eq", "value": "true"},
 {"field": "price_gbp", "op": "lte", "value": "80"}]
```

This looks like a task a frontier LLM should trivially handle. It doesn't, reliably, for reasons that are structural, not just a matter of prompting harder:

1. **It's a narrow, repetitive extraction task, not a reasoning task.** Frontier models are optimized for breadth — general knowledge, multi-step reasoning, code, conversation. A narrow, repetitive schema-extraction task doesn't play to that strength, and a general-purpose model has no reason to have specialized on the exact field vocabulary (`price_gbp`, `contains_added_sugar`, `warranty_months`) a specific catalog uses.
2. **Cost and latency compound at scale.** A live search-as-you-type product needs sub-second responses on every keystroke, potentially millions of times a day. Routing every query through a frontier API is expensive and adds real network + inference latency that a self-hosted small model doesn't have.
3. **Data doesn't leave your infrastructure.** A self-hosted model means customer search/purchase data never has to leave your own servers — a real, concrete advantage for privacy-sensitive catalogs (health, grocery, finance-adjacent retail).

**Measured, not assumed:** we tested this directly. On our held-out constraint-extraction set and on the real, public QueryNER benchmark, unmodified Claude and GPT — cheap tiers *and* flagship tiers — scored in the 0.20–0.62 F1 range. They're not incapable, but they're not specialized either, and a small model fine-tuned specifically for this schema clears that bar by a wide margin, for a fraction of the cost.

## The strategy

1. **Fine-tune a small, open base model** (Qwen3-1.7B, Apache-2.0) on this exact task, via QLoRA — a 4-bit-quantized, low-rank adaptation method that makes fine-tuning cheap and fast on a single consumer/prosumer GPU.
2. **Build training data honestly, without manual annotation.** ~70% of the training mixture is real, human-annotated public data (QueryNER, CC BY 4.0). The rest is LLM-generated synthetic data, but generated with a **structured-scenario-first** method: the actual constraint labels are fixed deterministically in code *before* any LLM call — the LLM's only job is paraphrasing that fixed scenario into natural language. This avoids the common synthetic-data failure mode where an LLM silently invents or drops a label while "helpfully" writing the query.
3. **Gate data quality mechanically, and catch failures by actually checking.** We built a validator that checks schema validity, evidence-span grounding, and execution correctness — and, after manually spot-checking output and finding 70% of an early synthetic batch used an unnatural, repetitive "products with X" template, added an explicit naturalness gate that blocks that failure mode going forward. This is disclosed, not hidden — data pipelines like this fail in ways that are easy to miss without directly reading the output.
4. **Track task performance *and* retention together, at every checkpoint.** Catastrophic forgetting — the model getting better at the new task while quietly getting worse at everything else — is a real risk in fine-tuning. We checkpoint and evaluate both signals every ~150 steps, and explicitly select the checkpoint with the best balance (not automatically the final one) once training completes.
5. **Merge into one portable artifact and prove it serves anywhere.** The trained LoRA adapter is merged into the base model weights, producing one self-contained model directory with no adapter-loading dependency — verified to load and run correctly on a RunPod GPU, on a local CPU-only Mac, and through a REST API, using the exact same code path.

## How it beat the benchmark (all numbers measured, not estimated)

### QueryNER — real, public, human-annotated benchmark (993 examples, full test set)

| System | F1 |
|---|---|
| GPT-5 (flagship) | 0.204 |
| Claude Haiku 4.5 (cheap tier) | 0.263 |
| GPT (cheap tier) | 0.294 |
| Claude Sonnet 5 (flagship) | 0.427 |
| **CommerceCore (this model)** | **0.498** — 95% CI [0.475, 0.523], 10,000-resample bootstrap |

CommerceCore beats every tested frontier tier, cheap and flagship, from both providers, on this exact real benchmark — verified on the complete 993-example test set, not a sample.

### Constraint extraction — held-out set, per-vertical (via the actual served/merged model, not just the raw checkpoint)

| Vertical | F1 |
|---|---|
| Fashion | 0.690 |
| Grocery | 0.667 |
| General merchandise | 0.694 |
| **Overall** | **0.683** |

Frontier baselines on the identical set, 3-repeat bootstrap CI: Claude Sonnet 5 0.592, GPT cheap tier 0.618, GPT-5 0.504, Claude Haiku 4.5 0.518 — all below CommerceCore.

### Every open-source model tested (unfine-tuned, few-shot, same or larger size class)

| Model | Params | License | F1 (held-out set) | F1 (fresh query set) |
|---|---|---|---|---|
| Qwen3.5-2B | 2B | Apache-2.0 | 0.05 | 0.00 |
| SmolLM3-3B | 3B | Apache-2.0 | 0.10 | 0.00 |

Both are larger than CommerceCore's 1.7B base and neither was fine-tuned for this task — both score far below CommerceCore on both eval sets. This confirms the gain comes from task-specific fine-tuning, not just model scale: a bigger unfine-tuned open model does not solve this task by default, and few-shot examples alone aren't enough to make it competitive here.

## Real cost (everything included)

| Item | Cost |
|---|---|
| Hardware profiling | ~$0.02 |
| Debug training run | $0.23 |
| Full multitask training run (2,130 examples, 1,200 steps, selected checkpoint at step 750) | $0.90 |
| Synthetic data generation (Claude API) | ~$1.00 |
| Frontier baseline comparisons (Claude + GPT API calls, multiple repeats + full QueryNER set) | ~$3–5 |
| RunPod setup/idle overhead across pod restarts | ~$1–2 |
| **Total** | **~$6–9** |

Hardware: single RunPod RTX PRO 4500 (32GB), $0.72/hr.

## Independent fresh-query test — important, honest result

To check the headline numbers weren't an artifact of the specific eval set, we ran a **second, independent comparison on 6 entirely new queries** (never used in training, in the held-out set above, or in any prior eval in this project) — deliberately including harder cases: negation ("not black"), unit conversion ("3 year" vs. months), and unusual comparison framing ("under X warranty").

| System | F1 on fresh queries |
|---|---|
| GPT-5 (flagship) | 0.167 |
| Qwen3.5-2B (unfine-tuned) | 0.000 |
| SmolLM3-3B (unfine-tuned) | 0.000 |
| Claude Haiku 4.5 | 0.298 |
| **CommerceCore (this model)** | **0.400** |
| GPT (cheap tier) | 0.428 |
| Claude Sonnet 5 (flagship) | 0.493 |

**On this harder, independent set, CommerceCore does NOT lead** — Claude Sonnet 5 and GPT's cheap tier both score higher. CommerceCore still clearly beats both unfine-tuned open models and GPT-5, and stays ahead of Claude Haiku, but the "beats every tested frontier model" claim from the held-out-set and QueryNER benchmarks above does not hold universally on more varied, harder phrasing.

**What this actually shows, concretely, from inspecting the failures directly**: the model reliably extracts 1-2 constraints per query but drops constraints when a query has 4+ of them ("green cotton hoodie size L under 40 pounds" → only price was extracted); it doesn't reliably normalize units/vocabulary it wasn't trained on ("wall powered" wasn't mapped to "mains", "3 year" wasn't converted to "36" months); and it mishandles negation ("not black" was kept as a literal value instead of being treated as an exclusion). These are genuine, disclosed limitations of a model trained on 2,130 examples with a narrow synthetic-scenario template set — not evidence the underlying approach is broken, but a clear, honest signal that **more diverse training data (more constraint combinations, explicit negation examples, unit-conversion examples) is the direct next step**, not a claim to paper over.

If you're evaluating this model yourself: expect strong, benchmark-beating performance on QueryNER-style entity segmentation and on straightforward single/double-constraint queries; expect degraded performance on complex multi-constraint queries, negation, and phrasing far from the training distribution.

## What this is NOT

- **Not a complete four-task system.** The original plan scoped four tasks (query-constraint extraction, catalog-attribute normalization, retrieval/matching, search-recovery). Only the first is fully trained and evaluated; catalog-attribute data was included in training but not separately benchmarked; retrieval and search-recovery are untouched.
- **Not "beats Claude/GPT" in general.** It beats the tested models on the tested tasks, in this evaluation. It has no general reasoning, conversation, or coding capability, and isn't meant to.
- **Not claiming to be first-of-its-kind.** A closely related approach (QLoRA + synthetic data + small model beating larger ones on ecommerce intent) is already published: [arXiv 2510.21970](https://arxiv.org/abs/2510.21970). We disclose this rather than overstate novelty — see the model card for the full comparison.
- **Real, disclosed failure modes**: occasional duplicate/conflicting constraints on complex multi-clause queries; imperfect recall on implicit (non-explicit) attribute values; retention (general-capability preservation) softened in later training steps, which is why an earlier checkpoint was deliberately selected over the final one.

## Repository structure

```
commercecore/
├── schemas/              # Pydantic contracts for each task type
├── data/                 # Synthetic generation, validation, partitioning
├── validators/           # Mechanical + naturalness quality gates
├── eval/                 # Scorers, baseline comparisons, benchmark scripts
├── train/                # Training scripts, run results, model-selection decision
├── serve/                # Inference class, REST API, Docker, merge-and-export
├── models/               # The merged, portable model (also on Hugging Face)
└── runpod_archive/       # Full logs, checkpoints, raw data pulled off the training pod
```

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("arghya2030/commercecore-qwen3-1.7b")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-qwen3-1.7b")

prompt = "Extract shopping constraints from this query: black waterproof trainers under 80 pounds\nConstraints:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
out = model.generate(input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True))
```

Or run the REST API (`serve/api.py`) — same code, runs on RunPod, AWS, Azure, GCP, or a local machine:

```bash
pip install -r serve/requirements.txt
COMMERCECORE_BASE_MODEL_PATH=arghya2030/commercecore-qwen3-1.7b uvicorn serve.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/parse-query -H "Content-Type: application/json" -d '{"query": "almonds with no added sugar"}'
```

**Note on speed**: the numbers above load and run fine on CPU, but generation is slow there (~130–145s/query, measured). For anything latency-sensitive, run on a GPU — 0.5–1s/query, measured on a RunPod RTX PRO 4500. CPU is fine for testing/verification, not for a live product.

## Reproducing the benchmarks yourself

Every benchmark number in this README/model card was produced by a script in `eval/`, using data in `data/queryner/` (the real QueryNER test set, included in this repo) — nothing needs the original training pod. Re-run any of them directly:

```bash
pip install -r serve/requirements.txt
python3 eval/full_queryner_eval.py       # full 993-example QueryNER benchmark + bootstrap CI
python3 eval/rigorous_baseline_comparison.py   # frontier model comparison (needs ANTHROPIC_API_KEY, OPENAI_API_KEY)
python3 eval/fresh_head_to_head_comparison.py  # the harder, independent fresh-query test
```

`eval/full_queryner_eval.py` loads the published model directly from Hugging Face — no local checkpoint needed. It will be slow on CPU; a GPU is recommended for the full 993-example run.

## License

Apache-2.0, matching the base model (Qwen3-1.7B).
