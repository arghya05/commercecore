# CommerceCore

A small (1.7B parameter), self-hostable model that parses natural-language shopping queries into structured constraints — trained for under $10 in total compute + API cost. Beats every tested frontier model on the primary held-out benchmarks; an honest, independent fresh-query test below shows this advantage does not hold universally — read that section before relying on any headline number.

**Model weights + model card:** https://huggingface.co/arghya2030/commercecore-qwen3-1.7b
**Source code + docs:** https://github.com/arghya05/commercecore
**Paper (PDF):** [`paper/CommerceCore_Paper.pdf`](paper/CommerceCore_Paper.pdf)

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

## Training logs — the real, complete checkpoint-by-checkpoint record

The final training run: Qwen3-1.7B, QLoRA (NF4 double-quant, BF16 compute, r=16/alpha=32/dropout=0.05), 2,130 examples, 1,200 steps, checkpointed and evaluated every 150 steps. Full raw logs are in `train/results_multitask_clean/` (loss every step, eval + retention every checkpoint) — this table is the checkpoint-level summary:

| Step | Constraint F1 | QueryNER F1 | Retention vs. frozen base | Loss (at that step) |
|---|---|---|---|---|
| 150 | 0.708 | 0.625 | 0.597 | 0.31 (at step 99) |
| 300 | 0.698 | 0.604 | 0.611 | 0.14 (at step 299) |
| 450 | 0.675 | 0.601 | 0.597 | 0.04 (at step 399) |
| 600 | 0.690 | 0.630 (peak) | 0.592 | 0.06 (at step 599) |
| **750 (selected)** | **0.725** | 0.556 | **0.614** (peak) | 0.05 (at step 799) |
| 900 | 0.725 | 0.614 | 0.490 (low point) | 0.03 (at step 899) |
| 1050 | 0.750 (peak) | 0.573 | 0.518 | 0.03 (at step 1099) |
| 1200 (final) | 0.740 | 0.530 | 0.489 (worst) | 0.02 (at step 1199) |

**Why step 750, not the final checkpoint 1200**: retention genuinely declines in the last ~450 steps (0.614 → 0.489) and never recovers, while task performance only marginally improves. That's real, measured forgetting — checked at every checkpoint, not assumed away. Full reasoning in `train/results_multitask_clean/MODEL_SELECTION_DECISION.md`.

Training cost: **$0.90**, 75.2 minutes, on a single RunPod RTX PRO 4500 (32GB). Peak VRAM: 2.13GB (the loaded 4-bit quantized model + LoRA adapter — well under the 32GB available, meaning a much smaller/cheaper GPU could run this same training).

### Loss by epoch (not just by step)

At grad-accumulation batch size 8 over 2,130 examples, one epoch ≈ 266 optimizer steps. 1,200 steps ≈ 4.5 epochs. Loss at approximately each epoch boundary (real values, nearest logged step):

| Epoch | Step | Loss |
|---|---|---|
| 1 | 266 | 0.083 |
| 2 | 532 | 0.073 |
| 3 | 798 | 0.047 |
| 4 | 1065 | 0.036 |

Loss drops fast and stays low from epoch 2 onward — consistent with a small, fairly repetitive dataset (2,130 examples over ~4.5 passes). This is also part of why retention degraded late in training (step 900+, epoch ~3.4 onward): the model had already converged on the task loss and additional steps mostly reinforced narrow patterns rather than learning anything new, which is exactly when general-capability drift shows up. Full per-step loss (1,200 rows) is in `train/results_multitask_clean/multitask_training_loss_log.csv`.

### Challenges faced in training — real, specific, not generic

1. **Data repetition vs. dataset size.** With only 2,130 examples and 1,200 steps (grad-accum 8), the model saw each example ~4.5 times. Loss converging to near-zero by epoch 2 is a symptom of a dataset too small/narrow for the step count used — the late-training retention drop (§ above) is a direct, measured consequence. The concrete fix, not yet done: either fewer steps (stop around epoch 2-3) or substantially more, more diverse data.
2. **Synthetic data quality wasn't caught by automated checks alone.** An early synthetic batch (see "Strategy") passed every mechanical validation check (schema, evidence-span binding, execution correctness) while 70% of it used a generic, repetitive "products with X" phrasing template — found only by manually reading the generated text, not by any automated gate that existed at the time. Fixed by adding an explicit naturalness check afterward; the underlying lesson is that mechanical checks prove internal consistency, not real-world quality, and spot-checking output by hand is not optional.
3. **A generation-time crash from a genuinely unrenderable scenario.** The synthetic-data generator's "messy/conflicting constraint" mode could produce a scenario like `contains_nuts=true AND contains_nuts=false` simultaneously — Claude correctly refused to paraphrase this into clean JSON and returned prose instead, which crashed the parser with an unhandled exception mid-run. Fixed by catching this as an expected reject-and-retry case, not a fatal error.
4. **API version mismatches deflated a real baseline score.** `claude-sonnet-5` rejects the `temperature` parameter outright (returns an HTTP 400), which silently zeroed out 2 of 3 repeat measurements in an early baseline run, deflating its measured score from a real 0.59 to a misleading 0.05. Caught by noticing an implausibly low number for a strong model and checking the raw API errors directly, not by trusting the aggregate score.

### Why these specific datasets

- **QueryNER** (1,500 of 2,130 training examples, real/human-annotated): chosen because it's a real, public, CC BY 4.0 licensed benchmark with an existing test split — this is what makes the "beats frontier models on QueryNER" claim independently verifiable by anyone, not just self-reported.
- **Synthetic query-constraint data** (450 examples): built rather than sourced, because no existing public dataset pairs natural-language shopping queries with structured field/operator/value constraint labels across fashion/grocery/general verticals at the granularity this task needs. Generated with labels fixed in code *before* any LLM call specifically to avoid an LLM silently inventing or dropping a constraint while writing the query.
- **Synthetic catalog-attribute data** (180 examples): included to test whether a second, related task (raw listing text → structured attributes) could be learned jointly without hurting the primary task — included in training, though not separately benchmarked in this release (see "What this is NOT").
- **Not used**: no proprietary/scraped ecommerce data. Every real-data source is publicly licensed and re-distributable; every synthetic source is disclosed as such, not blended in silently.

An earlier training run on lower-quality synthetic data (before a naturalness-quality gate was added — see "Strategy" above) is also fully logged in `train/results_scaled/`, kept for transparency rather than deleted once superseded.

## Serving logs — real, measured request/response pairs

Actual logged output from the deployed REST API (`serve/api.py`), running the merged model on a RunPod GPU:

```
POST /parse-query {"query": "black waterproof trainers under 80 pounds"}
→ {"query": "black waterproof trainers under 80 pounds",
   "constraints": [{"field": "color", "op": "eq", "value": "black"}]}

POST /parse-query {"query": "almonds with no added sugar"}
→ {"query": "almonds with no added sugar",
   "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"}]}

POST /parse-query {"query": "cordless drill, 2 year warranty minimum"}
→ {"query": "cordless drill, 2 year warranty minimum",
   "constraints": [{"field": "warranty_months", "op": "gte", "value": "24"},
                    {"field": "warranty_months", "op": "gte", "value": "12"}]}
   (note: real, disclosed imperfection — a duplicate/conflicting constraint,
   not hidden from this log)

GET /health → {"status": "ok", "model_loaded": true}
```

Measured latency, real requests, both environments:

| Environment | Load time | Latency/query |
|---|---|---|
| RunPod GPU (RTX PRO 4500) | 1.4s | 0.58–1.05s |
| Local CPU (Apple Silicon, no quantization) | 0.6s | 130–145s |

### Serving cost vs. frontier API token cost — an honest comparison, corrected once already

At GPU cost of $0.72/hr and our measured single-request throughput (3,429–6,207 queries/hour, no batching), self-hosted CommerceCore costs **$0.000116–$0.000210 per query**. For a typical request (~100 input tokens, ~30 output tokens), frontier API token costs are:

| System | Cost/query (approx., 100 in / 30 out tokens) |
|---|---|
| GPT cheap tier | **$0.000033** |
| CommerceCore (self-hosted, unbatched) | $0.000116–$0.000210 |
| Claude Haiku 4.5 | $0.000250 |
| Claude Sonnet 5 | $0.000750 |
| GPT-5 flagship | $0.000950 |

**Important, honest correction**: an earlier version of this comparison stated self-hosting was cheaper — that was wrong, caught and corrected here rather than left standing. At single-request, unbatched serving, GPT's cheap tier is actually the cheapest option on raw token cost. Self-hosting only beats it on cost past a breakeven of roughly 21,800 queries/hour on one GPU (achievable via request batching, which we did not measure) — below that, the honest cost argument for self-hosting is not "cheaper per token," it's **data locality** (queries never leave your infrastructure) and **beating GPT cheap tier's accuracy** on the tasks we tested (Section "How it beat the benchmark" above), not raw serving cost at low volume.

### Challenges faced in serving — real, specific, not generic

1. **`peft` was imported unconditionally, breaking the common case.** `serve/inference.py` originally did `from peft import PeftModel` at module level, even though most deployments should use the pre-merged model (no adapter, no PEFT needed at all). This meant a fresh install without `peft` — the exact "quick pip install and go" case the serving package is supposed to support — crashed on import. Fixed by moving the import inside the one code path that actually needs it.
2. **Local environment version mismatch, found only by actually testing on a second machine.** The Mac's default Python (3.9) could only install `transformers` 4.57, a version incompatible with how Qwen3's tokenizer config is structured — it crashed with an obscure `AttributeError` deep in `transformers` internals, not an obvious "wrong version" message. Fixed by using a newer Python (3.13) in a dedicated virtual environment. This would have looked like "it works on my machine" if the local CPU test had been skipped in favor of only testing on the pod.
3. **The REST API wrapper had no way to point at the merged model at all.** `serve/api.py` only supported the base HF model ID or an adapter path, with no way to load the actual portable, merged artifact this project produces — meaning the documented "one image, deploy anywhere" claim would have failed the moment someone tried it, because the config option to do so didn't exist. Fixed by adding a `COMMERCECORE_BASE_MODEL_PATH` environment variable.
4. **Eval scripts assumed the original training pod's file layout.** The full-benchmark reproduction script hardcoded paths like `/workspace/models/Qwen3-1.7B` and `/workspace/checkpoints_multitask/step_750` — paths that only existed on the specific RunPod instance used for training. Anyone cloning this repo would have hit an immediate `FileNotFoundError`. Fixed by loading the model directly from its published Hugging Face repo ID and reading eval data from a path relative to the repo itself.

Each of these was found by actually running the thing in a new context (a second machine, a fresh environment, someone else's hypothetical clone) rather than assuming the code that worked once would work everywhere — which is the entire point of testing "runs anywhere" as a claim, not just asserting it.

Full serving test logs (`serve/complete_inference_test.py`, `serve/full_inference_test.py`) run 8–20 queries end-to-end through the actual served model and are included in this repo, along with the exact scripts used to produce them — see "Reproducing the benchmarks yourself" below.

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
- **Not claiming to be first-of-its-kind** — see "Related work" below for the specific papers this overlaps with and how.
- **Real, disclosed failure modes**: occasional duplicate/conflicting constraints on complex multi-clause queries; imperfect recall on implicit (non-explicit) attribute values; retention (general-capability preservation) softened in later training steps, which is why an earlier checkpoint was deliberately selected over the final one.

## Related work — honest comparison, not a novelty claim

These are papers found and read during this project that overlap with parts of the approach here. Listed so a reader can judge the actual increment themselves, not take our word for it.

- **[Performance Trade-offs of Optimizing Small Language Models for E-Commerce](https://arxiv.org/abs/2510.21970)** (arXiv 2510.21970) — the closest match. Fine-tunes a 1B Llama 3.2 via QLoRA on synthetic ecommerce data, reports ~99% accuracy on intent classification, close to GPT-4.1. **How this project differs**: different task (multi-field constraint extraction + entity segmentation, not single-label intent classification), different base model, and this project additionally mixes in real human-annotated data (QueryNER) rather than using fully synthetic data. **What we do NOT claim**: that "small model + QLoRA + synthetic data beats bigger models on ecommerce" is a new idea — this paper already established it for a related task.
- **[EcomGPT: Instruction-tuning Large Language Models with Chain-of-Task Tasks for E-commerce](https://arxiv.org/abs/2308.06966)** (arXiv 2308.06966) — a genuinely multitask ecommerce LLM instruction-tuning approach, which overlaps with this project's decision to train query-constraint extraction and entity segmentation jointly rather than as separate models. Not independently re-benchmarked against this project — flagged here for a reader to compare directly, not ruled out as prior art.
- **[CuratorKIT: Data Curation and Synthetic Data Generation for LLM Post-Training](https://arxiv.org/abs/2606.21631)** (arXiv 2606.21631) — a published, general-purpose framework for exactly the kind of problem this project's naturalness-quality gate was built to solve (catching synthetic data that passes mechanical checks but fails a real quality bar). This project's gate is narrow and built for one specific failure mode found by manual inspection, not a general framework — CuratorKIT's approach is more mature and general.

If you're aware of closer prior art than what's listed here, it's a real gap in this review, not something to hide — this list reflects one focused research session, not an exhaustive literature survey.

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

## How to run inference (quick start)

**Option A — plain Python, 5 lines, downloads the model automatically from Hugging Face:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("arghya2030/commercecore-qwen3-1.7b")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-qwen3-1.7b")

prompt = "Extract shopping constraints from this query: black waterproof trainers under 80 pounds\nConstraints:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
out = model.generate(input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True))
# -> [{"field": "color", "op": "eq", "value": "black"}]
```

**Option B — the project's own inference class** (`serve/inference.py`, same repo), handles the prompt formatting and JSON parsing for you:

```python
from serve.inference import CommerceCoreQueryParser

parser = CommerceCoreQueryParser(base_model_path="arghya2030/commercecore-qwen3-1.7b")
result = parser.parse("black waterproof trainers under 80 pounds")
print(result)  # -> [{"field": "color", "op": "eq", "value": "black"}]
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
