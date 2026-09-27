# Capability 1 — Commerce-Understand

Implement after `00_expansion_core.md`. Priority 1 (`CAPABILITY_EXPANSION_PLAN.md` §3). Sibling LoRA adapter on the same frozen Qwen3-1.7B base as the shipped Commerce-Query model — never a continuation of its training run.

## Outcome

Convert messy supplier/catalog text into canonical structured product JSON: normalized attributes, explicit unit conversions, evidence spans, and `unknown`/`not_applicable` states for anything absent from the input. Target size band 0.5–1.5B (user's table) — the existing Qwen3-1.7B base already sits in this band, so no separate model is needed on size grounds alone; a sibling adapter is the correct choice per `CAPABILITY_EXPANSION_PLAN.md` §2b decision order.

Fashion needs composition, color, product type, size-system references. Grocery needs pack count, per-unit quantity, ingredient/declaration text. General merchandise needs model identifiers, dimensions, compatibility evidence. Do not infer material from an image, allergy safety from missing text, or compatibility from visual similarity — same rule as the original catalog use case (`ecommerce/plans/02_catalog_and_taxonomy.md`).

## Model and data

Base: `Qwen/Qwen3-1.7B`, same pinned revision as the shipped model (`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`). New sibling LoRA adapter `commercecore-understand-adapter`, loaded independently at serve time — never merged into or overwriting `models/commercecore_merged_step750/`.

**Mandatory baseline before training this adapter:** `fastino/gliner2.5-base-v1` (GLiNER2-base, 205M) — per the broader dossier, this must be beaten before a decoder-based approach is justified for an extraction-shaped task. Also compare `numind/NuExtract3` and an unfine-tuned Qwen3.5-2B with matched prompts.

Primary corpus: Amazon Berkeley Objects, https://registry.opendata.aws/amazon-berkeley-objects/, CC BY 4.0, 147,702 listings, ~83MB listings archive. Pull only the listings archive first. Verify field coverage locally before assuming completeness — do not claim all rows have all attributes. `item_id` and family/variant groups drive train/eval splitting, not random row sampling.

Reuse the existing 180 synthetic catalog-attribute rows already in this project (used as auxiliary signal in the original multitask run) — but check `train/build_multitask_mixture.py` for exactly which 180 rows were used, and exclude them from any new eval set. They may re-enter training, never eval.

Target record counts: 20,000 extraction/normalization records in the training mixture (matching the original `Commerce_SLM_Research_Plan.md` §9 allocation for this task). **ABO is the mandatory backbone — the majority of these 20,000 rows must come from real ABO listings, not synthetic generation.** Synthetic rows (the existing 180 plus any new ones) fill measured gaps only: rare attribute combinations, adversarial negation/omission cases, and verticals ABO underrepresents. Do not duplicate rows to hit a round number, and do not let synthetic volume exceed real volume without an explicit, disclosed reason.

**Any new synthetic row must pass the four-property gate before entering training** (`CAPABILITY_EXPANSION_PLAN.md` §2f): schema validity, source-span binding, execution correctness (does the extracted+normalized value round-trip against a synthetic catalog fixture consistently), and naturalness (mechanical blocklist plus a mandatory manual read of ≥10%/50 rows, whichever is larger, of any new batch). Required adversarial fixture category specific to this capability: **plausible-but-uncited-attribute** — a value true of the product category in general (e.g., "cotton items are usually machine-washable") but not stated for this specific listing; the correct output is `unknown`, and a naive model will confidently fill it in from category priors. Also required: negation, omitted-attribute, units-mismatch, conflicting-facts, missing-field (the five categories already established in `validators/adversarial_fixtures.py`), extended with catalog text specifically (not query text).

## API and schema

Implement `POST /v1/catalog/normalize`. Request: tenant context, source text (title + bullet points + any supplied structured evidence), requested schema fields, locale. Response: typed field facts (raw value, canonical value, unit, applicability state, source field, source span), taxonomy-adjacent metadata left to Commerce-Match/Commerce-Retrieve, not duplicated here.

Extend `schemas/catalog_taxonomy.py`'s existing contract rather than replacing it — the schema already exists from the original build; this capability is finishing the eval/training gap, not redesigning the contract.

Field states: `supported` / `contradicted` / `unknown` / `not_applicable`, matching the common representation already used project-wide. A value absent from title/bullets is `unknown`, never inferred as false. `2 x 500ml` produces `count:2, per_unit:500ml, total:1000ml` — distinct from a single 1-litre container; deterministic conversion code, not model arithmetic, produces the normalized value.

## Training

Reuse the exact settings that already worked for the shipped model, as the starting point, not a re-derivation: QLoRA NF4 double-quant, BF16 compute, LoRA r=16/alpha=32/dropout=0.05 on all supported linear layers, paged AdamW 8-bit, LR 1e-4, 3% warmup, cosine decay, microbatch 1 / accumulate 32, length 2048, gradient checkpointing, assistant-only loss, packing off for the first validated run. Deviate only if a smoke run shows a specific reason to.

Checkpoint and evaluate every 150 steps (same cadence as the shipped model), not only at the end. Track constraint/field F1 and retention against the current shipped Commerce-Query numbers (constraint F1 0.725, QueryNER F1 0.556) at every checkpoint — since this adapter shares the frozen base, per `CAPABILITY_EXPANSION_PLAN.md` §2b the regression gate applies at post-SFT, post-merge (if ever merged), and post-quantization stages, not just once.

## Evaluation and acceptance

Two eval sets minimum, ≥300 cases each (`CAPABILITY_EXPANSION_PLAN.md` §2c): (i) held-out ABO split, (ii) an out-of-distribution set stressing unseen product families/suppliers and short/inconsistent descriptions. Metrics: field-level micro/macro F1, supported-value precision, **missingness accuracy** (does the model correctly emit `unknown` rather than hallucinate — the single most important metric here, per the evidence-state rule), normalization exact-match on unit conversions.

**Breadth check before calling either eval set sufficient** (`CAPABILITY_EXPANSION_PLAN.md` §2g): confirm coverage across fashion, grocery, general merchandise, electronics/tech accessories and home/furniture — not just the three verticals already illustrated elsewhere in this plan. Confirm both fully-specified and partially-specified source text, and at least one adversarially-padded (irrelevant marketing copy) case per vertical.

Public benchmark gate (`CAPABILITY_EXPANSION_PLAN.md` §2e): no dedicated public catalog-extraction benchmark exists for this exact task — use the ABO held-out split plus GLiNER2-base's reported extraction numbers as the public reference point until a better one is identified; disclose this as a proxy, not a perfect substitute for a named benchmark like QueryNER.

Frontier LLM gate: GPT cheap tier + GPT-5 flagship, Claude Haiku 4.5 + Claude Sonnet 5, zero/few-shot, matched prompts, same eval sets. State which of the three claim types (model / system / economic, per §2d) is being made in the writeup.

Read a sample of actual failures by hand before declaring done — the original build's most useful discovery (the "products with X" synthetic-data template problem) was only caught this way, not by any automated gate.

## Deliverables

`train/results_understand/` with the same file shape as `results_multitask_clean/` (loss log, checkpoint eval log, retention log, `MODEL_SELECTION_DECISION.md`); `data/abo/` with manifest (source revision, SHA-256, license evidence, row count) matching `data/queryner/README.md`'s pattern; a serving smoke-test on both GPU and local CPU before calling this shipped (`CAPABILITY_EXPANSION_PLAN.md` §2c item 8).

Sources: `CAPABILITY_EXPANSION_PLAN.md` §4, §1b; `ecommerce/plans/02_catalog_and_taxonomy.md` (original catalog use-case spec, same task family); `ecommerce/Commerce_SLM_Research_Plan.md` §8–9; `additonal/Retail_Ecommerce_SLM_Research_Source.md` ch. 10 (GLiNER2 baseline requirement), ch. 11 (dataset licensing).
