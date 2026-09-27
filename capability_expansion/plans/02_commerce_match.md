# Capability 2 — Commerce-Match

Implement after `00_expansion_core.md` and `01_commerce_understand.md`. Priority 2 (`CAPABILITY_EXPANSION_PLAN.md` §3). Fully separate model — not a sibling adapter, not related to the shipped Commerce-Query checkpoint.

## Outcome

Classify a query–product or product–product relation into five classes: same product, variant, substitute, complement, unrelated. An accessory is not the product; a related item is not automatically a valid substitute; a complement is not interchangeable with the item it complements. This is the direct prerequisite for Commerce-Rank (needs relevance signal) and Commerce-Bundle (needs the complement/compatible split as its own starting point).

## Schema work required first — this is not purely a training-data gap

`schemas/retrieval_match.py`'s current `RelationClass` enum has only **4 values**: `EXACT_MATCH, SUBSTITUTE, COMPLEMENT, IRRELEVANT`. The user's target has **5**: same product / variant / substitute / complement / unrelated. Add a distinct `VARIANT` class before any training data work — "same product" (`EXACT_MATCH`, e.g. identical SKU) must be separated from "same product family, different variant" (new `VARIANT`, e.g. same shirt, different size/color, sharing a parent SKU or GTIN prefix). Write and run `schemas/migrate_relation_class.py` to confirm no existing serialized data breaks under the new enum before proceeding.

## Model and data

Base: `Qwen/Qwen3-0.6B` (target band 150–500M sits below the shared 1.7B base) or `answerdotai/ModernBERT-base` as a required baseline — train both and compare before committing to the generative approach; a non-generative classifier may win on cost/latency for a 5-class label task. `fastino/gliner2.5-base-v1` is a mandatory baseline per `00_expansion_core.md`.

Primary corpus: Amazon ESCI, https://github.com/amazon-science/esci-data, Apache-2.0, ~130k queries / 2.62M relevance judgments, E/S/C/I labels. **Not yet pulled onto disk anywhere in this repo** — pull into `data/esci/` with the same audit rigor already applied to QueryNER (split integrity, cross-split duplicate check, label distribution report, SHA-256 manifest).

**ESCI label mapping is an assumption to validate, not a given.** E→same-or-variant, S→substitute, C→complement, I→unrelated is plausible but must be hand-audited on a sample before trusting it. Critically, **ESCI does not distinguish "same product" from "variant" at all** — recovering that split needs either new annotation or a heuristic (shared parent SKU/GTIN prefix, matching title minus size/color tokens). Do not assume ESCI covers the 5-class target out of the box; the E-class rows need a second labeling pass.

Target: 12,000 relation/match records in the training mixture (matching the original plan's allocation for this task family), drawn from ESCI plus new same-vs-variant annotation on a sample of E-class rows.

## API and schema

Implement `POST /v1/match/classify`. Request: tenant context, query or product A, product B (or candidate product), locale. Response: `relation_class` (one of the 5), confidence (calibrated, not raw softmax presented as confidence), per-requirement pass/fail/unknown/not-applicable states for any explicit constraints carried from Commerce-Query's output.

Match states follow the existing common representation: `pass`, `fail`, `unknown`, `not_applicable` per requirement — a missing field is never a negative factual label. A hard contradiction always blocks a "satisfies requirement" claim; absence of evidence does not.

## Training

QLoRA settings as a starting point if using the Qwen3-0.6B route (r=16/alpha=32/dropout=0.05, NF4, paged AdamW 8-bit, LR 1e-4) — adjust for the smaller model's different optimal batch/LR if a smoke run indicates it; do not blindly copy the 1.7B settings without checking. If the ModernBERT-base baseline wins on the accuracy/latency/cost frontier, ship that instead — this is exactly the ablation `CAPABILITY_EXPANSION_PLAN.md` §2 rule 1 requires before committing to a decoder.

Checkpoint and evaluate on a real cadence, track per-class F1 (not just macro-F1 — a 5-class problem can hide a collapsed minority class, especially the new `VARIANT` class with the least labeled data) across checkpoints.

## Evaluation and acceptance

Two eval sets, ≥300 cases each: (i) held-out ESCI split, (ii) accessory-vs-product confusion, compatible-vs-similar, partial-evidence, and specifically same-vs-variant confusion slices (per `CAPABILITY_EXPANSION_PLAN.md` §5 item 5) — this last slice did not exist before the schema fix above and is the newest, least-tested part of this capability.

Public benchmark gate: **Amazon ESCI, official scoring recipe** (E=1/S=0.1/C=0.01/I=0 for the documented ranking track, per `Commerce_SLM_Research_Plan.md` §12) — use the pinned evaluator, do not silently substitute linear gains. This is a real public benchmark with an existing scorer, stronger evidence than a repurposed dataset.

Frontier LLM gate: same fixed matrix (GPT cheap/flagship, Claude Haiku/Sonnet), zero/few-shot 5-class classification, same eval sets, non-overlapping confidence intervals required to claim a win (`CAPABILITY_EXPANSION_PLAN.md` §2d).

## Deliverables

An ESCI-to-Match label-mapping document with a manually-audited sample (same rigor as the original project's "read 1,420 WDC rows by hand" audit) — produced *before* any training run starts, per `CAPABILITY_EXPANSION_PLAN.md` §5's first milestone. `train/results_match/` with loss/eval/retention-N/A logs (retention doesn't apply — separate model, no shared weights with Commerce-Query, record as `NOT_APPLICABLE` per `00_expansion_core.md`). Serving smoke test on GPU and local CPU.

Sources: `CAPABILITY_EXPANSION_PLAN.md` §5, §1b, §2e; `ecommerce/plans/03_retrieval_and_matching.md` (original relation-classification spec — note it uses the shipped model's 4-class scheme, this plan's 5-class target is new scope); `ecommerce/Commerce_SLM_Research_Plan.md` §12 (ESCI official scoring).
