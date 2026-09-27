# Implementation contract: CommerceCore capability expansion

This is a specification for a coding/training agent, not an already-trained set of models. Read this before plans 01–08. Use the same schema library, evaluation harness and release-gate discipline as the shipped Commerce-Query model (`00_shared_core.md` in `ecommerce/plans/`), extended to eight more capabilities. Where this file and `CAPABILITY_EXPANSION_PLAN.md` (one level up) disagree, `CAPABILITY_EXPANSION_PLAN.md` is the source of truth for policy (what must and must not happen); this file is the source of truth for exact implementation shape (repo layout, API surface, CLI, record counts).

**Non-negotiable, confirmed by the user, repeated here because it governs every file below:** the shipped model (`arghya2030/commercecore-qwen3-1.7b`, step-750 checkpoint) is never retrained, resumed, or overwritten. Every new capability is either (a) a new sibling LoRA adapter sharing only the frozen Qwen3-1.7B base, or (c) a fully separate model. A separate model is a normal, expected outcome, not a compromise. See `CAPABILITY_EXPANSION_PLAN.md` §2b.

## Product and experiment scope

Extend the shipped single-task query-constraint model into the full nine-capability portfolio: Commerce-Query (shipped), Commerce-Understand, Commerce-Match, Commerce-Retrieve, Commerce-Rank, Commerce-Bundle, Commerce-Agent, Commerce-Taste, Commerce-QA. Implementation order matches `CAPABILITY_EXPANSION_PLAN.md` §3: Understand → Match → Retrieve → Rank → Bundle → Agent → Taste → QA. Each capability's own plan file (01–08 below) is self-contained enough to hand to an agent without re-reading the others, but assumes the schemas and conventions fixed here.

Every capability must clear the joint gate in `CAPABILITY_EXPANSION_PLAN.md` §2e: beat a named public benchmark AND every frontier LLM in the fixed baseline matrix (GPT cheap tier, GPT-5 flagship, Claude Haiku 4.5, Claude Sonnet 5), both required, on its own eval sets, with confidence intervals, not point estimates.

## Model selection (per capability, restated from CAPABILITY_EXPANSION_PLAN.md §1b)

| Capability | Base model | Relationship to shipped checkpoint |
|---|---|---|
| Commerce-Understand | `Qwen/Qwen3-1.7B` (same frozen base as shipped) | New sibling LoRA adapter, `commercecore-understand-adapter` |
| Commerce-Match | `Qwen/Qwen3-0.6B` or `answerdotai/ModernBERT-base` | Fully separate model, `commercecore-match` |
| Commerce-Retrieve | `Qwen/Qwen3-Embedding-0.6B` (frozen initially) | Fully separate model — embedding architecture |
| Commerce-Rank | `Qwen/Qwen3-Reranker-0.6B` | Fully separate model — cross-encoder architecture |
| Commerce-Bundle | Same family as Match, warm-started from Match's checkpoint | Fully separate trained artifact, `commercecore-bundle` |
| Commerce-Agent (v0) | none — orchestration only | No model; v1 (later) is a sibling adapter, gated |
| Commerce-Taste | TBD pending Amazon-M2 license check | Fully separate model once unblocked |
| Commerce-QA | Deferred | Deferred |

Required baselines for every capability, no exceptions: `fastino/gliner2.5-base-v1` (GLiNER2-base) as the mandatory first non-generative bar for any extraction/classification-shaped task; `answerdotai/ModernBERT-base`; an unfine-tuned same/larger-size open model with matched prompts; GPT cheap tier + GPT-5 flagship; Claude Haiku 4.5 + Claude Sonnet 5. Record resolved model IDs, revisions and call dates, not display names.

## Repository layout to implement

```text
commercecore/
  data/{abo,esci,wands,amazon_m2}/         # new pulls, one dir per new source
  schemas/
    retrieval_match.py                      # MODIFY: add VARIANT to RelationClass (see plan 02)
    catalog_taxonomy.py                      # already exists — extend for Understand eval
  train/
    results_understand/                     # new, mirrors results_multitask_clean/ shape
    results_match/
    results_bundle/
  models/
    commercecore_understand_adapter/        # sibling adapter, separate from merged_step750
    commercecore_match/                      # standalone model directory
    commercecore_retrieve/                   # embedding model + index artifacts
    commercecore_rank/                       # reranker model directory
    commercecore_bundle/
  retrieve/
    index.py                                 # BM25 + dense + RRF fusion
    embed.py                                 # Qwen3-Embedding-0.6B wrapper
  serve/
    api.py                                   # EXTEND: new routes, existing /parse-query untouched
  eval/
    understand_eval.py
    match_eval.py                            # ESCI official E/S/C/I scoring recipe
    retrieve_eval.py                         # WANDS full-corpus + candidate reranking tracks
    gate_check_expansion.py                  # extends eval/gate_check.py for new capabilities
```

CLI to implement (mirrors the existing repo's script pattern, not a new framework):

```text
python3 train/build_understand_mixture.py --sources abo,synthetic_catalog
python3 train/train_understand_adapter.py --config configs/understand_sft.yaml
python3 eval/understand_eval.py --checkpoint <path> --report train/results_understand/

python3 data/pull_esci.py --audit                 # same audit rigor as data/queryner ingestion
python3 schemas/migrate_relation_class.py         # adds VARIANT, checks no serialized data breaks
python3 train/train_match.py --base Qwen3-0.6B --config configs/match_sft.yaml
python3 eval/match_eval.py --scoring esci_official --report train/results_match/

python3 retrieve/build_index.py --catalog data/abo/normalized.jsonl
python3 eval/retrieve_eval.py --track full_corpus --held_out data/wands/
```

## Hyperparameter starting points (confirmed against the dossier's own exact build plan, ch. 33/14/17 — not re-derived per capability)

These are starting points to smoke-test against, not settings to accept blindly — but they replace guessing from scratch for every new capability:

| Setting | Value | Source |
|---|---|---|
| LoRA rank screening range | 16, 32, 64 (alpha = 2× rank) | Dossier ch. 14 |
| Adapter LR (sibling adapters on Qwen3-1.7B) | 5e-5 and 1e-4 | Dossier ch. 33, narrowed specifically for this base model — matches the shipped model's own actual 1e-4 |
| Full-tuning LR (if ever used, not the default) | 5e-6, 1e-5, 3e-5 | Dossier ch. 14 |
| DAPT LR (continued pretraining, off by default per existing project policy) | 5e-6 to 3e-5 | Dossier ch. 13 |
| Dropout | ≈0.05 | Matches shipped model exactly |
| Sequence length | 2K–4K first | Dossier ch. 14, matches shipped model's 2048 |
| Epochs | Start at 1, max 2 | Dossier ch. 14, matches shipped model's "2 epochs maximum" |
| Warmup | 3% of updates | Matches shipped model exactly |
| Loss | Completion/assistant-only | Matches shipped model exactly |

**Memory/compute sizing formulas** (dossier ch. 17/13, useful for planning a training run before committing GPU spend, not just after): full-tuning Adam needs **~16 bytes/trainable parameter** (2 weight + 2 grad + 4 FP32 master copy + 8 for two FP32 Adam moments) — a 4B full fine-tune starts around 64GB before activations, which is why LoRA/QLoRA remains the default for every capability here, not a full-tune. Rough compute estimate: **FLOPs ≈ 6 × parameters × training tokens** — use this to sanity-check a training-time estimate before starting a paid run, not as a substitute for actually profiling on the target GPU (per the existing project's own "profile the real pod, don't guess" rule).

**KV-cache sizing for serving** (dossier ch. 18, relevant once Understand's sibling adapter or any generative capability is served at scale): `2 × attention_layers × KV_heads × head_dim × cached_tokens × concurrent_sequences × bytes_per_element`. Useful for capacity planning multiple concurrent requests against a served adapter — not needed for the embedding/reranker-based capabilities (Retrieve/Rank), which don't autoregressively generate.

## Contracts (shared conventions, all new capabilities must follow)

All new endpoints authenticate tenant context and record request ID, model SHA, adapter SHA (if applicable), schema version and evidence spans in the trace — identical to the existing `/parse-query` route's pattern in `serve/api.py`. No new endpoint may bypass tenant scoping to reuse code faster.

All new schemas add explicit `unknown` / `not_applicable` states; a missing field is never inferred as false or absent-therefore-safe. Half-open character spans `[start, end)` against immutable source text, matching the convention already used in `data/queryner/`.

## Data source gates (new sources only — QueryNER's gate is already satisfied)

| Source | License status | Gate before use |
|---|---|---|
| Amazon Berkeley Objects | CC BY 4.0 | Pull listings archive (~83MB), verify SHA-256, confirm field coverage locally — do not assume all 147,702 rows have all attributes |
| Amazon ESCI | Apache-2.0 | Pull, audit split integrity, check ABO/QueryNER product-ID overlap before mixing |
| WANDS | MIT | Pull, hold out entirely from tuning — locked-final partition, never touched mid-development |
| Amazon-M2 | **Unverified — check before any use** | Do not train on this until commercial license terms are confirmed in writing; this is a hard stop, not a formality |
| Manufacturer compatibility tables (Bundle) | No public dataset identified | Source per-merchant; scope cost/time before treating Bundle as "ready" |

Excluded from commercial default regardless of task: MAVE (CC BY-NC 4.0), Amazon Reviews 2023 (unresolved source rights), Dress Code (explicitly barred for private companies), Hopit MODA noncommercial-weight checkpoints.

## Release gates

Extend `ecommerce/configs/release_gates.yaml`'s pattern: PASS / FAIL / NOT_MEASURED states, no gate marked green without a measured value and confidence interval. Every new capability adds its own gate block; it does not relax the existing Commerce-Query gates. The retention gate from `CAPABILITY_EXPANSION_PLAN.md` §2b runs at every stage (post-SFT, post-merge, post-quantization) for any artifact that shares the frozen 1.7B base with Commerce-Query — for fully separate models (Match, Retrieve, Rank, Bundle), this gate is not applicable by construction, since no shared weights exist to regress; record that as `NOT_APPLICABLE`, not `NOT_MEASURED`, since it's a structural guarantee, not an unmeasured risk.

## Sources

`CAPABILITY_EXPANSION_PLAN.md` (this expansion's own policy document); `ecommerce/Commerce_SLM_Research_Plan.md` §5–13; `ecommerce/plans/00_shared_core.md` (structural template this file mirrors); `additonal/Retail_Ecommerce_SLM_Research_Source.md` ch. 10 (model landscape), ch. 11 (dataset inventory), ch. 28 (category-specialist decision rule), ch. 34 (forgetting/retention formal apparatus).
