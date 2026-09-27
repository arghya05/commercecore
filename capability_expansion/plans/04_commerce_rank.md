# Capability 4 — Commerce-Rank

Implement after `00_expansion_core.md`, `02_commerce_match.md` and `03_commerce_retrieve.md`. Priority 4 (`CAPABILITY_EXPANSION_PLAN.md` §3). Fully separate model — cross-encoder reranker architecture, not related to the shipped Commerce-Query checkpoint. Cannot be usefully trained before Match and Retrieve both exist: Rank consumes Retrieve's candidate lists and Match's relevance judgments as direct training inputs.

## Outcome

Given Retrieve's top-20 candidates for a parsed query, produce the final relevance-ordered result list, incorporating user/context signal where available (initially none — context-aware reranking is a later ablation once Commerce-Taste exists, not a v0 requirement). This is the last stage before results are shown; a mistake here is fully visible to the shopper, unlike an upstream retrieval miss that might be masked by a good rerank.

## Model and data

Base: `Qwen/Qwen3-Reranker-0.6B` (target band 100–400M, per user's table; this is the dossier's own v0 choice, `Commerce_SLM_Research_Plan.md` §6). Required baseline comparators: a simpler fine-tuned cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`-class model, named in the broader dossier's model landscape) and the frozen, untrained reranker as a floor.

Training data: **ESCI, reused from Commerce-Match's pull** (`data/esci/`, no second ingestion) — relevance-graded query/product pairs, official E/S/C/I labels, plus **Retrieve's own top-20 candidate lists as a hard-negative source** once Retrieve (capability 03) is deployed and producing real candidate sets. Do not train Rank on ESCI alone without also training on Retrieve's actual candidate distribution — a reranker tuned only on ESCI's pre-filtered pairs will not have seen the specific false-positive patterns Retrieve's BM25+dense fusion actually produces, and will underperform on the real pipeline it's meant to sit inside.

## API and schema

Implement `POST /v1/rank/reorder`. Request: tenant context, parsed query, candidate list (Retrieve's output schema, reused directly), optional context signal (empty in v0). Response: reordered candidate list with per-item relevance scores, calibrated (not raw cross-encoder logits presented as confidence).

No new schema needed — this capability consumes Commerce-Query's and Commerce-Retrieve's existing output types and produces a reordering of the same candidate list shape, not a new contract.

## Training

Cross-encoder pairwise/listwise training on ESCI-derived pairs, using the official ESCI ranking-track gain mapping (E=1/S=0.1/C=0.01/I=0) as the target signal, not a repurposed binary label. QLoRA is not the primary lever here — a 0.6B cross-encoder is typically fully fine-tuned or LoRA-tuned at a much smaller adapter size; use the dossier's full-tuning LR range (5e-6 to 3e-5, per `00_expansion_core.md`'s hyperparameter table) as the starting point if full-tuning, or the standard LoRA settings if adapting.

Checkpoint and evaluate on a real cadence (§2b item 3): log NDCG@10 against ESCI at every checkpoint, alongside the frontier-LLM reference score on the same benchmark, so no single lucky final checkpoint gets shipped without having been compared against the same yardstick throughout training.

## Evaluation and acceptance

Two eval sets, ≥300 cases each: (i) held-out ESCI reranking split, (ii) Retrieve's own candidate lists over a held-out query set, scored against adjudicated relevance — this second set is what actually validates the reranker in its real serving context, not just against ESCI's pre-filtered pairs.

Public benchmark gate (§2e): **ESCI reranking track**, same official scoring recipe as Commerce-Match, over the same judged candidate sets where possible — this keeps Match's and Rank's numbers directly comparable, since they're evaluated on overlapping data with different tasks (classification vs. ranking).

Frontier LLM gate: same fixed matrix, run as zero/few-shot rerankers over the identical candidate sets — not a different, easier prompt-based ranking task.

Breadth check (§2g): confirm the reranker doesn't systematically favor exact keyword matches over legitimate semantic matches (a known cross-encoder failure mode) — include at least one eval slice where the correct top result uses different vocabulary than the query.

## Deliverables

`train/results_rank/` with loss/eval logs (retention `NOT_APPLICABLE`, no shared weights with Query). Serving smoke test on GPU and local CPU, including a latency check under the existing p95 budget target (`Commerce_SLM_Research_Plan.md` §14 — 300ms for the normal search route) since Rank sits directly in the interactive request path, unlike Understand's largely offline/batch usage.

Sources: `CAPABILITY_EXPANSION_PLAN.md` §7, §1b, §2e; `ecommerce/plans/03_retrieval_and_matching.md` (original reranking step, step 7 of its pipeline); `Commerce_SLM_Research_Plan.md` §6, §12.
