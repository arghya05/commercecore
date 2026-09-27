# Capability 3 — Commerce-Retrieve

Implement after `00_expansion_core.md`, `01_commerce_understand.md` and `02_commerce_match.md`. Priority 3 (`CAPABILITY_EXPANSION_PLAN.md` §3). Fully separate model — embedding architecture, not a LoRA choice on any generative backbone. This is the first capability that actually composes Commerce-Query and Commerce-Understand's outputs into a working, testable search loop.

## Outcome

Query or image → correct products, via an index, not via a model memorizing a catalog. The model's job is producing embeddings/scores that plug into a retrieval pipeline; retrieval infrastructure (BM25, dense index, fusion) is separate from any model fine-tuning and must exist before training anything.

## Model and data

Base: `Qwen/Qwen3-Embedding-0.6B`, frozen initially (v0 default per the original research plan §6/§10 — do not fine-tune before establishing a frozen baseline). Compare a cheaper MiniLM-based encoder before committing to the larger model. Target band 150–400M, smallest in the whole capability table — this confirms the separate-model call, no ambiguity to resolve.

Indexed catalog: Commerce-Understand's structured output over ABO (requires capability 01 to exist first). Query side: Commerce-Query's already-shipped structured output — this capability is the first to actually depend on and compose two other capabilities' outputs, which is the point of the shared typed-contract design.

Held-out generalization test: **WANDS**, https://github.com/wayfair/WANDS, MIT, 480 queries / 42,994 products / 233,448 judgments — a different retailer's taxonomy entirely. Per `00_expansion_core.md`'s data-source gate table, this is a locked-final partition: pulled once, never touched mid-development, never used to inform tuning, mixture reweighting or architecture choices. Any violation of this invalidates the generalization claim.

## Retrieval pipeline (infrastructure to build, separate from model training)

1. Authenticate tenant, load model/schema/index versions atomically.
2. Parse query via the existing shipped Commerce-Query endpoint, or the exact-ID/rules fast path if applicable.
3. Apply verified index-supported hard filters from Commerce-Query's constraint output; preserve unresolved states rather than silently dropping them.
4. Retrieve top-100 lexical (BM25) and top-100 dense (Qwen3-Embedding-0.6B); deduplicate by product/variant identity.
5. Fuse with reciprocal rank fusion, k=60 as the starting setting — measure whether this actually helps, don't assume it.
6. Enforce typed constraints and the unknown-evidence policy; overfetch and post-filter only where the engine can't enforce a predicate directly, and measure the resulting recall loss.
7. Hand the top-20 candidates to Commerce-Rank (capability 04) for final ordering — Retrieve's job stops at producing a good candidate set, not the final ranked order.

Version indexes by embedding checkpoint and template. Querying an old index with new embeddings must be structurally impossible, not just discouraged — an atomic alias-switch on rebuild, not a manual step someone can forget.

## API and schema

Implement `POST /v1/retrieve/candidates`. Request: tenant context, parsed query (Commerce-Query's output schema, reused directly — not redefined here), candidate count, locale. Response: ranked candidate list with RRF scores, lexical rank, dense rank, and which hard filters were applied vs. left unresolved.

## Training (if fine-tuning beyond the frozen baseline is later justified)

Only after the frozen-embedding baseline is measured: optional research-variant shared projection head (256-d, cosine similarity, temperature 0.05) per `Commerce_SLM_Research_Plan.md` §6, trained with InfoNCE on ESCI-derived triplets. This is an ablation, not a v0 requirement — retain the frozen specialist embedding model if it already has a better quality/cost frontier. Gradient accumulation alone does not increase the number of in-batch contrastive negatives; use explicit negative batches or gradient caching if this path is pursued.

## Evaluation and acceptance

Two distinct experiments, not conflated (`Commerce_SLM_Research_Plan.md` §12): **candidate reranking** (rank the same judged candidates for every system, using the dataset's documented gain mapping) and **full-corpus retrieval** (index the complete corpus, disclose judgment coverage, add a blinded pooled-judgment sample). Do not label unjudged products irrelevant by default.

Metrics: NDCG@10, Recall@100, MRR, full-corpus vs. candidate-reranking tracks reported separately, domain slices (fashion/grocery/general), head/torso/tail query slices.

Public benchmark gate: **WANDS**, full-corpus track, held out entirely from tuning as stated above. Frontier LLM gate: same fixed matrix, run as zero/few-shot retrieval-and-rank baselines over the identical corpus and candidate sets — not a different, easier setup for the LLM comparators.

## Deliverables

A working `retrieve/` module taking one Commerce-Query output + one indexed catalog slice and returning a ranked candidate list with RRF scores, tested end-to-end on a small (~1,000 product) ABO slice before scaling to the full corpus. Index-versioning code with an atomic alias-switch mechanism, not just a versioning convention in documentation. Serving smoke test on GPU and local CPU, including a check that stale-index queries are structurally rejected, not just discouraged.

Sources: `CAPABILITY_EXPANSION_PLAN.md` §6, §1b, §2e; `ecommerce/plans/03_retrieval_and_matching.md` (original retrieval pipeline spec — steps above mirror it directly, extended for the new capability split between Retrieve and Rank); `Commerce_SLM_Research_Plan.md` §6, §9, §12.
