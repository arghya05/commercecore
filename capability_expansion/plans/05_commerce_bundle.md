# Capability 5 — Commerce-Bundle

Implement after `00_expansion_core.md` and `02_commerce_match.md`. Priority 5 (`CAPABILITY_EXPANSION_PLAN.md` §3). Fully separate model, warm-started from Match's checkpoint, not from Query's. **This capability is data-blocked, not model-blocked** — read the data-sourcing section before scoping any training work.

## Outcome

Two related but distinct judgments: (1) compatibility — does this specific accessory/part fit this specific product/model/version (the hardest, most specific case: "does this charger fit the 2023 or 2024 revision of this laptop," per the dossier's own U16 use case and its ch. 35 frontier-failure entry, both independently confirming this is harder than category-level relatedness); (2) complementary-product suggestion — what else pairs well with this product (basket completion, "frequently bought together"), a lower-stakes ranking problem building on Match's `COMPLEMENT` relation class.

A wrong compatibility judgment has real cost (a shopper buys an incompatible part) — this capability must abstain rather than guess when evidence is insufficient, per the project's evidence-state discipline (`CAPABILITY_EXPANSION_PLAN.md` §2, rule 3).

## Data-sourcing gap — resolve before scoping the model further

**No public dataset for compatibility-graph ground truth exists in either source dossier** (`CAPABILITY_EXPANSION_PLAN.md` §1b, §2f, §7 all confirm this independently). The dossier's own category-specialist chapter (ch. 28, category C03 — mobiles/chargers/accessories) states this is a "strong specialist-adapter opportunity because wrong answers have clear labels and commercially meaningful consequences," but also that WDC Products and ESCI are only "adjacent-only data," not comprehensive compatibility ground truth.

**Before writing a training script for this capability, produce a scoped estimate of:** (a) which merchant/manufacturer compatibility tables can realistically be licensed or accessed (structured data, e.g. device compatibility matrices from manufacturer spec sheets — often available as semi-structured web tables, not typically a clean bulk download), (b) the cost/time to acquire and clean a first tranche (target: at least 300 adjudicated compatibility pairs per major category, per `CAPABILITY_EXPANSION_PLAN.md` §2c's minimum eval-set floor, before any training decision), and (c) whether Match's own `COMPLEMENT`-labeled ESCI subset is sufficient to bootstrap the lower-stakes complementary-suggestion half of this capability while compatibility-checking remains blocked. Do not start full training on this capability until (a)/(b) produce a real number — an unscoped "we'll figure out the data later" is exactly what `CAPABILITY_EXPANSION_PLAN.md` §7 already flags as the reason Bundle should not be treated as equally ready as Match/Retrieve.

## Model and data

Base: same architecture family as Match (`Qwen/Qwen3-0.6B` or `ModernBERT-base`), warm-started from Match's trained checkpoint — same size band, adjacent task, but its own separately fine-tuned and evaluated artifact (`commercecore-bundle`), never presented as "Match's checkpoint, relabeled" (`CAPABILITY_EXPANSION_PLAN.md` §1c).

Bootstrap data, available today without new sourcing: Match's ESCI-derived `COMPLEMENT`-class examples, for the complementary-suggestion half only. Compatibility-checking has no bootstrap data path — this is the one sub-task of this capability that is genuinely blocked pending the data-sourcing estimate above.

## API and schema

Implement `POST /v1/bundle/compatible` (compatibility check, returns pass/fail/unknown against a specific target product, never a guessed pass) and `POST /v1/bundle/suggest` (complementary-product suggestions, returns a ranked list, lower-stakes than compatibility). Reuse `schemas/retrieval_match.py`'s relation types where applicable — this capability specializes Match's `COMPLEMENT` class, it does not invent a parallel schema.

## Training

For the complementary-suggestion half: same QLoRA/fine-tuning settings as Match (`00_expansion_core.md`'s hyperparameter table), warm-started from Match's checkpoint rather than the base model — measure whether warm-starting actually helps versus training from the base directly, don't assume it does.

For the compatibility-checking half: **do not begin training until the data-sourcing estimate above produces a real dataset.** If and when it does, this likely needs an entity-linking step first (resolving "this laptop" to an exact model/version identifier) before a relation classifier can be trained on top — the dossier's ch. 35 entry for U16 names "entity linking plus deterministic graph/rules" as the better-fit component, not a pure end-to-end learned classifier. Scope this as two sub-components (entity linking + compatibility lookup/classifier), not one model, once data exists.

## Evaluation and acceptance

Complementary-suggestion half: same eval structure as Match (§5 of this plan set) — two eval sets ≥300 cases, ESCI-derived held-out plus an out-of-distribution set, public benchmark gate via ESCI's `COMPLEMENT` slice specifically, frontier LLM gate on the same fixed matrix.

Compatibility-checking half: **no public benchmark exists to assign** (per `CAPABILITY_EXPANSION_PLAN.md` §2e's own table) — this half of the capability cannot claim to have cleared the two-part gate until either a benchmark is identified or a new one is built and explicitly disclosed as new, not borrowed. Do not report only the complementary-suggestion numbers and imply they cover the compatibility-checking claim too — these are different tasks with different risk profiles and must be reported separately.

Breadth check (§2g): compatibility-checking specifically needs an eval slice for "near-identical model numbers with a real functional difference" (the dossier's own stated failure mode) — this is the hardest and most safety-relevant case, and a capability that only tests obviously-different products has not demonstrated the thing that actually matters here.

## Deliverables

A scoped data-sourcing cost/time estimate (the immediate first milestone, before any code) — this is the deliverable that unblocks everything else in this file. Once unblocked: `train/results_bundle/` mirroring Match's structure, with the two sub-tasks (compatibility, complementary-suggestion) reported as separate rows, never combined into one headline number.

Sources: `CAPABILITY_EXPANSION_PLAN.md` §7, §1b, §2f, §2h (fold-in confirmation from two independent dossier chapters); `additonal/Retail_Ecommerce_SLM_Research_Source.md` ch. 28 (category C03 justification), ch. 35 (U16 frontier-failure entry).
