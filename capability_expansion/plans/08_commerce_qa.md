# Capability 8 — Commerce-QA

Implement after `00_expansion_core.md` and capability 01 (Commerce-Understand, whose structured output is this capability's grounding source). Priority 8, last (`CAPABILITY_EXPANSION_PLAN.md` §3) — lowest stated importance (★★★) of the nine capabilities, and the largest target size band (1–3B). Deferred by design: do not begin implementation before Rank 1–6 ship and a real product need is confirmed (`CAPABILITY_EXPANSION_PLAN.md` §1b, §7).

## Outcome, once scoped

Answer a specific shopper question about a product ("does this come in a size 10," "is this machine washable") grounded only in catalog/reviews/specification text — never inferring an answer from category priors or general world knowledge when the specific product's evidence is silent (the same evidence-state discipline as every other capability, `CAPABILITY_EXPANSION_PLAN.md` §2, rule 3).

Two output modes, per the fold-in decision from the gap analysis (§2h): (1) direct question answering, grounded with a citation to the source span; (2) grounded review-evidence synthesis (the U19 fold-in) — an aggregate summary of recurring product strengths/weaknesses with representative citations, a standing-summary mode rather than a single-question-answer mode, sharing the same grounded-citation infrastructure.

## Why this is deferred, stated plainly rather than skipped silently

This capability is not scoped in the same depth as capabilities 01–07 in this plan set, and that is intentional, not an oversight: (a) it is the lowest-priority capability by the user's own stated importance ranking, (b) no training data has been sourced for it at all, and reaching for a shortcut dataset here is a specific risk this project has already flagged — **do not reach for Amazon Reviews 2023, ECInstruct, or MAVE's underlying Amazon product text as a shortcut** when this capability is eventually scoped; all three have unresolved or explicitly denied commercial licensing per the dossier's dataset inventory (`CAPABILITY_EXPANSION_PLAN.md` §0/§2f, ch. 11's verbatim license findings), and (c) it depends on Commerce-Understand's structured output existing first as a grounding source, which is capability 01, not yet trained at the time this file is written.

## What must happen before this file can be expanded to full implementation depth

1. **Confirm a real product need exists** — this capability was explicitly ranked lowest by the user; do not build it merely because it's the last unaddressed row in the table.
2. **Source a licensed dataset for grounded Q&A pairs.** Candidate model per the original research plan's shortlist (`Commerce_SLM_Research_Plan.md` §7): `numind/NuExtract3`-class extractor for the grounding/citation step, or `Qwen/Qwen3.5-4B` (an escalation past the smaller models used elsewhere in this portfolio, justified only if the smaller models measurably miss required semantics — the dossier's own "not first run" framing for 4B-class models applies directly here). No dataset is currently vetted for this task in either source dossier — this is new sourcing work, not a pull from an already-identified source like ABO or ESCI.
3. **Design the evidence-citation contract** before any training: every answer must carry a source span (same half-open character-offset convention used project-wide) and an explicit `unanswerable`/`unknown` state when the catalog/reviews text doesn't support an answer — modeled on the dossier's own U41 use case ("answerable/unanswerable pairs and contradictory evidence").
4. **Assign a public benchmark.** None is currently named (`CAPABILITY_EXPANSION_PLAN.md` §2e) — Shopping MMLU is the dossier's own closest adjacent public benchmark (ch. 5, U17's evaluation reference), but it evaluates guided-shopping-mission difficulty broadly, not grounded single-product QA specifically; verify fit before adopting it as this capability's assigned benchmark, don't assume it transfers.

## Two-part gate reminder (§2e), applies here same as every other capability

Even though this capability is deferred, when it is eventually built it must clear the same joint gate as every other row: a named public benchmark **and** every frontier LLM in the fixed matrix, both required, non-overlapping confidence intervals to claim a win, and an explicit statement of which of the three claim types (model/system/economic, §2d) is being made. Grounded QA is exactly the kind of task where a frontier LLM with a good retrieval-augmented prompt may already perform very well — do not assume a small trained model wins here by default the way Commerce-Query did; this may be one of the capabilities where the honest outcome is "the frontier model wins on quality, the small model wins only on cost/deployment" (an economic claim, not a model claim) — report whichever is true, don't force a model-claim framing that the data doesn't support.

## Deliverables, once unblocked

A dataset-sourcing proposal (the actual first deliverable — this file is not build-ready today, unlike capabilities 01–06). Only after that: an implementation spec at the same depth as `01_commerce_understand.md` through `06_commerce_agent.md`, following the same section structure (Outcome → Model/data → API/schema → Training → Evaluation → Deliverables).

Sources: `CAPABILITY_EXPANSION_PLAN.md` §1b, §7; `Commerce_SLM_Research_Plan.md` §7; `additonal/Retail_Ecommerce_SLM_Research_Source.md` ch. 5 (U17, Shopping MMLU), ch. 8 (U41, product QA), ch. 11 (dataset licensing landmines to avoid).
