# Capability 7 — Commerce-Taste

Implement after `00_expansion_core.md`. Priority 7 (`CAPABILITY_EXPANSION_PLAN.md` §3). Fully separate model once unblocked. **This capability is license-blocked, not model-blocked or data-blocked** — a candidate public dataset exists (Amazon-M2), but its commercial license terms are not yet confirmed. Do not train on it before that confirmation happens — this is a hard stop, not a formality (`CAPABILITY_EXPANSION_PLAN.md` §2f, `00_expansion_core.md`'s data-source gate table).

## Outcome

User/session → shopper taste embedding, or a next-product prediction from current-session behavior. Two distinct sub-tasks worth separating, per the dossier's own use-case breakdown (ch. 6, U21–U24): (1) attribute affinity prediction (brand/color/size/category preference from interaction history — a lighter-weight, per-aspect classification problem) and (2) session next-item recommendation (predicting the next likely product from the current session's click/cart sequence — a heavier sequential-modeling problem). Do not conflate these into one undifferentiated "personalization model" — they have different architectures, different data requirements, and different failure costs.

## License gate — resolve before any other step in this file

**Amazon-M2's commercial license terms are unverified** (`CAPABILITY_EXPANSION_PLAN.md` §1b, §2f, §7). Before proceeding past this section: confirm in writing whether Amazon-M2 (the KDD Cup 2023 multilingual shopping session dataset) permits commercial model training, not just research/competition use. If it does not, this capability has no public dataset backbone at all, and the only remaining path is first-party merchant session data (with its own privacy/authorization requirements, `Commerce_SLM_Research_Plan.md` §1: "personalized recommendations... are separate projects") — do not substitute a different, unvetted session dataset as a workaround without running it through the same license-gate discipline applied to every other dataset in this project.

## Model and data

Base, per the dossier's own direct architecture evidence (ch. 3, E02 — DeepAffinity): a **tiny backbone with category-specific heads**, not a large generative model — DeepAffinity uses SmolLM2-135M and beats Gemini 1.5 Flash on a proprietary fashion attribute-affinity task (mean micro-F1 0.34 vs. 0.26), which is direct evidence that small-model-plus-heads is the right architecture family for the affinity-prediction sub-task specifically. For session next-item recommendation, the dossier separately points to sequential-recommendation architectures (HSTU, TIGER — ch. 7) as the relevant prior art, a materially different model shape (sequence model over item/action tokens, not a text decoder).

Target band: 100–500M (user's table). This confirms the small-backbone approach is appropriately sized, not a compromise.

Data: **Amazon-M2**, once license-confirmed, for session next-item recommendation. No public dataset is currently named for the attribute-affinity sub-task specifically — DeepAffinity's own data is proprietary; this sub-task may need first-party interaction data from the start, same as the cold-start caveat below.

## API and schema

Implement `POST /v1/taste/affinity` (per-aspect preference prediction: brand, color, size, category — returns a probability per aspect, with explicit low-confidence/cold-start signaling, not a forced guess) and `POST /v1/taste/next_item` (session-sequence → ranked next-item candidates, feeding into Commerce-Retrieve/Rank as an optional context signal once available).

New schema needed here — neither `retrieval_match.py` nor `catalog_taxonomy.py` covers user/session state. Define a minimal `UserSessionEvent` type (event type, product ID, timestamp, session ID) and a `TastePreference` output type (aspect, predicted value, confidence, cold-start flag) — keep this schema as small as the task requires; do not pre-build fields for capabilities not yet in scope.

## Training

Attribute-affinity sub-task: tiny backbone (SmolLM2-135M-class or equivalent) with category-specific classification heads, trained per-aspect — this is closer to a classical multi-task classifier than a generative model; standard classifier training (cross-entropy per head), not the QLoRA/generative pipeline used elsewhere in this project. Cache user embeddings and serve lightweight heads at inference, per the dossier's own stated pattern (ch. 6, U21).

Session next-item sub-task (only after license confirmation): sequential recommendation baseline first (a standard next-item transformer over session sequences), before considering a heavier generative/semantic-ID approach (TIGER-style) — the dossier's own frontier-failure audit (ch. 35, U23) explicitly warns that a generative retrieval approach "generates nonexistent or stale product IDs when item tokens and catalogue lifecycle are unmanaged" — this is a materially harder, higher-risk architecture than a ranking-based next-item model, and should not be the default starting point.

## Evaluation and acceptance

Two eval sets, ≥300 cases each, per sub-task: for affinity, per-aspect micro/macro F1, calibration, and an explicit **cold-start slice** (low-history users) — the dossier notes DeepAffinity itself excludes low-history users from its own headline number, which is a real limitation to test for here, not repeat silently. For next-item, Recall/NDCG, candidate validity (does the model ever recommend a discontinued or out-of-stock item — a direct evidence-state violation per this project's own rules), new-item coverage, and delayed-conversion handling, with temporal (not random) splits and explicit exposure-bias logging.

Public benchmark gate (§2e): Amazon-M2's own evaluation protocol, once license-confirmed — treat any result produced before that confirmation as provisional and undisclosed as final, per §1b/§7's existing flag.

Frontier LLM gate: same fixed matrix, run zero/few-shot on the same session data — note that this is a genuinely unusual task for a general LLM (no standard prompt format for "predict next item from this click sequence"), so the comparison protocol itself needs explicit design and disclosure, not an assumed-fair default prompt.

## Deliverables

The license confirmation itself (the actual first deliverable, blocking everything else). Once unblocked: `train/results_taste/` with the two sub-tasks reported separately, cold-start and exposure-bias slices explicitly called out, not averaged into a single headline number.

Sources: `CAPABILITY_EXPANSION_PLAN.md` §7, §1b, §2f; `additonal/Retail_Ecommerce_SLM_Research_Source.md` ch. 3 (E02 DeepAffinity), ch. 6 (U21–U24), ch. 7 (E06/E07 HSTU/TIGER), ch. 32 (V01, Amazon-M2 as adjacent research data), ch. 35 (U23 generative-recommendation risk).
