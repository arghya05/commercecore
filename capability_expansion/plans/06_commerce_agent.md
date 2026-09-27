# Capability 6 — Commerce-Agent

Implement after `00_expansion_core.md` and capabilities 01–04 (Understand, Match, Retrieve, Rank all individually evaluated). Priority 6 (`CAPABILITY_EXPANSION_PLAN.md` §3). **v0 requires no new trained model** — it is an orchestration layer over the four prior capabilities' typed contracts. v1 (only if measured need exists) is a sibling adapter, gated exactly like Understand.

## Outcome

Execute read-only, multi-step search/filter/compare workflows by composing Query → Understand → Match → Retrieve → Rank's existing typed outputs, plus the deterministic tool simulator already scoped in `schemas/search_recovery.py`. v0 explicitly excludes any write action (add-to-cart execution against a real cart, checkout, order modification) — those are a different, much higher-stakes product surface (`CAPABILITY_EXPANSION_PLAN.md` §0a, §2h: the transactional/operational surface is out of scope for this expansion).

v1 scope addition, folded in from the gap analysis (§2h): **multi-turn preference memory** — retaining an explicit constraint stated earlier in a session ("no leather," "under budget") across turns, handling corrections, and not silently treating a one-off preference as permanent. The dossier's own frontier-failure audit (ch. 35, U18) names the correct architecture for this: "consented structured memory with provenance and expiration" — this is a session-state design problem for the orchestration layer, not a new learned behavior, and should be built as such before considering a v1 trained adapter.

## Model and data

v0: **no model.** The orchestrator calls existing endpoints (`/parse-query`, `/v1/catalog/normalize`, `/v1/match/classify`, `/v1/retrieve/candidates`, `/v1/rank/reorder`) in sequence, validates each typed output against its schema before passing it to the next stage, and executes the deterministic tool simulator for read-only actions (`search_products`, `get_product_details`, `get_price`, `check_inventory`, `get_policy` — the exact tool set already implemented in `validators/mechanical_validator.py`'s `simulate_tool_plan_execution`, reused directly, not reinvented).

v1 (only if orchestration-level failures persist after v0 is measured and cannot be fixed by better deterministic routing logic): `Qwen/Qwen3-1.7B` or `Qwen/Qwen3.5-2B`, sibling LoRA adapter `commercecore-agent-adapter`, trained on the tool-plan records described below plus session-memory-update examples for the v1 scope addition.

Training data (v1 only): the 2,000-record tool-plan target already specified in `Commerce_SLM_Research_Plan.md` §9 ("read-only tool plans"), generated via the deterministic simulator with schema-verified outputs — these become real supervision only once v0's orchestration logic identifies specific failure patterns a learned policy would fix (e.g., deciding when to clarify vs. retry vs. abstain), not as a default v1 trigger.

## API and schema

Implement `POST /v1/agent/execute`. Request: tenant context, natural-language goal (e.g., "find me waterproof trainers under budget, compare the top 3"), session ID (for v1 memory), locale. Response: the composed result (search results, comparison table, or a clarification request), plus a full trace of which capabilities were called, in what order, with what inputs/outputs — this trace is required for debugging composed failures, since an agent's mistake could originate in any of the four upstream capabilities it calls.

Reuse `schemas/search_recovery.py`'s existing tool-plan contract for the read-only tool set — do not design a new tool-call schema; extend the existing one only if a genuinely new tool type is needed (none identified yet).

## Training (v0: none; v1 only if triggered)

If v1 is triggered: same QLoRA settings as Understand (`00_expansion_core.md`'s hyperparameter table), sibling adapter on the frozen base, subject to the exact same regression gate as any other sibling adapter (`CAPABILITY_EXPANSION_PLAN.md` §2b) — a v1 Agent adapter must not regress Commerce-Query's numbers any more than Understand's adapter is allowed to.

## Evaluation and acceptance

**Critical distinction, mandatory for this capability specifically**: report **pass^k** (succeeds on every one of *k* independent attempts) separately from **pass@k** (succeeds at least once within *k* attempts via retry). The dossier is explicit (ch. 20, confirmed by direct read): "a purchase workflow generally needs repeatable first-attempt correctness, not the ability to succeed once after many tries." Even though v0 is read-only (lower stakes than a purchase), report both metrics from the start — this discipline needs to be established before v1 or any future write-capable extension, not retrofitted after.

Public benchmark gate (§2e): **Tau2-bench** and/or **WebShop** as the public agent-environment benchmark, run with this project's own tool set substituted in where the environment allows it — do not silently adopt a different, easier tool set than the environment's own and still claim the benchmark score. **BFCL/Gorilla** additionally, for tool-call argument correctness specifically (a distinct failure mode from task-level success).

Frontier LLM gate: same fixed matrix, run as agents over the identical tool environment and tool set — not as plain chat completions given the same goal text, which is a materially easier and non-comparable setup.

Two eval sets, ≥300 cases each: (i) held-out goals over the deterministic simulator, (ii) Tau2-bench/WebShop as the out-of-distribution public environment test.

Breadth check (§2g): confirm the orchestrator correctly reports "no satisfiable result" rather than silently relaxing a hard constraint from Query's output when Retrieve/Rank return nothing — this is the exact failure mode the original project's own architecture diagram was built to prevent ("waterproof never becomes water-resistant").

## Deliverables

The orchestration code itself (composition logic, trace generation, tool-simulator integration) — this is the actual v0 deliverable, not a trained model. A Tau2-bench/WebShop reproduction report with both pass@k and pass^k reported. If v1 is triggered: `train/results_agent/` mirroring Understand's structure, same regression-gate discipline.

Sources: `CAPABILITY_EXPANSION_PLAN.md` §7, §1b, §2e, §2h; `ecommerce/plans/04_search_recovery_and_tools.md` (original tool-plan spec — reused directly for v0's tool set); `validators/mechanical_validator.py` (existing deterministic simulator, reused not reinvented); `additonal/Retail_Ecommerce_SLM_Research_Source.md` ch. 20 (pass@k/pass^k), ch. 35 (U18 memory architecture).
