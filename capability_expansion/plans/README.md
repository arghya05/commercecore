# Capability expansion — implementation plans

Same structure and density as `ecommerce/plans/` (the plans that actually drove the shipped Commerce-Query build): one shared-core contract file, then one self-contained implementation spec per capability (`Outcome → Model and data → API/schema → Training → Evaluation and acceptance → Deliverables/sources`).

Implement `00_expansion_core.md` first — it fixes repo layout, CLI, and shared conventions all other files assume.

| File | Capability | Status |
|---|---|---|
| `00_expansion_core.md` | Shared contract | Ready to implement |
| `01_commerce_understand.md` | Commerce-Understand | Ready to implement — Priority 1 |
| `02_commerce_match.md` | Commerce-Match | Ready to implement — Priority 2, includes required schema fix |
| `03_commerce_retrieve.md` | Commerce-Retrieve | Ready to implement — Priority 3 |
| `04_commerce_rank.md` | Commerce-Rank | Ready to implement — Priority 4, depends on 02+03 shipping first |
| `05_commerce_bundle.md` | Commerce-Bundle | Specified, but **data-blocked**: the first deliverable is a compatibility-table data-sourcing cost estimate, not code — see file for the exact gate |
| `06_commerce_agent.md` | Commerce-Agent | Ready to implement — v0 is pure orchestration (no new model), depends on 01–04 shipping first |
| `07_commerce_taste.md` | Commerce-Taste | Specified, but **license-blocked**: Amazon-M2's commercial terms must be confirmed before any training — see file for the exact gate |
| `08_commerce_qa.md` | Commerce-QA | Deliberately deferred — lowest priority; this file specifies exactly what's needed to unblock full scoping, not a build-ready spec |

All nine capabilities now have a plan file. Three (Bundle, Taste, QA) are honestly blocked on an external dependency (data sourcing, license confirmation, product-need confirmation respectively) rather than a planning gap — each file states its own unblock condition precisely. Build order still follows dependency order: Understand → Match → Retrieve → Rank → Bundle → Agent → Taste → QA.
