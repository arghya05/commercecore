# CommerceCore — which model to use, when

Quick-reference for building a product on top of CommerceCore. For the full policy rationale (why the boundaries are where they are, evaluation rigor, data gating), see `CAPABILITY_EXPANSION_PLAN.md`. For exact implementation specs, see `plans/`.

## The two model families — read this first

**Family 1 — shipped, frozen, never modified.**
- **Commerce-Query**: shopper text → structured constraints (color, price, size, exclusions, negation).
- Lives at `github.com/arghya05/commercecore`, weights at `huggingface.co/arghya2030/commercecore-qwen3-1.7b`, served at `/parse-query`.
- Nothing in this expansion folder ever retrains, resumes, or overwrites this. It is a fixed, versioned dependency, not a work-in-progress.

**Family 2 — this expansion, new capabilities, independently shippable.**
- Commerce-Understand, Commerce-Match, Commerce-Retrieve, Commerce-Rank, Commerce-Bundle, Commerce-Agent, Commerce-Taste, Commerce-QA.
- Each is either a sibling LoRA adapter (shares only frozen base weights with Query, no shared training) or a fully separate model (different architecture entirely). See `CAPABILITY_EXPANSION_PLAN.md` §1b, §2b for which is which and why.
- Each ships independently. Using one does not require deploying, retraining, or risking any other.

## When to use which capability

| If you need to... | Use | Depends on (must be deployed first) |
|---|---|---|
| Turn "black waterproof trainers under 80 pounds" into filterable constraints | **Commerce-Query** (shipped) | Nothing — standalone |
| Turn messy supplier text/listings into clean structured product JSON | **Commerce-Understand** | Nothing — standalone |
| Deduplicate near-identical listings across supplier feeds | **Commerce-Understand** (entity-dedup sub-task, see plan §2h) | Nothing — standalone |
| Decide if two products are the same/variant/substitute/complement/unrelated | **Commerce-Match** | Nothing — standalone |
| Find candidate products for a query, from an index | **Commerce-Retrieve** | Commerce-Query (query side) + Commerce-Understand (indexed catalog side) |
| Order a candidate list by relevance | **Commerce-Rank** | Commerce-Retrieve (candidates) + Commerce-Match (relevance signal) |
| Suggest compatible/complementary add-on products | **Commerce-Bundle** | Commerce-Match (complement relation as its starting point) |
| Recommend next products from session/user history | **Commerce-Taste** | Nothing — standalone (blocked on Amazon-M2 license check, see plan §1b) |
| Answer a specific product question with cited evidence | **Commerce-QA** | Commerce-Understand (grounding source) — deferred, not yet scoped |
| Execute a read-only multi-step search/filter/compare flow | **Commerce-Agent** (v0) | Commerce-Query + Commerce-Understand + Commerce-Match + Commerce-Retrieve, all deployed first — v0 is pure orchestration, no new model |

## Common product ideas → which capabilities to compose

| Product idea | Capabilities used |
|---|---|
| Smart search box / search-as-you-type | Query + Retrieve + Rank |
| Catalog cleanup / data-quality SaaS for merchants | Understand |
| "Frequently bought together" / compatible-accessories widget | Match + Bundle |
| Conversational shopping assistant | Query + Understand + Match + Retrieve + Agent |
| Personalized homepage / "recommended for you" | Taste + Rank |
| Product Q&A widget on a product page | QA (once scoped) |

A product idea outside this list (order status, returns, payments, fraud, warehouse/store ops, pricing analytics) is **not covered by either model family** — that's a different product surface (transactional/operational, not shopper-facing discovery), explicitly out of scope here. See `CAPABILITY_EXPANSION_PLAN.md` §2h/§0a for why, and what would be needed to build it as a separate effort.

## Status, right now

| Capability | Status |
|---|---|
| Commerce-Query | **Shipped** — trained, evaluated, published |
| Commerce-Understand | Scoped, implementation plan ready (`plans/01_commerce_understand.md`) — not yet trained |
| Commerce-Match | Scoped, implementation plan ready (`plans/02_commerce_match.md`), requires a schema fix first — not yet trained |
| Commerce-Retrieve | Scoped, implementation plan ready (`plans/03_commerce_retrieve.md`) — not yet built |
| Commerce-Rank, Bundle, Agent, Taste | Scope notes only (`CAPABILITY_EXPANSION_PLAN.md` §7) — implementation plans not yet written |
| Commerce-QA | Deferred — lowest priority, not yet scoped |

Build order matches dependency order above: Understand → Match → Retrieve → Rank → Bundle → Agent → Taste → QA. Don't skip ahead — most later capabilities are untrainable without an earlier one's output.
