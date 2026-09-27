# CommerceCore Serving

The concrete integration path for a startup adopting CommerceCore: parse natural-language shopping queries into structured constraints, self-hosted, low-resource.

## Quick start (Python library)

```python
from serve.inference import CommerceCoreQueryParser

parser = CommerceCoreQueryParser(adapter_path="path/to/trained/checkpoint")
result = parser.parse("black waterproof trainers under 80 pounds")
# -> [{"field": "color", "op": "eq", "value": "black"},
#     {"field": "waterproof", "op": "eq", "value": "true"},
#     {"field": "price_gbp", "op": "lte", "value": "80"}]
```

No adapter path given: runs the unmodified base model (Qwen3-1.7B), no fine-tuning applied — useful for testing the serving path before a trained checkpoint exists.

## Quick start (REST API / Docker)

```bash
docker build -t commercecore -f serve/Dockerfile .
docker run -p 8000:8000 -e COMMERCECORE_ADAPTER_PATH=/path/to/checkpoint commercecore
```

```bash
curl -X POST http://localhost:8000/parse-query \
  -H "Content-Type: application/json" \
  -d '{"query": "black waterproof trainers under 80 pounds"}'
```

Same image runs on a RunPod pod or a local machine — no code changes between environments, per the project's "self-hosted, not heavy" design decision.

## What this is validated on (be honest about scope)

- **Task**: query → structured shopping constraints (field/operator/value triples).
- **Verticals tested**: fashion, grocery, general merchandise (synthetic training data; see `EXECUTION_PLAN.md` for the synthetic-vs-real disclosure).
- **Public benchmark comparison**: QueryNER test set (real, human-annotated) — see `eval/queryner_baseline_results.json` for the exact Claude/GPT baseline numbers this was measured against.
- **Not yet validated**: full end-to-end retrieval, catalog normalization, or search-recovery tasks (these exist in the schema layer but aren't in this first trained checkpoint).

Don't claim more than what's actually been measured — see `EXECUTION_PLAN.md` section 0's claim-discipline convention (`MEASURED_HERE` / `EXTERNAL_REPORTED` / `TARGET` / `ESTIMATE`).
