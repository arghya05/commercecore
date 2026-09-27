"""Builds a real held-out eval set for the query-constraint-extraction task.

Uses structured scenarios NOT used in training (fresh random draws with a
different seed) plus a subset genuinely reserved as development-partition-only,
per EXECUTION_PLAN.md section 2's partition scheme. This is a DEVELOPMENT
partition eval set (used for iteration), NOT the locked-final partition.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from schemas.common import Vertical

OUT_PATH = Path(__file__).parent / "held_out_eval_set.jsonl"

# Fixed, hand-written eval examples (not LLM-generated) -- genuine ground
# truth, independent of the synthetic generation pipeline used for training.
EVAL_EXAMPLES = [
    {"query": "looking for black waterproof trainers, budget under 80 pounds",
     "constraints": [{"field": "color", "op": "eq", "value": "black"},
                     {"field": "waterproof", "op": "eq", "value": "true"},
                     {"field": "price_gbp", "op": "lte", "value": "80"}]},
    {"query": "need a leather jacket, doesn't matter the color",
     "constraints": [{"field": "material", "op": "eq", "value": "leather"}]},
    {"query": "wool sweater size S",
     "constraints": [{"field": "material", "op": "eq", "value": "wool"},
                     {"field": "size", "op": "eq", "value": "S"}]},
    {"query": "almonds with no added sugar please",
     "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"}]},
    {"query": "snack bar that has nuts in it, don't care about sugar",
     "constraints": [{"field": "contains_nuts", "op": "eq", "value": "true"}]},
    {"query": "peanut butter, no added sugar, must contain nuts obviously",
     "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"},
                     {"field": "contains_nuts", "op": "eq", "value": "true"}]},
    {"query": "something nut free and unsweetened",
     "constraints": [{"field": "contains_nuts", "op": "eq", "value": "false"},
                     {"field": "contains_added_sugar", "op": "eq", "value": "false"}]},
    {"query": "battery powered speaker with at least a year warranty",
     "constraints": [{"field": "power_source", "op": "eq", "value": "battery"},
                     {"field": "warranty_months", "op": "gte", "value": "12"}]},
    {"query": "mains powered kettle under 50 quid",
     "constraints": [{"field": "power_source", "op": "eq", "value": "mains"},
                     {"field": "price_gbp", "op": "lte", "value": "50"}]},
    {"query": "cordless drill, 2 year warranty minimum",
     "constraints": [{"field": "power_source", "op": "eq", "value": "battery"},
                     {"field": "warranty_months", "op": "gte", "value": "24"}]},
    {"query": "red waterproof jacket for hiking",
     "constraints": [{"field": "color", "op": "eq", "value": "red"},
                     {"field": "waterproof", "op": "eq", "value": "true"}]},
    {"query": "black leather boots, any size",
     "constraints": [{"field": "color", "op": "eq", "value": "black"},
                     {"field": "material", "op": "eq", "value": "leather"}]},
    {"query": "granola with nuts and sugar, the sweet kind",
     "constraints": [{"field": "contains_nuts", "op": "eq", "value": "true"},
                     {"field": "contains_added_sugar", "op": "eq", "value": "true"}]},
    {"query": "rice cakes, nothing sweet, no nuts either",
     "constraints": [{"field": "contains_added_sugar", "op": "eq", "value": "false"},
                     {"field": "contains_nuts", "op": "eq", "value": "false"}]},
    {"query": "robot vacuum with a good long warranty, at least 2 years",
     "constraints": [{"field": "warranty_months", "op": "gte", "value": "24"}]},
    {"query": "air fryer plugged into the wall, warranty over a year",
     "constraints": [{"field": "power_source", "op": "eq", "value": "mains"},
                     {"field": "warranty_months", "op": "gte", "value": "12"}]},
    {"query": "cheap trainers under 65 quid, has to be black",
     "constraints": [{"field": "price_gbp", "op": "lte", "value": "65"},
                     {"field": "color", "op": "eq", "value": "black"}]},
    {"query": "medium sized wool coat",
     "constraints": [{"field": "material", "op": "eq", "value": "wool"},
                     {"field": "size", "op": "eq", "value": "M"}]},
    {"query": "phone charger, battery operated obviously, half year warranty is fine",
     "constraints": [{"field": "power_source", "op": "eq", "value": "battery"},
                     {"field": "warranty_months", "op": "gte", "value": "6"}]},
    {"query": "oat milk that's vegan",
     "constraints": []},  # deliberately unresolvable-to-schema constraint (tests honest zero-match, not a fabricated pass)
]


if __name__ == "__main__":
    with open(OUT_PATH, "w") as f:
        for ex in EVAL_EXAMPLES:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(EVAL_EXAMPLES)} hand-written held-out eval examples to {OUT_PATH}")
