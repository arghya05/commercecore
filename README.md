# CommerceCore

Author: [Arghya Mukherjee](https://orcid.org/0009-0008-3423-8574) · [arghya05@gmail.com](mailto:arghya05@gmail.com) · ORCID: 0009-0008-3423-8574

CommerceCore is a Qwen3-1.7B research prototype for structured shopping-query extraction. It combines human-annotated QueryNER data with synthetic query and catalog supervision using QLoRA.

**The revised paper reports implementation feasibility and an evidence audit. Existing comparisons do not establish superiority over frontier models or specialized encoders.** The audit found incomplete baseline label instructions, evaluation examples used for checkpoint selection, and inconsistent synthetic targets. Earlier broad claims in this repository are superseded by the revised manuscript.

## Paper and source

- [Research paper PDF](paper/CommerceCore_Paper.pdf)
- [Main LaTeX source](paper/commercecore_paper.tex)
- [Complete arXiv source ZIP](paper/CommerceCore_arXiv_Source.zip)
- [Build instructions](paper/README.md)
- [Evidence review and publication status](paper/REVIEW_AND_PUBLICATION_STATUS.md)
- [Offline audit results and input hashes](paper/evidence/audit_results.json)

The PDF is a public preprint, not an accepted conference paper. GitHub publication is not an arXiv submission. The source archive includes all manuscript dependencies and can compile independently of the model and datasets.

## Supported results

| Result | Value | Scope |
|---|---:|---|
| Training mixture | 2,130 rows | 1,500 QueryNER, 450 synthetic queries, 180 synthetic listings; specified by the builder |
| Recorded run duration | 75.18 minutes | Training loop, periodic evaluation, and checkpointing |
| Estimated GPU charge | $0.9021 | Historical $0.72/hour rate; excludes startup and other project expenses |
| Constraint F1 at step 750 | 0.725 | 20 development queries evaluated during selection |
| QueryNER F1 | 0.498 | Mean per-query label/text-set F1 on 993 records; includes 30 selection-exposed examples |
| Six diagnostic queries | 0.400 mean F1 | Saved predictions rescored; micro-F1 0.381 and 0/6 exact-set matches |
| Synthetic target audit | 8 conflicting groups | 20 rows affected; 374 distinct queries among 450 rows |

The available QueryNER API baseline records cover only the first 50 examples and instruct models to emit two of 17 labels. Those results are not a matched full-benchmark comparison. The paper also distinguishes historical aggregate measurements from analyses that can be recomputed directly from saved predictions.

## Reproduce the offline audit and paper

Python 3.10+ is sufficient for the evidence audit; it makes no model or API calls:

```sh
python3 paper/audit_evidence.py
```

With Tectonic installed, build the PDF and arXiv source ZIP:

```sh
python3 paper/build_paper.py
```

The audit checks data counts, exact normalized query overlap, target-set conflicts, the baseline prompt's label ceiling, saved diagnostic scores, checkpoint trade-offs, and generation-cost arithmetic. It records SHA-256 hashes so the audited inputs remain identifiable.

## Model and serving code

The project's model release is [arghya2030/commercecore-qwen3-1.7b](https://huggingface.co/arghya2030/commercecore-qwen3-1.7b). The paper revision does not independently verify the hosted weights or rerun model inference.

- [Serving interface and instructions](serve/README.md)
- [Inference implementation](serve/inference.py)
- [Full QueryNER inference script](eval/full_queryner_eval.py)
- [Historical checkpoint decision](train/results_multitask_clean/MODEL_SELECTION_DECISION.md)

Some historical scripts still use original pod paths. Exact inference reproduction requires the appropriate model, environment, and path configuration; complete historical raw predictions and environment locks are not available. The inference parser maps malformed output to an empty list and does not establish semantic correctness. This is a research artifact, not a validated production filter.

## Research extensions

The revised paper specifies, but does not report completed results for, repaired supervision, a larger independently validated robustness benchmark, matched specialist and backbone baselines, data/task ablations, multiple seeds, and sustained serving measurements. Those experiments are necessary for stronger comparative claims. They cannot be replaced by rewording the existing results.

## Attribution

QueryNER is attributed to Chester Palen-Michel, Lizzie Liang, Zhe Wu, and Constantine Lignos. Its dataset card declares CC BY 4.0; retain its underlying Amazon ESCI provenance as well. Qwen3-1.7B is distributed under Apache 2.0. See [data provenance](data/queryner/README.md), the paper references, and the model card. No new blanket license for third-party assets is granted by this README.
