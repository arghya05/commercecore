# CommerceCore: evidence review and publication status

This revision prepares a grounded research preprint, with a compiled PDF and a self-contained LaTeX package. It does **not** certify acceptance at arXiv, NeurIPS, or another venue, and it does not represent unperformed experiments as completed. The current empirical record is insufficient for the original claim of superiority over all tested frontier models.

## Review finding and disposition

| Priority | Finding | Evidence | Revision |
|---|---|---|---|
| Critical | QueryNER scores were presented as a matched full-test comparison, but API results use the first 50 examples while CommerceCore uses 993. | `eval/queryner_baseline_results.json`; `eval/queryner_baseline_comparison.py`; `eval/full_queryner_eval_result.json` | Removed the superiority table and significance claim; reported sample scopes explicitly. |
| Critical | API QueryNER prompts permit only two labels, although the sample contains 13 and the full data contain 17. | Baseline prompt; local QueryNER JSONL | Computed the prompt-compliance oracle ceiling: 0.647810 on the first 50 examples. Of 119 spans, 52 are outside the allowed labels. |
| Critical | Test examples informed checkpoint selection. | `train/train_multitask.py`, first 30 QueryNER test records and all 20 constraint queries at every checkpoint | Reclassified the constraint set as development and disclosed selection exposure in the full QueryNER result. |
| Critical | Serialized synthetic targets differ from what was validated. | `data/generate_scaled_query_training_data.py` writes `scenario.constraints`, after validating a filtered contract | Described the actual implementation. The historical data/model were not silently changed. Retraining is needed after repair. |
| High | Same synthetic query can have conflicting target sets. | New deterministic audit of `data/synthetic_training_text.jsonl` | Reported 374 distinct queries, 76 repeated rows, 8 conflicting groups spanning 20 rows. List-order changes do not count as conflicts. |
| High | Several headline numbers lack local result artifacts. | Narrative claims without corresponding result files | Excluded merged-serving F1 0.683, flagship QueryNER F1 0.204/0.427, and open-model development F1 0.05/0.10 from primary results. |
| High | Five lexical-overlap probes were treated as measured catastrophic forgetting. | Training similarity implementation and retention CSV | Renamed the quantity as base-output similarity; added its exact formula and interpretation limits. |
| High | The checkpoint decision was retrospective and not uniquely optimal. | Eight checkpoint records | Added trajectories, full history, and a Pareto projection; identified six nondominated checkpoints across all three signals. |
| High | Three repetitions of 20 queries were pooled as 60 independent observations. | `eval/rigorous_baseline_comparison.py` | Removed invalid comparative uncertainty claims; specified paired query-cluster inference for future runs. |
| High | Tiny diagnostic set and incomplete gold labels cannot establish broad robustness. | Six query definitions and saved outputs | Rescored every case: mean F1 0.400, micro-F1 0.381, exact-set accuracy 0/6. Discussed omitted negative color and vegan labels. |
| High | No matched backbone/specialist baseline or data/task ablation. | Available evaluation files | Explicit specialist comparison and ablation protocol; no invented numerical rows. |
| Medium | Three tasks are trained, although the original paper called the formulation two-task. | Mixture builder and catalog data | Distinguished three training tasks from two evaluated tasks. |
| Medium | The scaled generator does not invoke the simulator. | Validator call omits `tool_calls` and `catalog` | Removed the claim that every scaled training example passed execution correctness. |
| Medium | Estimated training, generation, and total project costs were conflated. | Run summary and cumulative generation ledger | Verified $0.9021 run arithmetic; reported $1.882587 cumulative generation estimates over 1,630 calls; removed the unsupported total-project ceiling. |
| Medium | 2.13 GiB allocated memory was treated as full GPU memory/minimum hardware. | `torch.cuda.max_memory_allocated` in training script | Named the measured allocator statistic and its exclusions. |
| Medium | Public benchmark F1 was undefined and could be mistaken for official entity-level scoring. | Set scorers | Defined per-query exact-set averaging, text normalization, duplicate collapse, empty-set behavior, and differences from token-indexed evaluation. |
| Medium | References and novelty positioning were incomplete. | Primary QueryNER, EcomGPT, LoRA, QLoRA, Qwen3, and Licardo/Tankovic sources | Added/verified foundational references; removed unsupported concurrency/priority rhetoric and the unverified curation-framework citation. |

## What was actually done in this revision

- Inspected the local manuscript, training code/logs, data, validators, baseline scripts, and result artifacts.
- Rewrote the paper around the supported feasibility study and artifact audit.
- Added formal metrics, two checkpoint visualizations, all six diagnostic cases, provenance tables, and an experimental extension protocol.
- Added `audit_evidence.py`, reproducible offline results, input hashes, and reconstructed seed-42 QueryNER sample IDs.
- Preserved original experiment files. No model was retrained, no API baseline was rerun, and no new human annotation was collected.
- Prepared a public preprint using the official NeurIPS 2026 style in `preprint` mode. This does not imply submission or acceptance at NeurIPS.

## Research still needed for strong conference claims

The supplied publication guide is directionally correct. The order matters:

1. Repair label serialization and validate completeness, including catalog targets. Freeze the schema and resolve inconsistent annotations before training.
2. Establish a clean development/final-test protocol. Historical exposure cannot be undone by renaming files.
3. Build a separately frozen 300–1,000-query robustness benchmark with strata and independent label validation. Synthetic perturbations must be called synthetic stress tests; they are not automatically real out-of-distribution evidence.
4. Run matched QueryNER-style BERT, continued-pretraining BERT, a modern encoder, a small sequence-to-sequence extractor, unfine-tuned Qwen3-1.7B, and appropriately prompted API baselines. Use the same records, full ontology, scorer, and documented adaptation budgets.
5. Run multiple seeds and data/task ablations. “Synthetic only” and “no QueryNER multitask” are the same source condition in the current mixture and must not be presented as independent controls. “Real only” lacks direct constraint supervision and changes task coverage.
6. Measure sustained serving performance at feasible batch sizes 1/8/32/64, with latency percentiles, full memory, utilization, and cost. Use scored general-capability probes for retention claims.

These are experimental requirements, not editorial omissions. Adding empty tables or synthetic performance numbers would not close them.

## Anticipated reviewer questions

| Question | Answer in this revision |
|---|---|
| What is new? | A small, concrete implementation and reproducible analysis of target/evaluation integrity; no new architecture or adaptation algorithm. The significance is bounded by one project. |
| Does it outperform specialist encoders or frontier models? | The current evidence does not establish that. Baseline scope and missing comparisons are disclosed. |
| Is the final evaluation independent? | No: the historical constraint set was used for development and 30 QueryNER test examples informed selection. |
| Are synthetic labels reliable? | Mechanical acceptance is insufficient; eight inconsistent query groups and a writer/validator mismatch are documented. |
| Is checkpoint 750 objectively best? | It is a retrospective nondominated choice, not a unique optimum. |
| Does 0.400 F1 mean reliable execution? | No. All six diagnostic cases lack an exact-set match. |
| Can the reported numbers be reproduced? | New offline analyses and saved six-query predictions can be recomputed. Several historical aggregates cannot be regenerated offline without missing predictions and execution metadata. |
| Is deployment cheaper or faster? | No matched serving study supports such a claim. Only the recorded training cost arithmetic is verified. |

## Venue and upload boundaries

The main text targets a research preprint rather than an accepted conference paper. The official [NeurIPS 2026 handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) requires the year's style, a nine-page main text, anonymization for review, and a paper checklist. A public preprint uses the `preprint` option and may identify authors. The current PDF is public and **not anonymous**. The optional checklist is kept separately for a later conference preparation; it is not an author certification of unknown facts.

The [NeurIPS 2026 call](https://neurips.cc/Conferences/2026/CallForPapers) lists a May 6, 2026 full-paper deadline, which has passed as of this revision. A future conference submission must use that cycle's rules and template. The IR/search venues in the supplied guide are possibilities, not verified open submissions or guarantees of fit; no registration or external submission was performed.

For arXiv, [official submission instructions](https://info.arxiv.org/help/submit/index.html) favor source files and require source for a paper created in LaTeX. The supplied ZIP is the manuscript source, not this review guide. Category suggestions (`cs.IR`, cross-list `cs.CL`) are metadata suggestions; moderation determines acceptance/classification. The author must select a license, review the arXiv-rendered PDF, and complete account/endorsement requirements if requested. No arXiv submission is made by pushing to GitHub.
