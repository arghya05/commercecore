"""Recompute manuscript audits from local artifacts; no inference or API calls.

Run from any directory: python3 /path/to/commercecore/paper/audit_evidence.py
Historical measurements are preserved. New analyses are explicitly identified.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "evidence"
INPUTS: set[str] = set()


def read(path):
    INPUTS.add(path)
    return (ROOT / path).read_text()


def jsonl(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def constant(path, name):
    for node in ast.parse(read(path)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing constant {name} in {path}")


def normalize(text):
    return " ".join(text.casefold().split())


def triples(items):
    return frozenset((x["field"], x["op"], str(x["value"])) for x in items)


def set_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    return 2 * len(pred & gold) / (len(pred) + len(gold))


def spans(record):
    return frozenset(
        (span["label"], " ".join(record["tokens"][span["start_token"]:span["end_token"]]).lower().strip())
        for span in record["spans"]
    )


def main():
    OUT.mkdir(exist_ok=True)
    train = jsonl("data/queryner/train.jsonl")
    test = jsonl("data/queryner/test.jsonl")
    synthetic = jsonl("data/synthetic_training_text.jsonl")
    catalog = jsonl("data/catalog_training_text.jsonl")
    development = jsonl("data/held_out_eval_set.jsonl")
    rng = random.Random(42)
    sample = list(train)
    rng.shuffle(sample)
    sample = sample[:1500]
    archive_root = "paper/evidence/archive/"
    archive_manifest = json.loads(read(archive_root + "manifest.json"))
    for name, metadata in archive_manifest["files"].items():
        read(archive_root + name)
        assert hashlib.sha256((ROOT / archive_root / name).read_bytes()).hexdigest() == metadata["sha256"]
    archived_mixture = jsonl(archive_root + "multitask_train_mixture.jsonl")
    reconstructed_entities = [{
        "prompt": f"Extract entities from this query: {r['text_reconstructed']}",
        "completion": "\nEntities: " + json.dumps([
            {"label": s["label"], "text": " ".join(r["tokens"][s["start_token"]:s["end_token"]])}
            for s in r.get("spans", [])])} for r in sample]
    reconstructed_mixture = [{k: r[k] for k in ("prompt", "completion")}
                             for r in reconstructed_entities + synthetic + catalog]
    rng.shuffle(reconstructed_mixture)
    assert archived_mixture == reconstructed_mixture

    served = json.loads(read(archive_root + "complete_inference_test_results.json"))
    assert len(served["results"]) == served["n_examples"] == len(development)
    served_scores, served_verticals = [], defaultdict(list)
    served_tp = served_npred = served_ngold = served_exact = 0
    for row, expected in zip(served["results"], development):
        assert row["query"] == expected["query"] and row["vertical"] == expected["vertical"]
        pred, gold = triples(row["predicted"]), triples(row["truth"])
        assert gold == triples(expected["constraints"])
        score = set_f1(pred, gold)
        assert math.isclose(score, row["f1"], abs_tol=1e-12)
        served_scores.append(score); served_verticals[row["vertical"]].append(score)
        served_tp += len(pred & gold); served_npred += len(pred); served_ngold += len(gold)
        served_exact += pred == gold
    served_mean = sum(served_scores) / len(served_scores)
    assert math.isclose(served_mean, served["overall_f1"], abs_tol=1e-12)
    for vertical, scores in served_verticals.items():
        assert len(scores) == served["per_vertical_n"][vertical]
        assert math.isclose(sum(scores) / len(scores), served["per_vertical_f1"][vertical], abs_tol=1e-12)
    open_development = json.loads(read(archive_root + "open_model_comparison_results.json"))
    syntax_check = json.loads(read(archive_root + "full_inference_test_results.json"))
    assert len(syntax_check["results"]) == syntax_check["n_queries_tested"] == 8
    assert all(json.loads(r["raw_output"]) == r["parsed"] for r in syntax_check["results"])
    input_prompts = {normalize(x["text_reconstructed"]) for x in train}
    sampled_prompts = {normalize(x["text_reconstructed"]) for x in sample}
    overlap = [x["id"] for x in test if normalize(x["text_reconstructed"]) in input_prompts]
    sampled_overlap = [x["id"] for x in test if normalize(x["text_reconstructed"]) in sampled_prompts]

    allowed = {"core_product_type", "modifier"}
    label_audits = {}
    for n in (30, 50, len(test)):
        records = test[:n]
        counts = Counter(s["label"] for r in records for s in r["spans"])
        gold = [spans(r) for r in records]
        label_audits[str(n)] = {
            "n_records": n, "label_counts": dict(sorted(counts.items())),
            "n_labels": len(counts), "n_gold_spans": sum(counts.values()),
            "n_gold_spans_outside_baseline_prompt": sum(v for k, v in counts.items() if k not in allowed),
            "oracle_macro_f1_if_prompt_labels_obeyed": sum(
                set_f1(frozenset(s for s in g if s[0] in allowed), g) for g in gold
            ) / n,
        }

    groups = defaultdict(list)
    target_counts = Counter()
    for i, record in enumerate(synthetic, 1):
        query = record["prompt"].split(": ", 1)[1]
        target = json.loads(record["completion"].split(": ", 1)[1])
        groups[normalize(query)].append((i, triples(target)))
        target_counts[len(target)] += 1
    conflicts = []
    for query, entries in sorted(groups.items()):
        targets = {target for _, target in entries}
        if len(targets) > 1:
            conflicts.append({"query": query, "rows_1based": [i for i, _ in entries],
                              "distinct_target_sets": [sorted(t) for t in sorted(targets, key=lambda t: sorted(t))]})
    phrases = constant("validators/mechanical_validator.py", "GENERIC_TEMPLATE_PHRASES")
    blocked = [i for i, r in enumerate(synthetic, 1) if any(p in r["prompt"].lower() for p in phrases)]
    development_overlap = [r["query"] for r in development if normalize(r["query"]) in groups]
    fresh_queries = constant("eval/fresh_head_to_head_comparison.py", "FRESH_QUERIES")
    fresh_overlap = [r["query"] for r in fresh_queries if normalize(r["query"]) in groups]
    fresh_development_overlap = [r["query"] for r in fresh_queries if normalize(r["query"]) in {normalize(x["query"]) for x in development}]
    gpu = json.loads(read("eval/fresh_headtohead_gpu_results.json"))
    fresh_metrics = {}
    for name, result in gpu["results"].items():
        rows = result["details"]
        assert len(rows) == len(fresh_queries)
        pred, gold, scores = [], [], []
        for row, expected in zip(rows, fresh_queries):
            assert row["query"] == expected["query"]
            p, g = triples(row["predicted"]), triples(row["truth"])
            assert g == triples(expected["constraints"])
            score = set_f1(p, g)
            assert math.isclose(score, row["f1"], abs_tol=1e-12)
            pred.append(p); gold.append(g); scores.append(score)
        macro = sum(scores) / len(scores)
        assert math.isclose(macro, result["mean_f1"], abs_tol=1e-12)
        tp = sum(len(p & g) for p, g in zip(pred, gold))
        npred, ngold = sum(map(len, pred)), sum(map(len, gold))
        fresh_metrics[name] = {
            "n": len(rows), "macro_f1": macro, "per_query_f1": scores,
            "micro_precision": tp / npred if npred else 0,
            "micro_recall": tp / ngold if ngold else 0,
            "micro_f1": 2 * tp / (npred + ngold) if npred + ngold else 1,
            "exact_set_matches": sum(p == g for p, g in zip(pred, gold)),
            "true_positives": tp, "n_predicted": npred, "n_gold": ngold,
            "empty_parsed_outputs": sum(not p for p in pred),
        }

    checkpoints = list(csv.DictReader(read("train/results_multitask_clean/multitask_checkpoint_eval_log.csv").splitlines()))
    retention = {r["step"]: float(r["retention_similarity_vs_base"]) for r in csv.DictReader(read("train/results_multitask_clean/multitask_retention_log.csv").splitlines())}
    history = [{"step": int(r["step"]), "constraint_f1": float(r["constraint_task_f1"]),
                "queryner_f1": float(r["queryner_task_f1"]), "output_similarity": retention[r["step"]]}
               for r in checkpoints]
    pareto = [r["step"] for r in history if not any(
        all(other[k] >= r[k] for k in ("constraint_f1", "queryner_f1", "output_similarity")) and
        any(other[k] > r[k] for k in ("constraint_f1", "queryner_f1", "output_similarity")) for other in history
    )]
    run = json.loads(read("train/results_multitask_clean/multitask_run_summary.json"))
    loss_path = "train/results_multitask_clean/multitask_training_loss_log.csv"
    loss_rows = list(csv.DictReader(read(loss_path).splitlines()))
    losses = [{"step": int(r["step"]), "loss": float(r["loss"]),
               "elapsed_seconds": float(r["elapsed_seconds"])} for r in loss_rows]
    assert [r["step"] for r in losses] == list(range(1, run["n_steps"] + 1))
    assert all(math.isfinite(r["loss"]) and r["loss"] > 0 for r in losses)
    assert all(a["elapsed_seconds"] < b["elapsed_seconds"] for a, b in zip(losses, losses[1:]))
    accumulation = constant("train/train_multitask.py", "GRAD_ACCUM")
    steps_per_pass = run["n_training_examples"] / accumulation
    boundaries = [math.floor(k * steps_per_pass)
                  for k in range(1, math.floor(len(losses) / steps_per_pass) + 1)]
    if boundaries[-1] != len(losses):
        boundaries.append(len(losses))
    loss_windows = []
    start = 1
    for end in boundaries:
        window = losses[start - 1:end]
        loss_windows.append({"start_step": start, "end_step": end, "n_logged_steps": len(window),
                             "approximate_passes_at_end": end / steps_per_pass,
                             "mean_logged_loss": sum(r["loss"] for r in window) / len(window),
                             "end_step_loss": window[-1]["loss"]})
        start = end + 1
    loss_curve = [{**r, "trailing_50_mean": sum(x["loss"] for x in losses[max(0, i-49):i+1]) / min(i+1, 50)}
                  for i, r in enumerate(losses)]
    for row in history:
        loss = losses[row["step"] - 1]
        row.update({"training_loss": loss["loss"], "elapsed_seconds": loss["elapsed_seconds"],
                    "approximate_passes": row["step"] / steps_per_pass})
    ledger = json.loads(read("data/generation_cost_ledger.json"))
    ledger_recomputed = sum(x["cost_usd"] for x in ledger["calls"])
    assert math.isclose(ledger_recomputed, ledger["total_cost_usd"], abs_tol=1e-6)
    assert ledger["n_calls"] == len(ledger["calls"])
    full = json.loads(read("eval/full_queryner_eval_result.json"))
    assert full["n_examples"] == len(test)
    baseline = json.loads(read("eval/queryner_baseline_results.json"))
    assert baseline["n_sample"] == constant("eval/queryner_baseline_comparison.py", "N_EVAL_SAMPLE")
    # The historical numeric fields below remain stored measurements, not reruns.
    for path in ("train/train_multitask.py", "train/build_multitask_mixture.py", "data/generate_scaled_query_training_data.py", "eval/full_queryner_eval.py", "eval/rigorous_baseline_comparison.py", "eval/rigorous_baseline_results.json", "eval/fresh_headtohead_frontier_results.json"):
        read(path)
    report = {
        "analysis_type": "Offline reanalysis of existing artifacts; no new training or model inference",
        "archived_artifacts": {"manifest": archive_manifest, "mixture_n_rows": len(archived_mixture),
                               "mixture_matches_reconstructed_ordered_records": True,
                               "open_model_development_aggregates": open_development,
                               "syntax_check_n_valid_json_recomputed": len(syntax_check["results"])},
        "normalization": "Unicode casefold and whitespace collapse; no punctuation removal or semantic deduplication",
        "queryner": {"n_train_available": len(train), "n_train_sample_reconstructed": len(sample), "n_test": len(test),
                     "source_revisions": sorted({r["source_revision"] for r in train + test}),
                     "train_test_normalized_overlap_ids": overlap, "sample_test_overlap_ids": sampled_overlap,
                     "label_audits": label_audits,
                     "historical_full_result": full, "historical_baseline_results": baseline,
                     "selection_exposed_test_rows": constant("train/train_multitask.py", "N_QUERYNER_EVAL_SAMPLE")},
        "synthetic": {"n_query_rows": len(synthetic), "n_unique_normalized_queries": len(groups),
                      "n_repeated_rows_beyond_first": len(synthetic) - len(groups),
                      "n_conflicting_query_groups": len(conflicts),
                      "n_rows_in_conflicting_groups": sum(len(c["rows_1based"]) for c in conflicts),
                      "conflicts": conflicts, "target_length_counts": dict(sorted(target_counts.items())),
                      "blocklist_hit_rows_1based": blocked,
                      "query_vertical_counts": dict(Counter(r["vertical"] for r in synthetic)),
                      "catalog_vertical_counts": dict(Counter(r["vertical"] for r in catalog)),
                      "n_catalog_rows": len(catalog)},
        "constraints": {"n_development": len(development), "vertical_counts": dict(Counter(r["vertical"] for r in development)),
                        "empty_gold_development": sum(not r["constraints"] for r in development),
                        "development_training_exact_overlap": development_overlap,
                        "fresh_training_exact_overlap": fresh_overlap, "fresh_development_exact_overlap": fresh_development_overlap,
                        "fresh_recomputed": fresh_metrics,
                        "served_development_recomputed": {"n_queries": len(served_scores), "macro_f1": served_mean,
                            "per_query_f1": served_scores,
                            "per_vertical_f1": {v: sum(s)/len(s) for v, s in served_verticals.items()},
                            "per_vertical_n": {v: len(s) for v, s in served_verticals.items()},
                            "true_positives": served_tp, "n_predicted": served_npred, "n_gold": served_ngold,
                            "micro_f1": 2*served_tp/(served_npred+served_ngold), "exact_set_matches": served_exact}},
        "training": {"historical_run_summary": run, "checkpoint_history": history, "pareto_steps_on_three_logged_signals": pareto,
                     "loss_log": {"source": loss_path, "n_logged_steps": len(losses),
                                  "first_logged_loss": losses[0]["loss"], "last_logged_loss": losses[-1]["loss"],
                                  "meaning": "Mean completion-token loss over eight accumulated microbatches at each optimizer step; logged to four decimals",
                                  "window_rule": "Boundaries floor(k * n_training_examples / gradient_accumulation); final window is partial",
                                  "approximate_pass_windows": loss_windows,
                                  "curve_smoothing": "Trailing arithmetic mean over at most 50 logged optimizer steps"},
                     "approximate_passes_full_run": run["n_steps"] * 8 / run["n_training_examples"],
                     "approximate_passes_selected_checkpoint": 750 * 8 / run["n_training_examples"],
                     "historical_gpu_rate_usd_per_hour": 0.72,
                     "recomputed_run_cost_usd": run["total_time_seconds"] / 3600 * 0.72},
        "generation_ledger": {"n_calls": ledger["n_calls"], "historical_estimated_total_usd": ledger["total_cost_usd"],
                              "recomputed_sum_usd": round(ledger_recomputed, 6),
                              "scope": "cumulative recorded generation calls; not total project cost or invoices"},
        "inputs_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(INPUTS)},
    }
    (OUT / "audit_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "reconstructed_queryner_train_ids.txt").write_text("\n".join(r["id"] for r in sample) + "\n")
    with (OUT / "checkpoint_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(history)
    # Preserve the complete original loss CSV, and derive the plot and tables from it.
    (OUT / "training_loss_history.csv").write_bytes((ROOT / loss_path).read_bytes())
    with (OUT / "training_loss_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(loss_curve[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(loss_curve)
    (OUT / "checkpoint_rows.tex").write_text(
        "% Generated by audit_evidence.py; losses and elapsed times are from the exact checkpoint step.\n" +
        "\n".join(f"{r['step']:,} & {r['approximate_passes']:.2f} & {r['training_loss']:.4f} & "
                  f"{r['constraint_f1']:.4f} & {r['queryner_f1']:.4f} & {r['output_similarity']:.4f} & "
                  f"{r['elapsed_seconds']/60:.2f} " + r"\\" for r in history) + "\n")
    (OUT / "loss_window_rows.tex").write_text(
        "% Generated by audit_evidence.py; approximate pass windows, not exact epoch averages.\n" +
        "\n".join(f"{r['start_step']:,}--{r['end_step']:,} & {r['approximate_passes_at_end']:.2f} & "
                  f"{r['n_logged_steps']} & {r['mean_logged_loss']:.4f} & {r['end_step_loss']:.4f} " + r"\\"
                  for r in loss_windows) + "\n")
    vals = {
        "QueryNERTrainN": str(len(train)), "QueryNERTestN": str(len(test)),
        "SyntheticUniqueN": str(len(groups)), "SyntheticDuplicateN": str(len(synthetic)-len(groups)),
        "SyntheticConflictN": str(len(conflicts)), "SyntheticConflictRowsN": str(sum(len(c["rows_1based"]) for c in conflicts)),
        "RestrictedPromptCeiling": f"{label_audits['50']['oracle_macro_f1_if_prompt_labels_obeyed']:.3f}",
        "FreshMicroF": f"{fresh_metrics['commercecore']['micro_f1']:.3f}",
        "LedgerCost": f"{ledger['total_cost_usd']:.2f}",
    }
    (OUT / "numbers.tex").write_text("% Generated by audit_evidence.py; do not edit.\n" + "\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in vals.items()) + "\n")
    print(json.dumps({"status": "PASS", "synthetic_unique": len(groups), "conflicting_groups": len(conflicts),
                      "conflicting_rows": sum(len(c["rows_1based"]) for c in conflicts), "pareto_steps": pareto,
                      "train_test_exact_overlaps": len(overlap), "fresh_macro_f1": fresh_metrics['commercecore']['macro_f1'],
                      "outputs": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
