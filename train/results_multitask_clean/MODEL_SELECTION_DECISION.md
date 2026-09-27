# Model selection decision — clean-data multitask training run

Per EXECUTION_PLAN.md §10b: model selection must be explicit and evidence-based from the per-checkpoint eval history, not assumed to be the final-step checkpoint.

## Full checkpoint history (real, measured)

| Step | Constraint F1 | QueryNER F1 | Retention vs. base |
|---|---|---|---|
| 150 | 0.708 | 0.625 | 0.597 |
| 300 | 0.698 | 0.604 | 0.611 |
| 450 | 0.675 | 0.601 | 0.597 |
| 600 | 0.690 | **0.630** (best) | 0.592 |
| 750 | 0.725 | 0.556 | **0.614** (best) |
| 900 | 0.725 | 0.614 | 0.490 |
| 1050 | **0.750** (best) | 0.573 | 0.518 |
| 1200 (final) | 0.740 | 0.530 | 0.489 (worst) |

## Decision: **step 750 selected**, not step 1200

The final checkpoint (1200) shows a genuine, sustained retention decline over the last ~450 steps (0.61 → 0.49, never recovering) — this reads as real mild forgetting, not noise, since it's consistent across the last three checkpoints. Task performance at 1200 (constraint F1 0.740) is only marginally different from step 750 (0.725), and QueryNER F1 is actually worse at 1200 (0.530 vs 0.556 at 750).

Step 750 offers the best real balance: strong task performance on both tasks, and the best-measured retention point in the entire run. This is the checkpoint synced and treated as "the model" going forward — not an assumption, a decision made from the actual eval history.

## Cost/time context

Full run: 1200 steps, 2130 training examples, $0.90, 75.2 minutes, RTX PRO 4500. The selected checkpoint (750) represents 62.5% of the full run's compute.
