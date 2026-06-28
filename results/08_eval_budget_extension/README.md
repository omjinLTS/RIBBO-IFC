# 08 — Evaluation budget extension (does more budget close the WMMSE gap?)

**Meeting task**: "차원/evaluation 늘렸을 때 WMMSE랑 붙는지" (when dimension / evaluation
budget grows, does RIBBO catch up to WMMSE?).

RIBBO was trained with a 300-step optimization horizon. Here we roll the trained models
out far beyond that (1000 steps) on unseen channels and overlay WMMSE. Best-so-far is
monotone, so the question is the **asymptote**: does the gap to WMMSE keep shrinking?

## Setup
- Models: best SE model per dimension — D=3 3BA (`dt-ifc-run1/3000.ckpt`), D=10 5BA-filter
  (`dt-ifc-d10-run2b-5bafilter-resume/5000.ckpt`).
- 100 unseen channels × 1000 steps (vs 300 training horizon), stochastic policy.
- Reference: WMMSE per channel (Shi 2011, 10 restarts). Random baseline at the same budget.
- Script: `scripts/eval_ifc_budget.py` → `eval_ifc_budget.pdf`, `eval_ifc_budget.npz`.

## Result

| Setting | @300 steps | @1000 steps | gain |
|---|---|---|---|
| D=3 3BA | 87.0% of WMMSE | **91.7%** | +4.8 pts |
| D=10 5BA-filter | 72.9% of WMMSE | **81.0%** | +8.1 pts |

## Findings
1. **More evaluation budget DOES narrow the WMMSE gap** — RIBBO keeps improving past its
   300-step training horizon (best-so-far creeps up as the stochastic policy keeps exploring).
   It does not hard-plateau at 300.
2. **The higher dimension benefits more** (+8.1 pts at D=10 vs +4.8 pts at D=3). This is
   consistent with "high-dim search needs more function evaluations": at D=10 the 300-step
   budget was leaving more on the table.
3. Practical reading for the meeting: a cheap way to recover performance at deployment is
   simply to let RIBBO run longer at inference (it is cheap — no LLM API per step). The gap
   that remains (~8–18%) is the part that needs better training, not more test-time budget.

Note: absolute % at @300 here (87.0 / 72.9) matches the established eval pipeline numbers
(88.9% D=3, 71.4% D=10 in `results/FINAL_REPORT.md`) up to channel-sample and stochastic
-rollout variation (this run uses 100 channels; the headline used 200–500).
