# 09 — Energy Efficiency (EE) reward: SE vs EE

**Meeting task #2**: "SE는 어느정도 경향성이 있는 문제니 EE랑 비교" — SE has exploitable
structure, so compare against EE.

EE reward (Lee 2025 eq. 32), per-link rate divided by per-link power, summed:

> r_EE(x) = Σ_d log(1 + SINR_d) / (P_fix + P_tx · x_d),  P_fix=1, P_tx=10

Implemented in `problems/ifc.py` (`ifc_rates(..., reward_type="EE")`), search spaces
`IFCEE` (D=3) / `IFCEED10` (D=10). Same 3BA pool (Random, HillClimbing, CMAES) and training
protocol as the SE runs, so SE-vs-EE is apples-to-apples (only the reward changes).

## Why EE is the more interesting problem (structural finding)

From `ee_sanity_output.txt` (brute-force optima, D=3):

| channel | SE optimum x* | EE optimum x* |
|---|---|---|
| 0 | (1, 0, 0) | (0.18, 0, 0) |
| 1 | (0, 1, 0) | (0, 0.22, 0.08) |
| 4 | (0, 0, 1) | (0, 0.04, 0.10) |

- **SE optima sit at the corners** of [0,1]^D (turn links fully on/off) — simple, monotone-ish
  structure that even Random / simple baselines exploit.
- **EE optima are interior and low-power** (raising power helps rate but costs energy and adds
  interference → an interior trade-off). Uniform-random search almost never lands near the
  low-power interior optimum, especially as D grows.

## Optimum reference
- D=3: exhaustive grid (`ifc_optimum_bruteforce`, grid=51) — exact.
- D=10: multi-start projected gradient (`ifc_optimum_multistart`) — near-global, validated
  against brute force at D=3 (agreement ~1e-4).

## Result (200 unseen channels × 300 steps)

| | RIBBO-EE | Random | Optimum | RIBBO % | Random % | RIBBO−Random |
|---|---|---|---|---|---|---|
| **EE D=3**  | 0.559 | 0.480 | 0.580 | **96.5%** | 82.8% | +13.7 pts |
| **EE D=10** | 0.787 | 0.296 | 1.007 | **78.1%** | 29.4% | **+48.7 pts** |

For reference, the SE numbers (3BA, `FINAL_REPORT`): SE D=3 RIBBO beat Random by +5.9 pts,
SE D=10 (3BA) by +4.2 pts.

## Key findings

1. **RIBBO's advantage over Random is far larger on EE than on SE** — +48.7 pts at D=10 EE vs
   +4.2 pts at D=10 SE (same 3BA pool, same everything except the reward).
2. **Reason**: EE's interior/low-power optimum is invisible to uniform Random (only 29.4% of
   optimum at D=10 — it keeps sampling high-power, high-interference points). RIBBO, having
   absorbed the behavior algorithms' search toward the low-power region, navigates there (78.1%).
3. **This is the cleanest "why RIBBO" case for wireless**: on the well-structured SE problem a
   learned optimizer adds modest value (corner solutions are easy), but on the realistic,
   interior-optimum EE problem RIBBO's learned channel/search prior pays off dramatically.
4. At D=3, RIBBO-EE reaches 96.5% of the global optimum — higher than RIBBO-SE D=3 (88.9% of
   WMMSE) — so EE does not hurt RIBBO at low dimension either.

## Update (2026-06-25): EE pool ablation — curation HURTS here

Trained a **4BA-filter** pool (HC, RegEvol, Eagle, CMAES — Random excluded) for EE D=10:

| EE D=10 model | RIBBO % of optimum | Random % |
|---|---|---|
| 3BA (Random, HC, CMAES) | **78.1%** | 29.4% |
| 4BA-filter (HC, RegE, Eagle, CMAES) | 69.7% | 29.4% |

→ Unlike SE, **the curated pool is WORSE for EE** (69.7% < 78.1%, −8.4 pts from dropping Random).
Interpretation: EE's optimum is interior/low-power, and Random's broad coverage of the low-power
region is genuinely useful here, whereas the exploration-heavy evolutionary BAs (Eagle, RegEvol)
chase high-rate (SE-like) regions less suited to EE. This **reconfirms that BA-pool curation is
problem-dependent** (helps SE high-D, hurts EE) — don't apply the SE recipe blindly to EE.
The headline EE result (3BA: 78.1%, +48.7 pts over Random) is unaffected. (`eval_ifc_ee_d10_4ba.*`)

## Artifacts
- `eval_ifc_ee.pdf` / `eval_ifc_ee.npz` — 2-panel (D=3, D=10), RIBBO vs Random vs optimum.
- `eval_ifc_ee_d10_4ba.pdf` / `.npz` — EE D=10 4BA-filter (pool ablation).
- `eval_ifc_ee_budget.pdf` / `.npz` — EE budget extension (D=3, D=10 to 1000 steps).
- `eval_ifc_ee_d3.npz` — earlier D=3-only run (consistent: 96.6%).
- `ee_sanity_output.txt` — SE-corner-vs-EE-interior evidence + solver validation.
- Script: `scripts/eval_ifc_ee.py`, `scripts/sanity_ifc_ee.py`.

## Caveats
- D=3 EE model trained to epoch 2000 (converged: x_loss flat ~−16.2 from ep 1000); the run to
  3000 was cut by the 6/23 system freeze. D=10 EE trained full 3000 epochs.
- D=10 optimum is a strong near-global reference (multi-start gradient), not a certified global
  optimum (brute force is infeasible at D=10); validated against brute force at D=3.
