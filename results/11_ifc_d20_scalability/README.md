# 11 — Dimensional scalability: D=3 → D=10 → D=20 (IFC SE)

**Meeting task #4**: "차원/문제에 대한 scalability를 어떻게 가져가볼지" — how to take RIBBO's
scalability forward in dimension/problem.

D=20 SE trained with the same recipe as D=3/D=10 (3BA pool = Random+HillClimbing+CMAES,
20 channels × 300 steps = 60 trajectories, 3000 epochs) and evaluated vs WMMSE on 200 unseen
channels × 300 steps. Script: `scripts/eval_ifc_d20.py` (config `scripts/configs/dt/ifc_d20.py`).

## The dimension sweep (SE, 3BA pool)

| D | WMMSE (nats) | RIBBO (nats) | RIBBO % of WMMSE | Random % | RIBBO − Random |
|---|---|---|---|---|---|
| 3  | 2.864 | 2.547 | **88.9%** | 83.0% | +5.9 pts |
| 10 | 3.968 | 1.898 | **47.8%** | 43.6% | +4.2 pts |
| 20 | 4.764 | 1.574 | **33.0%** | 31.5% | +1.5 pts |

## Findings

1. **With a fixed small recipe, RIBBO's share of WMMSE falls monotonically with dimension**
   (88.9% → 47.8% → 33.0%) and its edge over Random collapses (+5.9 → +4.2 → +1.5 pts). At D=20
   the plain 3BA model is barely above Random — 60 trajectories cannot cover [0,1]^20.

2. **This is a recipe floor, not RIBBO's ceiling.** The levers that recover high-D performance
   were already demonstrated:
   - **Curated BA pool**: at D=10, 5BA-filter lifts RIBBO 47.8% → **71.4%** (+23.6 pts vs 3BA),
     and to +27.8 pts over Random (`results/07`).
   - **More evaluation budget**: extending 300 → 1000 steps adds **+8.1 pts** at D=10 (and the
     benefit grows with dimension) (`results/08`).

3. **Scalability strategy (the answer to task #4)**: dimension scaling is achievable, but the
   *recipe must scale with D* — you cannot keep the D=3 recipe and expect it to hold. Concretely,
   as D grows: (a) widen + curate the BA pool (well-performing AND diverse, drop Random/SGS),
   (b) generate more trajectories / channels, (c) give more inference budget. The D=20 3BA point
   here is the "do-nothing" baseline that motivates each lever.

4. **Problem axis matters too**: the EE reward (`results/09`) shows RIBBO's value is far larger on
   an interior-optimum problem than on corner-solution SE (+48.7 pts over Random at D=10 EE). So
   "scalability" is not only dimension — the *kind* of problem decides how much a learned optimizer
   helps. RIBBO scales best toward harder, non-corner objectives (which is where wireless EE lives).

## Update (2026-06-25): the recovery lever works at D=20

Trained a **4BA-filter** pool (HillClimbing, RegularizedEvolution, EagleStrategy, CMAES —
Random/ShuffledGridSearch excluded; BotorchBO dropped for D=20 GP cost) at D=20 and re-evaluated:

| D=20 model | RIBBO % of WMMSE | Random % | RIBBO − Random |
|---|---|---|---|
| 3BA (Random, HC, CMAES) | 33.0% | 31.5% | +1.5 pts |
| **4BA-filter (HC, RegE, Eagle, CMAES)** | **41.9%** | 31.5% | **+10.4 pts** |

→ The curated pool lifts RIBBO **33.0% → 41.9% (+8.9 pts)** at D=20 and restores a clear margin
over Random (+1.5 → +10.4 pts). So the "curate the BA pool as dimension grows" lever — first shown
at D=10 (5BA-filter 47.8%→71.4%) — **generalizes to D=20**. (`eval_ifc_d20_4ba.pdf/.npz`,
model `log/ifc/dt-ifc-d20-4ba`.) It does not fully close the WMMSE gap (still 41.9%), consistent
with also needing more data + budget as D grows.

## Artifacts
- `eval_ifc_d20.pdf` / `eval_ifc_d20.npz` — RIBBO 3BA vs Random vs WMMSE, D=20.
- `eval_ifc_d20_4ba.pdf` / `eval_ifc_d20_4ba.npz` — RIBBO 4BA-filter vs Random vs WMMSE, D=20.
- D=20 model: `log/ifc/dt-ifc-d20-3ba/.../ckpt/3000.ckpt` (converged, x_loss ≈ −85, slope ~0).
- Trained crash-free after the GSP-firmware fix (full 2h37m run; see PROGRESS 2026-06-25).
