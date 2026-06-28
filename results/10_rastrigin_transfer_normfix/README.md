# 10 — Rastrigin → IFC transfer: is RTG normalization the cause of failure?

**Meeting task**: "Rastrigin으로 학습된 모델을 IFC로 테스트 (차원 맞춰서)" — test a
Rastrigin-trained RIBBO model on IFC at matched dimension.

Earlier (`results/06`, `FINAL_REPORT` finding 4) the Rastrigin(D=10)→IFC(D=10) zero-shot
transfer failed (31.8% of WMMSE, below Random) and we *hypothesized* the cause was the
RTG-normalization mismatch (Rastrigin y∈[-3105,-109] vs IFC y∈[0.4,3.0], so IFC rewards
look "already optimal" under Rastrigin's scale and exploration is suppressed).

This experiment **tests that hypothesis directly** by feeding the same Rastrigin model the
IFC-correct normalization stats during rollout.

## Setup (IFC SE D=10, dimension-matched, 150 unseen channels × 150 steps)
- A. Rastrigin model + **Rastrigin** RTG stats (original mismatch — reproduces the failure)
- B. Rastrigin model + **IFC** RTG stats (the normalization "fix")
- C. Native IFC D=10 3BA model (reference)
- D. Random; + WMMSE band
- Script: `scripts/eval_ifc_rastrigin_normfix.py`

## Result

| Variant | Final SE (nats) | % of WMMSE |
|---|---|---|
| A. Rastrigin model + Rastrigin RTG (mismatch) | 1.267 | 31.8% |
| B. Rastrigin model + **IFC RTG (norm. fix)** | 1.241 | **31.2%** |
| C. Native IFC D=10 (3BA, reference) | 1.790 | 44.9% |
| D. Random | 1.672 | 42.0% |
| WMMSE | 3.983 | 100% |

## Finding (this overturns the earlier hypothesis)
- **Fixing RTG normalization does NOT rescue the transfer** (31.8% → 31.2%, unchanged, and
  both *below Random's 42%*). Normalization is **not** the bottleneck.
- The real conclusion: the Rastrigin-trained model **did not learn transferable black-box
  optimization behavior for IFC**, even at matched dimension and with corrected RTG scale.
  The policy it learned is specific to the Rastrigin landscape (additive, separable,
  many-local-minima) and does not carry over to IFC's interference-coupled sum-rate surface.
- **Implication for "why RIBBO" / scalability**: the channel-distribution prior that makes
  RIBBO useful for IFC must be learned from **in-domain IFC trajectories**. It cannot be
  borrowed for free from a generic synthetic benchmark — cross-problem zero-shot transfer
  fails here. (This is a cleaner, stronger statement than the earlier "normalization" story.)
