# 02 — 첫 IFC SE pipeline (3BA, 5/18 발표 base)

**목적**: Lee et al. 2025 Section V-A IFC SE 문제에 RIBBO 적용. 5/18 advisor 미팅 발표 기준 시점.

## Setup

- Problem: IFC SE, D=3, P_tx=10W, h_{d'd}~CN(0,1), reward r_SE in nats
- Behavior algorithms: **3** (Random, HillClimbing, CMAES)
- Train: 20 channels × 3 BA × 300 step = 60 trajectories
- Test: unseen channels (20-29 in some plots, 30-229 in others)
- Model: causal Transformer (paper default), 3000 epoch
- ckpt: `log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-12-50-15740/ckpt/3000.ckpt`

## Result (200 channels × 1 ep × 300 step, eval_ifc_matched 시점)

| Method | Final SE (nats) | bits |
|---|---|---|
| RIBBO best-so-far | 2.67 | 3.85 |
| Random best-so-far | 2.44 | 3.52 |

RIBBO +9.5% over random.

## Figures

- `convergence_ifc.pdf` — IFC SE 학습 시 x_loss 곡선 (3000 ep)
- `eval_ifc_matched.pdf` — 200 ch × 1 ep × 300 step, Lee et al. Fig 7(b) match setup
- `eval_ifc_ribbo_vs_baselines.pdf` — 50 ch × 3 ep × 300 step, Lee et al. reference 수평선들
- `eval_ifc_ribbo_vs_random.pdf` — 더 간소화된 RIBBO vs Random plot

## 한계

- Best-so-far만 표시 (instantaneous 미포함 → 03에서 추가)
- WMMSE 직접 구현 없음 (Lee 논문 수치만 hardcode → 04에서 추가)
- 3BA만 사용 — Song 2024 논문은 7BA → 05에서 확장
