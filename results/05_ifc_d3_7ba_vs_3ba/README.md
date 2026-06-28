# 05 — 7BA vs 3BA: BA pool diversity ablation (D=3 IFC SE)

**목적**: RIBBO 논문(Song 2024)이 명시한 7개 behavior algorithm을 사용해 학습하면 3BA 대비 성능이 향상되는지 확인. Advisor 미팅 next steps 1번.

## Setup

- Problem: 같은 D=3 IFC SE (04와 동일)
- **3BA** (control): Random, HillClimbing, CMAES — 60 trajectories
- **7BA** (treatment): Random, ShuffledGridSearch, RegularizedEvolution, HillClimbing, EagleStrategy, CMAES, BotorchBO — 140 trajectories
  - 원래 8BA(+Vizier)로 계획했으나 Vizier가 JAX/XLA OOM segfault로 미생성 → 7BA 확정
- Training: 동일 hyperparameter, 동일 epoch 비교 (3000) — but 7BA는 미수렴
- Eval: 500 unseen channels × 1 ep × 300 step (04와 같은 channel ID 사용 위해 매칭)

### ckpt
- 3BA: `log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-12-50-15740/ckpt/3000.ckpt`
- 7BA (3000 ep): `log/ifc/dt-ifc-run2-fullba/IFCSE-seed0-05-21-18-26-1352617/ckpt/3000.ckpt`
- 7BA (5000 ep): TBD — 재학습 진행 중

## 수렴 분석 (7BA 3000 epoch)

| Epoch 구간 | mean x_loss | std |
|---|---|---|
| 1-100 | -5.49 | 2.82 |
| 500-1000 | -14.16 | 1.49 |
| 1000-1500 | -15.43 | 1.09 |
| 1500-2000 | -16.51 | 1.02 |
| 2000-2500 | -17.33 | 1.04 |
| 2500-3000 | -17.78 | 0.99 |

- Last 1000 epoch slope = **-0.0009/epoch** — 여전히 단조 감소
- 3BA는 같은 hyperparameter로 epoch 500쯤에 plateau (수렴) 도달했음 → 7BA는 BA 다양성 ↑로 학습 어려워서 수렴 느림 (자연스러움)

## 비교 기준 — Same-epoch vs Converge

advisor 토의 결과 **converge 기준이 본 case에 더 fair**. 이유:
- 비교 핵심이 "BA pool 자체가 더 좋은 모델 만드나"
- 동일 epoch로 자르면 7BA가 underfit → 다양성 효과를 underfitting이 가림
- 학습 어려움 ↑는 BA 다양성의 자연스러운 부산물이지 단점 아님

다만 둘 다 보여주는 게 reporting 시 정직:
- **Main**: 7BA 5000 ep (converge) vs 3BA 3000 ep (converge)
- **Supplementary**: 7BA 3000 ep (미수렴) vs 3BA 3000 ep — "7BA가 학습 더 어렵다"는 finding 자체

## Result — Same-epoch (3000 ep), 7BA 미수렴

500 unseen channels × 1 ep × 300 step

| Method | Final SE (nats) | bits | % of WMMSE |
|---|---|---|---|
| WMMSE (ref) | **2.864** | 4.132 | 100% |
| RIBBO 3BA (3000 ep, conv) | 2.519 | 3.634 | **87.9%** |
| Random | 2.379 | 3.432 | 83.0% |
| RIBBO 7BA (3000 ep, **NOT conv**) | 2.281 | 3.291 | **79.6%** |
| WMMSE std | ±0.685 | | |

**관찰**: 7BA 3000 epoch는 미수렴 상태라 **Random보다도 못함** (2.28 < 2.38). 같은 epoch로 비교하면 BA 다양성 ↑이 오히려 손해처럼 보임 — 학습 어려움 미해결. → 5000 epoch에서 converge-based 비교 필요.

## Result — Converge-based (3BA 3000 ep vs 7BA 5000 ep)

500 unseen channels × 1 ep × 300 step. 7BA 5000ep 수렴 확인: epoch 4500-5000 평균 -18.32, last 1000ep slope -0.00016 (≈0)

| Method | Final SE (nats) | bits | % of WMMSE |
|---|---|---|---|
| WMMSE (ref) | **2.864** | 4.132 | 100% |
| **RIBBO 3BA (3000 ep, conv)** | 2.513 | 3.626 | **87.7%** |
| Random | 2.379 | 3.432 | 83.0% |
| **RIBBO 7BA (5000 ep, conv)** | 2.187 | 3.156 | **76.4%** |
| WMMSE std | ±0.685 | | |

## 통합 비교 (3 시나리오)

| 비교 | 3BA 점수 | 7BA 점수 | 결론 |
|---|---|---|---|
| Same-epoch (3000) | 2.519 | 2.281 (미수렴) | 7BA -9.4% |
| Converge | 2.513 | 2.187 (5000ep 수렴) | 7BA -13.0% |

**놀라운 발견**: 7BA가 수렴됐는데도 3BA보다 **나쁘다**, Random에도 진다 (2.19 < 2.38).

## 해석 (가설)

1. **BA 다양성이 RTG distribution을 노이즈화**: BotorchBO·EagleStrategy·ShuffledGridSearch 같은 매우 다른 exploration profile이 hindsight regret distribution을 어지럽혀 in-context learning을 어렵게 함
2. **Trajectory 다양성 ≠ 학습 유익**: 3BA(Random, HillClimbing, CMAES)는 비교적 일관된 step-by-step improvement trajectory를 보여 RIBBO가 imitate하기 좋음. 7BA에는 BotorchBO처럼 갑작스러운 exploit이 섞임
3. **데이터 양 부족 가설**: 7BA × 20 ch = 140 trajectory도 7개 알고리즘의 diversity 커버하기엔 부족할 수 있음 (Song 2024는 보통 더 큰 N)
4. **Random보다도 진다**는 점: RIBBO가 학습한 policy가 actively bad → 잘못된 RTG conditioning 가능성

## Figures

- `eval_ifc_3ba_vs_7ba_3000ep.pdf` — Same-epoch 비교 (3000 ep, 7BA 미수렴). 7BA 2.28 vs 3BA 2.52
- `eval_ifc_3ba_vs_7ba_converged.pdf` — Converge-based 비교 (3BA 3000 / 7BA 5000). 7BA 2.19 vs 3BA 2.51 (**메인**)
- raw data: `eval_ifc_3ba_vs_7ba_3000ep.npz`, `eval_ifc_3ba_vs_7ba_converged.npz`

## advisor 미팅 질문 후보

- "BA 다양성을 늘리면 더 좋아질 것"은 IFC setup에서 성립하지 않았다. 이게 IFC 문제의 특성인가 (corner solution 특성), 아니면 데이터 양/학습 부족인가?
- 만약 BA pool을 잘 골라야 한다면 selection 기준은? (Song 2024는 분석 미제공)
- Channel 수를 늘려야 하는가 (20 → 100 또는 그 이상)
- RTG normalization을 BA별로 따로 할지 (현재는 dataset-wide)


## Figures

- `eval_ifc_3ba_vs_7ba_3000ep.pdf` — Same-epoch 비교 (3000 ep). 두 모델 모두 같은 channel, 같은 eval setup
- `eval_ifc_3ba_vs_7ba_converged.pdf` — Converge-based 비교 (TBD)
- raw data: `eval_ifc_3ba_vs_7ba_3000ep.npz`

## Scripts

- `scripts/eval_ifc_3ba_vs_7ba.py` — 두 모델 동시 평가 + WMMSE overlay
- 데이터 생성:
  - 3BA 데이터는 02/04와 동일
  - 7BA 추가 데이터: `run_datagen_ifc_botorch_vizier.sh` (BotorchBO만 성공, Vizier OOM)
