# 07 — BA-pool × Dimension Ablation (Song 2024 권고 검증)

**목적**: 5/22 7BA D=3 실험에서 발견한 negative finding (BA 다양성 ↑ → 성능 ↓)을 Song 2024 논문의 권고 ("well-performing AND diverse BAs") 관점에서 재검증. Random + ShuffledGridSearch (underperforming) 제외한 **5BA-filter**가 BC Filter 패턴.

## Setup

- Problem: IFC SE, D ∈ {3, 10}
- 3가지 BA pool 비교:
  - **3BA**: Random, HillClimbing, CMAES (기존 base)
  - **5BA-filter**: HC + RegEvol + Eagle + CMAES + BotorchBO (Random + SGS 제외)
  - **7BA**: 5BA + Random + ShuffledGridSearch (Song 2024의 BC 등가)
- Training: 5000 epoch (5BA-filter), 3000 epoch (3BA), 5000 epoch (7BA). 모두 converged
- Evaluation: D=3 500 ch / D=10 200 ch, 1 ep × 300 step, WMMSE per-channel optimum overlay

### Models (ckpt)
| label | path | epochs |
|---|---|---|
| 3BA D=3 | `dt-ifc-run1/.../3000.ckpt` | 3000 (converged) |
| 5BA-filter D=3 | `dt-ifc-run4b-5bafilter-resume/.../5000.ckpt` | 5000 (converged) |
| 7BA D=3 | `dt-ifc-run3-fullba-5000ep/.../5000.ckpt` | 5000 (converged) |
| 3BA D=10 | `dt-ifc-d10-run1/.../3000.ckpt` | 3000 |
| 5BA-filter D=10 | `dt-ifc-d10-run2b-5bafilter-resume/.../5000.ckpt` | 5000 (converged) |

## Result — 2×2 Matrix (Final SE, % of WMMSE)

| Setup | Final (nats) | % of WMMSE |
|---|---|---|
| **WMMSE D=3 (mean)** | 2.864 | 100% (ref) |
| RIBBO **3BA D=3** | 2.547 | **88.9%** ✓ best @ D=3 |
| Random D=3 | 2.379 | 83.0% |
| RIBBO 5BA-filter D=3 | 2.256 | 78.8% |
| RIBBO 7BA D=3 | 2.214 | 77.3% |
| **WMMSE D=10 (mean)** | 3.968 | 100% (ref) |
| RIBBO **5BA-filter D=10** | 2.832 | **71.4%** ✓ best @ D=10 (**+23.6%p vs 3BA**) |
| RIBBO 3BA D=10 | 1.898 | 47.8% |
| Random D=10 | 1.729 | 43.6% |

WMMSE std: ±0.685 (D=3), ±0.727 (D=10).

## 핵심 발견

### (1) BA-pool 효과는 차원에 따라 정반대 방향

| | D=3 | D=10 |
|---|---|---|
| 3BA vs 5BA-filter | 3BA 우위 (+10.1%p) | **5BA-filter 우위 (+23.6%p)** |
| 5BA-filter vs 7BA | 5BA-filter 우위 (+1.5%p) | (7BA D=10 미실험) |

→ low-dim (D=3): 단순 BA 3개가 가장 좋음. RIBBO의 in-context BA-identification 복잡도가 손해
→ high-dim (D=10): BA pool curation 효과 극대. "well-performing + diverse"가 정확

### (2) Random 대비 RIBBO의 의미 있는 우위는 D=10 5BA-filter에서만

- D=3 3BA: +5.9%p
- D=3 5BA-filter: **−4.2%p** (Random보다 못함)
- D=3 7BA: **−5.7%p**
- D=10 5BA-filter: **+27.8%p** (압도적 우위)
- D=10 3BA: +4.2%p

→ **RIBBO의 진짜 효과는 (a) 적절한 BA pool + (b) 차원이 충분히 클 때 발휘**.

### (3) Song 2024 권고 ("well-performing AND diverse") 검증

논문 Section 4.2 / Appendix B.2:
> "BC tends to imitate the average behavior … inclusion of underperforming algorithms (Random Search, ShuffledGridSearch) may significantly degrade performance … BC Filter excludes these."

> "well-performing AND diverse behavior algorithms … if datasets are from suboptimal BAs predominantly, decline in performance even with RTG tokens."

우리 결과:
- D=10 5BA-filter (71.4%) > 3BA (47.8%) → 다양성 ↑ 효과 명확
- D=10 5BA-filter는 5개 모두 well-performing BA → 권고 충족
- 7BA(D=3, 77.3%) < 5BA-filter(D=3, 78.8%) → underperforming 빼면 미세 회복 (논문 BC vs BC Filter와 같은 방향)

### (4) RIBBO가 D=3 simple problem에서 underperform — 논문 SVM 결과 평행

논문 Section 4.3:
> "RIBBO does not perform well on the SVM problem, which may be due to the problem's low-dimensional nature (only three parameters) … BA can achieve good performance easily, while the complexity of RIBBO's training/inference processes could result in performance degradation."

- IFC SE D=3은 SVM과 같은 D=3 simple problem
- WMMSE 자체가 corner-solution이라 단순 BA(CMA-ES, HC)가 잘 한다
- RIBBO 학습 복잡도 손해 → 우리 D=3 3BA가 88.9%로 가장 좋고 5BA/7BA는 오히려 나쁨

## Figures

- **`ablation_2x2_summary.pdf`** — 2×2 bar chart (BA pool × dim) 핵심 요약
- **`ablation_curves_combined.pdf`** — 2 panel curve (D=3 / D=10), 각 panel에 3-4 model + Random + WMMSE
- 개별 curve plot:
  - `eval_ifc_d3_3way.pdf` — D=3 3-way (3BA/5BA-filter/7BA)
  - `eval_ifc_d10_3way.pdf` — D=10 2-way (3BA/5BA-filter)
- raw data: `eval_ifc_d3_3way.npz`, `eval_ifc_d10_3way.npz`

## advisor 미팅 talking points

1. **"BA 다양성 ↑ → 성능 ↑"는 차원 의존적**. D=3 IFC에선 안 통함, D=10에서 통함. 논문은 BBOB D=10에서 입증한 것 — 우리 D=3 IFC가 그 일반화의 boundary case.
2. **D=10 5BA-filter가 +23.6%p 도약** = RIBBO 잠재력 발휘. BA pool curation이 critical.
3. **Random + SGS는 underperforming → 제외해야** (논문 BC Filter 등가). D=10에서 명확.
4. **D=3 RIBBO는 단순 BA 묶음에 못 미침 (SVM 평행)**. 미팅에서 "왜 RIBBO인가" 질문에 대한 답: low-dim에서는 RIBBO의 학습 복잡도가 손해. 우리 IFC를 D≥10 setup으로 가는 게 paper-worthy.
5. **앞으로 권장**: D=3 IFC는 RIBBO 적용 의의가 약함. D=10 IFC + 5BA-filter + 적절한 hyperparameter가 가장 강한 candidate. EE reward도 D=10에서 testing.
