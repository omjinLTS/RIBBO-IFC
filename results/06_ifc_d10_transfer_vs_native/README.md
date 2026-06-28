# 06 — IFC D=10: Rastrigin transfer vs Native + WMMSE

**목적**: RIBBO의 problem-family간 zero-shot scalability 확인 + 동일 D=10에서 native 학습과의 성능 차이 정량화.

## Setup

- Problem: IFC SE, **D=10** (Lee 2025는 D=3이지만 RIBBO scalability 위해 차원↑)
- Channel: H_sq[d',d] = |h_{d'd}|² with h~CN(0,1), d',d ∈ {0..9}
- Eval: 200 unseen channels × 1 ep
  - Transfer: 150 step (Rastrigin ckpt의 pos-emb size 151 제약)
  - Native: 300 step (in-distribution)
- WMMSE per-channel optimum overlay (`problems/ifc.py:wmmse_sum_rate`)

### Models compared
- **Rastrigin→IFC transfer**: D=10 Rastrigin training의 1000-epoch ckpt (`log/synthetic/dt-convergence-check/.../1000.ckpt`). RTG normalization은 Rastrigin training stats(y_max≈-109, y_min≈-3105) 사용 — 의도적 mismatch
- **IFC D=10 native**: 새로 학습 (`log/ifc/dt-ifc-d10-run1/.../3000.ckpt`). 3BA × 20 ch × 300 step = 60 trajectory, 3000 epoch, ~2h47min

## Result

| Method | Final SE (nats) | bits | % of WMMSE |
|---|---|---|---|
| **WMMSE D=10 (mean)** | **3.968** | 5.724 | 100% (ref) |
| RIBBO D=10 native (best-so-far) | 1.935 | 2.792 | **48.8%** |
| Random D=10 (best-so-far) | 1.729 | 2.495 | 43.6% |
| RIBBO Rast→IFC transfer (@step 150) | 1.261 | 1.819 | 31.8% |
| WMMSE std | ±0.727 | | |

## 핵심 발견

1. **차원 증가에 따른 RIBBO 성능 급락**:
   - D=3 (04번): WMMSE의 **88.3%**
   - D=10 (이 폴더): WMMSE의 **48.8%** — 절반 이하
   - 가설: 300 step으로 [0,1]^10 공간 탐색 sample-inefficient
2. **Rastrigin transfer 실패**: native(@150)보다도 못함. RTG scale mismatch가 exploration 사실상 억제 (raw IFC y가 Rastrigin normalization 적용 시 "이미 최적"으로 보임)
3. Random과 gap 좁아짐: D=3 +6.1%p → D=10 +5.2%p

## Figures

- `eval_ifc_d10_from_rastrigin_with_instant.pdf` — Transfer only, 200 ch × 1 ep × 150 step
- `eval_ifc_d10_native.pdf` — Native + Transfer + Random + WMMSE 4-curve 비교 (메인 그림)
- raw data: `eval_ifc_d10_from_rastrigin.npz`, `eval_ifc_d10_native.npz`

## 한계 / 다음 단계

- D=10 native도 3000 epoch에서 수렴 미확정 (BA 7개로 늘리고 epoch ↑하면 더 좋아질 가능성)
- Transfer 실험을 RTG renormalization 강제하는 변형도 fair? (advisor 질문 후보)
