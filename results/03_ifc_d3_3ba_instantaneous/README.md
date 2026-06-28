# 03 — Instantaneous(current) reward overlay 추가

**목적**: 02의 best-so-far 곡선만으로는 exploration↔exploitation 트레이드오프 안 보임. 각 step의 raw reward 평균도 같이 그려 dynamics 시각화.

## Setup

같은 ckpt(`dt-ifc-run1/.../3000.ckpt`) + 같은 data·channel·eval count. 코드만 plot 추가.

수정 스크립트:
- `scripts/eval_ifc.py` → `eval_ifc_ribbo_vs_baselines_with_instant.pdf`
- `scripts/eval_ifc_matched.py` → `eval_ifc_matched_with_instant.pdf`
- `scripts/eval_ifc_d10_from_rastrigin.py` (D=10 — 06번 폴더에)

## Figures

- `eval_ifc_matched_with_instant.pdf` — 200 ch × 1 ep × 300 step. 실선=best-so-far, 점선=current
- `eval_ifc_ribbo_vs_baselines_with_instant.pdf` — 50 ch × 3 ep, Lee reference 수평선 + instantaneous overlay

## 관찰

- RIBBO의 current reward는 학습 진행에 따라 (= step 진행에 따라) best-so-far에 수렴. exploration→exploitation 자연스러운 패턴
- Random의 current는 평탄 (탐색만 함)
- Best-so-far와 current의 gap이 RIBBO < Random → RIBBO가 exploit phase에 들어감
