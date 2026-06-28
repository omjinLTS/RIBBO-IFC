# 04 — WMMSE optimum 직접 구현 + overlay (500 channels)

**목적**: 외부 reference 값 대신 per-channel WMMSE optimum을 직접 계산해 fair comparison. Eval scale도 확장 (200→500 ch, 1→3 ep).

## Setup

- 같은 ckpt(`dt-ifc-run1/.../3000.ckpt`, 3BA × 3000 ep)
- Eval: **500 unseen channels × 3 ep × 300 step**
- WMMSE: `problems/ifc.py:wmmse_sum_rate` (Shi 2011, box-constrained per-link power, 10 random restart)
  - Sanity: brute-force grid 20³/40³와 10/10 채널 모두 gap=0 (IFC sum-rate optimum의 corner-solution 특성)
- Eval script: `scripts/eval_ifc_matched.py`

## Result

| Method | Final SE (nats) | bits | % of WMMSE |
|---|---|---|---|
| **WMMSE (our impl, mean)** | **2.864** | 4.132 | 100% (ref) |
| RIBBO best-so-far | 2.529 | 3.649 | **88.3%** |
| Random best-so-far | 2.384 | 3.440 | 83.2% |
| WMMSE std | ±0.685 | | |

- RIBBO 3BA-3000ep는 WMMSE의 **88.3%** 수준 도달
- Lee 2025 "WMMSE/Local opt" 값(2.50) > 우리 WMMSE 평균(2.86) — channel distribution이 같지 않아 직접 비교 불가. 우리 setup에서는 fair

## Figures

- `eval_ifc_matched_with_wmmse.pdf` — best-so-far + current(점선) + WMMSE 수평선 + ±1σ band

## 한계

- 3BA만 (BA 다양성 효과 미반영 → 05번 폴더에서 7BA 비교)
- D=3만 (D=10 분석은 06에)
