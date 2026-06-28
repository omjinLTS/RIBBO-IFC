# 19 — Extended-budget baselines (D=10/20, 1000 evals)

선배 추가 피드백: **고차원(D=10/20) 문제는 더 어려우니 budget(function evaluations)을 길게 봐야 한다.**
→ 300 → **1000 eval**로 늘려 RIBBO vs CMA-ES/DE/Random 재비교.

## 설정
- RIBBO@1000, Random@1000은 기존 budget-확장 npz 재사용(동일 unseen 채널 cid 30+, optimum 동일 확인).
  - EE D=10: `results/09_ifc_ee/eval_ifc_ee_budget.npz`(100ch) · EE D=20: `results/16_ee_d20/eval_ee_d20_budget.npz`(150ch)
  - SE D=10 5BA-filter: `results/08_eval_budget_extension/eval_ifc_budget.npz`(100ch)
- DE@1000, CMA-ES@1000은 동일 채널에서 새로 계산. 동일 300-eval 설정의 자연 확장.
- **BO 제외**: 1000-점 GP는 채널당 비현실적 + 이미 300 eval서 정체(results/18)라 길게 줘도 의미 없음.
- 코드: `scripts/eval_ifc_budget_baselines.py`. 지표 gap to optimum %(낮을수록 좋음).

## 결과 (gap %, @300 → @1000)

| Case | RIBBO | DE | CMA-ES | Random |
|---|---|---|---|---|
| EE D=10 | 20.6 → **15.7** | 48.7 → 24.3 | 27.1 → **0.4** | 70.5 → 68.2 |
| EE D=20 | 70.0 → 69.8 | 74.3 → 63.9 | 64.7 → **8.1** | 82.8 → 81.9 |
| SE D=10 (5BA-filter) | 27.0 → 18.9 | 39.4 → 23.1 | 25.3 → **5.8** | 56.6 → 53.5 |

## 핵심 발견 (결정적)
1. **budget을 길게 주면 CMA-ES가 사실상 문제를 푼다**: 세 케이스 모두 gap < 8%(EE D=10은 0.4%≈최적).
   RIBBO는 정체(EE D=20 70%, SE/EE D=10 16~19%).
2. **300 eval서 RIBBO가 이기던 EE D=10이 1000 eval서 완전 역전.** 교차점이 **정확히 ~300 eval**(이전 budget 경계)
   에서 발생 — RIBBO는 단-budget 선두, CMA-ES는 장-budget 승자. RIBBO의 유일한 우위(EE D=10)가 **단-budget 아티팩트**였음.
3. **인사이트 강화**: IFC power control은 "충분한 평가를 주면 적응형 최적화기로 풀리는" benign 문제.
   학습형(RIBBO)의 니치는 **측정이 극히 적은 단-budget 구간**으로 한정. budget 여유 있으면 CMA-ES가 정답.
4. (부수) CMA-ES@1000이 독립적으로 gap≈0(EE D=10 0.4%)에 도달 → 우리 optimum 참조해가 tight함을 교차검증.

## 공정성 caveat (RIBBO에 유리하게 해석할 여지)
- RIBBO는 **horizon 300으로 학습**됨(max_input_seq_len=300); 300 이후는 timestep clamp(=299)로 외삽이라
  300 너머 정체는 일부 **학습-horizon 한계**일 수 있음(더 긴 horizon으로 재학습하면 개선 여지). 단 EE D=10 교차는
  학습 horizon 내(~300)에서 이미 일어나며, "충분 budget이면 CMA-ES가 푼다"는 결론은 그대로 유지.

## 그림
- `budget_baselines_gap.pdf` / `bgap-1.png` (gap, 점선=이전 budget 300 표시)
- `budget_baselines_raw.pdf` / `braw-1.png` (실제 best-so-far reward + optimum 선)

## horizon-1000 재학습 (fairness 보정, 학생 요청) — `scripts/eval_ee_d10_h1000.py`
선배/학생 지적: h300 모델을 1000 eval로 보는 건 외삽이라 불공정 → EE D=10을 **horizon 1000으로 재학습**
(`dt-ifc-ee-d10-3ba-h1000`, 1000-step 3BA 데이터, max_input_seq_len=1000, 3000ep). 평가 seq_len=1000, clamp 999.

- ★**평가 정규화 버그 수정**: `eval_ifc_ee.stats()`가 RTG ymax/ymin을 `max_input_seq_len=300`으로 하드코딩
  → 1000으로 학습된 h1000과 train/eval 정규화 불일치(scale 0.811 vs 0.893). `stats_at(...,1000)`로 학습 horizon에
  맞춰 재평가. **고쳐도 사실상 불변**(불일치 modest): 버그판 54.0→39.0 → 수정판 52.3→40.2 (차이 stochastic ±2%p).
- **plumbing 정상**: JSON·캐시 X=[1000,10] 길이 1000, ts clamp=999(300 클램프 아님). 300 넘어 계속 개선(정체 아님).
- **결과(@300 → @1000 gap, 수정판)**: RIBBO-h1000 **52.3 → 40.2** vs RIBBO-h300 20.6 → 15.7 vs
  CMA-ES 27.1 → **0.4** vs DE 48.7 → 24.3 vs Random 70.5 → 68.2.
- **반직관적 핵심**: 재학습한 h1000 모델이 원래 h300보다 **모든 구간에서 더 나쁨**(@1000 40.2 > 15.7, @300도 52.3 > 20.6).
  h300의 빠른 초반 하강(sample efficiency)을 잃음. 즉 "h300를 1000까지 외삽한 불공정"은 사실 RIBBO에 **유리**했던 것.
- **결론(강화)**: 공정하게 1000-horizon으로 학습해도 CMA-ES(0.4)가 RIBBO-h1000(40.2)을 압도. fairness 보정이
  결론을 약화시키긴커녕 **강화** — 장-budget은 CMA-ES의 영역, RIBBO 가치는 단-budget(빠른 초반)에 한정.
- **undertraining 아님(tb 확인)**: h1000 x_loss -38.4 plateau(last-20% slope ≈ 0), h300 -48.5로 더 낮게 수렴 →
  h1000은 "수렴했는데 더 나쁜 정책". (NLL은 horizon 다르면 직접 비교 불가 → 직접 증거는 eval gap.)
- **공정성 검증**: h300 baseline은 같은 100채널(30~129)·같은 optimum·같은 롤아웃, h300=seq_len300+ts clamp299(외삽)·
  h1000=1000, 각자 올바른 정규화 → h300>h1000은 아티팩트 아님.
- **why = 데이터 희석(측정 확인)**: 같은 3BA를 1000-step로 늘리면 BA가 초반 수렴 후 평탄 → 1000-step traj의
  **97.7%가 flat**, 학습용 50-step window 중 '새 best 개선 있는' 비율 h300데이터 ~58% → h1000데이터 **36%** 급감.
  h1000은 같은 3000ep에 탐색·개선 패턴을 덜 보고 배움 → 약한 explorer → 모든 budget서 나쁨.
  인사이트: horizon 늘리려 롤아웃만 길게 뽑으면 평탄 tail만 추가돼 역효과. 그림: `ee_d10_h1000.pdf`/`-1.png`.

## 한 줄
"고차원은 길게 봐야 한다"를 반영하니, **장-budget에선 CMA-ES가 거의 최적까지 가고 RIBBO는 정체** →
이 문제에서 학습형의 가치는 단-budget(적은 측정) 코너로 더욱 좁혀짐. horizon-1000 재학습으로 **공정하게**
비교해도 동일(오히려 RIBBO-h1000이 h300보다 나빠 결론 강화). (정직한, RIBBO-부정적 발견.)
