# 18 — Strong black-box baselines (DE, CMA-ES, BO)

선배 피드백("DE 그래프보다 RIBBO가 잘 나와야 의미 있다") → 강한 model-free black-box 최적화기들을
RIBBO/Random과 **완전히 동일 조건**으로 비교. Random < {DE, CMA-ES, BO} 스펙트럼 위에서 RIBBO 위치 평가.

## 설정 (apples-to-apples)
- 동일 unseen 채널(dataset_id 30–), 동일 per-channel optimum(SE=WMMSE, EE=brute/multistart), 동일 gap 지표.
- 동일 예산 **300 function evaluations**, best-so-far 추적. 전부 model-free(reward만 query, 채널모델/gradient X).
- RIBBO/Random/optimum은 기존 npz 재사용 → baseline만 계산. 메서드별 `bsf_cache/`에 캐싱.
- **DE**: scipy `differential_evolution`(rand/1/bin, polish=False, pop~24).
- **CMA-ES**: pycma(`cma`), x0=0.5, σ0=0.25, bounds[0,1], default popsize.
- **BO**: GP(SingleTaskGP) + Expected Improvement(botorch). n_init=2D, restarts=3, raw=32. 300 eval에서 GP가
  채널당 ~8분이라, **하이퍼파라미터를 10 step마다 재적합**(GP는 매 step 전 데이터 조건화; fit만 amortize)으로
  ~2.5분/채널로 단축. **BO만 n=30**(나머지 n=200).
- 코드: `scripts/eval_ifc_baselines.py`. (DE 단독 초판: `scripts/eval_ifc_de.py`.)

## BO 공정성 검증 (중요)
가속(amortized fit)이 BO를 불공정하게 약화시켰는지 확인: EE D=10 5채널에서 **full BO(매 step 재적합, restarts=5/raw=64)
53.1 vs 가속 BO 58.2** (~5pt, 채널별론 가속이 더 나은 곳도 있음). → 가속이 BO 패배 원인 아님. full BO도 ~53%로
RIBBO/CMA-ES에 크게 뒤짐. **비교는 공정.** (BO의 10차원 부진은 GP-EI가 moderate-D에서 degrade하는 알려진 현상.)

## 결과 1 — DE & CMA-ES 전 케이스 (n=200, final gap %, 낮을수록 좋음)

| Case | RIBBO | DE | CMA-ES | Random | RIBBO 승률 vs DE / CMA |
|---|---|---|---|---|---|
| EE D=3 | 3.2 | 0.8 | **-0.0** | 17.6 | 20% / 0% |
| **EE D=10** | **21.5** | 48.6 | 26.8 | 70.6 | **100% / 65%** ★ |
| EE D=20 | 69.8 | 74.4 | **64.8** | 82.8 | 86% / 20% |
| SE D=10 (3BA) | 52.2 | 40.9 | **26.4** | 56.7 | 22% / 5% |
| SE D=10 (5BA-filter) | 28.4 | 40.9 | **26.4** | 56.7 | 91% / 38% |
| SE D=20 (3BA) | 66.8 | 61.5 | **51.3** | 68.5 | 22% / 2% |
| SE D=20 (4BA-filter) | 58.0 | 61.5 | **51.3** | 68.5 | 75% / 12% |

## 결과 2 — BO 포함 4종 (n=30, 동일 30채널, final gap %)

| Case | RIBBO | DE | CMA-ES | BO | Random | RIBBO 승률 vs BO |
|---|---|---|---|---|---|---|
| EE D=3 | 3.6 | 1.8 | **-0.1** | 2.0 | 17.6 | (저차원 다 최적) |
| **EE D=10** | **20.2** | 46.7 | 26.6 | 52.6 | 70.3 | **100%** ★ |
| SE D=10 (5BA-filter) | 26.2 | 36.7 | **20.4** | 35.0 | 55.0 | 80% |

(그림: `baselines_final.pdf`/`final-1.png`. DE+CMA만: `baselines_ee_dc.pdf`/`baselines_se_dc.pdf`.)

## 핵심 발견 (정직)
1. **vs DE: RIBBO가 자기 무대(고차원·EE·curated)에서 전부 이김.** DE는 strawman 아님(Random보다 훨씬 셈).
2. **CMA-ES가 제일 센 경쟁자.** RIBBO가 CMA-ES를 "확실히" 이기는 곳은 **EE D=10 한 곳**(평균·중앙값·65%채널).
   SE D=10(5BA)·D=20·EE D=3 등은 CMA-ES 우세. RIBBO가 학습 prior를 갖고도 from-scratch CMA-ES에 대부분 밀림 = 솔직한 한계.
3. **BO는 저차원의 강자, 고차원서 붕괴.** EE D=3에선 BO가 초반 최강·거의 최적(2.0). 그러나 EE D=10/SE D=10에선
   일찍 plateau(GP-EI의 moderate-D degrade) → **RIBBO가 BO를 EE D=10서 전 채널(100%) 압도.** ★
4. **EE D=10 = RIBBO의 깨끗한 승리**: 3대 강한 black-box 최적화기(DE/CMA-ES/BO)를 모두 이김. interior·저전력 최적 +
   moderate 차원이 RIBBO 학습 prior가 빛나는 지점.
5. **★Sample-efficiency 각도(그림)**: RIBBO는 예산 초반(적은 eval)에 prior로 빠르게 내려감. 측정비용 큰 실시간
   무선에서 "적은 평가로 빨리"가 RIBBO 실질 강점.

## 한 줄
강한 baseline 3종까지 넣으니 "RIBBO가 다 이긴다"는 무너지지만, **RIBBO의 깨끗한 우위는 EE D=10(3종 모두 격파) +
적은-예산 영역**으로 또렷해짐. CMA-ES는 대부분 regime에서 RIBBO보다 강함(정직한 경계). 이 경계가 advisor 논의 핵심.
