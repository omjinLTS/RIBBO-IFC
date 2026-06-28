# 선배 발표초안 피드백 정리 + 답변 (2026-06-26)

대상: `presentation/progress_0625.pdf` (메인 8장: Title / Overview / Result 1~5 / Insights).
지표는 gap to optimum (낮을수록 좋음). 아래 답은 전부 코드·결과파일·슬라이드 소스(`progress_0625.tex`)로 확인한 사실.

---

## A. 슬라이드별 질문 — 답 + 슬라이드 수정 시사점

### 2p Overview
1. **corner / interior 뜻**: action x∈[0,1]^D 박스에서 최적해의 위치.
   - SE 최적 = **corner(꼭짓점)** = 강한 링크는 풀파워(x=1), 약한 링크는 끔(x=0). "켜고/끄기".
   - EE 최적 = **interior(내부)** = 중간 전력값(예: x≈0.18). 전력 올리면 rate는 늘지만 에너지·간섭 손해 → 적당한 중간 지점.
   - → 수정: 작은 그림(박스+점 위치) 또는 "on/off vs mid-power" 한 줄을 Overview에 박아두면 R2의 on/off·middle 혼란까지 해결.
2. **SE=WMMSE, EE=exhaustive?** 절반만 맞음. SE 기준해 = WMMSE. EE 기준해 = **D=3만 exhaustive grid(전수)**, **D=10/20은 multi-start gradient(40 restart, 전수 아님 = near-global 참조)**. 현재 슬라이드 "strong reference from many restarts (EE), checked against exhaustive search"가 뭉뚱그려져 오해 유발. → D별로 분리 명시.
3. **3BA = 3개 알고리즘으로 RIBBO 학습?** 맞음. 3 BA(Random, HillClimbing, CMA-ES)가 만든 **탐색 궤적**이 RIBBO 학습데이터. RIBBO 자체는 트랜스포머 1개.
4. **unseen channel = 같은 분포 vs 완전 다른 분포?** **같은 분포(Rayleigh CN(0,1))의 새 realization**. train=채널 0~19, eval=30번~ (전부 같은 분포, 다른 seed). "완전 다른 분포" 케이스는 Result 5의 Rastrigin transfer임. → "unseen = new Rayleigh realizations (same distribution)" 한 줄 명시.

### 3p Result 1 (BA-pool × Dimension)
1. **어떤 문제로 학습?** SE. (SE 궤적 학습 → SE 평가)
2. **그래프 문제?** SE. (라벨 없음 → "SE" 명시 필요)
3. **5BA에 Rand+SGS 추가?** **반대**. 5BA-filter는 7BA에서 Random+SGS를 **뺀** 것. (7BA = 5BA-filter + Random + SGS). 더 헷갈리는 점: 3BA={Rand,HC,CMA}와 5BA-filter={HC,RegEvol,Eagle,CMA,BoTorch}는 포함관계가 아님(3BA엔 Random 있고 5BA-filter엔 없음). → **BA 구성 표를 명시**하는 게 최선.
4. **Song 2024?** RIBBO 논문(Song et al., IJCAI 2025, "Reinforced In-Context BBO"). "well-performing AND diverse BA" 권고 출처. → "(the RIBBO paper)" 명시.
5. **왜 "best behavior"?** 슬라이드 문구 "The best behavior-algorithm (BA) pool flips with dimension" = "**최적 BA 풀**이 차원에 따라 뒤집힌다". "best behavior"가 아니라 "best BA **pool**". 표현 어색 → "Which BA pool works best depends on the dimension" 류로.

### 4p Result 2 (SE vs EE)
1. **어떤 상황 학습?** 같은 3BA 풀, **reward만 SE↔EE**. SE 모델/EE 모델 따로, 차원별로도 따로 학습. → 명시.
2. **on/off? middle?** on/off=corner(SE 최적), middle=interior(EE 최적). Overview 개념과 동일. Overview에 그림으로 한 번 깔면 해결.

### 5p Result 3 (Scalability)
1. **"Scalability" 표현 문제 — 선배 지적이 맞음.** 현재는 **차원마다 별도 모델**(입력차원=D라 dimension-agnostic 아님). 한 모델이 차원을 가로질러 일반화하는 게 아니므로 "scalability"는 과한 표현. → 제목을 **"Effect of problem dimension"** 또는 **"Performance across dimensions"**로. (진짜 scalability = dimension-agnostic 모델은 future work로.)
2. **"Ways to shrink the gap" = BA pool 조합 + step 늘리기?** 맞음. 회복 레버 둘: (a) BA 풀 curation(D=20 4BA-filter), (b) 평가 step 연장(budget). 단 EE D=20은 step 늘려도 회복 안 됨(데이터 필요).

### 6p Result 4 (Robustness)
1. **"Random은 naive하니 당연히 이겨야" — 일리 있음.** "Random 이긴다"만으론 약한 주장. 보강 포인트: (i) 우리 Random = Lee 2025의 "Brute-force" baseline과 **동일** → 의미 있는 비교점. (ii) 핵심은 "이긴다"가 아니라 **"Random은 예산 내 최적의 50%도 못 가고 plateau, RIBBO만 도달"** = 균등 샘플링이 구조적으로 못 가는 영역(특히 EE interior)을 학습 prior가 간다. → 프레이밍을 "beat Random" → "Random이 못 푸는 영역을 RIBBO가 푼다"로.
2. **small error bars 의미?** 95% 신뢰구간(±<1.5%, n=200~500). = 결과가 샘플 운이 아니라 통계적으로 안정적.
3. **마지막 문장(hybrid/local search) 의미?** result 맞음(정직한 한계). RIBBO 끝점에 local search 붙이면 gap≈0 도달하는데, **Random+local search도 ≈0 도달** → local search가 지배, 출발점 거의 무관 → **gradient(모델) 있으면 이 문제는 쉬움 → RIBBO 가치는 model-free 상황에 한정**. 현재 한 문장에 다 욱여넣어 모호 → 별도 bullet 분리 or appendix hybrid 슬라이드로.

### 7p Result 5 (Transfer)
1. **색 맞나?** 맞음. green=IFC 학습→IFC test(in-domain 참조), red/blue=Rastrigin 학습→IFC test(as-is/rescaled), gray=Random.
2. **문제 라벨?** SE D=10. (안 써있음 → 추가)
3. **왜 IFC model ≈ Random?** 이 슬라이드는 **SE D=10 3BA** 케이스 = RIBBO가 Random보다 몇 pts만 나은 **약한 영역**. 큰 격차(+48.7p)는 EE거나 5BA-filter. 즉 transfer 비교용으로 (우연히) 가장 약한 RIBBO config를 써서 둘이 비슷해 보임. 슬라이드 목적은 RIBBO-vs-Random이 아니라 **in-domain vs transfer**. → 라벨 + 한 줄 주석.

---

## B. 교수님 대비 마이너 코멘트 — 정리 + 액션

1. **약어 풀어쓰기**: 통신 약어(CSI/UE/BS/SINR/SE/EE/WMMSE)는 OK. **비통신(ML) 약어는 첫 등장 시 full term**: RIBBO, RTG, HRR, BA, CMA-ES, BO, SGS, FP, CI 등. 교수님 발표는 전부 풀어쓰기. (SE/EE도 첫 등장 시 spectral/energy efficiency로 한 번 풀어주면 안전.)
2. **optimum/optimal 신중**: **WMMSE는 local optimal** → "optimum"이라 단정하면 안 됨. Lee 본인도 "local optimal"로 표기. EE D=10/20도 near-global(certified global 아님). → "gap to optimum"의 기준해를 **"best-known / local-optimal reference"**로 relabel. 메트릭 이름 "optimality gap"은 Lee가 쓰므로 유지 가능하나, 기준해를 "the optimum"이라 부르지 말 것.
3. **그래프 legend/제목**: y축은 **"optimality gap (%)"** 으로 깔끔히, **그래프 위 제목 제거**(아래 캡션이 이미 설명). → `scripts/plot_gap_figs.py` 수정으로 전체 도면 일괄 반영 가능.
4. **숫자/퍼센트 과다는 old·피로**: 슬라이드엔 "더 우수한 성능" 등 **정성 표현**, 퍼센트는 **구두로**. 표/그림의 % 반복 톤다운. (핵심 1~2개만 남기는 것 고려.)
5. **영어 표현 다듬기 (연구실 논문 톤)**: 현재 deck의 구어체 예 — "flips with dimension", "RIBBO saves"(표 헤더), "it keeps trying high-power settings", "Fewer tries", "goes below", "as-is". → 격식체로. (원하면 슬라이드별로 대체 문장 제안 가능.)

---

## C. 권장 처리 순서 (수정 착수 시)
1. Overview에 corner/interior 그림 + reference(SE=WMMSE / EE=D3 exhaustive·D10·20 multistart) 분리 + unseen 정의 → R2·R4 혼란 동시 해소.
2. 각 Result에 **SE/EE·차원 라벨** 추가 + BA 구성 표.
3. R3 제목 "Scalability" → "Effect of problem dimension".
4. R4 hybrid 문장 분리, Random 프레이밍 보강.
5. plot_gap_figs.py에서 제목 제거·y축 라벨 통일.
6. "optimum" 용어 일괄 점검, ML 약어 풀어쓰기, 영어 톤 교정.
