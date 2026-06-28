# RIBBO × IFC Power Control — 통합 보고

**기간**: 2026-05-20 ~ 2026-05-27 (advisor 미팅 5/18 직후 8일)
**범위**: 5/18 미팅의 next-step 5건 + 후속 BA-pool × Dimension ablation

> **지표 안내(2026-06-26)**: 발표 deck/그림은 **gap to optimum (= 100 − %of-optimum, 낮을수록 좋음 = Lee의
> optimality gap)**으로 통일함. 본 리포트 본문의 과거 기록은 "% of optimum"(높을수록 좋음) 표기를 유지하며,
> 둘은 `gap = 100 − %` 로 1:1 변환됨. 발표용 gap 수치/그림은 `MASTER_GUIDE.md §8`, `CHEATSHEET.md §9`,
> `scripts/plot_gap_figs.py` 참조.

---

## 0. 2026-06-26 (밤샘 자율작업: 분석 A/B/C/D + EE D=20 학습)

학생 지시로 야간 자율 진행(GPU 1장, 순차). 결과 디렉토리 `12`~`16`, 슬라이드 `presentation/appendix_0625.pdf` + 메인 deck에 Result 5(robustness) 추가.

### (E) EE D=20 신규 학습 ★
3BA 풀(EE는 Random 커버리지가 도움)로 IFCEED20 추가(코드/config/datagen) → 60 traj 생성 → 3000ep 학습(2h44m, 무크래시) → 200 unseen ch 평가.

| EE D=20 (3BA) | % of optimum | 95% CI | win-rate |
|---|---|---|---|
| RIBBO | **30.2%** | ±0.8 | **100%** |
| Random | 17.2% | ±0.3 | — |

→ RIBBO−Random **+13.0p** (SE D=20 +10.4p보다 큼). 절대 %는 차원에 따라 하락(고정 3BA recipe), 하지만 EE의 interior 최적 때문에 학습 prior의 상대적 가치는 더 큼. (`results/16`)

**Budget 회복 레버는 EE D=20엔 안 통함 (negative, 검증 완료)**: 300→1000 step 연장해도 RIBBO 30.1%→30.3% (**+0.2p**). SE(D=10 +8.1p, D=3 +4.8p)와 대조적. RIBBO가 plateau에 saturate → EE D=20의 gap은 **budget 부족이 아니라 recipe/데이터 용량 한계**. 즉 "회복 레버" 주장 정밀화: budget은 *모델이 아직 개선 중일 때만* 효과. EE D=20 회복엔 더 많은 채널/trajectory(sample efficiency)가 필요하며, pool curation은 EE에 손해. (`results/16/eval_ee_d20_budget`)

### (A) Sample efficiency — Random이 못 가는 영역을 RIBBO는 간다
목표 %도달까지의 function eval 수. **SE D=10·EE D=10에서 Random은 예산 내에 최적의 50%에도 도달 못함**(plateau 47%/29%), RIBBO만 도달. SE D=3은 RIBBO가 70% 도달 9× 빠름. inference 싸므로 "더 돌리면 됨" 정량화. (`results/12`)

### (B) 채널별 통계 — 평균 너머
| Case | win-rate | RIBBO median % | Random median % |
|---|---|---|---|
| SE D=10 (5BA) | **100%** | 72.3 | 42.8 |
| EE D=10 (3BA) | **100%** | 79.4 | 29.1 |
| SE D=20 (4BA) | 99.5% | 41.5 | 31.9 |
| EE D=3 | 88.5% | 98.0 | 84.6 |
| SE D=3 | 73.6% | 96.3 | 84.6 |

→ D≥10에서 RIBBO가 **모든 채널**에서 Random을 이김(평균 우위가 아니라 분포 전체 이동). CDF는 `results/13`.

### (C) 신뢰구간 — 결과는 샘플링 아티팩트가 아님
모든 headline에 95% CI(±0.5~1.8, n=200). 핵심 2건 500채널 재평가로 재확인: EE D=10 79.3±0.7, SE D=20 41.4±0.6. (`results/14`)

### (D) Hybrid (RIBBO + local polish) — 솔직한 발견
RIBBO 최종점에서 L-BFGS-B 미세조정 시 EE D=10 99.8%, SE D=20 96.6% 도달. **단 Random+polish도 99.3%/97.4%로 동급** → polish가 지배적, init은 거의 무관. 해석: IFC reward가 gradient에 benign(=WMMSE/FP가 잘 되는 이유). **RIBBO의 고유 가치는 gradient-free black-box regime**이며, gradient가 있으면 local solver가 이미 ~최적. (`results/15`) → insight #2(model-free가 진짜 강점)를 직접 뒷받침.

### 산출물
- 코드: `problems/ifc.py`(IFCEED20 추가), `scripts/configs/dt/ifc_d20_ee.py`, `run_datagen_ifc_d20_ee.sh`, `run_train_ee_d20.sh`, `scripts/{analysis_sample_efficiency,analysis_per_channel,analysis_ci_from_npz,eval_ci,eval_hybrid,eval_ee_d20,plot_hybrid}.py`
- 슬라이드: `presentation/progress_0625.pdf`(9장, Result 5 추가), `presentation/appendix_0625.pdf`(10장 백업: 샘플효율/채널통계/WMMSE검증/EE pool/budget/full table/EE D=20/hybrid/CI)

---

## 0. 2026-06-25 업데이트 (지난 미팅 후속 4개 태스크 + 시스템 안정화)

지난 미팅 next-step 4건을 모두 진행. 새 결과 디렉토리 `08`~`11`.

### 핵심 신규 결과

**(#1) Evaluation budget ↑ → WMMSE에 더 붙는다** — `08/`. 학습 horizon(300) 너머 1000 step까지
연장 시 WMMSE gap 좁혀짐. D=3 3BA 87.0→**91.7%**(+4.8p), D=10 5BA-filter 72.9→**81.0%**(+8.1p).
고차원일수록 budget 효과 큼 (inference는 가벼우니 실전에서 그냥 더 돌리면 됨).

**(#2) SE vs EE — RIBBO는 EE에서 훨씬 강하다** — `09/`. EE(Lee eq.32) 최적해는 **interior·저전력**
(SE는 corner). uniform Random은 EE interior 최적을 거의 못 찾음. 결과(200 ch × 300 step):

| | RIBBO | Random | Optimum | RIBBO% | RIBBO−Random |
|---|---|---|---|---|---|
| EE D=3  | 0.559 | 0.480 | 0.580 (brute) | **96.5%** | +13.7p |
| EE D=10 | 0.787 | 0.296 | 1.007 (multistart) | **78.1%** | **+48.7p** |

→ D=10 EE에서 RIBBO 우위 +48.7p (vs D=10 SE 3BA +4.2p). **EE야말로 "왜 RIBBO인가"의 가장 강한
wireless 근거** (corner SE는 단순 baseline도 잘하지만, interior EE는 학습된 prior가 결정적).
참고: EE D=10 4BA-filter(Random 제외)는 69.7%로 **3BA(78.1%)보다 낮음** — EE는 interior 최적이라
Random의 넓은 저전력 커버리지가 도움. pool curation은 문제 의존적(SE 고차원엔 이득, EE엔 손해).

**(#3) Rastrigin→IFC transfer 실패는 normalization 탓이 아니다** — `10/`. RTG normalization을
IFC 값으로 고쳐도 transfer 안 살아남(31.8→31.2%, Random 미만). 기존 "normalization mismatch가 원인"
가설 **반증**. 차원 맞춰도 cross-domain zero-shot 실패 → **in-domain IFC 데이터 학습이 필수**.

**(#4) 차원 scalability D=3→10→20** — `11/`. 동일 recipe(3BA, 60 traj)로 차원만 키우면 RIBBO의
WMMSE 대비 성능이 단조 하락 (88.9→47.8→**33.0%**), Random 대비 우위도 붕괴(+5.9→+4.2→+1.5p).
단 이는 *recipe 바닥*이지 RIBBO 한계 아님 — D=10에서 보였듯 **curated BA pool(+23.6p) + budget(+8.1p)**
이 회복 레버. **D=20 4BA-filter로 직접 확인: 33.0%→41.9%(+8.9p), Random 대비 +1.5p→+10.4p** — 회복
레버가 D=20에서도 작동. 결론: **차원이 커지면 recipe(BA pool 질·데이터량·budget)도 같이 키워야** 함.

### 시스템 크래시 — 근본원인 해결 (GSP 펌웨어)

5월부터 4번 반복된 silent hard-freeze의 진짜 원인을 **black-box telemetry로 규명**: 6/24 freeze가
**유휴·냉각 상태**(util 4%, 9.6W, 45°C)에서 발생 → 열/전력/ASPM/sshfs/swap 전부 배제. 원인은
**NVIDIA GSP 펌웨어**(595.71.05). 조치 `NVreg_EnableGpuFirmware=0`(드라이버 버전 유지). 압축 스트레스
테스트(254W/77°C, ctx 70회) + D=20 풀학습 2h37m **무크래시 통과**로 확정. 상세는 `PROGRESS.md` 6/25.

---

## 1. 핵심 결과 (한 줄)

> RIBBO는 IFC에서 **D=3 simple problem에선 단순 BA(3BA)가 가장 좋고**, **D=10 비-trivial problem에선 well-curated BA pool(5BA-filter)이 +23.6%p 도약**하여 RIBBO 잠재력을 발휘한다. Song 2024의 "well-performing AND diverse BAs" 권고가 IFC에서 차원 의존적임을 정량 검증.

## 2. 작업 요약

| Task | 내용 | 산출물 | 상태 |
|---|---|---|---|
| 3 | best-so-far + current(instantaneous) reward plot | [03_*/](03_ifc_d3_3ba_instantaneous/) | done |
| 2 | WMMSE optimum 직접 구현 + 500 ch × 3 ep eval | `problems/ifc.py:wmmse_sum_rate`, `scripts/wmmse_sanity.py`, [04_*/](04_ifc_d3_3ba_wmmse/) | done |
| 4 | Rastrigin(D=10) → IFC(D=10) transfer + IFC D=10 native | `ifc_d10` 신규 pipeline, [06_*/](06_ifc_d10_transfer_vs_native/) | done |
| 1 | 7BA 데이터/학습/평가 (BA 다양성 확장) | Vizier 제외 7BA 확정, same-epoch + converge 비교, [05_*/](05_ifc_d3_7ba_vs_3ba/) | done (negative finding) |
| **후속** | **5BA-filter × {D=3, D=10} 2×2 ablation** | [07_*/](07_ifc_5ba_filter_ablation/) | **done (key finding)** |
| 5 | EE reward variant | — | 미진행 (다음 세션 권장) |

## 3. 통합 결과 표

### IFC SE D=3 (500 unseen ch × 1 ep × 300 step)

| Method | Final (nats) | bits | % of WMMSE |
|---|---|---|---|
| **WMMSE** | **2.864** | 4.132 | 100% (ref) |
| RIBBO 3BA (3000 ep, conv) | 2.547 | 3.675 | **88.9%** ✓ best @ D=3 |
| Random | 2.379 | 3.432 | 83.0% |
| RIBBO 5BA-filter (5000 ep, conv) | 2.256 | 3.255 | 78.8% |
| RIBBO 7BA (5000 ep, conv) | 2.214 | 3.193 | 77.3% |

### IFC SE D=10 (200 unseen ch × 1 ep × 300 step)

| Method | Final (nats) | bits | % of WMMSE |
|---|---|---|---|
| **WMMSE D=10** | **3.968** | 5.724 | 100% (ref) |
| **RIBBO 5BA-filter (5000 ep, conv)** | **2.832** | 4.085 | **71.4%** ✓ best @ D=10 |
| RIBBO 3BA (3000 ep) | 1.898 | 2.738 | 47.8% |
| Random | 1.729 | 2.495 | 43.6% |
| RIBBO Rast→IFC transfer (Rastrigin ckpt, no IFC training) | 1.261 | 1.819 | 31.8% |

WMMSE std: ±0.685 (D=3), ±0.727 (D=10).

## 4. 핵심 발견 — 미팅 talking points

### 발견 1. BA pool 효과는 **차원에 따라 정반대**

| 비교 | D=3 | D=10 |
|---|---|---|
| 3BA vs 5BA-filter | 3BA 우위 +10.1%p | **5BA-filter 우위 +23.6%p** |
| 3BA vs 7BA | 3BA 우위 +11.6%p | (미실험) |
| Random 대비 RIBBO 우위 | 3BA만 (+5.9%p). 5BA/7BA는 Random에 짐 | 5BA-filter +27.8%p (압도) |

→ **D=3 IFC는 RIBBO 적용 의의 약함** (논문의 SVM D=3 결과와 평행). **D=10에서 BA-pool curation의 효과 극대화**.

### 발견 2. Song 2024 권고 정량 검증

논문 인용:
- Section 4.2: *"inclusion of underperforming algorithms (Random Search, ShuffledGridSearch) may significantly degrade performance ... BC Filter excludes these."*
- Appendix B.2: *"well-performing AND diverse BAs ... if datasets from suboptimal BAs predominantly, decline in performance even with RTG tokens."*

검증:
- D=3 7BA (Random+SGS 포함) → 77.3% — 논문이 경고한 underperforming BA 포함의 함정
- D=3 5BA-filter (Random+SGS 제외) → 78.8% — 1.5%p 미세 회복 (BC vs BC Filter와 같은 방향)
- D=10 5BA-filter → 71.4% — well-performing + diverse 5개 BA의 효과 명확 (+23.6%p)

### 발견 3. RIBBO가 D=3 simple problem에서 underperform

논문 인용 (Section 4.3): *"RIBBO does not perform well on the SVM problem, which may be due to the problem's low-dimensional nature (only three parameters) ... BA can achieve good performance easily, while the complexity of RIBBO's training/inference processes could result in performance degradation."*

- IFC SE D=3 = SVM과 같은 D=3 simple problem
- WMMSE 자체가 corner-solution이라 단순 BA가 잘 함
- 우리 결과: D=3 3BA(88.9%) > D=3 7BA(77.3%) > D=3 Random(83.0%) 사이의 비-단조 패턴 = 논문 finding 재현

### 발견 4. Rastrigin → IFC D=10 zero-shot transfer 실패 (RTG mismatch)

- D=10 IFC native 5BA-filter: 71.4%
- D=10 Rastrigin transfer (no IFC training): 31.8% (Random에 짐)
- RTG normalization stat: Rastrigin y∈[-3100,-109] vs IFC y∈[0,30] — 의도된 mismatch가 exploration 사실상 억제

### 발견 5. WMMSE 본 구현 vs Lee 2025 보고값 차이

- 본 구현 (Shi 2011 box-constrained, 10 restart) D=3 mean: 2.864 nats
- Lee 2025 Fig 7(b) "WMMSE/Local opt": 2.50 nats
- Brute-force 20³/40³ grid와 10/10 채널 gap=0 → 본 구현 정확. 차이는 채널 분포 또는 setup 차이로 추정

## 5. Implementation 산출물

### 코드
- `problems/ifc.py`: `wmmse_sum_rate()`, `IFC_DIM_MAP` (search_space_id → dim)
- `data_gen/ifc_problem_statement.py`: dim 자동 추론
- `scripts/configs/dt/`: `ifc_d10.py`, `ifc_5bafilter.py`, `ifc_d10_5bafilter.py` 신규
- `scripts/configs/dataset_specs_ifc.py`: `IFCSED10` 추가
- 평가 스크립트:
  - `scripts/wmmse_sanity.py`
  - `scripts/eval_ifc.py`, `eval_ifc_matched.py`, `eval_ifc_d10_from_rastrigin.py` (instantaneous + WMMSE overlay)
  - `scripts/eval_ifc_d10_native.py`
  - `scripts/eval_ifc_3ba_vs_7ba.py`, `eval_ifc_3ba_vs_7ba_converged.py`
  - `scripts/eval_ifc_d3_3way.py`, `eval_ifc_d10_3way.py`
  - `scripts/plot_2x2_ablation.py`

### 데이터
- `data/generated_data/ifc/seed0/`: D=3 6BA × 20 ch = 120 traj (+ 5BA-filter symlink dir 100 traj)
- `data/generated_data/ifc_d10/seed0/`: D=10 6BA × 20 ch = 120 traj (+ 5BA-filter symlink dir 100 traj)
- 모두 BotorchBO 포함, Vizier 제외 (JAX OOM)

### 학습된 모델
- 3BA D=3: `dt-ifc-run1/.../3000.ckpt`
- 7BA D=3 (converged): `dt-ifc-run3-fullba-5000ep/.../5000.ckpt`
- 5BA-filter D=3 (converged): `dt-ifc-run4b-5bafilter-resume/.../5000.ckpt`
- 3BA D=10: `dt-ifc-d10-run1/.../3000.ckpt`
- 5BA-filter D=10 (converged): `dt-ifc-d10-run2b-5bafilter-resume/.../5000.ckpt`

### 디렉토리 구조
- `presentation/`: 5/18 발표 자료 (history)
- `results/01..07/`: 카테고리별 결과 + README + PDF + npz
- `results/FINAL_REPORT.md`: 본 보고서
- `PROGRESS.md`: 시간순 진행 로그
- `CLAUDE.md`: 영속 컨텍스트

## 6. 운영 이슈 — 시스템 crash 2회

### 시각
- 5/22 21:48 (학습 시작 후 ~3h 28min, NVRM API mismatch 상태)
- 5/24 07:30 (학습 시작 후 ~3h 38min, NVRM 통일 후에도 발생)

### 진단
- BERT(ACPI Boot Error Record Table) — 매 부팅 직후 "Skipped 1 error records" → 매 crash가 hardware-level error 동반 확정
- dmesg에 NVRM/Xid/MCE/AER 직접 메시지는 안 잡힘 → 너무 빠른 freeze로 ring buffer flush 못 함
- 패턴: 약 3.5시간 sustained GPU load 후 silent wedge

### 조치 (학생이 sudo로 적용)
- NVIDIA **Open KMM → Closed KMM** (proprietary) 교체 (`595.71.05 Open` → `595.71.05 closed`)
- sshfs-labftp 무한 restart 루프 stop (10초마다 fail 1300+회 누적 contributing factor)
- 환경변수 OMP/MKL/NUMEXPR threads=13 (80% CPU 제한)
- Python `torch.cuda.set_per_process_memory_fraction(0.8)` 80% GPU mem 제한

### 결과
- Closed driver 교체 후 D=10 5BA-filter 학습 1h resume → crash 없이 완료 (5/27)
- 직전 학습은 4000.ckpt 살아 있어서 resume으로 시간 손실 최소화

## 7. 권장 다음 단계 (미수행)

1. **EE reward variant** (Task 5 deferred) — Lee 2025 eq.32 EE 식. `problems/ifc.py`에 `IFCNumpyEE`, search_space_id `IFCEE`, 데이터 재생성 → 학습 → 평가. D=10 5BA-filter setup 그대로 가져가는 게 권장 (D=3은 의의 약함). 약 4-6h GPU.
2. **D=10 채널 ↑** — 20→100 ch, 100 traj → 500 traj로 sample efficiency 검증. 학습 시간 ↑이지만 paper-grade 신뢰성.
3. **D=20 or D=50 IFC** — 차원 더 늘려서 RIBBO 우위가 monotone하게 커지는지 확인. WMMSE 자체도 더 어려워지므로 RIBBO/WMMSE gap 비교가 더 의미.
4. **RTG normalization 비교** — 현재 dataset-wide. BA별 또는 channel별로 분리하면 Rastrigin transfer가 살아나는지 검토.
5. **GPU 안정성 확인** — closed driver로 다음 학습 1회 crash 없으면 환경 확정. 또 crash나면 추가 진단 (rasdaemon).

## 8. 미팅 발표 권장 슬라이드 순서

1. Problem (IFC SE D=3, Lee 2025 setup) — 기존 5/18 자료 재사용
2. Method (RIBBO) — 기존 자료
3. **WMMSE 직접 구현 + 88% 달성 (D=3 3BA)** — [04](04_ifc_d3_3ba_wmmse/)
4. **BA-pool ablation 핵심 — 2x2 plot** — [07/ablation_2x2_summary.pdf](07_ifc_5ba_filter_ablation/ablation_2x2_summary.pdf) ★ key slide
5. **차원 의존성 발견 + Song 2024 권고 검증** — talking points 위 4-(1)(2)(3)
6. (선택) Rastrigin transfer 실패 = RTG mismatch 한계 — [06](06_ifc_d10_transfer_vs_native/)
7. 다음 단계 — EE / D=20 / 채널 ↑

## 9. 결론

5/18 미팅 next-step 5건 중 4건 완료, 1건(EE) 보류. 핵심 finding은 advisor가 미팅에서 명시적으로 묻지 않은 **BA-pool curation × dimension 상호작용** — Song 2024 권고가 IFC에서 차원 의존적으로 작동함을 정량 검증. 이는 다음 미팅에서 "왜 RIBBO인가"의 정량적 답변 (low-dim에선 의의 약, high-dim + curated BA에서 +23.6%p 도약)과 "어떻게 setup해야 하는가"의 가이드 (Random/ShuffledGridSearch 제외, D≥10)를 제공.
