# RIBBO × IFC Power Control 프로젝트

이 파일은 Claude Code에게 주는 영속 컨텍스트다. 매 세션 시작 시 먼저 읽는다.

---

## 1. 목표

- **단기 (~2026-05-18 미팅)**: RIBBO를 IFC (interference channel) power control task distribution으로 학습시키는 end-to-end pipeline을 한 바퀴 돌리는 것. 결과 수치 안 좋아도 됨. "어디까지 갔고 어디서 막혔나"가 진행 보고의 가치.
- **최종**: Lee et al. (2025) Section V-A의 transmit power control in interference channels 문제를 RIBBO로 풀고, LLMO-E와 비교.

이 repo는 **RIBBO 공식 구현 (lamda-bbo/RIBBO, IJCAI'25)** clone이다.

---

## 2. 두 핵심 논문

### Song et al. (2024) — RIBBO

- 본 repo의 base method
- Causal transformer로 BBO 알고리즘을 offline data로부터 end-to-end 학습
- 핵심 요소: **RTG (regret-to-go) token**, **HRR (Hindsight Regret Relabelling)**
- 학습 데이터: K개 behavior algorithm × N task × M seed × T step의 optimization history
- 원 벤치마크: BBOB (synthetic), HPO-B, rover trajectory

### Lee et al. (2025) — LLMO 수렴성 (TCOM)

- 우리가 풀 problem 정의는 Section V-A
- **Setup**:
  - D = 3 transmitter-receiver pair
  - Action: x ∈ [0, 1]^D (power allocation ratio per pair)
  - 채널: Rayleigh fading, h\_{d'd} ~ CN(0, 1), 총 D² = 9개
  - Ptx = 10 W (transmit power budget), Pfix = 1 W (static power)
- **Reward functions (eq. 32)**:
  - SE: r*SE(x) = Σ_d log(1 + Ptx|h_dd|²x_d / (1 + Ptx Σ*{d'≠d} |h_d'd|²x_d'))
  - EE: r_EE(x) = Σ_d [같은 numerator] / (Pfix + Ptx·x_d)
- **첫 구현은 SE 먼저** (식이 더 단순, 분모 0 이슈 없음). EE는 SE 돌고 난 다음.

---

## 3. "왜 RIBBO인가" — 미팅 답변용 가설

학생이 advisor에게 "왜 이걸 합니까" 질문에 답하기 위한 working hypotheses (현재 미확정, advisor 의도 추측):

1. **Inference cost**: LLMO는 매 iteration LLM API 호출 (비용·지연). RIBBO는 학습 후 inference 가볍다 → wireless 실시간 운영에 더 practical.
2. **Channel distribution prior**: LLMO는 매 channel마다 prompt 루프 처음부터. RIBBO는 Rayleigh fading distribution을 학습에 흡수 → 새 channel에 더 빠른 수렴 기대.
3. **새 contribution 영역**: RIBBO는 BBOB/HPO/rover에서만 검증. Wireless로 가져온 사례 거의 없음 → 논문거리.

이 중 무엇이 advisor의 의도인지 학생이 확인 필요. Claude Code는 이 가설들을 당연한 것처럼 다루지 말고, 모호한 점이 나오면 학생에게 묻는다.

---

## 4. 7일 plan

| Day | 날짜 | 목표                                                           |
| --- | ---- | -------------------------------------------------------------- |
| 1   | 5/11 | Repo 정찰 + BBOB toy로 pipeline 동작 검증                      |
| 2   | 5/12 | IFC task class 작성 + sanity check (random action reward 분포) |
| 3   | 5/13 | Offline data 생성 (Random Search, CMA-ES, Hill Climbing 3개)   |
| 4   | 5/14 | RIBBO 학습 (5090 ×2, 기본 hyperparameter)                      |
| 5   | 5/15 | Test channel 30개 evaluation + LLMO 비교 (Lee 논문 수치 인용)  |
| 6   | 5/16 | 디버깅 버퍼 + 슬라이드 시작                                    |
| 7   | 5/17 | 진행 보고 슬라이드 (5장) 마무리                                |

각 Day 시작 시 학생이 그 Day의 구체 지시를 준다. Claude Code는 Day 범위를 멋대로 넘어서 미리 작업하지 않는다.

---

## 5. 환경

- **하드웨어 (현재 실사용)**: GPU 1장 — **RTX 4070 Ti (12GB)**. 모든 학습/평가는 이 1장 기준으로 순차 진행.
- **하드웨어 (목표 배포처)**: 연구실 **시뮬컴 = RTX 5090 × 2**. 단 현재 다른 사람이 사용 중이라 사용 불가. 가용해지면 그쪽으로 옮김.
- **Python**: conda env `RIBBO`. 비대화형 셸에서는 PATH에 안 잡히므로 `/home/myoungjin/miniconda3/envs/RIBBO/bin/python` 직접 호출(또는 PATH 앞에 추가). bare `python`으로 스크립트 돌리면 conda 밖 인터프리터라 실패.
- **OS**: Linux (kernel 6.8). GPU 안정성: GSP 펌웨어 off (`NVreg_EnableGpuFirmware=0`)로 freeze 해결됨.

---

## 6. Claude Code 작업 원칙

### 6.1 추측 금지

- Repo 구조, library API, 함수 signature, hyperparameter 기본값은 **반드시 코드/문서/터미널로 확인** 후 사용
- 모르면 모른다고 말하고, `grep`/`cat`/`find`로 확인하거나 학생에게 물어본다
- "아마 이럴 것이다" 류 표현 금지. 확인하든지 묻든지 둘 중 하나.

### 6.2 작게 먼저

- 큰 학습/데이터 생성 돌리기 전에 toy 규모로 pipeline 동작 확인
- 예: N=2 task, M=2 seed, T=10 step
- Toy가 안 도는데 큰 실험 시작하지 않는다

### 6.3 Destructive change는 사전 확인

다음 작업 전에는 학생에게 먼저 보고/허락:

- 기존 파일 대량 수정·삭제 (10줄 이상 또는 함수 시그니처 변경)
- 새 dependency 설치 (`pip install`)
- 대용량 데이터/모델 생성 (디스크 1GB 이상)
- `git push`, `git rebase`, `git reset --hard` 등 history 변경
- 환경변수·설정 영구 변경

### 6.4 Hyperparameter 튜닝 금지

- 7일 안에 끝내는 게 우선. Paper/repo 기본값 100% 그대로 사용
- 학습이 안 되면 hyperparameter 의심하기 전에 **데이터/입출력 shape/normalization** 먼저 의심
- 튜닝 시작하면 시간 다 날아간다. 학생이 명시적으로 지시할 때만.

### 6.5 솔직한 보고

- 막힌 지점, 못 한 것, 헷갈리는 부분은 **즉시** 명시
- 잘 되는 척, 다 안 척 금지
- 부분 성공도 부분 성공이라고 말한다 (e.g., "pipeline은 도는데 loss가 감소 안 함")

### 6.6 Commit hygiene

- 의미 있는 단위로 commit, 커밋 메시지는 무엇이 왜 바뀌었는지
- 새 branch는 학생이 지시할 때만 생성
- `.gitignore`에 데이터/모델/로그 디렉토리 포함되어 있는지 확인

### 6.7 Reproducibility

- 모든 실험: seed 고정, config 파일에 hyperparameter 명시
- 출력 경로 일관성: `experiments/<exp_name>/{config.yaml, logs/, ckpts/, results/}` 류
- 그림 그리면 옆에 그림 그린 script도 commit

---

## 7. PROGRESS.md 유지

매 작업 세션 끝에 repo 루트의 `PROGRESS.md`를 업데이트한다. 형식:

```
## YYYY-MM-DD (Day N)
**완료**:
- ...
**막힘**:
- ...
**다음 세션 시작 시 필요**:
- ...
**advisor 미팅에 가져갈 질문 후보**:
- ...
```

미팅 자료는 PROGRESS.md를 기반으로 학생이 작성한다. Claude Code는 PROGRESS.md를 정확하게 유지하는 데 책임이 있다.

---

## 8. 헷갈릴 때 학생에게 물어볼 것 (예시)

- "이 부분 IFC reward를 추가하려면 `benchmarks/synthetic/` 아래에 추가하는 게 맞는지, 별도 `benchmarks/wireless/` 디렉토리를 새로 만들지 어떤 게 repo 컨벤션과 더 일관됩니까?"
- "Behavior algorithm 3개로 줄여서 시작하라고 했는데, repo에 이 3개가 다 구현되어 있나 확인했더니 X는 없습니다. 어떻게 처리할까요?"
- "RTG normalization에서 y_max proxy를 어떻게 잡을지 — IFC는 채널마다 optimum이 다른데, behavior algorithm들의 best observed value 평균을 쓰는 게 paper 방식과 맞는지 advisor에게 확인할 만한 포인트입니다."
