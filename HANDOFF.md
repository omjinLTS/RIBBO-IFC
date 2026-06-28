# HANDOFF — RIBBO × IFC Power Control (발표 준비용 단일 진입점)

> 이 문서 하나 + `results/FINAL_REPORT.md` + `PROGRESS.md` 만 읽으면 전체 맥락이 복원됩니다.
> 새 세션에서 작업 시작할 때 여기부터 읽으세요.

---

## ⚡ 최신 상태 (2026-06-26)

- **지표**: 발표는 **gap to optimum (= 100 − %of-optimum, 낮을수록 좋음, Lee의 optimality gap)** 으로 통일.
- **발표자료**: `presentation/progress_0625.pdf` (메인 **8장**), `presentation/appendix_0625.pdf` (백업 **12장**).
  그림은 `scripts/plot_gap_figs.py` 로 npz에서 재생성(gap 버전).
- **발표 보조문서**: `MASTER_GUIDE.md`(발표 숙지/Q&A), `CHEATSHEET.md`(실험환경·코드·숫자 즉답).
- **신규 결과**: EE D=20 학습/평가(`results/16`), 분석 A/B/C/D(`results/12~15`), synthesis 그림(`results/17`).
- **핵심 메시지**: RIBBO는 (고차원 + EE interior-optimum + curated pool + model-free 상황)에서 가치 큼.
  EE D=10에서 Random 대비 gap 48.7pts 작음. hybrid(local search)는 모델 있으면 쉬움을 보임(정직한 한계).
- **세션 재시작 스크립트 수정(2026-06-26)**: `scripts/restart_session.sh` 버그픽스. 기존 pkill이
  텔레그램 **bun poller**(`claude-plugins-official/telegram`)는 안 죽여 새 세션 getUpdates **409 충돌** →
  (1) bun poller까지 종료, (2) 프로세스 사라질 때까지 최대 15s 폴링+강제킬+3s settle, (3) claude 경로를
  `command -v`로 해석(conda bin엔 claude 없음). 가장 확실한 실행은 **터미널 직접 `bash scripts/restart_session.sh`**(branch A, TTY exec).
- 상세 시간순은 `PROGRESS.md` 최상단, 종합은 `results/FINAL_REPORT.md` 최상단.

---

## 🔁 세션 재시작 프로토콜

**트리거**: 사용자가 텔레그램으로 "세션 새로 시작해"(류) 라고 하면, **현재 세션**이 다음을 수행한다:

1. **HANDOFF.md 갱신** — 위 "최신 상태"를 현재 시점으로 업데이트(최근 작업/산출물/다음 할 일).
2. **`bash scripts/restart_session.sh` 실행** — 기존 telegram claude 세션을 종료하고 새 세션을 띄움.
   - 새 세션은 부트스트랩 프롬프트(`scripts/restart_bootstrap.txt`)를 받아:
     (a) 텔레그램으로 "새 세션이 시작됐습니다. 내용 파악 중입니다." 전송 →
     (b) HANDOFF/PROGRESS/FINAL_REPORT/MASTER_GUIDE/CHEATSHEET/CLAUDE.md 읽고 파악 →
     (c) 텔레그램으로 "진행상황 파악 완료" + 요약 전송.
   - 스크립트는 터미널 직접 실행(TTY) 시 `exec`로 교체, 세션 내부 트리거 시 `setsid`로 분리 실행 후 기존 종료.
   - 새 세션 로그: `.session_restart.log`, 재시작 시각: `.last_session_restart`.

> 참고: claude 세션은 자기 자신을 완벽히 self-restart 하기 어렵다(TTY/텔레그램 폴링 충돌). 가장 확실한 건
> 터미널에서 `bash scripts/restart_session.sh` 직접 실행. 텔레그램 트리거(분리 모드)는 새 세션이 TTY 없이
> 떠야 하므로 환경에 따라 실패할 수 있음 — 그 경우 터미널 실행으로 폴백.

---

## 0. 한 줄 요약

RIBBO(causal-transformer in-context BBO)를 IFC(간섭채널) 전력제어에 적용. WMMSE를 기준으로,
**RIBBO의 가치는 (1) 고차원, (2) interior-optimum 문제(EE), (3) curated BA pool에서 극대화**된다.
저차원·corner-solution(SE D=3)에선 단순 baseline도 잘해서 RIBBO 의의가 약하다.

## 1. 문제 정의 (Lee et al. 2025, Sec V-A)

- IFC D개 송수신 pair, action x∈[0,1]^D (pair별 전력비), Rayleigh fading h~CN(0,1), Ptx=10, Pfix=1.
- **SE** r(x)=Σ log(1+SINR_d) — 최적해가 **corner**(켜고/끄기). WMMSE가 최적 baseline.
- **EE** r(x)=Σ log(1+SINR_d)/(Pfix+Ptx·x_d) — 최적해가 **interior·저전력**. (식: `problems/ifc.py:ifc_rates`)
- 우리 WMMSE 구현(2.864 nats D=3)은 brute-force로 검증됨(gap 0/10). Lee 논문값(2.50)과 차이는 채널분포/setup 차이.

## 2. 전체 결과 표

### (A) 차원 scalability — IFC SE, 3BA pool (results/11)
| D | WMMSE | RIBBO % of WMMSE | Random % | RIBBO−Random |
|---|---|---|---|---|
| 3  | 2.864 | **88.9%** | 83.0% | +5.9p |
| 10 | 3.968 | 47.8% | 43.6% | +4.2p |
| 20 | 4.764 | 33.0% | 31.5% | +1.5p |
→ 고정 recipe는 차원↑에 단조 하락, Random 대비 우위 붕괴.

### (B) BA-pool curation (회복 레버) — IFC SE
| | D=3 | D=10 |
|---|---|---|
| 3BA (R,HC,CMA) | **88.9%** | 47.8% |
| 5BA-filter (HC,RegE,Eagle,CMA,BoTorch) | 78.8% | **71.4%** (+23.6p) |
| 7BA (5BA+R+SGS) | 76.4% | — |
| **D=20 4BA-filter (HC,RegE,Eagle,CMA)** | — | **41.9%** (3BA 33.0% 대비 **+8.9p**) ← 신규 |
→ D=3은 단순 3BA가 best(다양성↑ 손해), D=10+는 curated pool이 큰 도약. (results/07)
→ **D=20: curated 4BA-filter가 3BA 33.0%→41.9% 회복(+8.9p), Random 대비 +10.4p. scalability 회복 레버가 D=20에서도 확인됨.** (results/11)

### (C) Evaluation budget ↑ (results/08)
| 모델 | @300 step | @1000 step |
|---|---|---|
| SE D=3 3BA | 87.0% | **91.7%** (+4.8p) |
| SE D=10 5BA-filter | 72.9% | **81.0%** (+8.1p) |
→ inference 가벼우니 그냥 더 돌리면 gap 좁혀짐. 고차원일수록 효과 큼.

### (D) SE vs EE — RIBBO의 진짜 무대 (results/09) ★핵심
| | RIBBO | Random | RIBBO−Random |
|---|---|---|---|
| EE D=3 (3BA) | 96.5% | 82.8% | +13.7p |
| EE D=10 (3BA) | **78.1%** | 29.4% | **+48.7p** |
| **EE D=10 (4BA-filter)** | **69.7%** | 29.4% | +40.3p (3BA보다 낮음) |
| EE D=10 budget @1000 | 84.2% | — | (+4.9p vs @300) |
→ EE 최적이 interior라 Random이 못 찾음(D=10서 29.4%). RIBBO는 학습된 prior로 도달.
→ **D=10 EE RIBBO 우위 +48.7p (SE 3BA는 +4.2p)** = "왜 RIBBO" 가장 강한 근거.
→ **EE 4BA-filter(69.7%)는 3BA(78.1%)보다 낮음**: EE는 interior·저전력 최적이라 Random의 넓은 커버리지가
  도움 → Random 빼면 손해. pool curation 효과가 **문제 의존적**임을 재확인(SE 고차원엔 이득, EE엔 손해).

### (E) Cross-domain transfer 실패 (results/10)
| | % of WMMSE |
|---|---|
| Rastrigin→IFC D=10, Rastrigin RTG | 31.8% |
| Rastrigin→IFC D=10, **IFC RTG (norm fix)** | 31.2% (변화 없음, Random 미만) |
| Native IFC D=10 (참조) | 44.9% |
→ normalization mismatch가 원인이라던 기존 가설 **반증**. in-domain IFC 데이터 학습 필수.

## 3. 발표 슬라이드 순서 (제안)
1. Problem: IFC power control (Lee 2025), SE→EE — 기존 5/18 자료 재사용
2. Method: RIBBO (RTG token + HRR)
3. WMMSE 직접 구현 + D=3서 88.9% 달성 — `results/04`
4. **BA-pool × 차원 2×2** (key) — `results/07/ablation_2x2_summary.pdf`
5. **SE vs EE — interior optimum에서 RIBBO +48.7p** (★ 핵심 슬라이드) — `results/09`
6. Scalability D=3→10→20 + 회복 레버(pool·budget) — `results/11`, `results/08`
7. Transfer 실패 = in-domain 필수 — `results/10`
8. 결론: 고차원+EE+curated pool에서 RIBBO 가치 극대화, inference도 가벼움

## 4. 그림/파일 위치
- `results/0X_*/`: 각 실험 PDF + npz + README (X=1~11)
- `results/FINAL_REPORT.md`: 전체 종합 (맨 위 "0. 2026-06-25 업데이트")
- `results/07_*/ablation_2x2_summary.pdf`: 기존 핵심 슬라이드
- `presentation/`: 5/18 발표 자료 (LaTeX, 재사용 가능)
- 코드: `problems/ifc.py`(SE/EE/WMMSE/optimum solvers), `scripts/eval_ifc_*.py`, `scripts/configs/dt/ifc_*.py`

## 5. 주의점 (질문 대비)
- WMMSE 우리값 2.864 ≠ Lee 2.50: 채널분포/setup 차이. 우리는 동일 채널 fair comparison + brute-force 검증.
- D=10/D=20 EE 최적은 multistart(near-global, D=3서 brute-force와 1e-4 일치). certified global 아님.
- 데이터 작음(채널 20, 60~100 traj). 평가는 200~500채널이라 통계 OK. "더 키우면 더 좋아질 여지"가 곧 scalability 메시지.
- 4BA-filter = 5BA-filter에서 BotorchBO 제외(D=20 GP가 traj당 ~9.5분이라 시간상 제외). 기존 D=10 5BA엔 BotorchBO 포함.
- D=3 EE 모델은 2000ep(수렴 확인, 6/23 크래시로 3000 못 채움). 결과 영향 없음.
- 수치는 stochastic policy라 재평가 시 ±1~2%p 변동.

## 6. 새 세션에서 이어가는 법
- 새 `claude` 세션 시작 → 메모리 자동 로드 + 이 `HANDOFF.md`, `PROGRESS.md`, `results/FINAL_REPORT.md` 읽으면 맥락 복원.
- 발표 자료는 `presentation/` LaTeX 재사용 + 위 슬라이드 순서로 `results/` PDF 끼우면 됨.
- 학습 재현/추가: `run_train_*.sh`(telemetry+resume+250ep ckpt 내장), eval은 `scripts/eval_ifc_generic.py`.
- ⚠️ 터미널 세션은 conda RIBBO env 명시 필요: `/home/myoungjin/miniconda3/envs/RIBBO/bin/python`.

## 7. 시스템 (크래시 해결됨)
- 5월부터 4회 silent hard-freeze = **NVIDIA GSP 펌웨어**. 해결 `NVreg_EnableGpuFirmware=0`.
- 검증: 스트레스 70cyc + 풀학습 2.5h + 이번 4BA 체인까지 무크래시.
- 보험: `scripts/setup_watchdog.sh`(하드웨어 워치독, 발표 후 sudo 1줄), `scripts/gpu_blackbox.sh`(telemetry).
