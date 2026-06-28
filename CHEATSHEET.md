# 치트시트 — 실험환경 & 코드 디테일 (선배 Q&A 대비)

> 선배가 setup/코드/숫자 물어보면 바로 답하기 위한 빠른 참조. 근거 파일 경로 포함.
> 작성 2026-06-26. 지표는 **gap to optimum (%) = 100·(opt−achieved)/opt, 낮을수록 좋음**.

---

## 1. 실험 환경

| 항목 | 값 |
|---|---|
| GPU | **1× RTX 4070 Ti (12GB)** (시뮬컴 5090×2는 현재 타인 사용중이라 불가) |
| Env | conda `RIBBO`. 비대화형 셸에선 `python`이 conda 밖 → **`/home/myoungjin/miniconda3/envs/RIBBO/bin/python`** 직접 호출(또는 PATH 앞에 추가) |
| 필수 env var | `PYTHONPATH=.`, `WANDB_MODE=disabled` |
| OS | Linux kernel 6.8. GPU freeze는 **GSP 펌웨어 off**(`NVreg_EnableGpuFirmware=0`)로 해결 |
| 학습 시간 | 3000 epoch ≈ **2h40~50m** (D=20 기준). 250 epoch마다 체크포인트(resume 가능) |
| 평가 시간 | 200~500채널 × 300 step ≈ 10~20분 (EE는 multistart optimum 포함) |

---

## 2. 문제 정의 (IFC, Lee 2025 Sec V-A)

- 송수신 쌍 **D ∈ {3, 10, 20}**, action **x ∈ [0,1]^D** (쌍별 전력비).
- 채널 **h ~ CN(0,1)** (Rayleigh fading), 잡음 σ²=1, **Ptx=10 W, Pfix=1 W**.
- **SE** = Σ_d log(1+SINR_d) — 최적해 corner(켜고/끄기).
- **EE** = Σ_d log(1+SINR_d)/(Pfix+Ptx·x_d) — 최적해 interior(중간 전력).
- SINR_d = Ptx·|h_dd|²·x_d / (1 + Ptx·Σ_{d'≠d}|h_d'd|²·x_d').
- 코드: `problems/ifc.py:ifc_rates(H_sq, X, Ptx, Pfix, reward_type)`.
- **채널 생성**: `dataset_id`가 seed. train=채널 0–19, test=20–29, **평가(eval)=30번부터** (전부 unseen).
  `IFCNumpy(search_space_id, dataset_id, dim)`가 채널 행렬 H 고정.

---

## 3. RIBBO 모델 / 학습 설정 (`scripts/configs/dt/ifc_*.py`)

| 하이퍼파라미터 | 값 |
|---|---|
| backbone | causal transformer (GPT-2 류), add_bos=True, mix=concat |
| embed_dim / layers / heads | **256 / 12 / 8** |
| pos_encoding | embed (learned), use_abs_timestep=True |
| input_seq_len (context) | **50** (학습/평가 공통; config엔 300이나 run script가 50으로 override) |
| max_input_seq_len (horizon) | 300 |
| batch / dropout | 64 / 0.1 (attn·resid·embed) |
| optimizer | Adam, **lr 2e-4**, weight_decay 1e-2, betas (0.9,0.999), **warmup 10k steps** |
| epochs / step_per_epoch | **3000 / 100** (= 300k step) |
| init_regrets (RTG) | [0, 20, 50] |
| normalize_method | "random" (RTG 정규화에 y_max/y_min 사용) |
| x_type | stochastic (평가 시 deterministic=False) |
| y_loss_coeff / n_block | 0 / 100 |

- **RTG 정규화**: `info["y_max_mean"], info["y_min_mean"]` (dataset-wide). 정규화 y = (y−ymin)/(ymax−ymin), regret = (ymax−y)/scale.
- **평가 rollout**: 300 step, deterministic=False, regret_strategy="none", init_regret=0. (`scripts/eval_ifc_ee.py:rollout_ribbo`)
- 학습 자기 한 줄: `python scripts/run_dt.py --config <cfg> --name <run> --id <SSID> --device cuda:0 ...` (telemetry+resume는 `run_train_*.sh`에 내장).

---

## 4. Behavior Algorithm (BA) 풀 — 학습 데이터 생성기

| 풀 | 구성 | 비고 |
|---|---|---|
| **3BA** | Random, HillClimbing, CMAES | 기본. EE는 이게 best |
| **5BA-filter** | HillClimbing, RegularizedEvolution, EagleStrategy, CMAES, BotorchBO | Random+SGS 제외(Song BC Filter). SE 고차원 best |
| **7BA** | 5BA + Random + ShuffledGridSearch | Song의 BC 등가 |
| **4BA-filter** | HillClimbing, RegEvol, Eagle, CMAES | BotorchBO 제외(D=20 GP 느림). SE D=20용 |

- **Vizier(VizierGPBandit) 제외**: JAX/XLA "Cannot allocate memory" segfault (단일프로세스서도 재현). IFC엔 7개 BA만 동작.
- 데이터 규모: 풀당 **(BA수)×20 채널 × 300 step** trajectory. 3BA→60 traj, 5BA→100 traj.
- 생성: `run_datagen_ifc*.sh` (내부 `data_gen/data_gen_main.py`). 출력 `data/generated_data/<set>/seed0/<BA>_<SSID>_<ch>_0.json`.

---

## 5. 기준선(baseline) & 최적해(optimum)

| | 방법 | 코드 |
|---|---|---|
| **SE optimum** | **WMMSE** (Shi 2011), box-constrained, **10 restarts**, max_iter 200 | `problems/ifc.py:wmmse_sum_rate` |
| **EE optimum (D=3)** | brute-force 그리드 (grid=51, 전수) — 정확 | `ifc_optimum_bruteforce` |
| **EE optimum (D=10/20)** | multi-start projected gradient (L-BFGS-B, **40 restarts**: all-1, all-0.5, 강링크, 나머지 랜덤) | `ifc_optimum_multistart` |
| **Random** | 매 step 균등 [0,1]^D 샘플 (= Lee의 "brute-force"와 동일) | `rollout_random` |

- **검증**: D=3에서 multistart vs brute-force 일치(~1e-4); WMMSE vs brute-force gap 0/10 채널. EE optimum은 restart 40→200 늘려도 변화 0.000%(수렴), RIBBO가 한 번도 안 넘음 → 견고한 상한.
- **지표**: gap to optimum = 100·(opt−achieved)/opt. (Lee 논문도 "optimality gap |r*−r̄|"을 주 지표로 사용. Lee의 "local optimal" = WMMSE(SE)/FP(EE)를 50 random init 중 best — 우리 multistart와 같은 방식, 전수조사 아님.)

---

## 6. 학습된 모델 체크포인트 (`log/ifc/<run>/<...>/ckpt/<ep>.ckpt`)

| 모델 | run dir | ep |
|---|---|---|
| SE D=3 3BA | `dt-ifc-run1` | 3000 |
| SE D=3 7BA | `dt-ifc-run3-fullba-5000ep` | 5000 |
| SE D=3 5BA-filter | `dt-ifc-run4b-5bafilter-resume` | 5000 |
| SE D=10 3BA | `dt-ifc-d10-run1` | 3000 |
| SE D=10 5BA-filter | `dt-ifc-d10-run2b-5bafilter-resume` | 5000 |
| SE D=20 3BA | `dt-ifc-d20-3ba` | 3000 |
| SE D=20 4BA-filter | `dt-ifc-d20-4ba` | 3000 |
| EE D=3 3BA | `dt-ifc-ee-d3-3ba` | 2000(수렴; 6/23 크래시로 3000 미달, 결과 영향 없음) |
| EE D=10 3BA | `dt-ifc-ee-d10-3ba` | 3000 |
| EE D=10 4BA-filter | `dt-ifc-ee-d10-4ba` | 3000 |
| EE D=20 3BA (신규) | `dt-ifc-ee-d20-3ba` | 3000 |

---

## 7. 코드 맵 (어디에 뭐가 있나)

- `problems/ifc.py` — 핵심: `ifc_rates`(SE/EE 보상), `wmmse_sum_rate`, `ifc_optimum_bruteforce/multistart`, `IFC_DIM_MAP`/`IFC_REWARD_MAP`(SSID→dim/reward), `IFCNumpy`/`IFCTorch`/`IFCMetaProblem`.
- `scripts/configs/dt/ifc_*.py` — 학습 config (ifc, ifc_d10, ifc_d20, ifc_d20_ee, ifc_5bafilter, ifc_d10_ee, ifc_d10_ee_4bafilter 등).
- `scripts/configs/dataset_specs_ifc.py` — SSID별 train/test 채널 인덱스.
- `data_gen/data_gen_main.py`, `data_gen/ifc_problem_statement.py` — 데이터 생성.
- `run_datagen_ifc*.sh`, `run_train_*.sh` — 생성/학습 런처(telemetry+resume 내장).
- `scripts/eval_ifc_*.py` — 평가 (eval_ifc_ee, eval_ifc_d20, eval_ci, eval_hybrid, eval_ee_d20, eval_ee_d20_budget).
- `scripts/plot_gap_figs.py` — **발표용 gap 그림 전부 재생성**(npz→presentation/fig_*.pdf). `plot_synthesis/plot_hybrid/analysis_*` 도 있음.

---

## 8. 재현 명령 (conda python 직접 호출)

```bash
PY=/home/myoungjin/miniconda3/envs/RIBBO/bin/python
export PYTHONPATH=. WANDB_MODE=disabled

# 데이터 생성 (예: EE D=20 3BA)
PATH=$(dirname $PY):$PATH bash run_datagen_ifc_d20_ee.sh

# 학습 (체크포인트+resume 내장)
PATH=$(dirname $PY):$PATH bash run_train_ee_d20.sh

# 평가 (예: EE D=20 곡선)
$PY scripts/eval_ee_d20.py --n 200

# 발표 그림 전부 gap 버전으로 재생성
$PY scripts/plot_gap_figs.py
```

---

## 9. 핵심 결과 (gap to optimum %, 낮을수록 좋음)

| | D=3 | D=10 | D=20 |
|---|---|---|---|
| SE RIBBO (3BA) | 11.1 | 52.2 | 67.0 |
| SE RIBBO (best filter) | — | 28.6 | 58.1 |
| SE Random | 17.0 | 56.4 | 68.5 |
| EE RIBBO (3BA) | 3.5 | **21.9** | 69.8 |
| EE Random | 17.2 | 70.6 | 82.8 |
| **EE: RIBBO가 Random보다 작은 gap** | 13.7 | **48.7** | 13.0 |

- win-rate(채널별 RIBBO<Random gap): SE D=10 100%, EE D=10 100%, SE D=20 99.5%, EE D=20 100%.
- 95% CI(±, n=200): 전부 ±0.5~1.8. 핵심 2건 500채널 재확인(EE D=10 gap 20.7±0.7, SE D=20 58.6±0.6).
- budget(1000 step): SE D=3 gap 13→8.3, SE D=10 27.1→19.0, **EE D=20 69.9→69.7(회복 안 됨)**.
- hybrid(+local search): RIBBO·Random 모두 gap ~0~3 (모델 있으면 쉬움 → RIBBO는 no-model용).
- WMMSE 절대값(nats): D=3 2.864, D=10 3.968, D=20 4.764. (Lee 보고 2.50 ≠ 우리 2.864: 채널/세팅 차이, brute-force 검증.)

---

## 10. 알려진 이슈 / 주의

- bare `python`으로 스크립트 돌리면 conda 밖이라 **조용히 실패**(데이터 0개 생성 등). 반드시 conda python.
- Vizier BA는 IFC에서 깨짐(JAX OOM) → 제외.
- EE D=10/20 optimum은 'certified global' 아님(near-global 강한 참조). 단 RIBBO vs Random 격차 주장엔 무관.
- 수치는 stochastic policy라 재평가 시 ±1~2%p 변동.
- 데이터 작음(채널 20, traj 60~100). 평가는 200~500채널이라 통계 OK. "더 키우면 더 좋아질 여지"가 곧 scalability 메시지.
- 발표 deck은 `presentation/progress_0625.pdf`(8장) + `presentation/appendix_0625.pdf`(12장 백업). 모두 gap 지표.
