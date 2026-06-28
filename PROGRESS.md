## 2026-06-29 (밤샘 자율) — 발표 마감 패키지: GA baseline + 폴리시 + 팔로업 + 런북

학생 지시: 8시 출근 후 2시간 내 선배에게 [답변+팔로업+발표자료] 보낼 수 있게, 자는 동안 "과하다 싶을 정도로" 고품질 준비.

**1) GA(유전알고리즘) baseline 추가** — 선배가 "GA랑 비교스킴" 직접 언급:
- `scripts/eval_ifc_baselines.py:rollout_ga` 신규(numpy-only, 새 의존성 X). 토너먼트+SBX(η15)+다항변이(η20,p=1/D)+elitism, pop20, pymoo 기본 연산자, 튜닝 안 함. (pymoo/deap 미설치 확인 → pip install은 학생 부재로 회피)
- n=200 전 케이스. 결과(gap%): EE D10 RIBBO 21.5 / **GA 71.2** / DE 48.6 / CMA 26.8 / Rand 70.6. GA는 빠듯 예산서 가장 약함(고차원 Random과 동급), D=3만 거의 최적. **RIBBO가 GA를 EE D10/D20·curated SE D10/D20서 100% 격파.**
- 그림 `results/18_de_baseline/baselines_with_ga.pdf/-1.png`, README §결과 1b 추가.

**2) 발표자료 폴리시 마무리**(선배 §B): ML 약어 첫등장 풀이(CMA-ES/BO/SBX 등), 숫자 톤다운(표 유지·본문 % 절제), 영어 격식화. overflow 0, 8장 유지. R4 그림 가독성(상하 스택)도 반영됨.

**3) 선배 전송 패키지 신규**:
- `presentation/followup_senior_0629.{tex,pdf}`(2쪽, 한글) — 피드백 반영요약 + GA 포함 baseline 표 + 정직한 한계 + 다음(LLMO).
- `presentation/followup_senior_0629.txt` — 선배 보낼 커버 메시지 복붙용(긴/짧은).
- `MORNING_RUNBOOK.md`(루트) — 8–11시 스텝바이스텝, 파일 인벤토리, 검증 체크리스트, 선배 되물음 즉답카드, 재컴파일 명령.

**다음 세션/아침**: 학생이 런북 따라 검증→전송. 남은 결정 3건(통신약어 더 풀지/숫자 톤다운 정도/영어 추가)은 학생 확인 대기. GitHub(omjinLTS/RIBBO-IFC main)에도 push 예정.

---

## 2026-06-28 (새벽) — 메인 deck에 선배 피드백 반영 (progress_0625)

학생 결정: h1000(A) finding은 발표에 안 씀(기록만), 메인 흐름 유지 + 선배 피드백 적용.
`presentation/senior_feedback_0626.md` §C 액션을 `progress_0625.tex`에 반영(8장, overflow 0, pdflatex+Fira).
- Overview: SE corner/EE interior 한 줄 설명, 기준해 D별 분리(SE=WMMSE / EE D3=전수 / D10·20=multistart near-global), unseen=같은 분포 새 realization 명시, 'optimum'→'reference(local-optimal)' 용어 정리, ML 약어 풀어쓰기(RIBBO/RTG/HRR/BA).
- R1: 제목 (SE) + heading "Which BA pool works best depends on the dimension"(flips 제거) + BA 구성 3줄 명시(비포함관계). R2: 헤더 Δ(Random−RIBBO), "model per reward·dim" 명시. R3: 제목 "Scalability"→"Effect of Problem Dimension" + "차원마다 별도 모델" 명시. R4: Random=Lee Brute-force 동치 + "못 가는 영역" 프레이밍 + hybrid 별도 bullet. R5: "SE D=10" 라벨 + 약한 config 주석, 'as-is'→'unchanged'.
- 그림(`scripts/plot_gap_figs.py` 수정 후 재생성): 메인 도면 위 제목 제거, y축 'optimality gap (%)' 통일, 기준선 'reference (gap 0)' relabel.
- 학생 지적("R4 그림 안 보임"): per_channel CDF 2패널을 좌우(13×5)→상하 스택(7.2×8.6)으로 바꾸고 폰트 키움(title20/label18/legend17), 슬라이드는 figure height 0.86\textheight·캡션 제거로 컬럼 꽉 채움. 가독성 확보.
- overflow 잡느라 R3 본문/그림 height 트림(여러 패스). 최종 0.
- **미적용(주관적, 학생 확인 대기)**: B4 숫자 톤다운(표는 유지), B5 영어 톤 추가 정리, 통신/공통 약어(CMA-ES/BO/WMMSE/SINR) 풀어쓰기.
- 산출물: `presentation/progress_0625.pdf` 갱신. 학생 텔레그램 전송.
- 학생 요청("정리해서 답변"): 선배 질문별 **답변지** 작성 `presentation/senior_feedback_answers_0628.{tex,pdf}`(xelatex, 3쪽, Q→답→반영완료, 남은결정 빨강). 선배 회신/숙지용. 남은 결정 3건: 통신약어 풀이 / 숫자 톤다운 정도 / 영어 추가 손질 — 학생 결정 대기.

### GitHub 백업/싱크 (다른 컴 이어작업용)
- 학생 요청: 코드+데이터+모델 전부 본인 repo에 올려 다른 컴에서 이어작업. remote 구성: `origin`=git@github.com:omjinLTS/RIBBO-IFC(SSH, omjinLTS 인증 확인), 기존 `lamda-bbo/RIBBO`는 `upstream`으로. branch master→**main**.
- `.gitignore` 재작성: data/cache/results 포함, **log/**/*.ckpt 무시 후 최종 ckpt만 force-add**, 논문 PDF(`/*.pdf`)·세션로그 제외.
- 환경: pip freeze가 conda `file://` 경로라 비포터블 → **environment.yml**(conda env export) 생성이 정답(`conda env create -f environment.yml`). requirements.txt(핵심 pip 5개)도 유지.
- commit `94a9559`: 1440파일 / **785MB** / 최종 체크포인트 **17개**(TOY/smoke/1-epoch 크래시 잔여는 제외). push 성공(main→origin/main).
- 주의: run_*.sh·CHEATSHEET의 python 경로 `/home/myoungjin/miniconda3/...` 하드코딩 → 타 머신선 conda activate 후 `python` 사용. 중간 체크포인트 4.3GB(resume전용)는 미포함(원하면 추가 push).

---

## 2026-06-27 (오후) — horizon-1000 재학습 완료 + 공정 비교 (deferred P1 해결)

- EE D=10 horizon-1000 재학습 완료(`dt-ifc-ee-d10-3ba-h1000`, 3000ep, 3000.ckpt, config.txt: max_input_seq_len=1000). 평가 `scripts/eval_ee_d10_h1000.py`.
- ★**평가 정규화 버그 발견·수정(이번 세션)**: `eval_ifc_ee.stats()`가 RTG 정규화 ymax/ymin을 `max_input_seq_len=300`으로 하드코딩 → 1000으로 학습된 h1000 모델에 **train/eval 정규화 불일치**(scale 0.811 vs 0.893, ymin 0.103 vs 0.034)로 regret 신호 왜곡. 공용 함수는 안 건드리고 h1000 전용 `stats_at(...,1000)` 추가해 학습 horizon과 일치시킴.
  - **고쳐도 결론 불변**(불일치가 modest): h1000 gap **52.3(@300)→40.2(@1000)** (버그판 54.0→39.0과 사실상 동일, 차이는 stochastic ±2%p). 즉 반직관 결과는 정규화 아티팩트가 **아님**.
- 데이터/plumbing 정상 검증: JSON·캐시 모두 X=[1000,10] 길이 1000, ts clamp=999(300 클램프 아님).
- **반직관 결과(검증됨)**: 재학습 h1000(@1000 gap **40.2**)이 원래 h300(**15.7**)보다 **모든 구간에서 더 나쁨**(300 이내도 52.3 vs 20.6). h300의 빠른 초반 하강을 잃음.
  → "h300를 1000까지 외삽=불공정" 우려는 사실 RIBBO에 유리했던 것. 공정 비교(h1000)는 결론을 **강화**:
  CMA-ES(0.4)가 RIBBO-h1000(40.2) 압도. 장-budget=CMA-ES 영역, RIBBO 가치=단-budget 한정.
- **undertraining 아님(tb 확인)**: h1000 x_loss가 -38.4에서 plateau(last-20% slope ≈ +0.006/ep ≈ 0), h300은 -48.5로 더 낮게 수렴. h1000은 "수렴했는데 더 나쁜 정책"으로 수렴. (단 NLL은 horizon 다르면 직접 비교 불가 → 격차의 직접 증거는 eval gap.)
- **비교 공정성 검증(학생 "h300이 더 나은 게 말 안 됨" 지적 대응)**: h300 baseline(results/09 npz)은 같은 100채널(30~129)·같은 optimum(reuse)·같은 롤아웃 코드, h300은 seq_len=300+timestep clamp 299(외삽), h1000은 1000, 각자 올바른 정규화. → h300>h1000은 아티팩트 아님(진짜).
- **why = 데이터 희석(측정 확인)**: 둘은 "같은 모델 길게 본 것"이 아니라 서로 다른 데이터로 학습된 다른 모델. 같은 3BA를 1000-step로 늘리면 BA가 초반 수렴 후 평탄 → 1000-step traj의 **97.7%가 수렴 후 flat**, 학습용 50-step window 중 '새 best 개선 있는' 비율이 h300데이터 **~58%** → h1000데이터 **36%**로 급감. h1000은 같은 3000ep에 "탐색·개선" 패턴을 훨씬 덜 보고 배움 → 약한 explorer → 모든 budget서 나쁨.
- **인사이트**: horizon 늘리려 BA 롤아웃만 길게 뽑으면 평탄 tail만 추가돼 역효과. long-horizon RIBBO엔 1000-step 내내 개선하는 데이터(더 센/다양한 BA, 채널↑) 필요. (측정 스크립트: 임시, data y-array의 best-so-far 개선 window 비율.)
- 결과: `results/19_budget_baselines/ee_d10_h1000.{pdf,png,npz}`(정규화 수정판으로 재생성). deferred P1 메모 완료 처리.

---

## 2026-06-27 — 세션 재시작 스크립트 진짜 원인 규명 + 수정 (검증됨)

기존 "미검증" 수정에도 재시작이 계속 실패하던 진짜 원인을 `.session_restart.log`에서 발견:
- **근본원인**: `claude --channels` 가 **variadic 옵션**이라, `--channels plugin:telegram@... "<BOOT_PROMPT>"`
  형태에서 BOOT_PROMPT까지 채널 항목으로 삼켜 `--channels entries must be tagged` 파싱 에러 → 새 세션이
  뜨기도 전에 죽음. (실제 동작 중인 세션 cmdline 은 trailing prompt 없이 `claude --channels plugin:telegram@...`.)
- **수정**(`scripts/restart_session.sh` do_launch): usage `claude [options] [prompt]` 에 맞춰 프롬프트를
  `--channels` **앞**으로 이동: `exec claude "$BOOT_PROMPT" --channels plugin:telegram@claude-plugins-official`.
- **검증**: 새 순서를 **bogus 채널**(`plugin:__nope__@__nope__`, 실제 텔레그램 미연결→409 없음)로 실행 →
  프롬프트가 정상적으로 positional prompt 로 파싱됨(claude 가 응답), "must be tagged" 에러 사라짐. `bash -n` 통과,
  기존 세션(PID 2297798) 무영향, 잔존 프로세스 없음. 실제 채널 end-to-end 는 현재 세션 교체 위험으로 미실행
  (가장 확실한 검증·복구 = 터미널에서 `bash scripts/restart_session.sh` 직접 실행).
- 참고: 이 시점에 텔레그램 bun poller 가 또 스스로 종료(MCP 끊김) — 알려진 flakiness, 스크립트와 무관.

---

## 2026-06-26 (오후) — 선배 발표초안 피드백 정리 + DE baseline 추가

세션 재시작(부트스트랩) 후 텔레그램으로 학생 운영.

**1) 선배 피드백 정리** (`presentation/senior_feedback_0626.{md,pdf}`, `_phone.tex`/`_wide.tex`):
- 선배가 progress_0625 draft 보고 슬라이드별 질문 19개 + 교수님 대비 마이너 코멘트 5개 줌.
- 슬라이드 소스(`progress_0625.tex`)·결과파일로 사실 확인 후 "질문→답→수정 액션" 표/카드로 정리.
  폰에서 보기 좋게 세로 카드형 PDF(3p, Noto Sans CJK, xelatex)로 제작.
- 핵심: 질문 절반은 슬라이드 라벨/정의 누락(SE/EE·차원·BA구성·reference종류·unseen정의)이 원인.
  사실정정 1건(5BA-filter는 Rand+SGS "추가"가 아니라 7BA에서 "뺀" 것). 선배 지적 타당: "Scalability"
  표현 과함(차원마다 별도 모델, dim-agnostic 아님), WMMSE를 "optimum"이라 단정 금지(local optimal).
- **아직 슬라이드 본문 미수정** — 학생 지시 대기.

**2) DE(Differential Evolution) baseline 추가** (`scripts/eval_ifc_de.py`, `results/18_de_baseline/`):
- 선배가 "DE 그래프보다 잘 나와야 의미 있다" → DE를 강한 black-box baseline으로 비교.
- scipy DE(rand/1/bin, polish=False=순수 black-box), 300 eval 예산, 동일 unseen채널/optimum/gap지표.
  RIBBO/Random/optimum은 기존 npz 재사용 → DE만 계산(apples-to-apples). pop~24(tight budget용).
- **결과(final gap %, 낮을수록 좋음, 200ch×300eval)**:

| Case | RIBBO | DE | Random | RIBBO<DE 승률 |
|---|---|---|---|---|
| EE D=3 | 3.2 | **0.8** | 17.6 | 20% (DE 우세) |
| EE D=10 | **21.5** | 48.6 | 70.6 | **100%** |
| EE D=20 | **69.8** | 74.4 | 82.8 | 86% |
| SE D=10 3BA | 52.2 | **40.9** | 56.7 | 22% (DE 우세) |
| SE D=10 5BA-filter | **28.4** | 40.9 | 56.7 | **91%** |
| SE D=20 3BA | 66.8 | **61.5** | 68.5 | 22% (DE 우세) |
| SE D=20 4BA-filter | **58.0** | 61.5 | 68.5 | **75%** |

- **핵심**: DE는 Random보다 훨씬 셈(=진짜 경쟁자). RIBBO는 자기 무대(고차원+EE+curated pool)에서 DE
  격파(EE D=10 전채널 승). SE는 **curated pool이 DE를 역전시키는 레버**(3BA는 DE 우세 → 5BA/4BA-filter로
  RIBBO 역전). DE가 이기는 곳(EE D=3 저차원, 약한 3BA SE)은 RIBBO 원래 약한 영역 = 기존 메시지와 일치.

**3) CMA-ES + BO 추가** (학생 "(b) CMA-ES+BO 해봐", `scripts/eval_ifc_baselines.py`):
- 동일 300 eval·동일 채널·gap. CMA-ES=pycma(빠름), BO=botorch GP-EI(느림: amortized hyper-fit로 ~2.5분/ch, n=30만).
- **CMA-ES가 제일 센 경쟁자**(DE보다 강함): RIBBO가 CMA-ES를 "확실히" 이기는 건 **EE D=10 한 곳**(21.5 vs 26.8,
  중앙값·65%채널). 나머지(SE 전부, EE D=20, EE D=3)는 CMA-ES 우세. RIBBO가 학습 prior 갖고도 from-scratch
  CMA-ES에 대부분 밀림 = **솔직한 한계**. (전 케이스 n=200 표는 results/18 README.)
- **BO**: 저차원 강자(EE D=3 2.0, 거의 최적)지만 **고차원서 붕괴**(EE D=10 52.6, GP-EI의 moderate-D degrade) →
  RIBBO가 EE D=10서 BO 전 채널(100%) 압도. **공정성 검증**: full BO(53.1)≈amortized BO(58.2), 가속이 패배 원인 아님.
- **EE D=10 = RIBBO의 깨끗한 승리**: DE/CMA-ES/BO 3대 강한 black-box 최적화기 모두 격파(20.2 vs 46.7/26.6/52.6).
  + **sample-efficiency**(초반 적은 eval서 RIBBO가 앞섬) = "왜 RIBBO" 가장 강한 프레이밍.
- 그림: `results/18/baselines_final.pdf`(RIBBO/DE/CMA/BO/Random), `baselines_ee_dc/se_dc.pdf`. 텔레그램 전송.

**4) 확장-budget 비교 (선배 피드백 "D=10/20은 budget 길게 봐야")** (`scripts/eval_ifc_budget_baselines.py`, `results/19`):
- 300→1000 eval. RIBBO@1000/Random@1000은 budget npz 재사용(동일 채널), DE/CMA-ES@1000만 새 계산. BO 제외(1000-점 GP 비현실 + 300서 정체).
- **결정적 발견**: 길게 주면 CMA-ES가 사실상 문제를 풂(gap @1000: EE D=10 **0.4**, EE D=20 **8.1**, SE D=10 **5.8**). RIBBO는 정체(15.7/69.8/18.9).
- **300서 RIBBO가 이기던 EE D=10이 1000서 완전 역전**(CMA-ES 0.4 vs RIBBO 15.7). 교차점이 정확히 ~300 eval(이전 budget 경계). → RIBBO 유일 우위(EE D=10)가 **단-budget 아티팩트**.
- 인사이트 강화: IFC는 "budget 충분하면 적응형 최적화기로 풀리는" benign 문제 → 학습형 니치 = 측정 극소 단-budget 코너로 한정.
- caveat(정직): RIBBO는 horizon 300 학습이라 300 너머 외삽 → 정체에 학습-horizon 한계 일부 기여(더 긴 horizon 재학습 시 개선 여지). 단 EE D=10 교차는 horizon 내 발생, 결론 유지.
- 학생 입장 재확인: RIBBO 방어 아님, "쓸 만한가+인사이트" 목적. 현재 정직한 결론 = 이 문제엔 대체로 CMA-ES면 충분; 진짜 연구질문은 CMA-ES vs LLMO(학습형 전체의 정당성).

**5) collapse 시각화 + horizon-1000 재학습 (학생 지시, 진행 중)**:
- **collapse 확인 완료** (`scripts/eval_ribbo_collapse.py`, `results/19/ribbo_collapse.pdf`): RIBBO 제안 x의 탐색반경(15-step std)을 step별로. EE D=20는 step~100에 0.04로 붕괴(uniform 0.289 대비), EE D=10는 0.25로 넓게 유지. → D=20 정체는 objective local-optima가 아니라 **정책의 early collapse**(고차원 capacity 한계). CMA-ES는 같은 지형 92% 도달.
- **horizon-1000 재학습 (진행 중)**: 학생이 "300 넘어 외삽이라 unfair, 1000으로 재학습해 비교"를 정확히 지적 → EE D=10 재학습.
  - 1000-step 3BA 데이터 생성 완료(`run_datagen_ifc_d10_ee_h1000.sh` → `data/generated_data/ifc_d10_ee_h1000`, 60 traj len=1000).
  - config `scripts/configs/dt/ifc_d10_ee_h1000.py`(max_input_seq_len=1000, 새 cache). ⚠️ 핵심: `datasets.py:68` max_input_seq_len이 traj 자름 → 1000 필수. `problem.seq_len=X.shape[0]`라 데이터 길이가 임베딩 크기 결정.
  - 학습: `dt-ifc-ee-d10-3ba-h1000`, 3000ep, ~2.9h, 진행 중(2026-06-27 오전).
  - 평가 스크립트 준비됨 `scripts/eval_ee_d10_h1000.py`(seq_len=1000, clamp 999, 1000 step 롤아웃, 동일 100채널/optimum, h300·CMA-ES@1000 비교).
- **예상(미검증)**: 재학습해도 CMA-ES@1000(gap 0.4) 못 이길 것(distillation 천장 + 채널20). 학습 끝나면 검증.

**다음 세션 시작 시 필요**:
- 학생 결정: (a) 슬라이드 수정 착수 여부/순서, (b) baseline 비교(DE/CMA-ES/BO)를 deck/appendix 어디에 넣을지.
- **h1000 학습 끝났는지 확인** → `scripts/eval_ee_d10_h1000.py` 실행해 공정 1000-eval 그림 생성. (ckpt: `log/ifc/dt-ifc-ee-d10-3ba-h1000`)
- (선택) SE D=20도 1000 eval로. LLMO 비교(다음 메인).
- 학생 입장: RIBBO 방어 아님 / 인사이트 목적. 현재 정직한 결론 = 이 문제는 대체로 CMA-ES면 충분, 학습형 니치는 단-budget·고차원 EE 코너로 한정.
- 권장: EE D=10 슬라이드에 4종 baseline 그림(RIBBO가 DE/CMA/BO 다 이기는 유일·핵심 케이스) + "왜 RIBBO=고차원EE+적은예산"
  메시지. CMA-ES가 대부분 regime서 RIBBO보다 센 건 정직하게 한계로 명시.
- (선택) BO를 n=200으로 확대(시간 큼) 또는 SE D=20/EE D=20에도 BO 추가.

---

## 2026-06-26 (오전5) — 세션 재시작 스크립트 버그픽스

학생: "bash로 실행해도 오류나서 수동으로 프롬프트 넣어 재시작" → 원인 규명 + 수정.
- **근본원인**: `scripts/restart_session.sh`의 `pkill -f 'claude --channels plugin:telegram'`가
  claude 부모만 죽이고 텔레그램 polling을 도는 **bun 자식**(`claude-plugins-official/telegram`)은
  살려둠 → 새 세션의 Telegram getUpdates **409 conflict**로 채널 초기화 실패(=학생이 본 "오류"로 추정).
  추가로 대기 1~2s 부족, claude 경로를 conda bin(claude 없음)에만 의존.
- **수정**: bun poller까지 종료 + 프로세스 사라질 때까지 ≤15s 폴링 + 강제킬(-9) + 3s settle,
  `command -v claude`로 경로 해석(폴백 `~/.local/bin/claude`), 터미널(A)/분리(B) 모드가
  `do_launch`/`kill_old_sessions` 공유하도록 리팩터.
- **검증(안전 범위)**: `bash -n` 통과, claude 경로 `/home/myoungjin/.local/bin/claude` 확인,
  kill 패턴이 claude 부모/bun poller 정확 구분, claude positional-prompt 파싱 정상(print 테스트).
  **미검증**: 실제 exec(현재 세션 죽여야 해서). 가장 확실=터미널 직접 `bash scripts/restart_session.sh`.
- 작업 중 텔레그램 bun poller가 스스로 종료 → MCP 끊김. 재시작으로 채널 복구 겸 스크립트 검증.

**다음 세션 시작 시**: 재시작이 깨끗이 떴는지 확인. 또 오류면 정확한 에러 메시지 받아서 마저 수정.

---

## 2026-06-26 (오전4) — 발표 톤 정리 + 세션 재시작

- 선배 피드백 반영(메인 deck): Result 3 제목 "(key)" 제거; Result 4에서 "not just luck" 삭제,
  "never"→"stays above 50% gap within budget", "Honest check" 라벨 제거, 주관어 "easy" 제거;
  Insights를 단정/추론 줄이고 관찰형(과거형)으로, 주관 평가 축소. appendix도 "easy/never" 정리.
- 둘 다 재컴파일(8장/12장, overflow 0).
- 작업 후 세션 재시작(출장 대비, 이후 텔레그램 운영). 업데이트 문서 텔레그램 전송.

---

## 2026-06-26 (오전3) — 세션 재시작 스크립트

텔레그램 "세션 새로 시작해" → 재시작 플로우 구축:
- `scripts/restart_session.sh` (TTY면 exec 교체 / 세션 내부면 setsid 분리 후 기존 pkill + 새 세션 실행)
- `scripts/restart_bootstrap.txt` (새 세션 부트스트랩: 텔레그램 "시작됐습니다 파악중" 전송 → HANDOFF/PROGRESS/
  FINAL_REPORT/MASTER_GUIDE/CHEATSHEET/CLAUDE 읽기 → "파악 완료" + 요약 전송)
- `HANDOFF.md` 상단에 "최신 상태" + "세션 재시작 프로토콜" 추가(트리거 시 현재 세션이 HANDOFF 갱신 후 스크립트 실행).
- 한계: TTY 없이 새 telegram 세션이 떠야 하므로 환경따라 분리모드 실패 가능 → 터미널 직접 실행이 가장 확실.

---

## 2026-06-26 (오전2) — 지표 gap 전환 + deck 정리 + 치트시트

선배 초안 검토 대비 정리.
- **지표를 gap to optimum(=100−%, 낮을수록 좋음, Lee optimality gap)으로 전환**: 그림 전부 재작도
  (`scripts/plot_gap_figs.py`, npz→presentation/fig_*.pdf, 곡선은 gap이 0으로 감소하는 Lee 스타일),
  deck/appendix 표·숫자·라벨 전부 flip. 근거: Lee 본문 주 지표가 optimality gap(이론/수렴/Fig16),
  "% of optimal"은 1회뿐.
- 메인 deck: **Next steps&Ideas 슬라이드 삭제(9→8장)**, Insights 간결화(서브불릿 제거, 5줄).
- overflow/깨짐 전부 수정(메인·appendix 모두 overfull-vbox 0). 메인 8장 / appendix 12장.
- **`CHEATSHEET.md` 신규**(루트): 실험환경/문제setup/모델config/BA풀/baseline·optimum/체크포인트/코드맵/
  재현명령/핵심숫자(gap)/주의 — 선배 Q&A 즉답용.
- `MASTER_GUIDE.md`(8장 반영 + gap 안내 + §8 gap표), `FINAL_REPORT.md`(상단 gap 안내; 과거 % 기록은 유지)
  정합성 정리.

---

## 2026-06-26 (오전) — EE D=20 budget extension (negative finding)

학생 "2.5h 안에 끝날 것 아무거나" → EE D=20 budget 회복레버 테스트(eval-only).
**결과(negative)**: 300→1000 step 연장해도 RIBBO 30.1→30.3% (**+0.2p**). SE(D=10 +8.1p)와 대조.
RIBBO가 plateau saturate → EE D=20 gap은 budget 부족이 아니라 recipe/데이터 한계. "회복 레버"
주장 정밀화: budget은 모델이 개선 중일 때만 효과. (results/16/eval_ee_d20_budget)
appendix_0625.pdf에 슬라이드 추가(11장), FINAL_REPORT/이전 EE D=20 "회복 기대" 문구 정정.
→ 솔직한 negative finding. 다음: EE D=20 회복은 채널수↑/데이터↑로 시도해야.

추가(synthesis 그림): `scripts/plot_synthesis.py` → `results/17_synthesis/`. % of optimum vs 차원(3/10/20)
을 SE·EE × RIBBO·Random 한 그림으로 종합(전부 3BA). appendix 2번 슬라이드 + **메인 deck Result 3
(scalability)를 이 그림으로 교체/업그레이드** (SE-only 표+D20 곡선 → SE&EE 종합도 + 회복레버 bullet,
EE budget 미회복 caveat 포함). 메인 deck Result 번호 gap(3→5→6) 정정 → Result 1~5 연속.

(추가 조정) 학생 피드백 반영: 슬라이드2를 "지난 미팅" 서술식 → "Overview" 간략식으로. 전반적으로
청중(한국인)·발표자(석사 1학기) 눈높이에 맞게 어려운/생소한 용어 평이화: in-context/learned prior/
gradient-free/L-BFGS-B/zero-shot/CDF/sampling artifacts/no free lunch/curation/recovery levers 등을
풀어쓰거나 제거. 도메인 용어(WMMSE/SINR/SE·EE/RIBBO/RTG/HRR)는 유지. 사실오류 2건도 정정:
Ideas의 "EE엔 clean classical solver 없음"(틀림) 제거, "RIBBO warm-start"(hybrid 결과로 반증됨) 제거.

(추가) 발표 숙지용 `MASTER_GUIDE.md`(루트, 한국어) 작성: 한줄메시지/큰그림/핵심용어 쉬운설명/슬라이드별
말할거리/예상Q&A 10개+정직한답변/한계/appendix 사용시점/발표흐름/숫자 참조표. 발표자(석사1학기) 숙지용.

---

## 2026-06-26 — 밤샘 자율작업 (분석 A/B/C/D + EE D=20 학습)

학생이 "내일 10시까지 알아서 고도화" 지시 → 후보 제시 후 A+B+C+D+H+E 승인받아 진행.
GPU 1장(4070 Ti)이라 순차 실행.

**완료**:
- **E (EE D=20)**: IFCEED20 추가(ifc.py 맵, dataset_specs, dt config, datagen/train 스크립트) →
  토이검증 → 60 traj 생성 → 3000ep 학습(23:29~02:13, 2h44m, 무크래시) → 평가.
  RIBBO 30.2%±0.8 vs Random 17.2%±0.3, **+13.0p, win-rate 100%**. (results/16)
- **A 수렴속도**: SE/EE D=10에서 Random은 예산 내 최적 50%도 미달, RIBBO만 도달. (results/12)
- **B 채널별 통계**: D≥10 win-rate 100%(모든 채널), CDF 분포 이동. (results/13)
- **C 신뢰구간**: 전 결과 95% CI(±0.5~1.8) + 핵심 2건 500채널 재평가 확인. (results/14)
- **D Hybrid**: RIBBO+polish 99.8/96.6% 도달하나 Random+polish도 99.3/97.4% → polish 지배,
  init 무관. IFC가 gradient에 benign → RIBBO 가치는 gradient-free regime. (results/15) (솔직한 발견)
- **H 통합**: 메인 deck에 Result 5(robustness) 추가→9장. appendix_0625.pdf 10장(백업).
  FINAL_REPORT 맨 위 "0. 2026-06-26" 섹션 추가.
- CLAUDE.md/메모리: GPU 실사용 1장(4070Ti) 정정, bare-python 함정 명기.

**막힘**:
- 없음. (EE D=20 절대 %가 30%로 낮음 — recipe decay, 의도된 한계. 회복 레버는 미실행.)

**다음 세션 시작 시 필요**:
- 학생 검토: 새 슬라이드/insight 채택 여부, EE D=20 회복(4ba/budget) 추가 실험할지
- (선택) hybrid 결과를 limitation으로 명시할지 vs "near-optimal 솔루션 싸게 얻음"으로 포지셔닝할지

**advisor 미팅 질문 후보**:
- hybrid 발견(gradient 있으면 local solver로 충분)을 어떻게 프레이밍? → RIBBO는 model-free/CSI-free
  상황(LLMO와 같은 전제)에서 정당화. LLMO 정면비교가 다음 핵심.

---

## 2026-06-25 — 발표자료 작성 (progress_0625 deck)

HANDOFF.md + results/FINAL_REPORT.md 기반으로 advisor 발표 deck 작성.

**완료**:
- `presentation/progress_0625.tex` → `progress_0625.pdf` (10장, AICL beamer, 영어) 컴파일 성공
- 기존 5/18 `presentation/main.tex`는 그대로 보존(history). 새 파일로 분리.
- 그림 6개를 `results/{04,07,08,09,10,11}`에서 `presentation/fig_*.pdf`로 복사해 self-contained화
  (fig_wmmse_d3, fig_ablation_2x2, fig_ee_se, fig_d20, fig_budget, fig_transfer)
- 1차 10장 작성 후, 학생 피드백("지난번 발표 내용은 크게 넣지 말고 progress/insight/추가아이디어
  에 포커스")에 따라 **8장으로 재구성**:
  Title / Since-last-meeting(recap 1장) / Result1 BA-pool×차원(key) / Result2 SE-vs-EE(★) /
  Result3 Scalability+회복레버 / Result4 Transfer 실패 / **Insights(통합 5개)** /
  **Additional ideas(LLMO 정면비교 등)**
- 지난번(5/18)에 보여준 Problem/Method/pipeline/WMMSE-D3 슬라이드는 recap 한 장으로 압축
- Insights #2 초안("EE엔 깔끔한 고전 solver 없음")은 **Lee 논문으로 검증 결과 틀림** → 수정.
  Lee 2025 p.11394: "Local optimal: fractional programming [54](Shen&Yu 2018) for EE, WMMSE [53] for SE."
  즉 EE도 깔끔한 고전 해법(FP)이 존재. 진짜 차별점은 "model-based(WMMSE/FP/GNN, 채널모델 필요) vs
  model-free(RIBBO/LLMO, reward만 query)". Insight #2를 이 model-free 프레이밍으로 교체(정확+더 강함).
- 모든 수치 FINAL_REPORT·results README와 대조 확인 (88.9/47.8/33.0, EE +48.7p, D=20 4BA 41.9%, budget 91.7/81.0 등)
- 완성 PDF는 Telegram으로 학생에게 전송함

- 학생 피드백("결과 그래프가 단순하다, 설명 붙여라")에 따라 결과 그림 4개를 npz 원본에서
  재생성: `scripts/make_slide_figs.py`. 1차로 화살표·callout 텍스트를 넣었으나 학생이 "글씨
  겹쳐서 오히려 별로"라 해서 **그래프 내 텍스트 주석 전부 제거** → 최종은 큰 폰트 + gap 음영
  band + 범례 % 표기만. 설명은 슬라이드 그림 캡션 1줄 + 왼쪽 bullet으로 분리(겹침 없음).
  재생성 %는 report와 정확히 일치(검증). 시스템/구조(TikZ) 그림은 학생 요청대로 미변경.

**막힘**:
- 없음.

**다음 세션 시작 시 필요**:
- 학생 피드백: insight #2(EE 정당화 논리) 채택 여부, 슬라이드 추가/삭제 의견
- (선택) fig_budget.pdf 복사돼 있으나 현재 deck 미사용 — budget 별도 슬라이드 원하면 추가

**advisor 미팅에 가져갈 질문 후보**:
- "왜 RIBBO인가"의 답을 EE(interior-optimum, +48.7p)로 잡는 프레이밍이 advisor 의도와 맞는지
- D=3 corner-solution(SE)에서 RIBBO 의의 약함을 limitation으로 솔직히 제시하는 게 맞는지

---

## 2026-06-25 — 크래시 근본원인 해결(GSP 펌웨어) + EE 결과 완성

### 시스템 크래시 — 근본원인 규명 및 해결 ★

4번째 크래시(6/24 20:09) 때 처음으로 **black-box 로그가 freeze 직전을 포착**:

```
20:08:44  66°C  util82%  173W   ← D=10 EE 학습 막판 (정상 부하)
20:08:46  53°C  util 1%   79W   ← 부하 종료, D=20으로 전환 시작
20:09:06  45°C  util 4%  9.6W   ← ★마지막 기록: 유휴·냉각 상태에서 freeze
```

→ **freeze가 유휴·저온·저전력 상태에서, 학습→학습 전환(CUDA context 재초기화) 구간에 발생.**
이로써 **열/전력/ASPM/sshfs/swap 전부 배제** 확정 (완화책 다 적용했는데도 터졌고, 게다가 부하 중이 아니었음).

**진짜 원인 = NVIDIA GSP 펌웨어.** `nvidia-smi -q`에 `GSP Firmware Version: 595.71.05` 활성 확인.
최신 5xx 드라이버는 GPU 스케줄링을 GPU 내부 GSP 마이크로컨트롤러로 offload하는데, RTX 40 + 신규
드라이버에서 GSP가 간헐적으로 멈추면 호스트가 PCIe에서 무한대기 → 커널 로그 없이 전체 freeze.
4번 모두 hung_task/Xid/MCE 로그 0건인 것과 정확히 일치.

**조치 (드라이버 버전 유지, 모듈 옵션만)**:
```bash
echo 'options nvidia NVreg_EnableGpuFirmware=0' | sudo tee /etc/modprobe.d/nvidia-gsp.conf
sudo update-initramfs -u && sudo reboot
# 확인: nvidia-smi -q | grep "GSP Firmware"  →  N/A (꺼짐)
```
(드라이버 535 다운그레이드는 커널 6.8 DKMS 빌드 실패로 블랙스크린 → 롤백. GSP off가 정답.)

**검증 — 압축 스트레스 테스트** (`scripts/gpu_stress_test.sh`): CUDA context 생성/파괴 70회 +
풀로드 버스트 + 부하↔유휴 전환 반복(=크래시 4건의 조건 압축)을 **풀파워 254W / 77°C**로 ~15분.
→ **70/70 무사 통과, freeze 없음.** 이후 EE 평가(~10분)도 무탈. GSP off가 해결책으로 확정적.

운영 도구: `scripts/gpu_blackbox.sh`(2초마다 fsync telemetry), `scripts/train_resilient.sh`,
`run_train_ifc_d20.sh`(telemetry+resume+250ep 체크포인트 내장). 학습은 전부 telemetry 켜고 돌림.

### EE 결과 완성 (Task #2)

크래시 직전 살아남은 D=10 EE ckpt(3000, 무결) + D=3 EE(2000, 수렴) 평가 완료.
`results/09_ifc_ee/` (200 unseen ch × 300 step):

| | RIBBO-EE | Random | Optimum | RIBBO% | Rand% | RIBBO−Rand |
|---|---|---|---|---|---|---|
| EE D=3  | 0.559 | 0.480 | 0.580 (brute) | **96.5%** | 82.8% | +13.7p |
| EE D=10 | 0.787 | 0.296 | 1.007 (multistart) | **78.1%** | 29.4% | **+48.7p** |

**핵심 발견**: SE 최적해는 corner(켜고/끄기), **EE 최적해는 interior·저전력**. uniform Random은
EE의 interior 최적을 거의 못 찾음(D=10서 29.4%). → **RIBBO 우위가 EE에서 훨씬 큼** (D=10 EE +48.7p
vs D=10 SE 3BA +4.2p). EE야말로 "왜 RIBBO인가"의 가장 깨끗한 wireless 사례.

### D=20 + 추가 4BA-filter 실험 (완료)

- ✓ D=20 SE 3BA 학습+평가: RIBBO **33.0%** of WMMSE (Random 31.5%) — scalability 단조 하락 확인 (`results/11`)
- ✓ EE budget extension: EE D=10 79.3%→**84.2%** @1000 step (`results/09`)
- ✓ **D=20 SE 4BA-filter**(HC,RegE,Eagle,CMA): RIBBO **41.9%** (3BA 33%→+8.9p) — **회복 레버 D=20 확증**
- ✓ **EE D=10 4BA-filter**: RIBBO **69.7%** — 3BA(78.1%)보다 낮음 → EE는 Random 포함이 유리(pool은 문제 의존적)
- ✓ HANDOFF.md(발표용 단일 진입점) 작성, FINAL_REPORT/results/09·11 갱신
- 운영: GSP fix 후 D=20 풀학습 2.5h + 4BA 체인 + 평가 전부 무크래시 (uptime 10h+). 세션 재시작 시 conda RIBBO env 명시 필요 이슈 1건(스크립트에 explicit python 경로로 해결). Telegram 채널 연동 성공.

---

## 2026-06-24 — 야간 자율실험 + 시스템 하드 크래시 (3번째)

### 진행 status (크래시 시점 22:13까지)

오버나이트로 5/18 미팅 후속 4개 태스크 진행. 22:13에 시스템 freeze로 중단.

**완료 (크래시에도 살아남음, 디스크에 결과 있음)**:

- ✓ **EE reward 구현** (`problems/ifc.py`): SE/EE 벡터화 코어 + `Pfix`, search space `IFCEE`/`IFCEED10`,
  exact brute-force + 검증된 multistart EE optimum solver. SE 수치 불변(2.7e-15), WMMSE sanity 10/10 유지.
  - **정성 발견**: SE 최적해는 corner `(1,0,0)`, EE 최적해는 interior `(0.18,0,0)` → advisor의
    "SE는 경향성, EE랑 비교" 동기를 정량 뒷받침. (`results/09_ifc_ee/ee_sanity_output.txt`)
- ✓ **Task #1 (evaluation budget)** `results/08_eval_budget_extension/`: 300→1000 step 연장 시 WMMSE
  gap 좁혀짐. D=3 3BA 87.0%→91.7%(+4.8p), D=10 5BA-filter 72.9%→81.0%(+8.1p). 고차원일수록 budget 효과 큼.
- ✓ **Task #3 (Rastrigin→IFC transfer)** `results/10_rastrigin_transfer_normfix/`: RTG normalization을
  IFC 값으로 고쳐도 transfer 안 살아남(31.8%→31.2%, Random 미만). → 기존 "normalization mismatch가 원인"
  가설 **반증**. 차원 맞춰도 cross-domain zero-shot transfer 실패 = in-domain IFC 데이터 학습 필수.
- ✓ **Task #4 (D=20 scalability) 준비 완료**: 데이터/config/cache/train·eval 스크립트 ready.
  D=20에서 random=WMMSE의 31% (D=10 44%, D=3 83%) — 차원 stress 확인.

**미완 (크래시로 중단)**:

- 🟡 **Task #2 EE 학습**: D=3 EE는 epoch ~2194까지 학습되어 **수렴**(x_loss -16.2 flat, ckpt 2000 사용가능).
  **D=10 EE 미시작, D=20 미학습.** EE eval 미실행.

### 시스템 크래시 진단 (3번째 — 5/22, 5/24, 6/23)

**사실**:

- 학습 시작 20:22:56 → **22:13:38 freeze** (sustained GPU load **1h51m** 후). 12.5h frozen 후 학생이 10:38 리부팅.
- journald persistent 로그 살아있음 (boot -1). **freeze 직전 커널 에러 0건** — 마지막 줄은 평범한
  userspace 서비스 실패. Xid/NVRM error/MCE/thermal/AER **전부 없음**. → **silent instantaneous hard-hang**
  (커널이 로그 flush 전에 즉사). nvidia-smi -q: throttle/retired-page/reset 플래그 전부 clean.
- **드라이버**: closed/proprietary **595.71.05** in-use (open 패키지 rc=제거됨). 즉 5월의 Open→Closed
  교체는 적용됐으나 **여전히 크래시** + 이번엔 더 빨리(1h51m). → 모듈 flavor 아니라 **드라이버 버전 595.71.05
  자체가 의심**. 커널 6.8.0-124, Ubuntu 22.04.5.
- **sshfs-labftp**: FINAL_REPORT는 "stop/disable"이라 했으나 **여전히 살아서 restart 루프** —
  crash boot에서 **restart counter 3386회** (`-o foreground` fuse 옵션 오류로 10초마다 실패→재시작).
  **user 레벨 서비스**(`systemctl --user`)라 system-level에서 안 보였던 것. 지금도 looping 중(리부팅 5분에 39회).
- **swap 0B** (FINAL_REPORT "swap 추가"도 미적용). RAM 31G 중 13G free라 이번 OOM은 아님.
- **lm-sensors 미설치** → CPU/board 온도 telemetry 없음 (thermal black-box 부재).

**원인 가설 (우선순위)**:

1. **NVIDIA 595.71.05 드라이버 sustained-load hard-hang (최유력)**. silent, no-Xid, 가변 시간(1.8~3.6h),
   sustained CUDA load에서만 재현. open→closed로 안 고쳐짐 → **버전 다운그레이드(535 LTS/550 production)** 필요.
2. **전원(PSU/VRM) transient spike (유력)**. 4070 Ti는 순간 2×TDP 스파이크. marginal PSU/12VHPWR면
   sustained load + spike에서 무로그 즉시 freeze 설명됨. → **power cap (`nvidia-smi -pl 200`)** 으로 싸게 검증.
3. **thermal (가능성 낮음)**. 보통 throttle 먼저 로그 남기는데 없음. but VRM hotspot은 edge sensor 없이 trip 가능.
4. **sshfs-labftp restart loop (기여 요인)**. 단독 hard-freeze 유발은 어렵지만 FUSE/systemd/네트워크 churn 가중 + 로그 오염. 무조건 제거.

### 재발 방지 대책

**학생이 sudo로 실행 (높은 레버리지 순)**:

```bash
# 1) 드라이버 다운그레이드 (595.71.05 → 535 LTS 또는 550 production). 최우선.
sudo apt install nvidia-driver-535   # 또는 550. 설치 후 재부팅, nvidia-smi 버전 확인
# 2) GPU 전원 캡 (transient spike 완화 — 싸고 효과 빠른 검증)
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 200   # 250→200W
# 3) sshfs-labftp 영구 제거 (USER 서비스!)
systemctl --user disable --now sshfs-labftp.service   # sudo 아님, user 레벨
# 4) swap 추가 (OOM hang 방지)
sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
# 5) 온도 telemetry 설치
sudo apt install lm-sensors && sudo sensors-detect --auto
```

**코드/운영 측 (sudo 불필요, 이미 준비됨)**:

- `scripts/gpu_blackbox.sh`: 2초마다 temp/power/clock/throttle를 디스크에 fsync 기록 = 다음 freeze의
  유일한 forensic 단서 (커널이 안 남기므로). **모든 학습 동안 같이 띄울 것.**
- `scripts/train_resilient.sh`: 최신 ckpt 자동 resume + black-box 동시 기동 + (가능 시) power cap.
  save_interval 250(=12.5분)으로 단축 → 크래시 손실 최소화.
- 다음 long run 전에 **짧은 통제된 stress 재현** (gpu-burn 등 + black-box) 으로 thermal/power vs driver 판별.

---

## 2026-05-24 — 5BA-filter ablation (D=3 학습 완료, D=10 진행 예정)

**진행 status** (자동 chain 중):

- ✓ D=10 6BA × 20ch × 300step = 120 trajectory 생성 완료 (BotorchBO 포함, Vizier 제외)
- ✓ 5BA-filter D=3 dataset 셋업 (symlink, Random/ShuffledGridSearch 제외 100 traj)
- ✓ 5BA-filter D=10 dataset 셋업 (symlink, 100 traj)
- ✓ 5BA-filter × D=3 학습 5000ep 완료 (run4 1-2500 + run4b 2500-5000, system crash 후 resume)
  - 수렴: epoch 4500-5000 mean x_loss = -18.15, slope -0.00019/ep
- ▶ 5BA-filter D=3 eval 진행 중 (3BA vs 5BA vs 7BA + Random + WMMSE, 500ch × 1ep × 300step)
- ☐ 5BA-filter × D=10 학습 5000ep — eval 끝나면 자동 시작
- ☐ D=10 eval + 2x2 ablation plot
- ☐ FINAL_REPORT 갱신 (논문 비교, 보완점, 미팅 관점)

**환경 이슈 (5/22 발생, 학생이 reboot으로 복구)**:

- 시스템 crash @ 5/22 21:48 (학습 3.5h 시점). 학습 진행도 epoch ~2700.
- 원인 가설:
  - sshfs-labftp 서비스 `-o foreground` 옵션 오류로 10초마다 무한 restart (3.5h × 6/min = ~1300회) → 누적 contributing factor
  - NVIDIA Open Kernel Module 595.71.05 (Apr 2026 build) 장시간 학습 안정성 불확실
  - Swap 0 → OOM hang 위험
- 조치:
  - sshfs-labftp.service stop 완료. 영구 fix는 학생이 sed 또는 disable
  - swap 추가, persistence mode는 sudo 필요 (학생 보류)
- crash 후 resume: 2500.ckpt → 5000.ckpt 재진행 완료, 동일 환경에서 약 2h17min 추가 학습 — 이번엔 안 뻗음

---

## 2026-05-27 — 5BA-filter ablation 완료: 결정적 finding

**완료**:

- 5BA-filter D=10 학습 5000ep 완료 (run2 1-4000 + run2b 4000-5000 resume, NVIDIA driver fix 후 진행)
  - 수렴: epoch 4500-5000 mean x_loss = -44.48, slope -0.00046/ep
- D=10 3-way eval 완료 (3BA-D=10 / 5BA-filter-D=10 / Random + WMMSE) — 200 ch × 1 ep × 300 step
- 2x2 ablation plot + results/07 README + FINAL_REPORT 작성

### 결정적 결과 (2×2 매트릭스)

| | D=3 (% of WMMSE) | D=10 (% of WMMSE) |
|---|---|---|
| 3BA (R, HC, CMA) | **88.9%** ✓ | 47.8% |
| 5BA-filter (HC, RegE, Eagle, CMA, BoTorch) | 78.8% | **71.4%** ✓ (+23.6%p vs 3BA) |
| 7BA (5BA + R + SGS) | 77.3% | (미실험) |
| Random baseline | 83.0% | 43.6% |

### 핵심 발견

1. **BA-pool 효과는 차원에 따라 정반대**:
   - D=3 simple problem: 단순 3BA가 가장 좋음. 다양성 ↑이 오히려 손해 (논문 SVM D=3 결과와 평행)
   - D=10 nontrivial: 5BA-filter가 +23.6%p 도약. Random에 +27.8%p 압도

2. **Song 2024 권고 ("well-performing AND diverse") 정량 검증**:
   - Section 4.2 BC vs BC Filter: underperforming BA(Random + SGS) 빼면 성능 ↑
   - Appendix B.2: "well-performing + diverse" 강조
   - D=10에서 정확히 그대로 재현됨. D=3은 boundary case

3. **D=3 IFC는 RIBBO 적용 의의 약함** = 미팅 "왜 RIBBO인가" 정량 답변. D=10 + curated BA pool이 RIBBO의 진짜 효과 영역

### 운영 — NVIDIA driver 이슈 해소

- 5/22, 5/24 두 차례 시스템 freeze (학습 ~3.5h 후, BERT hardware error)
- 원인: NVIDIA Open KMM 595.71의 sustained-load wedge 추정
- 조치: Open → Closed KMM 교체 + sshfs-labftp stop + 80% thread/mem 제한
- 결과: 5/27 D=10 resume 학습 (1h) crash 없이 완료. 환경 확정

**다음 세션 권장**:

- Task 5 (EE reward variant) — D=10 5BA-filter setup 그대로 가져가서 reward만 EE로
- D=10 channel 확장 (20→100) 또는 더 큰 차원 (D=20/50)
- 새 미팅 발표: 5/18 자료 + 04 + **07 (key slide)** 순서

---

## 다음 세션 TODO — 2x2 ablation: (BA pool) × (dimension)

7BA negative finding(D=3에서 88% → 76%)의 해석 후속. Song 2024 논문 명시 권고(Appendix B.2 + Section 4.2 BC vs BC Filter)에 따라 **Random + ShuffledGridSearch 제외한 5BA-filter**로 재학습. 차원도 같이 늘려 RIBBO의 D=10 잠재력 검증.

### 비교 매트릭스 (목표)

| | D=3 | D=10 |
|---|---|---|
| 3BA (Random, HC, CMAES) | 87.9% ✓ 기존 | 48.8% ✗ 기존 |
| 5BA filter (HC, RegEvol, Eagle, CMAES, BotorchBO) | **TBD** | **TBD** |
| 7BA (5BA + Random + ShuffledGridSearch) | 76.4% ✗ 기존 | — |

### 실행 항목 (우선순위 순)

1. **D=10 추가 데이터 생성** — 현재 `data/generated_data/ifc_d10/seed0/`에 3BA만. RegularizedEvolution / EagleStrategy / BotorchBO 3 designer × 20 channel × 300 step = 60 trajectory 추가. BotorchBO는 D=10 GP scaling으로 시간 ↑ 예상 (~30분-1h)
2. **5BA filter × D=10 학습** — 100 trajectory, 5000 epoch, ~5h. run name: `dt-ifc-d10-run2-5bafilter`
3. **5BA filter × D=3 데이터셋 셋업** — 새 데이터 생성 불필요. `data/generated_data/ifc/seed0/`에서 Random/ShuffledGridSearch 빼는 cache 또는 filter_data 옵션 활용. dataset_specs 한 줄 추가로 가능할 듯
4. **5BA filter × D=3 학습** — 100 trajectory, 5000 epoch, ~5h. run name: `dt-ifc-run4-5bafilter`
5. **평가 + 2x2 비교 plot** — `scripts/eval_ifc_2x2_ablation.py` (가칭). 500 ch × 1 ep × 300 step. WMMSE overlay. 결과를 `results/07_ifc_5ba_filter_ablation/` 신규 디렉토리에 정리 + README
6. **FINAL_REPORT.md 갱신** — Section 3에 (2) BA pool ↔ dimension interaction 추가. 4번 finding(D=10 scalability limit)도 dataset size로 재해석
7. **PROGRESS.md 갱신** — 이 항목 결과로 갱신
8. **(미수행) Task 5 — EE reward variant** — Lee 2025 eq.32 EE 식 구현 + 데이터 재생성 + 학습. 이 세션에서 deferred. ~4-6h GPU

### 가설

- **D=10 RIBBO의 48% 결과는 차원 자체 아니라 데이터 부족** — 60 traj로 [0,1]^10 공간 학습 어려움. 100 traj + 5BA로 의미 있게 개선 기대
- **D=3에서 5BA filter는 3BA(87.9%)와 비슷하거나 약간 향상** — Random은 어차피 약한 BA지만 ShuffledGridSearch 빼는 것의 효과는 IFC corner-solution 특성과 어떻게 상호작용할지 불확실
- **D=10에서 5BA filter는 3BA(48.8%) 대비 큰 향상** — 데이터 양 1.7× + 더 강한 BA mix 효과 결합

### advisor 미팅 후보 (이번 분석 통합)

- "BA 다양성 ↑이 IFC에서 안 통한 이유": 논문 권고("well-performing AND diverse")에 underperforming(Random/ShuffledGridSearch) 포함이 원인이라는 해석. 우리 case는 BC와 평행 (BC vs BC Filter 차이가 RIBBO에서도 나타남)
- "RIBBO가 D=3 IFC에서 88% 도달은 운 좋은 BA pool 덕"이라는 frame. 5BA filter로 검증 시도
- D=10이 더 어려운 게 차원 자체인지 데이터 부족인지 — 이번 ablation으로 분리

---

## 2026-05-22 (Post-meeting Day 4) — Task 1 (7BA): 핵심 negative finding + 디렉토리 정리

**완료**:

- IFC D=3 7BA 데이터 생성 (140 trajectory, BotorchBO 포함; Vizier는 JAX OOM segfault로 제외)
- 7BA × 3000 epoch 학습 → 미수렴(loss slope -0.0009/ep) → 5000 epoch 재학습 → 수렴(slope -0.00016/ep)
- Same-epoch + Converge 두 기준으로 3BA vs 7BA 비교 (500 ch × 1 ep × 300 step)
- `results/` 디렉토리 재구조: 6개 카테고리(01~06) + 각각 README + 마스터 README + `FINAL_REPORT.md`
- `presentation/`은 5/18 발표 자료(main.tex/pdf + 그 그림) 그대로 보존
- Memory 업데이트: `env_setup.md`에 Vizier 사용 불가/JSON metadata 동기화 주의 기록, `feedback_workflow.md`에 "프로젝트 내 묻지 말고 진행" 규칙 추가

### 핵심 결과 (500 unseen ch × 1 ep × 300 step, IFC SE D=3)

| Method | Final SE (nats) | % of WMMSE |
|---|---|---|
| **WMMSE (our impl)** | **2.864** | 100% (ref) |
| RIBBO 3BA (3000 ep, converged) | 2.519 | **87.9%** |
| Random | 2.379 | 83.0% |
| RIBBO 7BA (3000 ep, NOT conv) | 2.281 | 79.6% |
| RIBBO 7BA (5000 ep, converged) | 2.187 | **76.4%** |

### 핵심 finding (Negative result)

- **BA pool을 3→7개로 늘리면 IFC SE D=3에서 성능 하락** (88% → 76%)
- 5000 epoch 수렴 후에도 동일 — 단순 학습 부족이 원인 아님
- 7BA는 Random보다도 못함 (2.19 < 2.38) → RIBBO가 잘못된 RTG conditioning을 학습한 것으로 추정
- 가설:
  1. BotorchBO/EagleStrategy/ShuffledGridSearch 같은 다양한 exploration profile이 RTG distribution을 노이즈화
  2. 140 trajectory가 7BA diversity coverage에 부족
  3. IFC sum-rate가 corner solution 특성이라 simple BA(직선적 improvement) trajectory가 학습에 유리

**막힘**:

- 학습 초반 두 번 시도 시행착오 (`IFCSE_D10` underscore 파서 깨뜨림 → `IFCSED10`, JSON metadata 별도 동기화 필요)
- Vizier(VizierGPBandit) JAX/XLA OOM 으로 8BA 불가, 7BA 확정

**다음 세션 시작 시 필요**:

- 미진행 Task 5 (EE reward variant) 진행 여부 결정 (4-6h GPU 시간 예상)
- D=10 7BA + 더 긴 학습으로 D=10 RIBBO 성능 향상 시도
- BA selection ablation (어떤 BA를 빼면 성능 회복?)

**advisor 미팅에 가져갈 질문 후보**:

- "BA 다양성 ↑ → 성능 ↑" 가설이 IFC에서 깨진 점: 문제 특성(corner solution)인지 데이터 양 부족인지
- BA pool selection 기준 제시 가능성 (Song 2024는 분석 미제공)
- WMMSE 우리 값(2.86) vs Lee 2025 보고값(2.50) 차이의 원인 (channel 분포만?)
- RTG normalization을 BA별로 따로 하면 7BA 학습이 안정화될지

---

## 2026-05-21 (Post-meeting Day 3) — Task 4: IFC D=10 native 학습 + Rastrigin transfer 비교

**완료**:

- IFCNumpy/IFCMetaProblem **dim parameterize** via `IFC_DIM_MAP` (search_space_id → dim). 기존 `IFCSE` 작동에 영향 없음.
- 새 config: `scripts/configs/dataset_specs_ifc.py`에 `IFCSED10` 추가, `scripts/configs/dt/ifc_d10.py`, `run_datagen_ifc_d10.sh`
- **데이터 생성** (D=10, 3BA × 20ch × 300step = 60 trajectory, ~3분, `data/generated_data/ifc_d10/seed0/`)
- **학습** (`dt-ifc-d10-run1`, 3000 epoch, **2h 47min** on RTX 4070 Ti). ckpt: `log/ifc/dt-ifc-d10-run1/IFCSED10-seed0-05-21-10-34-541982/ckpt/3000.ckpt`
- **평가** (`scripts/eval_ifc_d10_native.py`, 200 unseen channel × 1 ep × 300 step + WMMSE D=10 overlay + Rastrigin transfer overlay)

### 결과 수치 (200 unseen channel, D=10)

| Method | Final SE (nats) | (bits) | % of WMMSE |
|---|---|---|---|
| **WMMSE D=10 (직접 구현)** | **3.968** | 5.724 | reference |
| RIBBO D=10 native best-so-far | 1.935 | 2.792 | **48.8%** |
| Random D=10 best-so-far | 1.729 | 2.495 | 43.6% |
| RIBBO Rastrigin → IFC transfer (@step 150) | 1.261 | 1.819 | 31.8% |
| WMMSE std | ±0.727 | | |

- 그림: `presentation/eval_ifc_d10_native.pdf` (4 곡선 + WMMSE band)
- raw rollout: `presentation/eval_ifc_d10_native.npz`

### 주요 발견 (advisor 미팅 토픽)

1. **차원 증가에 따른 RIBBO 성능 약화**:
   - D=3: RIBBO 88.3% of WMMSE
   - D=10: RIBBO **48.8%** of WMMSE — 절반 이하로 떨어짐
   - 같은 training protocol(3BA × 20ch × 300step × 3000 epoch). 데이터 규모는 동일, 차원만 증가.
   - 가설: D=10에서는 sample efficiency 부족 — 300 step으로 [0,1]^10 공간 탐색 불충분
2. **Rastrigin transfer는 동일 차원이라도 실패**:
   - D=10 native (1.93) > D=10 Rastrigin transfer (1.26)
   - 같은 step 150에서 비교해도 native가 우월
   - RTG normalization mismatch가 transfer를 실질적으로 막음 (Rastrigin y range ≈ [-3100, -109], IFC y ≈ [0, 30])
3. Random과의 gap도 좁아짐 (D=3: +6.1%p, D=10: +5.2%p) — exploration이 사실상 random에 가까워짐

**막힘**:

- 학습 초기 두 번 실패: (a) `IFCSE_D10` underscore가 load_datasets 파서 깨뜨림 → `IFCSED10`로 변경, (b) JSON metadata 안의 search_space_id도 같이 업데이트 필요 (파일명만 sed로 바꿔서는 안 됨). 합계 ~30분 낭비. CLAUDE.md Day 1 메모의 underscore 경고 못 챙김.

**다음 세션 시작 시 필요**:

- Task 1 (7BA 데이터 재생성 + 재학습) 진행 OK 여부

**advisor 미팅에 가져갈 질문 후보**:

- D=10 RIBBO가 WMMSE 절반밖에 못 따라가는 것 — 데이터 규모를 늘려야 하는가, 다른 architectural 변경이 필요한가
- 차원이 더 커져도 (D=20, D=50 등) 같은 패턴인가 vs D=3에서 잘 되는 게 우연인가
- Rastrigin transfer 실험을 "RTG normalization을 IFC stats로 강제 재조정"하면 다른 결과 나올지

---

## 2026-05-20 (Post-meeting Day 2) — Task 2: WMMSE 비교 + eval 확장

**완료**:

- WMMSE 구현 (Shi 2011 box-constrained, per-link power, 10 restart):
  - `problems/ifc.py`: `wmmse_sum_rate(H_sq, Ptx, ...)` + helper `_ifc_sum_rate()`
  - Sanity check `scripts/wmmse_sanity.py`: 10 채널에서 WMMSE rate = brute-force(20³, 40³) best rate **gap=0/10** (IFC sum-rate 최적해의 corner property)
- `scripts/eval_ifc_matched.py` 확장: 200×1 → **500 channels × 3 episodes × 300 steps**, WMMSE per-channel optimum overlay

### 결과 수치 (500 unseen channel, D=3)

| Method | Final SE (nats) | Final SE (bits) | WMMSE gap |
|---|---|---|---|
| WMMSE (our impl, mean) | **2.864** | 4.132 | (reference) |
| RIBBO best-so-far | 2.529 | 3.649 | −0.335 (**88.3%** of WMMSE) |
| Random best-so-far | 2.384 | 3.440 | −0.480 (83.2%) |
| WMMSE std | ±0.685 | | |

- 그림: `presentation/eval_ifc_matched_with_wmmse.pdf`

### 주요 발견

- RIBBO는 WMMSE의 **88.3%** 수준으로 근접. Random 대비 +6.1% 개선.
- 우리 WMMSE 구현 평균(2.86 nats) > Lee 2025 Fig 7(b) "WMMSE/Local opt" 값(2.50 nats). Lee 논문의 setup(다른 channel 분포/iteration 수)과 직접 비교 불가. 우리는 동일 채널에서 fair comparison.
- WMMSE std 큰 편(±0.685) — 채널 realization에 따라 optimum 변동 큼.

**막힘**:

- 없음. WMMSE 정확성 brute-force로 검증됨.

**다음 세션 시작 시 필요**:

- Task 4 진행 OK 여부 확인 (Rastrigin→IFC transfer + IFC D=10 native 학습)

**advisor 미팅에 가져갈 질문 후보**:

- Lee 2025 "WMMSE/Local opt" 값(2.50)이 우리 구현(2.86)과 차이 — channel sample 분포가 같지 않은지 등 setup 확인 필요
- WMMSE std 큰 점(±0.685) → 200 channel에서는 confidence interval 좁히려면 더 많은 sample 필요할 수도

---

## 2026-05-20 (Post-meeting Day 2) — Task 3: instantaneous plot 추가

**완료**:

- 5/18 미팅 후속 작업 5건의 plan 수립 (`/home/myoungjin/.claude/plans/lively-cooking-neumann.md`)
- 우선순위: 3 → 2 → 4 → 1 → 5
- **Task 3 (instantaneous plot)** 완료: 3개 eval 스크립트에 best-so-far 외에 current-step reward 곡선 overlay 추가
  - `scripts/eval_ifc.py` → `presentation/eval_ifc_ribbo_vs_baselines_with_instant.pdf` (50 ch × 3 ep × 300 step)
  - `scripts/eval_ifc_matched.py` → `presentation/eval_ifc_matched_with_instant.pdf` (200 ch × 1 ep × 300 step)
  - `scripts/eval_ifc_d10_from_rastrigin.py` → `presentation/eval_ifc_d10_from_rastrigin_with_instant.pdf` (200 ch × 1 ep × 150 step)

### 결과 수치

| 스크립트 | RIBBO best-so-far | Random best-so-far | 비고 |
|---|---|---|---|
| eval_ifc_matched (D=3) | **2.560 nats** | 2.394 | 이전 발표(2.67 vs 2.44) 대비 약간 낮음 — stochastic policy 변동 |
| eval_ifc_d10_from_rastrigin (D=10) | **1.261 nats** | **1.672** | ⚠ RIBBO < Random — transfer 실패 |

### 주요 발견

- **Rastrigin → IFC D=10 zero-shot transfer가 Random보다 못함** (1.26 vs 1.67 nats).
  - Rastrigin 학습 시 `y_max_mean=-108.7, y_min_mean=-3104.6` (negative reward, BBOB 표준)
  - IFC raw SE는 [0, ~24]로 양수 → Rastrigin normalization 적용 시 "이미 최적"으로 보여 exploration 억제
  - Task 4-3 (IFC D=10 native 학습) 결과와 비교 시 RTG scale mismatch 영향 정량화 가능

**막힘**:

- 없음

**다음 세션 시작 시 필요**:

- Task 2 (WMMSE 비교) 진행 OK 여부 확인
- WMMSE power constraint: per-link box (each `p_d ≤ Ptx=1`, since our `x_d ∈ [0,1]`) — Lee 2025 setup과 일관

**advisor 미팅에 가져갈 질문 후보**:

- Rastrigin → IFC D=10 transfer가 Random 미만인 점 — RTG normalization fair transfer 방법?
- "모든 BA" 범위: Vizier 포함 8개인지, Song 2024 7개인지

---

## 2026-05-18 (Day 7)

**완료**:

- BBOB Rastrigin 수렴성 확인 실험 완료 (1000 epoch, GPU, `dt-convergence-check`)
- IFC 훈련/테스트 pipeline 코드 구현 완료:
  - `problems/ifc.py`: `IFCMetaProblem` 추가
  - `scripts/configs/dt/ifc.py`: 학습 config 생성
  - `scripts/configs/dataset_specs_ifc.py`: train/test task 목록 생성
  - `scripts/run_dt.py`: `"ifc": IFCMetaProblem` 등록

### 수렴성 분석 결과 (Rastrigin, x_loss = Gaussian policy NLL)

| epoch | x_loss |
|-------|--------|
| 1     | +10.78 |
| 100   | -27.39 |
| 200   | +7.21  |
| 300   | -26.48 |
| 400   | -39.12 |
| 500   | -41.21 |
| 600   | -42.45 |
| 700   | -42.42 |
| 800   | -42.29 |
| 900   | -44.14 |
| 1000  | -44.94 |

- **수렴 판정**: ✓ epoch 500~600 이후 안정적으로 수렴 확인
- **세팅**: 20 task × 3 designer (Random, HillClimbing, CMAES) × 300 step, paper 기본 하이퍼파라미터
- **비고**: epoch 200까지 진동 있음 (warmup 10,000 step 효과), 이후 단조 감소

---

### IFC 데이터 생성 세팅

**문제 정의 (Lee et al. 2025, Section V-A, SE)**

$$r_{SE}(x) = \sum_{d=1}^{D} \log\!\left(1 + \frac{P_{tx} |h_{dd}|^2 x_d}{1 + P_{tx} \sum_{d' \neq d} |h_{d'd}|^2 x_{d'}}\right)$$

- D = 3 (송수신 pair 수)
- x ∈ [0, 1]^3 (전력 배분 비율)
- h_{d'd} ~ CN(0, 1): 복소 가우시안, real/imag 각 N(0, 1/√2) → E[|h|²] = 1
- P_tx = 10 W
- 구현 log 단위: 자연로그 (nats). 최적해는 log₂와 동일, 수치 비교 시 주의.

**채널 행렬 생성 (task별 고정)**

```python
rng = np.random.default_rng(int(dataset_id))   # dataset_id = 채널 seed
real = rng.normal(0, 1/sqrt(2), (3, 3))
imag = rng.normal(0, 1/sqrt(2), (3, 3))
H_sq = real**2 + imag**2   # |h_{d'd}|^2, shape [3,3], [row=tx, col=rx]
```

**데이터 생성 세팅**

| 항목 | 값 |
|------|----|
| Train channel (task) | 20개 (dataset_id 0~19) |
| Test channel (task) | 10개 (dataset_id 20~29, eval 시 online 생성) |
| Behavior algorithm | 3개: Random, HillClimbing, CMAES |
| Steps / trajectory | 300 |
| Seed | 0 |
| 총 trajectory | 60개 (20 × 3) |
| 병렬 실행 | 12 프로세스 (16코어 중 4코어 여유) |
| 소요 시간 | ~2분 |
| 출력 경로 | `data/generated_data/ifc/seed0/` |
| 파일명 규칙 | `{designer}_IFCSE_{dataset_id}_0.json` |

---

### IFC 학습 세팅

**실행 커맨드**

```bash
PYTHONPATH=. WANDB_MODE=disabled python -u scripts/run_dt.py \
    --config scripts/configs/dt/ifc.py \
    --name dt-ifc-run1 \
    --embed_dim 256 --num_heads 8 --num_layers 12 \
    --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch 3000 --batch_size 64 \
    --eval_interval 500 --save_interval 500 \
    --id IFCSE --seed 0 --device cuda:0
```

**모델 하이퍼파라미터**

| 항목 | 값 | 비고 |
|------|----|------|
| Architecture | Causal Transformer (GPT-2 backbone) | RIBBO 논문 그대로 |
| embed_dim | 256 | 논문값 |
| num_layers | 12 | 논문값 |
| num_heads | 8 | 논문값 |
| input_seq_len | 50 | 학습 시 샘플 윈도우 크기 |
| max_input_seq_len | 300 | trajectory 전체 길이 활용 |
| x_type | stochastic (Gaussian) | NLL loss |
| pos_encoding | embed | 논문값 |

**학습 하이퍼파라미터**

| 항목 | 값 | 비고 |
|------|----|------|
| num_epoch | 3000 | Rastrigin 수렴 epoch 기준 (~600) × 5 여유 |
| batch_size | 64 | 논문값 |
| optimizer | AdamW | lr=2e-4, weight_decay=1e-2 |
| lr schedule | Linear warmup + Cosine annealing | warmup 10,000 step |
| eval_interval | 500 | test channel 10개 rollout |
| normalize_method | random | RTG normalization (HRR) |
| GPU | RTX 4070 Ti (12GB) | 시뮬컴 사용 불가 |
| 예상 학습 시간 | ~2.5시간 | Rastrigin 51min/1000epoch 기준 |

**평가 세팅**

| 항목 | 값 |
|------|----|
| Train eval channels | 0~14 (15개) |
| Test eval channels | 20~29 (10개, unseen) |
| eval_episodes | 5 |
| init_regrets | [0, 20, 50] |
| Test normalization | y_max_mean fallback (훈련 채널 평균으로 대리) |

**막힘**:

- 없음

**advisor 미팅에 가져갈 질문 후보**:

- log 단위: 구현은 nats (ln), Lee et al. 보고값이 bits (log₂)인지 확인 필요 (수치 비교 시 ÷ ln2 해야 할 수 있음)
- Test channel의 regret normalization을 y_max_mean으로 대리하는 방식이 fair comparison인지
- Rastrigin 수렴은 확인됐으나 IFC는 reward scale이 채널마다 다름 → RTG normalization의 per-channel y_max 방식이 paper 의도와 맞는지

---

## 2026-05-11 (Day 1)

**완료**:

- Repo 정찰 완료 (전체 구조 파악)
- 환경 검증 및 의존성 수정 완료
- BBOB Rastrigin mini-scale smoke test 성공 (data gen → train, 5 epoch, CPU)

---

### Repo 구조 요약

```
RIBBO/
├── problems/           ← 벤치마크 task 정의
│   ├── base.py         ← ProblemBase, MetaProblemBase (abstract interface)
│   ├── bbob.py         ← BBOB 함수 19개 (Rastrigin 등) numpy 구현
│   ├── synthetic.py    ← SyntheticNumpy / SyntheticTorch / SyntheticMetaProblem
│   ├── hpob_problem.py ← HPOBMetaProblem
│   └── real_world_problem.py + real_world_utils/rover_function.py 등
│
├── data_gen/           ← 오프라인 데이터 생성
│   ├── data_gen_main.py       ← 단일 (designer, task, seed) → JSON 1개
│   ├── synthetic_problem_statement.py  ← Vizier 검색 공간 wrap
│   ├── hill_climbing_designer.py
│   ├── botorch_designer.py
│   └── regularized_evolution_designer.py
│
├── datasets/           ← 데이터 로딩 / RTG(regret) 계산
│   ├── datasets.py     ← TrajectoryDataset (캐시 자동 생성), TrajectoryIterableDataset
│   ├── load_datasets.py ← JSON → Trajectory 파싱 (파일명 규칙 중요, 아래 참고)
│   └── trajectory.py   ← Trajectory 클래스
│
├── algorithms/         ← 모델 & 학습
│   ├── modules/dt.py   ← DecisionTransformer (GPT2 backbone, offlinerllib 기반)
│   ├── designers/dt_designer.py  ← RTG 초기화(reset), 평가 루프(evaluate_decision_transformer_designer)
│   └── data_augment/ : HRR은 datasets.py set_regrets() + normalize_y_and_regrets()에서 처리
│
└── scripts/
    ├── run_dt.py              ← 학습 entry point (parse_args + NameSpace config)
    ├── rollout_and_vis.py     ← inference / 시각화
    ├── run_data_gen.py        ← 병렬 data gen 오케스트레이터
    └── configs/dt/synthetic.py ← hyperparameter (Python NameSpace, yaml 아님)
```

**파일명 규칙** (`load_datasets.py` 기준):
- 형식: `{designer}_{search_space_id}_{dataset_id}_{seed}.json`
- 위치: `data/generated_data/{problem}/seed{N}/`
- 주의: `search_space_id` 또는 `designer`에 `_` 포함 시 파서 오류 발생

**Behavior algorithm 목록** (학습 데이터에 실제 사용):
`Random`, `ShuffledGridSearch`, `RegularizedEvolution`, `HillClimbing`, `EagleStrategy`, `CMAES`, `BotorchBO`

---

### IFC task 추가 시 수정/추가해야 할 파일 (코드 근거 기반)

#### 새로 추가

| 파일 | 내용 |
|---|---|
| `problems/ifc.py` | `IFCNumpy`, `IFCTorch`, `IFCMetaProblem` 클래스. `SyntheticMetaProblem` 참고. dim=3, lb=0, ub=1, dataset_id=채널 seed |
| `data_gen/ifc_problem_statement.py` | Vizier 검색 공간 wrap. `synthetic_problem_statement.py` 참고 |
| `scripts/configs/dt/ifc.py` | `synthetic.py` 복사 후 id="IFC", data_dir, cache_dir 수정 |
| `scripts/configs/dataset_specs_ifc.py` | train/test dataset_id 목록 |

#### 수정

| 파일 | 수정 내용 |
|---|---|
| `data_gen/data_gen_main.py:88-114` | `elif args.problem == 'ifc':` 브랜치 추가 |
| `scripts/run_data_gen.py:39` | choices에 `'ifc'` 추가 + elif 브랜치 |
| `scripts/run_dt.py:29-34` | `problem_cls` dict에 `"ifc": IFCMetaProblem` 추가 |

#### IFC reward (Lee et al. 2025, eq. 32 SE)

```python
# D=3, Ptx=10, 채널 h[d'][d] ~ CN(0,1)
def forward(self, X):  # X: [B, 3], x_d ∈ [0,1]
    se = sum(
        log(1 + Ptx * |h[d][d]|^2 * x[d]
            / (1 + Ptx * sum(|h[d'][d]|^2 * x[d'] for d'≠d)))
        for d in range(D)
    )
    return se  # maximize
```

---

### 환경 확정 사항

- **conda env**: `RIBBO` (`conda run -n RIBBO`)
- **PYTHONPATH**: 반드시 `/home/myoungjin/01_Research/01_Projects/RIBBO` 절대경로 명시
- **WANDB**: 학습 시 `WANDB_MODE=disabled` 필요 (API key 없음)
- **GPU**: RTX 4070 Ti 12GB × 1 (CLAUDE.md의 5090×2와 다름 — 로컬 PC)

#### 환경 수정 이력 (Day 1에 해결한 것들)

| 이슈 | 해결책 |
|---|---|
| `chex` 없음 | `pip install chex` |
| `cvxpy` 없음 | `pip install cvxpy` |
| `equinox`, `optax`, `flax` 없음 | `pip install equinox optax flax` |
| `evojax` 없음 | `pip install evojax` |
| vizier `designers/__init__.py` 전부 eager import | optional import들 try/except로 패치 (conda env 내 파일 직접 수정) |
| `datasets/`, `problems/`, `algorithms/` 에 `__init__.py` 없음 → HuggingFace `datasets` 패키지와 충돌 | 각 폴더에 빈 `__init__.py` 추가 |
| NumPy 2.2.6 → PyTorch `.numpy()` 불가 | `pip install "numpy<2"` → 1.26.4 |

---

### Smoke Test 결과

**커맨드**:
```bash
# Phase 1: data gen (3 designers × 2 tasks, 50 steps, ~3분)
export PYTHONPATH=/home/myoungjin/01_Research/01_Projects/RIBBO
export WANDB_MODE=disabled
cd /home/myoungjin/01_Research/01_Projects/RIBBO
for designer in Random HillClimbing CMAES; do
  for dataset_id in 0 1; do
    python data_gen/data_gen_main.py \
      --problem=synthetic --designer=$designer \
      --search_space_id=Rastrigin --dataset_id=$dataset_id \
      --out_name=data/generated_data/synthetic/seed0/${designer}_Rastrigin_${dataset_id}_0.json \
      --length=50 --seed=0
  done
done

# Phase 2: train (5 epoch, CPU, ~수초)
python scripts/run_dt.py \
  --config scripts/configs/dt/synthetic.py \
  --name mini-smoke --id Rastrigin \
  --embed_dim 64 --num_layers 2 --num_heads 4 \
  --num_epoch 5 --step_per_epoch 5 \
  --batch_size 4 --input_seq_len 10 --max_input_seq_len 50 \
  --n_block 1 --eval_interval 9999 \
  --seed 0 --device cpu --debug
```

**결과**: ✓ data gen 6 JSON 생성, ✓ cache 자동 생성, ✓ 5 epoch 학습 완료, ✓ 체크포인트 저장

---

**막힘**:

- 없음 (환경 이슈는 모두 해결)

---

**다음 세션(Day 2) 시작 시 필요한 결정/입력**:

1. **IFC 채널 파라미터 확인**: Lee et al. (2025) Section V-A의 Rayleigh fading 채널 생성 방식 — `dataset_id`를 seed로 쓸 때 채널 행렬 H를 어떻게 고정할지 (seed per realization)
2. **behavior algorithm 3개 확정**: smoke test에서는 Random/HillClimbing/CMAES 사용. Day 3 실제 데이터 생성 시 몇 개 쓸지 결정 필요 (7개 전부 vs 3개)
3. **데이터 규모 결정**: N=몇 개 task (channel realization), M=몇 개 seed, T=몇 step. CLAUDE.md에는 미정

---

**advisor 미팅에 가져갈 질문 후보**:

1. "왜 RIBBO인가" 가설 3개 (CLAUDE.md 섹션 3) 중 어느 것이 주요 동기인지 advisor 확인 필요
2. BotorchBO는 NumPy/JAX 의존성 이슈로 당분간 제외 — 동의 여부
3. IFC task에서 RTG normalization: 채널마다 optimal이 다른데, behavior algo의 best-observed로 per-channel y_max 잡는 방식이 맞는지
