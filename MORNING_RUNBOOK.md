# ☀️ 아침 런북 — 선배에게 답변+팔로업+발표자료 보내기 (2026-06-29)

> 목표: 8:00 출근 → 2시간 안에 **네 눈으로 검증하면서** 고품질로 선배에게 전송.
> 모든 자료는 이미 만들어져 있음. 너가 할 일 = **읽고 → 확인 → 보내기**.

---

## 🎯 한눈에 (TL;DR)

| 무엇 | 파일 | 언제 | 받는이 |
|---|---|---|---|
| ① 피드백 답변지 | `presentation/senior_feedback_answers_0628.pdf` (3p) | ~9:45 | 선배 |
| ② 팔로업(반영+GA실험) | `presentation/followup_senior_0629.pdf` (2p) | ~9:45 (①과 함께) | 선배 |
| ③ 업데이트 발표자료 | `presentation/progress_0625.pdf` (8p) | ~10:50 | 선배 |
| (첨부용) baseline 그림 | `results/18_de_baseline/baselines_with_ga-1.png` | ②와 함께 선택 | 선배 |

- **보낼 메시지 텍스트(복붙용)**: `presentation/followup_senior_0629.txt` (커버 메시지 긴/짧은 버전 둘 다)
- 핵심 메시지 한 줄: **"피드백 다 반영 + GA 포함 강한 baseline 비교까지 했고, RIBBO 우위는 EE D=10·적은예산으로 또렷, 한계는 정직하게."**

---

## ⏱️ 스텝 바이 스텝 (8:00–11:00)

### STEP 0 — 환경 (8:00, 1분)
```bash
cd /home/myoungjin/01_Research/01_Projects/RIBBO
conda activate RIBBO    # 또는 PATH에 /home/myoungjin/miniconda3/envs/RIBBO/bin
```

### STEP 1 — 답변지 검증 (8:05–8:25)
- 파일 열기: `presentation/senior_feedback_answers_0628.pdf`
- **확인 포인트**: 각 질문의 [답]이 사실인지, [반영 완료]가 실제 deck과 일치하는지.
  - 특히 빨강 "남은 결정" 3건(통신약어 풀이 / 숫자 톤다운 / 영어 추가손질) — 지금 상태로 보낼지 결정.
- 고칠 점 있으면 → 텔레그램으로 나한테 알려줘 → 내가 즉시 수정.

### STEP 2 — 팔로업 검증 (8:25–8:50)
- 파일 열기: `presentation/followup_senior_0629.pdf`
- **확인 포인트**: 2번 섹션 GA/DE/CMA-ES/BO 표 숫자, "정직한 한계" 톤.
  - 표 숫자 근거: `results/18_de_baseline/README.md` (결과 1 + 결과 1b GA).
- 그림도 볼 거면: `results/18_de_baseline/baselines_with_ga-1.png` (EE D=10·SE D=10, 5종 곡선).

### STEP 3 — 발표자료 훑기 (8:50–9:30)  ← 큰 화면 권장
- 파일 열기: `presentation/progress_0625.pdf` (8장)
- **확인 포인트(선배 피드백 반영분)**:
  1. Overview: SE corner/EE interior, 기준해 D별 분리, unseen 정의 ✔
  2. R1 제목 "(SE)", BA 구성 3줄 ✔
  3. R3 제목 "Effect of Problem Dimension" ✔
  4. R4 그림(상하 2패널, 글씨 크게) 잘 보이는지 ✔
  5. 모든 그래프: 위 제목 없음, y축 "optimality gap (%)" ✔
- 고칠 점 → 텔레그램으로. (deck은 11시 전송이라 여유 1시간 있음)

### STEP 4 — 답변+팔로업 전송 (9:45) ✅ 1차 데드라인(10시 전)
- 선배에게 메시지(복붙: `followup_senior_0629.txt`) + 첨부 2개:
  - `senior_feedback_answers_0628.pdf`
  - `followup_senior_0629.pdf`
  - (선택) `baselines_with_ga-1.png`

### STEP 5 — 발표자료 최종본 전송 (10:50) ✅ 2차 데드라인(11시쯤)
- 선배에게 `progress_0625.pdf` 전송. (STEP 3 수정 반영본)

---

## 🔧 수정이 필요하면 (내가 자는 동안 못 고친 것)

**나(클로드)한테 텔레그램으로 말하면 됨.** 직접 고치고 싶을 때만 아래:

```bash
# 발표자료 수정 후 재컴파일 (2번 돌려 목차 갱신)
cd presentation && pdflatex -interaction=nonstopmode progress_0625.tex && pdflatex -interaction=nonstopmode progress_0625.tex

# 답변지 / 팔로업 (한글 → xelatex)
xelatex -interaction=nonstopmode senior_feedback_answers_0628.tex
xelatex -interaction=nonstopmode followup_senior_0629.tex

# 그래프 다시 그리기 (gap 그림 전부)
cd .. && PYTHONPATH=. WANDB_MODE=disabled python scripts/plot_gap_figs.py
```

---

## 💬 선배가 되물을 만한 것 (즉답 카드)

- **"GA가 왜 이렇게 약해?"** → 300 eval 빠듯한 예산 + pop20×15세대라 10차원 수렴 부족. GA가 CMA-ES/DE보다 sample-efficiency 낮은 건 알려진 특성. 저차원(D=3)은 GA도 거의 최적(6.0). 표준 연산자(SBX+다항변이+토너먼트, pymoo 기본값) 그대로 씀, 튜닝 안 함.
- **"그럼 RIBBO가 GA 이긴 게 의미 있나?"** → GA만이면 약한 주장 맞음. 핵심은 **EE D=10에서 가장 센 CMA-ES·BO·DE까지 다 이긴 것**. GA는 "비교 스킴 다 넣었다"는 완결성 차원.
- **"CMA-ES가 더 센데 RIBBO 왜 함?"** → (1) RIBBO는 적은 예산서 빠르게 수렴(sample-efficiency), (2) 최종 비교 대상은 CMA-ES가 아니라 같은 model-free인 **LLMO**. CMA-ES 우위는 정직한 한계로 명시.
- **"WMMSE 값이 Lee랑 다른데?"** → 채널분포/세팅 차이. 동일 채널 공정비교 + brute-force로 우리 WMMSE 검증함(gap 0/10).
- **"unseen이 같은 분포면 너무 쉬운 것 아냐?"** → 학습 채널과 겹치지 않는 새 Rayleigh realization(train 0–19 / eval 30–). 완전 다른 분포 transfer는 R5(Rastrigin)에서 다룸, 실패함.
- (더 깊은 Q&A는 `MASTER_GUIDE.md` §4, 숫자는 `CHEATSHEET.md` §9.)

---

## 📋 검증 체크리스트 (보내기 전 마지막)

- [ ] 답변지 PDF 3쪽 다 열림, 깨진 글자 없음
- [ ] 팔로업 PDF 표 숫자 = README와 일치 (EE D10 RIBBO 21.5 / GA 71.2 / DE 48.6 / CMA 26.8)
- [ ] 발표자료 8쪽, 그래프 제목 없음·y축 통일·R4 글씨 보임
- [ ] 선배 메시지에 첨부 빠진 것 없는지 (답변지+팔로업 / 발표자료)
- [ ] 숫자 입으로 말할 것 vs 슬라이드 표시 구분 (선배 코멘트 B4)

---

## 📁 만들어둔 자료 전체 목록

| 파일 | 내용 |
|---|---|
| `presentation/progress_0625.pdf` | 업데이트 발표자료 (메인 8장, 선배 피드백 전부 반영) |
| `presentation/senior_feedback_answers_0628.pdf` | 질문별 답변지 (Q→답→반영, 3쪽) |
| `presentation/followup_senior_0629.pdf` | 팔로업 (반영요약 + GA 포함 baseline, 2쪽) |
| `presentation/followup_senior_0629.txt` | 선배 보낼 커버 메시지 (복붙용, 긴/짧은 버전) |
| `results/18_de_baseline/baselines_with_ga-1.png` | RIBBO vs GA/DE/CMA-ES/Random 그림 (EE D10·SE D10) |
| `results/18_de_baseline/README.md` | 모든 baseline 숫자·해석 (GA는 §결과 1b) |
| `MASTER_GUIDE.md` / `CHEATSHEET.md` | 발표 Q&A 대비 / 숫자 즉답 |

> 모든 코드/데이터/모델/이 자료들은 GitHub `omjinLTS/RIBBO-IFC` (branch main)에도 백업됨.
