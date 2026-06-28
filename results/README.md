# Results Index

실험별 결과를 카테고리로 나눠 정리. 각 디렉토리에 README.md로 setup·그림 설명·핵심 수치 기록.

| Dir | Topic | Status | Date |
|---|---|---|---|
| [01_rastrigin_convergence](01_rastrigin_convergence/) | BBOB Rastrigin x_loss 수렴 검증 (D=10) | done | 2026-05-18 |
| [02_ifc_d3_3ba_initial](02_ifc_d3_3ba_initial/) | 첫 IFC SE pipeline — 3BA × 20 ch × 300 step | done (5/18 발표) | 2026-05-18 |
| [03_ifc_d3_3ba_instantaneous](03_ifc_d3_3ba_instantaneous/) | best-so-far 외 raw(current) reward overlay | done | 2026-05-20 |
| [04_ifc_d3_3ba_wmmse](04_ifc_d3_3ba_wmmse/) | WMMSE optimum overlay (500 ch × 3 ep) | done | 2026-05-20 |
| [05_ifc_d3_7ba_vs_3ba](05_ifc_d3_7ba_vs_3ba/) | 7BA vs 3BA training (BA diversity ablation) | done | 2026-05-22 |
| [06_ifc_d10_transfer_vs_native](06_ifc_d10_transfer_vs_native/) | Rastrigin→IFC D=10 zero-shot transfer + native | done | 2026-05-21 |
| [07_ifc_5ba_filter_ablation](07_ifc_5ba_filter_ablation/) | BA-pool × Dimension 2×2 ablation (key finding) | **done** | 2026-05-27 |
| [08_eval_budget_extension](08_eval_budget_extension/) | (#1) eval budget 300→1000 vs WMMSE (+4.8p D=3, +8.1p D=10) | **done** | 2026-06-23 |
| [09_ifc_ee](09_ifc_ee/) | (#2) SE vs EE — RIBBO +48.7p over Random on EE D=10 (key) | **done** | 2026-06-25 |
| [10_rastrigin_transfer_normfix](10_rastrigin_transfer_normfix/) | (#3) transfer 실패 ≠ normalization (가설 반증) | **done** | 2026-06-23 |
| [11_ifc_d20_scalability](11_ifc_d20_scalability/) | (#4) 차원 sweep D=3→10→20 (88.9→47.8→33.0%) | **done** | 2026-06-25 |

## 통합 비교 — D=3 IFC SE, 500 ch × 3 ep × 300 step (where applicable)

| Setup | Final SE (nats) | % of WMMSE | 비고 |
|---|---|---|---|
| **WMMSE (our impl)** | **2.864** | 100% (ref) | per-channel optimum |
| RIBBO 3BA (3000 ep, conv) | 2.519 | **87.9%** | dt-ifc-run1, best of our trained models |
| RIBBO 7BA (3000 ep, NOT conv) | 2.281 | 79.6% | dt-ifc-run2-fullba (same-epoch) |
| RIBBO 7BA (5000 ep, conv) | 2.187 | **76.4%** | dt-ifc-run3 (converged but worse than 3BA) |
| Random | 2.379 | 83.0% | uniform [0,1]^3 baseline |

**핵심 finding**: BA pool을 3→7로 늘리는 게 IFC SE D=3에서는 오히려 성능 하락. 5000 epoch까지 학습해도 3BA의 88%에서 76%로 떨어짐. 자세한 가설은 [05_*/README.md](05_ifc_d3_7ba_vs_3ba/README.md).

D=10 결과는 [06_*/README.md](06_ifc_d10_transfer_vs_native/README.md).

## 디렉토리 규칙

- 각 폴더 = 한 가지 비교/실험. PDF + npz + README.
- 같은 모델 ckpt를 여러 polot에서 쓰면 가장 처음 등장한 폴더에 raw npz, 이후는 참조만.
- 새 실험 추가 시 번호 증가(`07_...`)로.
- 발표 자료(`presentation/main.tex`)는 5/18 시점 그대로 보존. 새 결과는 여기 results/에 정리하고 발표 자료는 새로 만들 때 참조.
