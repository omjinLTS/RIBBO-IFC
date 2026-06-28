# 01 — BBOB Rastrigin x_loss 수렴 검증

**목적**: RIBBO 학습 파이프라인이 paper 기본 hyperparameter로 정상 수렴하는지 검증. IFC로 가기 전 sanity check.

## Setup

- Problem: BBOB Rastrigin (D=10)
- Behavior algorithms: 3 (Random, HillClimbing, CMAES)
- Tasks (channels): 20 train × 300 step
- Model: causal Transformer, embed_dim=256, num_layers=12, num_heads=8
- Training: 1000 epoch, AdamW lr=2e-4, cosine annealing + warmup 10k step
- GPU: RTX 4070 Ti 12GB, ~51min
- run: `log/synthetic/dt-convergence-check/Rastrigin-seed0-05-18-10-28-26005/`

## Result

- x_loss (Gaussian policy NLL): epoch 1 = +10.78 → epoch 1000 = **-44.94**
- ~epoch 500-600 이후 plateau (수렴 확인)
- Epoch 100~200 진동은 LR warmup(10k step)의 자연스러운 효과

## Figures

- `convergence_rastrigin.pdf` — x_loss vs epoch 곡선
