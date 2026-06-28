"""
Same-epoch comparison: 3BA-trained vs 7BA-trained RIBBO on IFC SE D=3.

Both models trained 3000 epochs with identical hyperparameters; only the
behavior-algorithm pool differs (3 vs 7).

NOTE: The 7BA model is NOT converged at 3000 epochs (x_loss still
trending down). This plot is the *same-epoch* comparison, NOT the
converge-based comparison. The converge-based comparison will follow
after retraining 7BA at 5000 epochs.

Outputs:
  results/05_ifc_d3_7ba_vs_3ba/eval_ifc_3ba_vs_7ba_3000ep.pdf
  results/05_ifc_d3_7ba_vs_3ba/eval_ifc_3ba_vs_7ba_3000ep.npz
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCNumpy, IFCMetaProblem, wmmse_sum_rate

CKPT_3BA = "./log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-12-50-15740/ckpt/3000.ckpt"
CKPT_7BA = "./log/ifc/dt-ifc-run2-fullba/IFCSE-seed0-05-21-18-26-1352617/ckpt/3000.ckpt"
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"

N_CHANNELS = 500
CHANNEL_IDS = [str(i) for i in range(30, 30 + N_CHANNELS)]
N_EPISODES = 1
N_STEPS    = 300
INIT_REGRET = 0
INPUT_SEQ_LEN = 50
X_DIM, Y_DIM = 3, 1
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8

OUT_DIR = "./results/05_ifc_d3_7ba_vs_3ba"


def build_designer(ckpt_path):
    transformer = DecisionTransformer(
        x_dim=X_DIM, y_dim=Y_DIM,
        embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=300,
        num_heads=NUM_HEADS, add_bos=True,
        mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1,
        pos_encoding="embed",
    )
    d = DecisionTransformerDesigner(
        transformer=transformer,
        x_dim=X_DIM, y_dim=Y_DIM,
        embed_dim=EMBED_DIM, seq_len=300,
        input_seq_len=INPUT_SEQ_LEN,
        x_type="stochastic", y_loss_coeff=0.0,
        use_abs_timestep=True, device=DEVICE,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    d.load_state_dict(ckpt)
    d.eval()
    return d


def rollout_ribbo(designer, func, problem, n_steps, n_episodes, init_regret=0):
    B = n_episodes
    designer.past_x       = torch.zeros([B, n_steps+1, X_DIM], dtype=torch.float).to(DEVICE)
    designer.past_y       = torch.zeros([B, n_steps+1, 1],      dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps+1, 1],      dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    ts = torch.arange(n_steps+1).clamp(max=299).long()
    designer.timesteps    = ts.to(DEVICE).reshape(1, n_steps+1).repeat(B, 1)
    designer.step_count   = 0

    raw_y_all = np.zeros([B, n_steps])
    last_x = last_y = last_regret = None

    y_max_mean = problem.dataset.sp_id2info["IFCSE"]["y_max_mean"]
    y_min_mean = problem.dataset.sp_id2info["IFCSE"]["y_min_mean"]
    scale = max(y_max_mean - y_min_mean, 1e-6)

    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(
                last_x=last_x, last_y=last_y, last_onestep_regret=last_regret,
                deterministic=False, regret_strategy="none",
            )
        x_real = last_x.cpu().numpy() / 2 + 0.5
        raw_y_np = np.array([func(x_real[[b], :]).item() for b in range(B)])
        norm_y      = (raw_y_np - y_min_mean) / scale
        norm_regret = (y_max_mean - raw_y_np) / scale
        last_y      = torch.tensor(norm_y,     dtype=torch.float).reshape(B, 1).to(DEVICE)
        last_regret = torch.tensor(norm_regret, dtype=torch.float).reshape(B, 1).to(DEVICE)
        raw_y_all[:, i] = raw_y_np
    return raw_y_all


def rollout_random(func, n_steps, n_episodes, seed=42):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, size=(n_episodes, n_steps, X_DIM))
    raw_y = np.zeros([n_episodes, n_steps])
    for b in range(n_episodes):
        for i in range(n_steps):
            raw_y[b, i] = func(xs[b, i:i+1, :]).item()
    return raw_y


def best_so_far(y_arr):
    return np.maximum.accumulate(y_arr, axis=1)


def main():
    # 3BA and 7BA models share data_dir; only the trajectory pool differs.
    # Use 7BA data_dir (full 7BA cache) so y_max_mean reflects training stats
    # for both rollouts (RTG normalization is per-model anyway).
    problem = IFCMetaProblem(
        search_space_id="IFCSE", root_dir=None,
        data_dir="./data/generated_data/ifc", cache_dir="./cache/ifc/",
        input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=300,
        normalize_method="random", n_block=100,
    )

    print(f"Loading 3BA model: {CKPT_3BA}")
    d_3ba = build_designer(CKPT_3BA)
    print(f"Loading 7BA model: {CKPT_7BA}")
    d_7ba = build_designer(CKPT_7BA)

    raw_3ba = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    raw_7ba = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    raw_rand = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    wmmse_rates = np.zeros(N_CHANNELS)

    print(f"\nEvaluating {N_CHANNELS} channels × {N_EPISODES} ep × {N_STEPS} steps")
    for i, cid in enumerate(CHANNEL_IDS):
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{N_CHANNELS}]")
        func = IFCNumpy("IFCSE", cid, dim=3)
        raw_3ba[i] = rollout_ribbo(d_3ba, func, problem, N_STEPS, N_EPISODES, INIT_REGRET)
        raw_7ba[i] = rollout_ribbo(d_7ba, func, problem, N_STEPS, N_EPISODES, INIT_REGRET)
        raw_rand[i] = rollout_random(func, N_STEPS, N_EPISODES, seed=int(cid)+1000)
        wmmse_rates[i], _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)

    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, "eval_ifc_3ba_vs_7ba_3000ep.npz")
    np.savez(npz_path,
             raw_3ba=raw_3ba, raw_7ba=raw_7ba, raw_rand=raw_rand,
             wmmse=wmmse_rates, channel_ids=np.array(CHANNEL_IDS))
    print(f"\nSaved raw rollouts: {npz_path}")

    bsf_3ba = best_so_far(raw_3ba.mean(axis=1))
    bsf_7ba = best_so_far(raw_7ba.mean(axis=1))
    bsf_rand = best_so_far(raw_rand.mean(axis=1))

    wmmse_mean, wmmse_std = wmmse_rates.mean(), wmmse_rates.std()

    print("\n" + "="*72)
    print(f"{'Method':<28} {'Final (nats)':<14} {'(bits)':<10} {'% of WMMSE':<10}")
    print("-"*72)
    for name, arr in [("RIBBO 3BA (3000 ep, conv)", bsf_3ba),
                      ("RIBBO 7BA (3000 ep, *not conv*)", bsf_7ba),
                      ("Random", bsf_rand)]:
        fn = arr[:, -1].mean()
        print(f"{name:<28} {fn:<14.4f} {fn/np.log(2):<10.4f} {100*fn/wmmse_mean:<10.1f}")
    print(f"{'WMMSE (mean)':<28} {wmmse_mean:<14.4f} {wmmse_mean/np.log(2):<10.4f} (ref)")
    print(f"{'WMMSE (std)':<28} ±{wmmse_std:.4f}")
    print("="*72)

    evals = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evals, bsf_3ba.mean(0),  color='#2563EB', lw=2.5, label='RIBBO 3BA (3000 ep, converged)')
    ax.plot(evals, bsf_7ba.mean(0),  color='#16A34A', lw=2.5, label='RIBBO 7BA (3000 ep, NOT converged)')
    ax.plot(evals, bsf_rand.mean(0), color='#9CA3AF', lw=1.8, ls='--', label='Random')

    ax.axhline(wmmse_mean, color='#000000', lw=1.8, label='WMMSE (mean)')
    ax.fill_between(evals, wmmse_mean - wmmse_std, wmmse_mean + wmmse_std,
                    color='#000000', alpha=0.10, label='WMMSE ±1σ')

    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Average Best SE (nats)', fontsize=12)
    ax.set_title(f'Same-Epoch Comparison: 3BA vs 7BA at 3000 epochs (IFC SE D=3)\n'
                 f'{N_CHANNELS} unseen channels × {N_EPISODES} ep — 7BA NOT yet converged',
                 fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(1, N_STEPS)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls='--', alpha=0.3)

    out = os.path.join(OUT_DIR, "eval_ifc_3ba_vs_7ba_3000ep.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
