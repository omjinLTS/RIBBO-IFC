"""
Evaluation matched to Lee et al. (2025) Fig. 7(b) setup:
  - Channel dist : h_{d'd} ~ CN(0,1)  [same as our IFCNumpy]
  - # channels   : 200 random realizations (IDs 30-229)
  - # evals      : 500 (= Lee+25: 100 iterations × P=5 actions)
  - x-axis       : Function Evaluations (Lee+25 iter t ≡ x = t*5)
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCNumpy, IFCMetaProblem, wmmse_sum_rate

# ── config ───────────────────────────────────────────────────────────────────
CKPT   = "./log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-12-50-15740/ckpt/3000.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

N_CHANNELS  = 500          # extended: 200 → 500
CHANNEL_IDS = [str(i) for i in range(30, 30 + N_CHANNELS)]   # unseen
N_EPISODES  = 3            # extended: 1 → 3 (stochastic policy variance reduction)
N_STEPS     = 300          # in-distribution (trained on 300-step trajectories)
INIT_REGRET = 0
INPUT_SEQ_LEN = 50
X_DIM, Y_DIM  = 3, 1
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8


# ── model ────────────────────────────────────────────────────────────────────
def build_designer():
    # Must use seq_len=300 to match checkpoint architecture
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
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    d.load_state_dict(ckpt)
    d.eval()
    return d


# ── custom eval loop (supports n_steps > seq_len) ────────────────────────────
def rollout_ribbo(designer, func, problem, n_steps, n_episodes, init_regret=0):
    """
    Run RIBBO for n_steps. designer.reset() uses seq_len+1 buffer;
    we override the buffers to n_steps+1 so step_count never overflows.
    """
    B = n_episodes
    designer.past_x       = torch.zeros([B, n_steps+1, X_DIM],  dtype=torch.float).to(DEVICE)
    designer.past_y       = torch.zeros([B, n_steps+1, 1],       dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps+1, 1],       dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    # cap timestep at 299 (trained max) so positional embedding stays in-distribution
    ts = torch.arange(n_steps+1).clamp(max=299).long()
    designer.timesteps    = ts.to(DEVICE).reshape(1, n_steps+1).repeat(B, 1)
    designer.step_count   = 0

    raw_y_all = np.zeros([B, n_steps])
    last_x = last_y = last_regret = None

    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(
                last_x=last_x,
                last_y=last_y,
                last_onestep_regret=last_regret,
                deterministic=False,
                regret_strategy="none",
            )

        # evaluate on IFCNumpy directly (raw r_SE, no normalization needed)
        x_np = last_x.cpu().numpy()  # [B, 3]
        # unnormalize: designer outputs in [-1,1] → [0,1]
        x_real = x_np / 2 + 0.5     # matches MetaProblemBase.transform_x
        raw_y_np = np.array([func(x_real[[b], :]).item() for b in range(B)])  # [B]

        # normalized y & regret for the designer's context update
        y_max_mean = problem.dataset.sp_id2info["IFCSE"]["y_max_mean"]
        y_min_mean = problem.dataset.sp_id2info["IFCSE"]["y_min_mean"]
        scale = max(y_max_mean - y_min_mean, 1e-6)
        norm_y     = (raw_y_np - y_min_mean) / scale
        norm_regret = (y_max_mean - raw_y_np) / scale

        last_y      = torch.tensor(norm_y,     dtype=torch.float).reshape(B, 1).to(DEVICE)
        last_regret = torch.tensor(norm_regret, dtype=torch.float).reshape(B, 1).to(DEVICE)
        raw_y_all[:, i] = raw_y_np

    return raw_y_all   # [B, n_steps]


def rollout_random(func, n_steps, n_episodes, seed=42):
    rng = np.random.default_rng(seed)
    xs  = rng.uniform(0, 1, size=(n_episodes, n_steps, X_DIM))
    raw_y = np.zeros([n_episodes, n_steps])
    for b in range(n_episodes):
        for i in range(n_steps):
            raw_y[b, i] = func(xs[b, i:i+1, :]).item()
    return raw_y


def best_so_far(y_arr):
    return np.maximum.accumulate(y_arr, axis=1).mean(axis=0)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # build problem only for normalization stats
    problem = IFCMetaProblem(
        search_space_id="IFCSE", root_dir=None,
        data_dir="./data/generated_data/ifc", cache_dir="./cache/ifc/",
        input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=300,
        normalize_method="random", n_block=100,
    )

    designer = build_designer()

    ribbo_bsf = []
    rand_bsf  = []
    ribbo_inst = []
    rand_inst  = []
    wmmse_rates = []

    print(f"Evaluating {N_CHANNELS} channels × {N_EPISODES} episodes × {N_STEPS} steps ...")
    for i, cid in enumerate(CHANNEL_IDS):
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{N_CHANNELS}]")
        func = IFCNumpy("IFCSE", cid, dim=3, lb=0.0, ub=1.0)

        ribbo_raw = rollout_ribbo(designer, func, problem, N_STEPS, N_EPISODES, INIT_REGRET)
        rand_raw  = rollout_random(func, N_STEPS, N_EPISODES, seed=int(cid)+1000)
        r_wmmse, _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)

        ribbo_bsf.append(best_so_far(ribbo_raw))
        rand_bsf.append(best_so_far(rand_raw))
        ribbo_inst.append(ribbo_raw.mean(0))   # instantaneous mean over episodes [N_STEPS]
        rand_inst.append(rand_raw.mean(0))
        wmmse_rates.append(r_wmmse)

    ribbo_bsf = np.stack(ribbo_bsf)   # [N_CH, N_STEPS]
    rand_bsf  = np.stack(rand_bsf)
    ribbo_inst = np.stack(ribbo_inst)
    rand_inst  = np.stack(rand_inst)
    wmmse_rates = np.array(wmmse_rates)   # [N_CH]

    # ── print ─────────────────────────────────────────────────────────────────
    wmmse_mean = wmmse_rates.mean()
    wmmse_std  = wmmse_rates.std()
    print("\n" + "="*70)
    print(f"{'Method':<22} {'Final (nats)':<16} {'Final (bits)':<14} {'gap to WMMSE':<14}")
    print("-"*70)
    for name, arr in [("RIBBO best-so-far", ribbo_bsf), ("Random best-so-far", rand_bsf)]:
        fn = arr[:, -1].mean()
        print(f"{name:<22} {fn:<16.4f} {fn/np.log(2):<14.4f} {fn - wmmse_mean:<+14.4f}")
    print(f"{'WMMSE (mean)':<22} {wmmse_mean:<16.4f} {wmmse_mean/np.log(2):<14.4f} {'(reference)':<14}")
    print(f"{'WMMSE (std)':<22} ±{wmmse_std:.4f}")
    print("="*70)

    # reference values from Lee et al. Fig 7(b)
    ref = {
        "GNN [Lee+25]":               2.70,
        "WMMSE/Local opt [Lee+25]":   2.50,
        "LLMO-E [Lee+25]":            2.50,
        "GA [Lee+25]":                2.40,
        "BO [Lee+25]":                2.20,
        "Brute-force [Lee+25]":       1.50,
    }
    print("\nLee et al. (2025) reference @ iteration 100 (500 evals):")
    for k, v in ref.items():
        print(f"  {k:<32} {v:.2f} nats  ({v/np.log(2):.2f} bits)")

    # ── plot ──────────────────────────────────────────────────────────────────
    # x-axis: function evaluations (1-500)
    # secondary x-axis: Lee+25 iterations (evals / 5)
    evals = np.arange(1, N_STEPS + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(evals, ribbo_bsf.mean(0), color='#2563EB', lw=2.5, label='RIBBO best-so-far')
    ax.plot(evals, rand_bsf.mean(0),  color='#9CA3AF', lw=1.8, ls='--', label='Random best-so-far')
    ax.plot(evals, ribbo_inst.mean(0), color='#2563EB', lw=1.0, ls=':', alpha=0.6, label='RIBBO current')
    ax.plot(evals, rand_inst.mean(0),  color='#9CA3AF', lw=1.0, ls=':', alpha=0.6, label='Random current')

    # WMMSE optimum overlay (per-channel optimum, averaged across channels)
    ax.axhline(wmmse_mean, color='#000000', lw=1.8, ls='-', alpha=0.85, label='WMMSE (mean)')
    ax.fill_between(evals, wmmse_mean - wmmse_std, wmmse_mean + wmmse_std,
                    color='#000000', alpha=0.10, label='WMMSE ±1σ')

    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Average SE (nats)', fontsize=12)
    ax.set_title(f'Convergence Behavior — IFC SE (D=3, $P_{{tx}}$=10 W)\n'
                 f'{N_CHANNELS} channels × {N_EPISODES} ep (solid=best-so-far, dotted=current, black=WMMSE)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(1, N_STEPS)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls='--', alpha=0.3)

    out = './presentation/eval_ifc_matched_with_wmmse.pdf'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
