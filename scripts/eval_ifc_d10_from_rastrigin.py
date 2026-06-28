"""
T4: Zero-shot transfer test — Rastrigin D=10 trained RIBBO model
evaluated on IFC D=10 power control channels.

Tests RIBBO's scalability / generalization across problem families
at the same input dimension.

⚠️ Reward magnitude mismatch:
- Rastrigin y range ≈ [-90, 0] (negated, maximized)
- IFC SE D=10 expected ≈ [0, 20]
- Model conditioned on Rastrigin RTG scale → IFC y appears "already optimal"
  under Rastrigin normalization, may suppress exploration.
- Per plan: report this as scalability/transfer finding regardless of outcome.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCNumpy
from problems.synthetic import SyntheticMetaProblem

# ── config ───────────────────────────────────────────────────────────────────
CKPT   = "./log/synthetic/dt-convergence-check/Rastrigin-seed0-05-18-10-28-26005/ckpt/1000.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

N_CHANNELS  = 200
CHANNEL_IDS = [str(i) for i in range(30, 30 + N_CHANNELS)]
N_EPISODES  = 1
N_STEPS     = 150           # cap at Rastrigin's max_input_seq_len (model pos-emb size 151)
INIT_REGRET = 0
INPUT_SEQ_LEN = 50          # Rastrigin run_main.sh override
SEQ_LEN       = 150         # Rastrigin model arch (pos_encoding size 151 ⇒ seq=150)

X_DIM, Y_DIM  = 10, 1       # D=10 for both Rastrigin and IFC
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8


# ── model ────────────────────────────────────────────────────────────────────
def build_designer():
    transformer = DecisionTransformer(
        x_dim=X_DIM, y_dim=Y_DIM,
        embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
        num_heads=NUM_HEADS, add_bos=True,
        mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1,
        pos_encoding="embed",
    )
    d = DecisionTransformerDesigner(
        transformer=transformer,
        x_dim=X_DIM, y_dim=Y_DIM,
        embed_dim=EMBED_DIM, seq_len=SEQ_LEN,
        input_seq_len=INPUT_SEQ_LEN,
        x_type="stochastic", y_loss_coeff=0.0,
        use_abs_timestep=True, device=DEVICE,
    )
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    d.load_state_dict(ckpt)
    d.eval()
    return d


# ── rollout (Rastrigin-conditioned model on IFC D=10) ────────────────────────
def rollout_ribbo(designer, func, y_stats, n_steps, n_episodes, init_regret=0):
    """
    y_stats: dict with 'y_max_mean', 'y_min_mean' from Rastrigin training distribution.
    Feeds IFC raw y through Rastrigin normalization (intentional, transfer test).
    """
    B = n_episodes
    designer.past_x       = torch.zeros([B, n_steps+1, X_DIM],  dtype=torch.float).to(DEVICE)
    designer.past_y       = torch.zeros([B, n_steps+1, 1],       dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps+1, 1],       dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    ts = torch.arange(n_steps+1).clamp(max=SEQ_LEN - 1).long()  # cap at model's pos-emb range
    designer.timesteps    = ts.to(DEVICE).reshape(1, n_steps+1).repeat(B, 1)
    designer.step_count   = 0

    raw_y_all = np.zeros([B, n_steps])
    last_x = last_y = last_regret = None

    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(
                last_x=last_x, last_y=last_y, last_onestep_regret=last_regret,
                deterministic=False, regret_strategy="none",
            )

        # designer outputs in [-1,1]; IFC expects [0,1]
        x_np = last_x.cpu().numpy()                   # [B, 10]
        x_real = x_np / 2 + 0.5
        raw_y_np = np.array([func(x_real[[b], :]).item() for b in range(B)])  # [B]

        # Normalize using Rastrigin stats (transfer test — intentional mismatch)
        scale = max(y_stats['y_max_mean'] - y_stats['y_min_mean'], 1e-6)
        norm_y      = (raw_y_np - y_stats['y_min_mean']) / scale
        norm_regret = (y_stats['y_max_mean'] - raw_y_np) / scale

        last_y      = torch.tensor(norm_y,     dtype=torch.float).reshape(B, 1).to(DEVICE)
        last_regret = torch.tensor(norm_regret, dtype=torch.float).reshape(B, 1).to(DEVICE)
        raw_y_all[:, i] = raw_y_np

    return raw_y_all


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
    # Build Rastrigin meta-problem only to grab normalization stats from training data
    rast_problem = SyntheticMetaProblem(
        search_space_id="Rastrigin", root_dir=None,
        data_dir="./data/generated_data/synthetic", cache_dir="./cache/synthetic/",
        input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=SEQ_LEN,
        normalize_method="random", n_block=100,
    )
    y_stats = {
        'y_max_mean': rast_problem.dataset.sp_id2info["Rastrigin"]["y_max_mean"],
        'y_min_mean': rast_problem.dataset.sp_id2info["Rastrigin"]["y_min_mean"],
    }
    print(f"Rastrigin training stats: {y_stats}")

    designer = build_designer()

    ribbo_raw_all = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    rand_raw_all  = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))

    print(f"\nEvaluating {N_CHANNELS} ch × {N_EPISODES} ep × {N_STEPS} steps "
          f"(IFC D={X_DIM} from Rastrigin D={X_DIM} ckpt)")
    for i, cid in enumerate(CHANNEL_IDS):
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{N_CHANNELS}]")
        func = IFCNumpy("IFCSE", cid, dim=X_DIM, lb=0.0, ub=1.0)
        ribbo_raw = rollout_ribbo(designer, func, y_stats, N_STEPS, N_EPISODES, INIT_REGRET)
        rand_raw  = rollout_random(func, N_STEPS, N_EPISODES, seed=int(cid)+1000)
        ribbo_raw_all[i] = ribbo_raw
        rand_raw_all[i]  = rand_raw

    # Save .npz so T3-style current-y plot can re-use this data
    os.makedirs("./presentation", exist_ok=True)
    npz_path = "./presentation/eval_ifc_d10_from_rastrigin.npz"
    np.savez(npz_path,
             ribbo_raw=ribbo_raw_all, rand_raw=rand_raw_all,
             channel_ids=np.array(CHANNEL_IDS),
             n_steps=N_STEPS, x_dim=X_DIM)
    print(f"Saved raw rollouts → {npz_path}")

    # Per-channel best-so-far (averaged over episodes), then mean across channels
    ribbo_bsf = np.stack([best_so_far(ribbo_raw_all[i]) for i in range(N_CHANNELS)])
    rand_bsf  = np.stack([best_so_far(rand_raw_all[i])  for i in range(N_CHANNELS)])
    # Instantaneous (per-step) mean across episodes
    ribbo_inst = ribbo_raw_all.mean(axis=1)   # [N_CH, N_STEPS]
    rand_inst  = rand_raw_all.mean(axis=1)

    print("\n" + "="*55)
    print(f"{'Method':<22} {'Final SE (nats)':<16} {'Final (bits)':<14}")
    print("-"*55)
    for name, arr in [("RIBBO/Rast→IFC D=10", ribbo_bsf), ("Random D=10", rand_bsf)]:
        fn = arr[:, -1].mean()
        print(f"{name:<22} {fn:<16.4f} {fn/np.log(2):<14.4f}")
    print("="*55)

    # Plot
    evals = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(evals, ribbo_bsf.mean(0), color='#2563EB', lw=2.5,
            label=f'RIBBO best-so-far (Rastrigin→IFC D={X_DIM})')
    ax.plot(evals, rand_bsf.mean(0),  color='#9CA3AF', lw=1.8, ls='--',
            label=f'Random best-so-far D={X_DIM}')
    ax.plot(evals, ribbo_inst.mean(0), color='#2563EB', lw=1.0, ls=':', alpha=0.6,
            label='RIBBO current')
    ax.plot(evals, rand_inst.mean(0),  color='#9CA3AF', lw=1.0, ls=':', alpha=0.6,
            label='Random current')
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel(f'Average SE (nats), D={X_DIM}', fontsize=12)
    ax.set_title(f'Zero-shot Transfer: Rastrigin D={X_DIM} → IFC SE D={X_DIM}\n'
                 f'{N_CHANNELS} Rayleigh channels (solid=best-so-far, dotted=current)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_xlim(1, N_STEPS)
    ax.grid(True, ls='--', alpha=0.3)
    out = './presentation/eval_ifc_d10_from_rastrigin_with_instant.pdf'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
