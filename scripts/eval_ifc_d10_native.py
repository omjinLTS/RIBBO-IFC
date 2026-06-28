"""
IFC D=10 native evaluation:
  - RIBBO trained on IFC D=10 (60 trajectories, 3BA, 3000 epoch)
  - vs WMMSE optimum (per-channel)
  - vs Random
  - overlays Rastrigin-D=10 transfer result from prior run

Output:
  presentation/eval_ifc_d10_native.pdf — 4-curve comparison plot
  presentation/eval_ifc_d10_native.npz — raw rollouts
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
CKPT   = "./log/ifc/dt-ifc-d10-run1/IFCSED10-seed0-05-21-10-34-541982/ckpt/3000.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

N_CHANNELS  = 200
CHANNEL_IDS = [str(i) for i in range(30, 30 + N_CHANNELS)]
N_EPISODES  = 1
N_STEPS     = 300
INIT_REGRET = 0
INPUT_SEQ_LEN = 50
X_DIM, Y_DIM  = 10, 1
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8

TRANSFER_NPZ = "./presentation/eval_ifc_d10_from_rastrigin.npz"


def build_designer():
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

    y_max_mean = problem.dataset.sp_id2info["IFCSED10"]["y_max_mean"]
    y_min_mean = problem.dataset.sp_id2info["IFCSED10"]["y_min_mean"]
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
    problem = IFCMetaProblem(
        search_space_id="IFCSED10", root_dir=None,
        data_dir="./data/generated_data/ifc_d10", cache_dir="./cache/ifc_d10/",
        input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=300,
        normalize_method="random", n_block=100,
    )
    designer = build_designer()

    ribbo_raw_all = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    rand_raw_all  = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    wmmse_rates   = np.zeros(N_CHANNELS)

    print(f"Evaluating {N_CHANNELS} channels × {N_EPISODES} ep × {N_STEPS} steps (D=10 native)")
    for i, cid in enumerate(CHANNEL_IDS):
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{N_CHANNELS}]")
        func = IFCNumpy("IFCSED10", cid, dim=10)
        ribbo_raw_all[i] = rollout_ribbo(designer, func, problem, N_STEPS, N_EPISODES, INIT_REGRET)
        rand_raw_all[i]  = rollout_random(func, N_STEPS, N_EPISODES, seed=int(cid)+1000)
        wmmse_rates[i], _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)

    # save raw
    os.makedirs("./presentation", exist_ok=True)
    np.savez("./presentation/eval_ifc_d10_native.npz",
             ribbo_raw=ribbo_raw_all, rand_raw=rand_raw_all, wmmse=wmmse_rates,
             channel_ids=np.array(CHANNEL_IDS))

    # per-channel curves
    ribbo_bsf = best_so_far(ribbo_raw_all.mean(axis=1))   # [N_CH, N_STEPS]
    rand_bsf  = best_so_far(rand_raw_all.mean(axis=1))
    ribbo_inst = ribbo_raw_all.mean(axis=1)
    rand_inst  = rand_raw_all.mean(axis=1)

    # load Rastrigin transfer results (D=10, N_STEPS_TRANSFER=150)
    transfer_avail = os.path.exists(TRANSFER_NPZ)
    if transfer_avail:
        t = np.load(TRANSFER_NPZ)
        transfer_raw = t['ribbo_raw']         # [N_CH, N_EP, 150]
        transfer_bsf = best_so_far(transfer_raw.mean(axis=1))
        transfer_inst = transfer_raw.mean(axis=1)
        n_steps_transfer = transfer_raw.shape[-1]
        print(f"Loaded Rastrigin transfer results ({transfer_raw.shape})")

    wmmse_mean, wmmse_std = wmmse_rates.mean(), wmmse_rates.std()

    print("\n" + "="*72)
    print(f"{'Method':<32} {'Final (nats)':<16} {'Final (bits)':<14} {'% WMMSE':<8}")
    print("-"*72)
    for name, arr in [("RIBBO D=10 native best-so-far", ribbo_bsf),
                      ("Random D=10 best-so-far", rand_bsf)]:
        fn = arr[:, -1].mean()
        print(f"{name:<32} {fn:<16.4f} {fn/np.log(2):<14.4f} {100*fn/wmmse_mean:<8.1f}")
    if transfer_avail:
        fn = transfer_bsf[:, -1].mean()
        print(f"{'RIBBO Rast→IFC transfer (@150)':<32} {fn:<16.4f} {fn/np.log(2):<14.4f} {100*fn/wmmse_mean:<8.1f}")
    print(f"{'WMMSE D=10 (mean)':<32} {wmmse_mean:<16.4f} {wmmse_mean/np.log(2):<14.4f} (ref)")
    print(f"{'WMMSE D=10 (std)':<32} ±{wmmse_std:.4f}")
    print("="*72)

    # ── plot ─────────────────────────────────────────────────────────────────
    evals = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(evals, ribbo_bsf.mean(0), color='#2563EB', lw=2.5,
            label='RIBBO D=10 native (best-so-far)')
    ax.plot(evals, rand_bsf.mean(0),  color='#9CA3AF', lw=1.8, ls='--',
            label='Random D=10 (best-so-far)')
    ax.plot(evals, ribbo_inst.mean(0), color='#2563EB', lw=0.9, ls=':', alpha=0.6,
            label='RIBBO native (current)')
    ax.plot(evals, rand_inst.mean(0),  color='#9CA3AF', lw=0.9, ls=':', alpha=0.6,
            label='Random (current)')

    if transfer_avail:
        evals_t = np.arange(1, n_steps_transfer + 1)
        ax.plot(evals_t, transfer_bsf.mean(0), color='#DC2626', lw=2.0, ls='-',
                label=f'RIBBO Rast→IFC transfer (best-so-far, ≤{n_steps_transfer})')
        ax.plot(evals_t, transfer_inst.mean(0), color='#DC2626', lw=0.9, ls=':', alpha=0.6,
                label='Rast→IFC (current)')

    ax.axhline(wmmse_mean, color='#000000', lw=1.8, label='WMMSE D=10 (mean)')
    ax.fill_between(evals, wmmse_mean - wmmse_std, wmmse_mean + wmmse_std,
                    color='#000000', alpha=0.10, label='WMMSE ±1σ')

    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Average SE (nats), D=10', fontsize=12)
    ax.set_title(f'IFC SE D=10 — Native training vs Rastrigin transfer vs WMMSE\n'
                 f'{N_CHANNELS} unseen Rayleigh channels', fontsize=11)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(1, N_STEPS)
    ax.grid(True, ls='--', alpha=0.3)
    out = './presentation/eval_ifc_d10_native.pdf'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
