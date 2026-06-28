"""
D=10 3-way comparison: 3BA-native vs 5BA-filter vs Random + WMMSE.
Companion to scripts/eval_ifc_d3_3way.py (D=3 version with 7BA).

Outputs:
  results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.pdf
  results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCNumpy, IFCMetaProblem, wmmse_sum_rate

CKPT_3BA = "./log/ifc/dt-ifc-d10-run1/IFCSED10-seed0-05-21-10-34-541982/ckpt/3000.ckpt"
CKPT_5BA = "./log/ifc/dt-ifc-d10-run2b-5bafilter-resume/IFCSED10-seed0-05-27-01-41-1171473/ckpt/5000.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

N_CHANNELS = 200
CHANNEL_IDS = [str(i) for i in range(30, 30 + N_CHANNELS)]
N_EPISODES = 1
N_STEPS = 300
INIT_REGRET = 0
INPUT_SEQ_LEN = 50
X_DIM, Y_DIM = 10, 1
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8

OUT_DIR = "./results/07_ifc_5ba_filter_ablation"


def build_designer(ckpt):
    transformer = DecisionTransformer(
        x_dim=X_DIM, y_dim=Y_DIM, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=300, num_heads=NUM_HEADS, add_bos=True, mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1, pos_encoding="embed",
    )
    d = DecisionTransformerDesigner(
        transformer=transformer, x_dim=X_DIM, y_dim=Y_DIM, embed_dim=EMBED_DIM, seq_len=300,
        input_seq_len=INPUT_SEQ_LEN, x_type="stochastic", y_loss_coeff=0.0,
        use_abs_timestep=True, device=DEVICE,
    )
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    d.load_state_dict(sd); d.eval()
    return d


def rollout_ribbo(designer, func, y_max, y_min, n_steps, n_episodes, init_regret=0):
    B = n_episodes
    designer.past_x = torch.zeros([B, n_steps+1, X_DIM], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps+1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps+1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    ts = torch.arange(n_steps+1).clamp(max=299).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps+1).repeat(B, 1)
    designer.step_count = 0
    raw_y_all = np.zeros([B, n_steps])
    last_x = last_y = last_regret = None
    scale = max(y_max - y_min, 1e-6)
    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(last_x=last_x, last_y=last_y, last_onestep_regret=last_regret,
                                      deterministic=False, regret_strategy="none")
        x_real = last_x.cpu().numpy() / 2 + 0.5
        raw_y = np.array([func(x_real[[b], :]).item() for b in range(B)])
        norm_y = (raw_y - y_min) / scale
        norm_r = (y_max - raw_y) / scale
        last_y = torch.tensor(norm_y, dtype=torch.float).reshape(B, 1).to(DEVICE)
        last_regret = torch.tensor(norm_r, dtype=torch.float).reshape(B, 1).to(DEVICE)
        raw_y_all[:, i] = raw_y
    return raw_y_all


def rollout_random(func, n_steps, n_episodes, seed=42):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, size=(n_episodes, n_steps, X_DIM))
    raw_y = np.zeros([n_episodes, n_steps])
    for b in range(n_episodes):
        for i in range(n_steps):
            raw_y[b, i] = func(xs[b, i:i+1, :]).item()
    return raw_y


def best_so_far(y):
    return np.maximum.accumulate(y, axis=1)


def get_stats(data_dir, cache_dir, label):
    print(f"  loading {label}: {data_dir}")
    prob = IFCMetaProblem(
        search_space_id="IFCSED10", root_dir=None,
        data_dir=data_dir, cache_dir=cache_dir,
        input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=300,
        normalize_method="random", n_block=100,
    )
    info = prob.dataset.sp_id2info["IFCSED10"]
    return info["y_max_mean"], info["y_min_mean"]


def main():
    print("Loading per-model normalization stats")
    y3_max, y3_min = get_stats("./data/generated_data/ifc_d10",            "./cache/ifc_d10/",            "3BA")
    y5_max, y5_min = get_stats("./data/generated_data/ifc_d10_5bafilter",  "./cache/ifc_d10_5bafilter/",  "5BA")
    print(f"  3BA: y_max={y3_max:.4f}, y_min={y3_min:.4f}")
    print(f"  5BA: y_max={y5_max:.4f}, y_min={y5_min:.4f}")

    print("Loading models")
    d_3ba = build_designer(CKPT_3BA)
    d_5ba = build_designer(CKPT_5BA)

    raw_3 = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    raw_5 = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    raw_r = np.zeros((N_CHANNELS, N_EPISODES, N_STEPS))
    wmm = np.zeros(N_CHANNELS)

    print(f"\nEvaluating {N_CHANNELS} ch × {N_EPISODES} ep × {N_STEPS} steps (D=10)")
    for i, cid in enumerate(CHANNEL_IDS):
        if (i+1) % 25 == 0: print(f"  [{i+1}/{N_CHANNELS}]")
        func = IFCNumpy("IFCSED10", cid, dim=10)
        raw_3[i] = rollout_ribbo(d_3ba, func, y3_max, y3_min, N_STEPS, N_EPISODES, INIT_REGRET)
        raw_5[i] = rollout_ribbo(d_5ba, func, y5_max, y5_min, N_STEPS, N_EPISODES, INIT_REGRET)
        raw_r[i] = rollout_random(func, N_STEPS, N_EPISODES, seed=int(cid)+1000)
        wmm[i], _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)

    os.makedirs(OUT_DIR, exist_ok=True)
    npz = os.path.join(OUT_DIR, "eval_ifc_d10_3way.npz")
    np.savez(npz, raw_3ba=raw_3, raw_5ba=raw_5, raw_rand=raw_r, wmmse=wmm,
             channel_ids=np.array(CHANNEL_IDS))
    print(f"\nSaved: {npz}")

    bsf3 = best_so_far(raw_3.mean(axis=1))
    bsf5 = best_so_far(raw_5.mean(axis=1))
    bsfr = best_so_far(raw_r.mean(axis=1))
    wmm_mean, wmm_std = wmm.mean(), wmm.std()

    print("\n" + "="*72)
    print(f"{'Method':<34} {'Final (nats)':<14} {'(bits)':<10} {'% WMMSE':<10}")
    print("-"*72)
    for n, arr in [("RIBBO 3BA D=10 (3000 ep)", bsf3),
                   ("RIBBO 5BA-filter D=10 (5000 ep)", bsf5),
                   ("Random D=10", bsfr)]:
        fn = arr[:, -1].mean()
        print(f"{n:<34} {fn:<14.4f} {fn/np.log(2):<10.4f} {100*fn/wmm_mean:<10.1f}")
    print(f"{'WMMSE D=10 (mean)':<34} {wmm_mean:<14.4f} {wmm_mean/np.log(2):<10.4f} (ref)")
    print(f"{'WMMSE std':<34} ±{wmm_std:.4f}")
    print("="*72)

    evals = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evals, bsf3.mean(0), color='#2563EB', lw=2.5, label='RIBBO 3BA (R, HC, CMA)')
    ax.plot(evals, bsf5.mean(0), color='#16A34A', lw=2.5, label='RIBBO 5BA-filter (HC, RegE, Eagle, CMA, BoTorch)')
    ax.plot(evals, bsfr.mean(0), color='#9CA3AF', lw=1.8, ls='--', label='Random baseline')
    ax.axhline(wmm_mean, color='#000000', lw=1.8, label='WMMSE (mean)')
    ax.fill_between(evals, wmm_mean - wmm_std, wmm_mean + wmm_std, color='#000000', alpha=0.10, label='WMMSE ±1σ')

    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Average Best SE (nats), D=10', fontsize=12)
    ax.set_title(f'BA-pool ablation — IFC SE D=10 (3BA / 5BA-filter)\n'
                 f'{N_CHANNELS} unseen channels × {N_EPISODES} ep × {N_STEPS} steps', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(1, N_STEPS); ax.grid(True, ls='--', alpha=0.3)
    out = os.path.join(OUT_DIR, "eval_ifc_d10_3way.pdf")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
