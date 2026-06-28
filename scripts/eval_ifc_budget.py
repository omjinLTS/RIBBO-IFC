"""
Task: "does RIBBO close the WMMSE gap when the evaluation budget is increased?"
Re-evaluate the best SE models (D=3 3BA, D=10 5BA-filter) far beyond the
300-step training horizon (N_STEPS=1000) and overlay WMMSE.

Two panels: D=3 (3BA) and D=10 (5BA-filter). Vertical line marks the 300-step
training horizon. Best-so-far is monotone, so the question is the asymptote.

Outputs:
  results/08_eval_budget_extension/eval_ifc_budget.pdf
  results/08_eval_budget_extension/eval_ifc_budget.npz
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCNumpy, IFCMetaProblem, wmmse_sum_rate, ifc_rates

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT_D3_3BA = "./log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-12-50-15740/ckpt/3000.ckpt"
CKPT_D10_5BA = "./log/ifc/dt-ifc-d10-run2b-5bafilter-resume/IFCSED10-seed0-05-27-01-41-1171473/ckpt/5000.ckpt"

N_CHANNELS = 100
N_STEPS = 1000
TRAIN_HORIZON = 300
INPUT_SEQ_LEN = 50
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8
OUT_DIR = "./results/08_eval_budget_extension"


def build_designer(ckpt_path, x_dim):
    transformer = DecisionTransformer(
        x_dim=x_dim, y_dim=1, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=300, num_heads=NUM_HEADS, add_bos=True, mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1,
        pos_encoding="embed",
    )
    d = DecisionTransformerDesigner(
        transformer=transformer, x_dim=x_dim, y_dim=1, embed_dim=EMBED_DIM,
        seq_len=300, input_seq_len=INPUT_SEQ_LEN, x_type="stochastic",
        y_loss_coeff=0.0, use_abs_timestep=True, device=DEVICE,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    d.load_state_dict(ckpt)
    d.eval()
    return d


def rollout_ribbo(designer, func, x_dim, y_max_mean, y_min_mean, n_steps, init_regret=0):
    B = 1
    designer.past_x = torch.zeros([B, n_steps + 1, x_dim], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    ts = torch.arange(n_steps + 1).clamp(max=299).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps + 1).repeat(B, 1)
    designer.step_count = 0

    raw_y = np.zeros(n_steps)
    last_x = last_y = last_regret = None
    scale = max(y_max_mean - y_min_mean, 1e-6)
    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(last_x=last_x, last_y=last_y,
                                      last_onestep_regret=last_regret,
                                      deterministic=False, regret_strategy="none")
        x_real = last_x.cpu().numpy() / 2 + 0.5
        y = float(func(x_real)[0, 0])
        last_y = torch.tensor([[(y - y_min_mean) / scale]], dtype=torch.float).to(DEVICE)
        last_regret = torch.tensor([[(y_max_mean - y) / scale]], dtype=torch.float).to(DEVICE)
        raw_y[i] = y
    return raw_y


def rollout_random(func, x_dim, n_steps, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_steps, x_dim))
    return func(X).ravel()


def stats(data_dir, cache_dir, ssid):
    prob = IFCMetaProblem(search_space_id=ssid, root_dir=None, data_dir=data_dir,
                          cache_dir=cache_dir, input_seq_len=INPUT_SEQ_LEN,
                          max_input_seq_len=300, normalize_method="random", n_block=100)
    info = prob.dataset.sp_id2info[ssid]
    return info["y_max_mean"], info["y_min_mean"]


def run_panel(ckpt, x_dim, ssid, data_dir, cache_dir, label):
    print(f"\n=== {label} (D={x_dim}) ===")
    ymax, ymin = stats(data_dir, cache_dir, ssid)
    print(f"  norm stats: y_max={ymax:.4f} y_min={ymin:.4f}")
    d = build_designer(ckpt, x_dim)
    raw_ribbo = np.zeros((N_CHANNELS, N_STEPS))
    raw_rand = np.zeros((N_CHANNELS, N_STEPS))
    wmmse = np.zeros(N_CHANNELS)
    for i in range(N_CHANNELS):
        cid = str(30 + i)
        func = IFCNumpy(ssid, cid, dim=x_dim)
        raw_ribbo[i] = rollout_ribbo(d, func, x_dim, ymax, ymin, N_STEPS)
        raw_rand[i] = rollout_random(func, x_dim, N_STEPS, seed=int(cid) + 1000)
        wmmse[i], _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{N_CHANNELS}]")
    return raw_ribbo, raw_rand, wmmse


def bsf(y):
    return np.maximum.accumulate(y, axis=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r3, rr3, w3 = run_panel(CKPT_D3_3BA, 3, "IFCSE",
                            "./data/generated_data/ifc", "./cache/ifc/", "RIBBO 3BA")
    r10, rr10, w10 = run_panel(CKPT_D10_5BA, 10, "IFCSED10",
                               "./data/generated_data/ifc_d10_5bafilter",
                               "./cache/ifc_d10_5bafilter/", "RIBBO 5BA-filter")

    np.savez(os.path.join(OUT_DIR, "eval_ifc_budget.npz"),
             r3=r3, rr3=rr3, w3=w3, r10=r10, rr10=rr10, w10=w10)

    evals = np.arange(1, N_STEPS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (rb, rr, w, dlabel, mlabel) in zip(axes, [
        (r3, rr3, w3, "D=3", "RIBBO 3BA"),
        (r10, rr10, w10, "D=10", "RIBBO 5BA-filter"),
    ]):
        bsf_rb, bsf_rr = bsf(rb), bsf(rr)
        wm, ws = w.mean(), w.std()
        ax.plot(evals, bsf_rb.mean(0), color="#16A34A", lw=2.5, label=f"{mlabel}")
        ax.plot(evals, bsf_rr.mean(0), color="#9CA3AF", lw=1.8, ls="--", label="Random")
        ax.axhline(wm, color="#000000", lw=1.8, label="WMMSE (mean)")
        ax.fill_between(evals, wm - ws, wm + ws, color="#000000", alpha=0.10, label="WMMSE ±1σ")
        ax.axvline(TRAIN_HORIZON, color="#DC2626", lw=1.2, ls=":", label="train horizon (300)")
        ax.set_title(f"IFC SE {dlabel} — evaluation budget extended to {N_STEPS}\n"
                     f"final RIBBO = {100*bsf_rb[:,-1].mean()/wm:.1f}% of WMMSE "
                     f"(@300: {100*bsf_rb[:,TRAIN_HORIZON-1].mean()/wm:.1f}%)", fontsize=10)
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Average Best SE (nats)")
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eval_ifc_budget.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    # summary table
    print("\n" + "=" * 64)
    print(f"{'Setting':<22}{'@300':>10}{'@1000':>10}{'gain':>10}")
    print("-" * 64)
    for rb, w, lab in [(r3, w3, "D=3 3BA"), (r10, w10, "D=10 5BA-filter")]:
        b = bsf(rb)
        p300 = 100 * b[:, TRAIN_HORIZON - 1].mean() / w.mean()
        p1000 = 100 * b[:, -1].mean() / w.mean()
        print(f"{lab:<22}{p300:>9.1f}%{p1000:>9.1f}%{p1000-p300:>9.1f}p")
    print("=" * 64)


if __name__ == "__main__":
    main()
