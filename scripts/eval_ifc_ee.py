"""
SE vs EE comparison. Evaluate RIBBO trained on the EE reward (Lee 2025 eq.32)
at D=3 and D=10 vs Random, against the (near-)global EE optimum.

Optimum reference:
  D=3  : exhaustive grid (ifc_optimum_bruteforce, grid=51) -> exact.
  D=10 : multi-start projected gradient (ifc_optimum_multistart) -> near-global,
         validated against brute force at D=3 in scripts/sanity (diff ~1e-4).

Auto-discovers the latest checkpoint under each run dir. Override with --d3_run / --d10_run.

Outputs:
  results/09_ifc_ee/eval_ifc_ee.pdf
  results/09_ifc_ee/eval_ifc_ee.npz
"""
import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import (IFCNumpy, IFCMetaProblem, ifc_optimum_bruteforce,
                          ifc_optimum_multistart)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
INPUT_SEQ_LEN = 50
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8
N_STEPS = 300
OUT_DIR = "./results/09_ifc_ee"


def latest_ckpt(run_dir):
    cks = glob.glob(os.path.join(run_dir, "*", "ckpt", "*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no ckpt under {run_dir}")
    return max(cks, key=lambda p: int(os.path.basename(p).split(".")[0]))


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


def rollout_ribbo(designer, func, x_dim, ymax, ymin, n_steps, init_regret=0):
    B = 1
    designer.past_x = torch.zeros([B, n_steps + 1, x_dim], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    ts = torch.arange(n_steps + 1).clamp(max=299).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps + 1).repeat(B, 1)
    designer.step_count = 0
    raw = np.zeros(n_steps)
    last_x = last_y = last_regret = None
    scale = max(ymax - ymin, 1e-6)
    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(last_x=last_x, last_y=last_y,
                                      last_onestep_regret=last_regret,
                                      deterministic=False, regret_strategy="none")
        x_real = last_x.cpu().numpy() / 2 + 0.5
        y = float(func(x_real)[0, 0])
        last_y = torch.tensor([[(y - ymin) / scale]], dtype=torch.float).to(DEVICE)
        last_regret = torch.tensor([[(ymax - y) / scale]], dtype=torch.float).to(DEVICE)
        raw[i] = y
    return raw


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


def opt_value(func, x_dim):
    if x_dim <= 3:
        v, _ = ifc_optimum_bruteforce(func.H_sq, Ptx=func.Ptx, Pfix=func.Pfix,
                                      reward_type="EE", grid=51)
    else:
        v, _ = ifc_optimum_multistart(func.H_sq, Ptx=func.Ptx, Pfix=func.Pfix,
                                      reward_type="EE", n_restart=40)
    return v


def run_panel(ckpt, x_dim, ssid, data_dir, cache_dir, n_channels, label):
    print(f"\n=== {label} (D={x_dim}) ckpt={ckpt} ===")
    ymax, ymin = stats(data_dir, cache_dir, ssid)
    print(f"  EE norm stats: y_max={ymax:.4f} y_min={ymin:.4f}")
    d = build_designer(ckpt, x_dim)
    raw_rb = np.zeros((n_channels, N_STEPS))
    raw_rd = np.zeros((n_channels, N_STEPS))
    opt = np.zeros(n_channels)
    for i in range(n_channels):
        cid = str(30 + i)
        func = IFCNumpy(ssid, cid, dim=x_dim)
        raw_rb[i] = rollout_ribbo(d, func, x_dim, ymax, ymin, N_STEPS)
        raw_rd[i] = rollout_random(func, x_dim, N_STEPS, seed=int(cid) + 1000)
        opt[i] = opt_value(func, x_dim)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_channels}]")
    return raw_rb, raw_rd, opt


def bsf(y):
    return np.maximum.accumulate(y, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3_run", default="./log/ifc/dt-ifc-ee-d3-3ba")
    ap.add_argument("--d10_run", default="./log/ifc/dt-ifc-ee-d10-3ba")
    ap.add_argument("--n_d3", type=int, default=200)
    ap.add_argument("--n_d10", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    c3 = latest_ckpt(args.d3_run)
    c10 = latest_ckpt(args.d10_run)
    rb3, rd3, o3 = run_panel(c3, 3, "IFCEE", "./data/generated_data/ifc_ee",
                             "./cache/ifc_ee/", args.n_d3, "RIBBO-EE 3BA")
    rb10, rd10, o10 = run_panel(c10, 10, "IFCEED10", "./data/generated_data/ifc_d10_ee",
                                "./cache/ifc_d10_ee/", args.n_d10, "RIBBO-EE 3BA")

    np.savez(os.path.join(OUT_DIR, "eval_ifc_ee.npz"),
             rb3=rb3, rd3=rd3, o3=o3, rb10=rb10, rd10=rd10, o10=o10,
             ckpt_d3=c3, ckpt_d10=c10)

    evals = np.arange(1, N_STEPS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary = []
    for ax, (rb, rd, o, dl) in zip(axes, [
        (rb3, rd3, o3, "D=3"), (rb10, rd10, o10, "D=10")]):
        b_rb, b_rd = bsf(rb), bsf(rd)
        om, osd = o.mean(), o.std()
        p_rb = 100 * b_rb[:, -1].mean() / om
        p_rd = 100 * b_rd[:, -1].mean() / om
        summary.append((dl, b_rb[:, -1].mean(), b_rd[:, -1].mean(), om, p_rb, p_rd))
        ax.plot(evals, b_rb.mean(0), color="#7C3AED", lw=2.5, label=f"RIBBO-EE ({p_rb:.0f}% of opt)")
        ax.plot(evals, b_rd.mean(0), color="#9CA3AF", lw=1.8, ls="--", label=f"Random ({p_rd:.0f}% of opt)")
        ax.axhline(om, color="#000000", lw=1.8, label="Global EE optimum (mean)")
        ax.fill_between(evals, om - osd, om + osd, color="#000000", alpha=0.10, label="optimum ±1σ")
        ax.set_title(f"IFC EE {dl} — RIBBO vs Random vs optimum\n"
                     f"{rb.shape[0]} unseen channels × {N_STEPS} steps", fontsize=10)
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Average Best EE")
        ax.legend(fontsize=8, loc="lower right")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.set_xlim(1, N_STEPS)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eval_ifc_ee.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    print("\n" + "=" * 70)
    print(f"{'Setting':<10}{'RIBBO-EE':>12}{'Random':>12}{'Optimum':>12}{'RIBBO%':>10}{'Rand%':>10}")
    print("-" * 70)
    for dl, rb, rd, om, prb, prd in summary:
        print(f"{dl:<10}{rb:>12.4f}{rd:>12.4f}{om:>12.4f}{prb:>9.1f}%{prd:>9.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
