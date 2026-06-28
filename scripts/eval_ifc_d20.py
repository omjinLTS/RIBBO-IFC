"""
Scalability point D=20: RIBBO (3BA) vs Random vs WMMSE on IFC SE D=20.
Completes the dimension sweep D=3 -> D=10 -> D=20 (all 3BA, 300-step horizon).

Auto-discovers latest checkpoint under --run.

Outputs:
  results/11_ifc_d20_scalability/eval_ifc_d20.pdf / .npz
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
from problems.ifc import IFCNumpy, IFCMetaProblem, wmmse_sum_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
X_DIM = 20
N_STEPS = 300
INPUT_SEQ_LEN = 50
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8
OUT_DIR = "./results/11_ifc_d20_scalability"


def latest_ckpt(run_dir):
    cks = glob.glob(os.path.join(run_dir, "*", "ckpt", "*.ckpt"))
    return max(cks, key=lambda p: int(os.path.basename(p).split(".")[0]))


def build(ckpt):
    tr = DecisionTransformer(x_dim=X_DIM, y_dim=1, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
                             seq_len=300, num_heads=NUM_HEADS, add_bos=True, mix_method="concat",
                             attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1,
                             pos_encoding="embed")
    d = DecisionTransformerDesigner(transformer=tr, x_dim=X_DIM, y_dim=1, embed_dim=EMBED_DIM,
                                    seq_len=300, input_seq_len=INPUT_SEQ_LEN, x_type="stochastic",
                                    y_loss_coeff=0.0, use_abs_timestep=True, device=DEVICE)
    d.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    d.eval()
    return d


def rollout(designer, func, ymax, ymin, n_steps, init_regret=0):
    B = 1
    designer.past_x = torch.zeros([B, n_steps + 1, X_DIM], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    ts = torch.arange(n_steps + 1).clamp(max=299).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps + 1).repeat(B, 1)
    designer.step_count = 0
    raw = np.zeros(n_steps)
    lx = ly = lr = None
    scale = max(ymax - ymin, 1e-6)
    for i in range(n_steps):
        with torch.no_grad():
            lx = designer.suggest(last_x=lx, last_y=ly, last_onestep_regret=lr,
                                  deterministic=False, regret_strategy="none")
        x_real = lx.cpu().numpy() / 2 + 0.5
        y = float(func(x_real)[0, 0])
        ly = torch.tensor([[(y - ymin) / scale]], dtype=torch.float).to(DEVICE)
        lr = torch.tensor([[(ymax - y) / scale]], dtype=torch.float).to(DEVICE)
        raw[i] = y
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="./log/ifc/dt-ifc-d20-3ba")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    ck = latest_ckpt(args.run)
    print(f"ckpt: {ck}")
    p = IFCMetaProblem(search_space_id="IFCSED20", root_dir=None,
                       data_dir="./data/generated_data/ifc_d20", cache_dir="./cache/ifc_d20/",
                       input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=300,
                       normalize_method="random", n_block=100)
    info = p.dataset.sp_id2info["IFCSED20"]
    ymax, ymin = info["y_max_mean"], info["y_min_mean"]
    print(f"D=20 stats: y_max={ymax:.3f} y_min={ymin:.3f}")
    d = build(ck)
    rb = np.zeros((args.n, N_STEPS))
    rd = np.zeros((args.n, N_STEPS))
    wm = np.zeros(args.n)
    for i in range(args.n):
        cid = str(30 + i)
        func = IFCNumpy("IFCSED20", cid, dim=X_DIM)
        rb[i] = rollout(d, func, ymax, ymin, N_STEPS)
        rng = np.random.default_rng(int(cid) + 1000)
        rd[i] = func(rng.uniform(0, 1, (N_STEPS, X_DIM))).ravel()
        wm[i], _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{args.n}]")
    np.savez(os.path.join(OUT_DIR, "eval_ifc_d20.npz"), rb=rb, rd=rd, wm=wm, ckpt=ck)

    bsf_rb = np.maximum.accumulate(rb, axis=1)
    bsf_rd = np.maximum.accumulate(rd, axis=1)
    wmean, wstd = wm.mean(), wm.std()
    evals = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    prb = 100 * bsf_rb[:, -1].mean() / wmean
    prd = 100 * bsf_rd[:, -1].mean() / wmean
    ax.plot(evals, bsf_rb.mean(0), color="#2563EB", lw=2.5, label=f"RIBBO 3BA ({prb:.0f}% of WMMSE)")
    ax.plot(evals, bsf_rd.mean(0), color="#9CA3AF", lw=1.8, ls="--", label=f"Random ({prd:.0f}% of WMMSE)")
    ax.axhline(wmean, color="#000000", lw=1.8, label="WMMSE (mean)")
    ax.fill_between(evals, wmean - wstd, wmean + wstd, color="#000000", alpha=0.10, label="WMMSE ±1σ")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Average Best SE (nats)")
    ax.set_title(f"IFC SE D=20 scalability — RIBBO vs Random vs WMMSE\n{args.n} unseen channels × {N_STEPS} steps", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(1, N_STEPS)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eval_ifc_d20.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
    print("=" * 56)
    print(f"{'Method':<24}{'Final(nats)':>14}{'%WMMSE':>10}")
    print(f"{'RIBBO 3BA':<24}{bsf_rb[:,-1].mean():>14.3f}{prb:>9.1f}%")
    print(f"{'Random':<24}{bsf_rd[:,-1].mean():>14.3f}{prd:>9.1f}%")
    print(f"{'WMMSE':<24}{wmean:>14.3f}{100:>9.1f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
