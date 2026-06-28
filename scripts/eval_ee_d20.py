"""
(E-eval) Evaluate the trained EE D=20 RIBBO model vs Random vs near-global optimum
(multistart gradient). Saves full best-so-far curves + a curve figure consistent with the
deck style. Run after training completes.

  python scripts/eval_ee_d20.py --n 200
Outputs: results/16_ee_d20/{eval_ee_d20.npz, eval_ee_d20.pdf, README.md}
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scripts.eval_ifc_ee import build_designer, rollout_ribbo, rollout_random, stats, latest_ckpt
from problems.ifc import IFCNumpy, ifc_optimum_multistart

OUT = "results/16_ee_d20"
RUN = "./log/ifc/dt-ifc-ee-d20-3ba"
SSID = "IFCEED20"
DATA = "./data/generated_data/ifc_d20_ee"
CACHE = "./cache/ifc_d20_ee/"
XDIM = 20
NSTEP = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_restart", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ckpt = latest_ckpt(RUN)
    ymax, ymin = stats(DATA, CACHE, SSID)
    d = build_designer(ckpt, XDIM)
    print(f"[EE D=20] ckpt={ckpt} ymax={ymax:.4f} ymin={ymin:.4f} n={args.n}")

    rb = np.zeros((args.n, NSTEP))
    rd = np.zeros((args.n, NSTEP))
    opt = np.zeros(args.n)
    for i in range(args.n):
        cid = str(30 + i)
        func = IFCNumpy(SSID, cid, dim=XDIM)
        rb[i] = rollout_ribbo(d, func, XDIM, ymax, ymin, NSTEP)
        rd[i] = rollout_random(func, XDIM, NSTEP, seed=int(cid) + 1000)
        opt[i], _ = ifc_optimum_multistart(func.H_sq, func.Ptx, func.Pfix, "EE",
                                           n_restart=args.n_restart)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{args.n}]")

    np.savez(os.path.join(OUT, "eval_ee_d20.npz"), rb=rb, rd=rd, opt=opt, ckpt=ckpt)
    b_rb = np.maximum.accumulate(rb, 1).mean(0)
    b_rd = np.maximum.accumulate(rd, 1).mean(0)
    om = opt.mean()
    p_rb, p_rd = 100 * b_rb[-1] / om, 100 * b_rd[-1] / om
    evals = np.arange(1, NSTEP + 1)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    plt.rcParams.update({"font.size": 15})
    ax.axhline(om, color="#111111", lw=2.0, label="optimum (mean)")
    ax.fill_between(evals, om - opt.std(), om + opt.std(), color="#111111", alpha=0.07)
    ax.fill_between(evals, b_rd, b_rb, color="#DC2626", alpha=0.09)
    ax.plot(evals, b_rb, color="#7C3AED", lw=3.0, label=f"RIBBO ({p_rb:.0f}% of opt)")
    ax.plot(evals, b_rd, color="#9CA3AF", lw=2.2, ls="--", label=f"Random ({p_rd:.0f}%)")
    ax.set_title(f"IFC EE D=20  ({args.n} unseen channels x {NSTEP} steps)")
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("Average best EE")
    ax.set_xlim(1, NSTEP); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eval_ee_d20.pdf"), dpi=150, bbox_inches="tight")

    win = 100 * np.mean(np.maximum.accumulate(rb, 1)[:, -1] > np.maximum.accumulate(rd, 1)[:, -1])
    line = (f"EE D=20 (3BA), n={args.n}: RIBBO {p_rb:.1f}% of opt, Random {p_rd:.1f}%, "
            f"RIBBO-Random +{p_rb-p_rd:.1f} pts, win-rate {win:.1f}%")
    print("\n" + line)
    with open(os.path.join(OUT, "README.md"), "w") as fp:
        fp.write("# (E) EE D=20 evaluation\n\n" + line + "\n")


if __name__ == "__main__":
    main()
