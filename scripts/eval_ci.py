"""
(C) Tighten error bars: re-evaluate a trained model on many unseen channels and report
mean +/- 95% CI of (% of optimum) for RIBBO and Random, plus win-rate. Reuses the verified
rollout/optimum code from scripts.eval_ifc_ee and problems.ifc.

Example:
  python scripts/eval_ci.py --run_dir ./log/ifc/dt-ifc-ee-d10-3ba --ssid IFCEED10 \
      --data_dir ./data/generated_data/ifc_d10_ee --cache_dir ./cache/ifc_d10_ee/ \
      --x_dim 10 --reward EE --n 500 --tag ee_d10
Outputs append to results/14_error_bars/{tag}.npz and results/14_error_bars/README.md
"""
import os
import argparse
import numpy as np

from scripts.eval_ifc_ee import build_designer, rollout_ribbo, rollout_random, stats, latest_ckpt
from problems.ifc import IFCNumpy, wmmse_sum_rate, ifc_optimum_bruteforce, ifc_optimum_multistart

OUT = "results/14_error_bars"
os.makedirs(OUT, exist_ok=True)


def optimum(func, x_dim, reward):
    if reward == "SE":
        v, _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx)
        return v
    if x_dim <= 3:
        v, _ = ifc_optimum_bruteforce(func.H_sq, func.Ptx, func.Pfix, "EE", grid=51)
    else:
        v, _ = ifc_optimum_multistart(func.H_sq, func.Ptx, func.Pfix, "EE", n_restart=40)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ssid", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--x_dim", type=int, required=True)
    ap.add_argument("--reward", choices=["SE", "EE"], required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ckpt = latest_ckpt(args.run_dir)
    ymax, ymin = stats(args.data_dir, args.cache_dir, args.ssid)
    d = build_designer(ckpt, args.x_dim)
    print(f"[{args.tag}] ckpt={ckpt} ymax={ymax:.4f} ymin={ymin:.4f} n={args.n}")

    pr, pd = np.zeros(args.n), np.zeros(args.n)
    fr, fd = np.zeros(args.n), np.zeros(args.n)
    for i in range(args.n):
        cid = str(30 + i)
        func = IFCNumpy(args.ssid, cid, dim=args.x_dim)
        rb = rollout_ribbo(d, func, args.x_dim, ymax, ymin, args.n_steps)
        rd = rollout_random(func, args.x_dim, args.n_steps, seed=int(cid) + 1000)
        o = optimum(func, args.x_dim, args.reward)
        fr[i], fd[i] = np.maximum.accumulate(rb)[-1], np.maximum.accumulate(rd)[-1]
        pr[i], pd[i] = 100 * fr[i] / o, 100 * fd[i] / o
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.n}] RIBBO {pr[:i+1].mean():.1f}%  Random {pd[:i+1].mean():.1f}%")

    def ci(a):
        return 1.96 * a.std(ddof=1) / np.sqrt(len(a))

    win = 100 * np.mean(fr > fd)
    np.savez(os.path.join(OUT, f"{args.tag}.npz"), pr=pr, pd=pd, fr=fr, fd=fd, ckpt=ckpt)
    line = (f"{args.tag:<14} n={args.n}: RIBBO {pr.mean():.1f}% +/-{ci(pr):.1f}  "
            f"Random {pd.mean():.1f}% +/-{ci(pd):.1f}  win-rate {win:.1f}%")
    print("\n" + line)
    with open(os.path.join(OUT, "README.md"), "a") as fp:
        fp.write(line + "\n")


if __name__ == "__main__":
    main()
