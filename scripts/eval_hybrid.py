"""
(D) Hybrid RIBBO + local refinement. Roll out RIBBO, take its best point per channel, then
run a cheap local polish (L-BFGS-B on the true reward, bounds [0,1]) from that init. Measure
how much of the remaining gap to the optimum the polish closes. Random+polish is the control
(does the local optimizer alone explain any gain, or does RIBBO's init matter?).

Example:
  python scripts/eval_hybrid.py --run_dir ./log/ifc/dt-ifc-ee-d10-3ba --ssid IFCEED10 \
      --data_dir ./data/generated_data/ifc_d10_ee --cache_dir ./cache/ifc_d10_ee/ \
      --x_dim 10 --reward EE --n 150 --tag ee_d10
Outputs: results/15_hybrid/{tag}.npz, results/15_hybrid/README.md
"""
import os
import argparse
import numpy as np
import torch
from scipy.optimize import minimize

from scripts.eval_ifc_ee import build_designer, rollout_random, stats, latest_ckpt, DEVICE
from problems.ifc import (IFCNumpy, ifc_rates, wmmse_sum_rate,
                          ifc_optimum_bruteforce, ifc_optimum_multistart)

OUT = "results/15_hybrid"
os.makedirs(OUT, exist_ok=True)


def rollout_ribbo_x(designer, func, x_dim, ymax, ymin, n_steps):
    """Mirror of eval_ifc_ee.rollout_ribbo but also return the queried X (n_steps, x_dim)."""
    B = 1
    designer.past_x = torch.zeros([B, n_steps + 1, x_dim], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = 0
    ts = torch.arange(n_steps + 1).clamp(max=299).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps + 1).repeat(B, 1)
    designer.step_count = 0
    raw = np.zeros(n_steps)
    X = np.zeros((n_steps, x_dim))
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
        X[i] = x_real[0]
    return raw, X


def optimum(func, x_dim, reward):
    if reward == "SE":
        v, _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx)
        return v
    if x_dim <= 3:
        v, _ = ifc_optimum_bruteforce(func.H_sq, func.Ptx, func.Pfix, "EE", grid=51)
    else:
        v, _ = ifc_optimum_multistart(func.H_sq, func.Ptx, func.Pfix, "EE", n_restart=40)
    return v


def polish(func, x0, reward):
    def neg(x):
        return -float(ifc_rates(func.H_sq, x[None, :], func.Ptx, func.Pfix, reward)[0, 0])
    res = minimize(neg, np.clip(x0, 0, 1), method="L-BFGS-B",
                   bounds=[(0.0, 1.0)] * len(x0), options={"maxiter": 200, "ftol": 1e-10})
    return -res.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ssid", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--x_dim", type=int, required=True)
    ap.add_argument("--reward", choices=["SE", "EE"], required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ckpt = latest_ckpt(args.run_dir)
    ymax, ymin = stats(args.data_dir, args.cache_dir, args.ssid)
    d = build_designer(ckpt, args.x_dim)
    print(f"[{args.tag}] ckpt={ckpt} n={args.n}")

    cols = {k: np.zeros(args.n) for k in
            ["ribbo", "ribbo_polish", "random", "random_polish"]}
    for i in range(args.n):
        cid = str(30 + i)
        func = IFCNumpy(args.ssid, cid, dim=args.x_dim)
        o = optimum(func, args.x_dim, args.reward)
        rb, X = rollout_ribbo_x(d, func, args.x_dim, ymax, ymin, args.n_steps)
        j = int(np.argmax(rb))
        rd = rollout_random(func, args.x_dim, args.n_steps, seed=int(cid) + 1000)
        # random best x: regenerate same X as rollout_random
        rng = np.random.default_rng(int(cid) + 1000)
        Xr = rng.uniform(0, 1, (args.n_steps, args.x_dim))
        jr = int(np.argmax(rd))
        cols["ribbo"][i] = 100 * rb[j] / o
        cols["ribbo_polish"][i] = 100 * polish(func, X[j], args.reward) / o
        cols["random"][i] = 100 * rd[jr] / o
        cols["random_polish"][i] = 100 * polish(func, Xr[jr], args.reward) / o
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{args.n}] RIBBO {cols['ribbo'][:i+1].mean():.1f} -> "
                  f"+polish {cols['ribbo_polish'][:i+1].mean():.1f} | "
                  f"Random {cols['random'][:i+1].mean():.1f} -> +polish {cols['random_polish'][:i+1].mean():.1f}")

    np.savez(os.path.join(OUT, f"{args.tag}.npz"), ckpt=ckpt, **cols)
    line = (f"{args.tag:<14} n={args.n}: "
            f"RIBBO {cols['ribbo'].mean():.1f}% -> +polish {cols['ribbo_polish'].mean():.1f}%  |  "
            f"Random {cols['random'].mean():.1f}% -> +polish {cols['random_polish'].mean():.1f}%  (% of optimum)")
    print("\n" + line)
    with open(os.path.join(OUT, "README.md"), "a") as fp:
        fp.write(line + "\n")


if __name__ == "__main__":
    main()
