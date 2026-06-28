"""
Add Differential Evolution (DE) as a black-box baseline, matched to RIBBO/Random:
  - same unseen channels (dataset_id = 30 + i), same per-channel optimum,
  - same budget of 300 function evaluations, best-so-far tracked,
  - DE = scipy.optimize.differential_evolution (standard rand/1/bin), polish=False
    so it stays a pure model-free black-box optimizer (no gradient / L-BFGS polish).

RIBBO, Random and optimum are reused from the npz produced by the per-case eval
scripts (identical channels), so only DE is computed here -> exact apples-to-apples.

Metric (matches the deck): gap to optimum (%) = 100 * (opt - achieved) / opt, lower better.

Usage:
  PY=/home/myoungjin/miniconda3/envs/RIBBO/bin/python
  PYTHONPATH=. $PY scripts/eval_ifc_de.py --cases ee_d3 ee_d10 --n 200
"""
import os
import argparse
import numpy as np
from scipy.optimize import differential_evolution

from problems.ifc import IFCNumpy

N_STEPS = 300
OUT_DIR = "./results/18_de_baseline"

# case -> (npz path, RIBBO key, Random key, optimum key, ssid, dim, label)
CASES = {
    "ee_d3":  ("results/09_ifc_ee/eval_ifc_ee.npz", "rb3", "rd3", "o3",
               "IFCEE", 3, "EE D=3"),
    "ee_d10": ("results/09_ifc_ee/eval_ifc_ee.npz", "rb10", "rd10", "o10",
               "IFCEED10", 10, "EE D=10"),
    "ee_d20": ("results/16_ee_d20/eval_ee_d20.npz", "rb", "rd", "opt",
               "IFCEED20", 20, "EE D=20"),
    "se_d10_3ba": ("results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz",
                   "raw_3ba", "raw_rand", "wmmse", "IFCSED10", 10, "SE D=10 (3BA)"),
    "se_d10_5ba": ("results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz",
                   "raw_5ba", "raw_rand", "wmmse", "IFCSED10", 10, "SE D=10 (5BA-filter)"),
    "se_d20_3ba": ("results/11_ifc_d20_scalability/eval_ifc_d20.npz",
                   "rb", "rd", "wm", "IFCSED20", 20, "SE D=20 (3BA)"),
    "se_d20_4ba": ("results/11_ifc_d20_scalability/eval_ifc_d20_4ba.npz",
                   "rb", "rd", "ref", "IFCSED20", 20, "SE D=20 (4BA-filter)"),
}

# small populations suited to a tight 300-eval budget (a practitioner with 300
# evals would not use scipy's default popsize=15*D). Target total population ~24.
NP_TARGET = 24


def rollout_de(func, x_dim, budget, seed):
    """Standard scipy DE, capped at `budget` function evaluations, returns
    best-so-far reward over evaluations (length = budget)."""
    bounds = [(0.0, 1.0)] * x_dim
    best = -np.inf
    bsf = []

    class _Stop(Exception):
        pass

    def obj(x):
        nonlocal best
        if len(bsf) >= budget:
            raise _Stop()
        y = float(func(x[None, :]).ravel()[0])
        best = y if y > best else best
        bsf.append(best)
        if len(bsf) >= budget:
            raise _Stop()
        return -y

    popmult = max(1, round(NP_TARGET / x_dim))
    try:
        differential_evolution(
            obj, bounds, maxiter=100000, popsize=popmult,
            mutation=0.5, recombination=0.9, init="latinhypercube",
            polish=False, tol=0.0, seed=seed, vectorized=False,
            workers=1, updating="immediate",
        )
    except _Stop:
        pass

    arr = np.asarray(bsf[:budget], dtype=float)
    if arr.size < budget:  # DE converged before budget (rare) -> hold last value
        pad = np.full(budget - arr.size, arr[-1] if arr.size else 0.0)
        arr = np.concatenate([arr, pad])
    return arr


def bsf(y):
    return np.maximum.accumulate(y, axis=1)


def gap(curve_bsf, opt):
    """curve_bsf: [n_ch, T] best-so-far; opt: [n_ch] -> mean gap%(T) over channels."""
    g = 100.0 * (opt[:, None] - curve_bsf) / opt[:, None]
    return g.mean(0)


def run_case(case_key, n):
    npz_path, kb, kd, ko, ssid, dim, label = CASES[case_key]
    d = np.load(npz_path, allow_pickle=True)
    rb_raw = d[kb][:n]                 # RIBBO raw per-step reward
    rd_raw = d[kd][:n]                 # Random raw per-step reward
    opt = d[ko][:n]                    # [n] per-channel optimum
    if rb_raw.ndim == 3:              # some npz store an episode axis [n,1,300]
        rb_raw = rb_raw[:, 0, :]
    if rd_raw.ndim == 3:
        rd_raw = rd_raw[:, 0, :]
    n = rb_raw.shape[0]
    print(f"\n=== {label}  (D={dim}, n={n} channels, ssid={ssid}) ===")

    de_bsf = np.zeros((n, N_STEPS))
    for i in range(n):
        cid = str(30 + i)
        func = IFCNumpy(ssid, cid, dim=dim)
        de_bsf[i] = rollout_de(func, dim, N_STEPS, seed=int(cid) + 2000)
        if (i + 1) % 25 == 0:
            print(f"  DE [{i+1}/{n}]")

    rb_bsf, rd_bsf = bsf(rb_raw), bsf(rd_raw)
    res = {
        "label": label, "dim": dim, "n": n, "opt": opt,
        "rb_bsf": rb_bsf, "rd_bsf": rd_bsf, "de_bsf": de_bsf,
        "gap_rb": gap(rb_bsf, opt), "gap_rd": gap(rd_bsf, opt),
        "gap_de": gap(de_bsf, opt),
    }
    # final-step gap + per-channel win-rate (RIBBO smaller gap than DE)
    final = {
        "gap_rb": res["gap_rb"][-1], "gap_rd": res["gap_rd"][-1],
        "gap_de": res["gap_de"][-1],
    }
    g_rb_ch = 100.0 * (opt - rb_bsf[:, -1]) / opt
    g_de_ch = 100.0 * (opt - de_bsf[:, -1]) / opt
    final["win_vs_de"] = 100.0 * np.mean(g_rb_ch < g_de_ch)
    res["final"] = final
    print(f"  final gap%%: RIBBO {final['gap_rb']:.1f} | DE {final['gap_de']:.1f} | "
          f"Random {final['gap_rd']:.1f}  | RIBBO<DE win-rate {final['win_vs_de']:.0f}%")
    return res


def plot(results, out_pdf):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    evals = np.arange(1, N_STEPS + 1)
    k = len(results)
    fig, axes = plt.subplots(1, k, figsize=(6.2 * k, 4.8), squeeze=False)
    axes = axes[0]
    for ax, r in zip(axes, results):
        ax.plot(evals, r["gap_rb"], color="#7C3AED", lw=2.6,
                label=f"RIBBO (gap {r['final']['gap_rb']:.1f}%)")
        ax.plot(evals, r["gap_de"], color="#D97706", lw=2.2,
                label=f"DE (gap {r['final']['gap_de']:.1f}%)")
        ax.plot(evals, r["gap_rd"], color="#9CA3AF", lw=1.8, ls="--",
                label=f"Random (gap {r['final']['gap_rd']:.1f}%)")
        ax.set_title(f"IFC {r['label']}  ({r['n']} unseen channels)", fontsize=11)
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("optimality gap (%)")
        ax.legend(fontsize=9, loc="upper right")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.set_xlim(1, N_STEPS)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["ee_d10"], choices=list(CASES))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    results = [run_case(c, args.n) for c in args.cases]

    tag = ("_" + args.tag) if args.tag else ""
    save = {}
    for c, r in zip(args.cases, results):
        for key in ("gap_rb", "gap_de", "gap_rd", "de_bsf", "opt"):
            save[f"{c}__{key}"] = r[key]
        save[f"{c}__final"] = np.array([r["final"]["gap_rb"], r["final"]["gap_de"],
                                        r["final"]["gap_rd"], r["final"]["win_vs_de"]])
    np.savez(os.path.join(OUT_DIR, f"de_baseline{tag}.npz"), **save)
    plot(results, os.path.join(OUT_DIR, f"de_baseline{tag}.pdf"))

    print("\n" + "=" * 64)
    print(f"{'case':<10}{'RIBBO gap':>11}{'DE gap':>10}{'Random gap':>12}{'RIBBO<DE':>11}")
    print("-" * 64)
    for c, r in zip(args.cases, results):
        f = r["final"]
        print(f"{c:<10}{f['gap_rb']:>10.1f}%{f['gap_de']:>9.1f}%{f['gap_rd']:>11.1f}%{f['win_vs_de']:>10.0f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
