"""
Extended-budget comparison for the HARD (D=10, D=20) cases.
Senior feedback: high-dim problems are harder -> look at a longer horizon
(more function evaluations) than 300.

Budget = 1000 function evaluations. RIBBO@1000 and Random@1000 are reused from the
existing budget-extension npz (same unseen channels, cid 30+); DE@1000 and CMA-ES@1000
are computed here on the identical channels. BO is excluded: a 1000-point GP is
intractable per channel AND BO already plateaus by ~300 evals (see results/18).

Metric: optimality gap (%) = 100*(opt-achieved)/opt, lower better.
Outputs: results/19_budget_baselines/{budget_baselines.pdf, .npz, README via separate}
"""
import os
import argparse
import numpy as np

from problems.ifc import IFCNumpy
from scripts.eval_ifc_baselines import rollout_de, rollout_cmaes  # reuse identical rollouts

BUDGET = 1000
OUT_DIR = "./results/19_budget_baselines"
CACHE_DIR = os.path.join(OUT_DIR, "cache")

CASES = {
    "ee_d10": dict(npz="results/09_ifc_ee/eval_ifc_ee_budget.npz", rb="rb10", opt="o10",
                   rand=None, ssid="IFCEED10", dim=10, n=100, label="EE D=10"),
    "ee_d20": dict(npz="results/16_ee_d20/eval_ee_d20_budget.npz", rb="rb", opt="opt",
                   rand="rd", ssid="IFCEED20", dim=20, n=150, label="EE D=20"),
    "se_d10_5ba": dict(npz="results/08_eval_budget_extension/eval_ifc_budget.npz", rb="r10",
                       opt="w10", rand="rr10", ssid="IFCSED10", dim=10, n=100,
                       label="SE D=10 (5BA-filter)"),
}

STYLE = {"ribbo": ("#7C3AED", "RIBBO"), "cmaes": ("#059669", "CMA-ES"),
         "de": ("#D97706", "DE"), "rand": ("#9CA3AF", "Random")}


def rollout_random(func, dim, budget, seed):
    rng = np.random.default_rng(int(seed))
    return func(rng.uniform(0, 1, (budget, dim))).ravel()


def bsf(y):
    return np.maximum.accumulate(y, axis=1)


def compute(case, method, n, ssid, dim, force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{case}_{method}_n{n}_b{BUDGET}.npy")
    if os.path.exists(path) and not force:
        return np.load(path)
    fn = {"de": rollout_de, "cmaes": rollout_cmaes, "rand": rollout_random}[method]
    out = np.zeros((n, BUDGET))
    import time
    t0 = time.time()
    for i in range(n):
        cid = str(30 + i)
        out[i] = fn(IFCNumpy(ssid, cid, dim=dim), dim, BUDGET, seed=int(cid) + 2000)
        if (i + 1) % max(1, n // 5) == 0:
            print(f"    {method} [{i+1}/{n}] ({time.time()-t0:.0f}s)")
    np.save(path, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--methods", nargs="+", default=["de", "cmaes"], choices=["de", "cmaes"])
    ap.add_argument("--metric", choices=["gap", "raw"], default="gap")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for case in args.cases:
        c = CASES[case]
        d = np.load(c["npz"], allow_pickle=True)
        n = c["n"]
        opt = d[c["opt"]][:n]
        rb = bsf(d[c["rb"]][:n])
        rd = bsf(d[c["rand"]][:n]) if c["rand"] else bsf(
            compute(case, "rand", n, c["ssid"], c["dim"], args.force))
        curves = {"ribbo": rb, "rand": rd}
        for m in args.methods:
            curves[m] = bsf(compute(case, m, n, c["ssid"], c["dim"], args.force))
        gap = {k: (100 * (opt[:, None] - v) / opt[:, None]).mean(0) for k, v in curves.items()}
        raw = {k: v.mean(0) for k, v in curves.items()}
        results.append(dict(case=case, label=c["label"], n=n, opt=float(opt.mean()),
                            optstd=float(opt.std()), gap=gap, raw=raw))
        print(f"\n=== {c['label']} (n={n}) — gap%% @300 -> @1000 ===")
        for k in ["ribbo"] + args.methods + ["rand"]:
            print(f"  {STYLE[k][1]:8s}: {gap[k][299]:5.1f} -> {gap[k][-1]:5.1f}")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    ev = np.arange(1, BUDGET + 1)
    isr = args.metric == "raw"
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.8), squeeze=False)
    for ax, r in zip(axes[0], results):
        if isr:
            ax.axhline(r["opt"], color="#111", lw=1.4, ls=":", label=f"optimum ({r['opt']:.2f})")
        for k in ["ribbo"] + args.methods + ["rand"]:
            col, lbl = STYLE[k]
            ls = "--" if k == "rand" else "-"
            lw = 2.6 if k == "ribbo" else 2.0
            y = r["raw"][k] if isr else r["gap"][k]
            val = r["raw"][k][-1] if isr else r["gap"][k][-1]
            unit = f"{val:.2f}" if isr else f"{val:.1f}%"
            ax.plot(ev, y, color=col, lw=lw, ls=ls, label=f"{lbl} ({unit})")
        ax.axvline(300, color="#888", lw=1.0, ls=":", alpha=0.7)
        ax.text(300, ax.get_ylim()[1], " prev budget (300)", fontsize=7, color="#666",
                va="top", ha="left")
        ax.set_title(f"IFC {r['label']} ({r['n']} ch) — budget 1000", fontsize=11)
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("best-so-far reward" if isr else "optimality gap (%)")
        ax.legend(fontsize=8.5, loc=("lower right" if isr else "upper right"))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.set_xlim(1, BUDGET)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"budget_baselines_{args.metric}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
