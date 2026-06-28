"""
Strong black-box baselines for RIBBO comparison, all matched to the same setup:
  - same unseen channels (dataset_id = 30 + i), same per-channel optimum,
  - same budget of 300 function evaluations, best-so-far tracked,
  - all model-free (query reward only; no channel model / gradients).

Methods:
  de    : scipy.optimize.differential_evolution (rand/1/bin, polish=False)
  cmaes : pycma CMA-ES (cma package), bounded [0,1]^D
  bo    : Bayesian optimization, GP + Expected Improvement (botorch/gpytorch)

RIBBO / Random / optimum are reused from each case's existing npz (identical
channels), so only the baselines are computed here -> exact apples-to-apples.
Each (case, method) best-so-far matrix is cached to .npy so reruns are cheap
(important for the slow BO).

Metric (matches the deck): gap to optimum (%) = 100*(opt-achieved)/opt, lower better.

Usage:
  PY=/home/myoungjin/miniconda3/envs/RIBBO/bin/python
  PYTHONPATH=. $PY scripts/eval_ifc_baselines.py --cases ee_d10 --methods de cmaes bo --n 50
"""
import os
import argparse
import time
import warnings
import numpy as np
from scipy.optimize import differential_evolution

from problems.ifc import IFCNumpy

warnings.filterwarnings("ignore")

N_STEPS = 300
OUT_DIR = "./results/18_de_baseline"
CACHE_DIR = os.path.join(OUT_DIR, "bsf_cache")

# case -> (npz path, RIBBO key, Random key, optimum key, ssid, dim, label)
CASES = {
    "ee_d3":  ("results/09_ifc_ee/eval_ifc_ee.npz", "rb3", "rd3", "o3", "IFCEE", 3, "EE D=3"),
    "ee_d10": ("results/09_ifc_ee/eval_ifc_ee.npz", "rb10", "rd10", "o10", "IFCEED10", 10, "EE D=10"),
    "ee_d20": ("results/16_ee_d20/eval_ee_d20.npz", "rb", "rd", "opt", "IFCEED20", 20, "EE D=20"),
    "se_d10_3ba": ("results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz",
                   "raw_3ba", "raw_rand", "wmmse", "IFCSED10", 10, "SE D=10 (3BA)"),
    "se_d10_5ba": ("results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz",
                   "raw_5ba", "raw_rand", "wmmse", "IFCSED10", 10, "SE D=10 (5BA-filter)"),
    "se_d20_3ba": ("results/11_ifc_d20_scalability/eval_ifc_d20.npz",
                   "rb", "rd", "wm", "IFCSED20", 20, "SE D=20 (3BA)"),
    "se_d20_4ba": ("results/11_ifc_d20_scalability/eval_ifc_d20_4ba.npz",
                   "rb", "rd", "ref", "IFCSED20", 20, "SE D=20 (4BA-filter)"),
}

METHOD_STYLE = {  # color, label
    "ribbo": ("#7C3AED", "RIBBO"),
    "bo":    ("#2563EB", "BO (GP-EI)"),
    "cmaes": ("#059669", "CMA-ES"),
    "de":    ("#D97706", "DE"),
    "rand":  ("#9CA3AF", "Random"),
}
NP_TARGET = 24  # DE population target for a tight 300-eval budget


# ----------------------------- baseline rollouts -----------------------------
def rollout_de(func, x_dim, budget, seed):
    bounds = [(0.0, 1.0)] * x_dim
    best, bsf = -np.inf, []

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
        differential_evolution(obj, bounds, maxiter=100000, popsize=popmult,
                               mutation=0.5, recombination=0.9, init="latinhypercube",
                               polish=False, tol=0.0, seed=seed, vectorized=False,
                               workers=1, updating="immediate")
    except _Stop:
        pass
    return _pad(bsf, budget)


def rollout_cmaes(func, x_dim, budget, seed):
    import cma
    best, bsf = -np.inf, []
    x0 = np.full(x_dim, 0.5)
    es = cma.CMAEvolutionStrategy(x0, 0.25, {
        "bounds": [0.0, 1.0], "seed": int(seed) + 1, "verbose": -9,
        "verb_log": 0, "verb_disp": 0, "maxfevals": budget * 4,
        "tolfun": 0, "tolx": 0, "tolfunhist": 0,
    })
    while len(bsf) < budget and not es.stop():
        X = es.ask()
        F = []
        full = True
        for x in X:
            y = float(func(np.asarray(x)[None, :]).ravel()[0])
            best = y if y > best else best
            bsf.append(best)
            F.append(-y)
            if len(bsf) >= budget:
                full = False
                break
        if full:
            es.tell(X, F)
        else:
            break
    return _pad(bsf, budget)


def rollout_bo(func, x_dim, budget, seed, n_init=None, restarts=3, raw_samples=32,
               refit_every=10):
    """GP + Expected Improvement BO. To stay tractable at a 300-eval budget, GP
    hyperparameters are re-fit every `refit_every` steps (the GP still conditions
    on ALL observed data every step; only the marginal-likelihood fit is amortized).
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(int(seed))
    dtype = torch.double
    rng = np.random.default_rng(int(seed) + 3000)
    if n_init is None:
        n_init = min(2 * x_dim, budget // 3)
    bounds = torch.stack([torch.zeros(x_dim, dtype=dtype),
                          torch.ones(x_dim, dtype=dtype)])

    Xnp = rng.uniform(0, 1, (n_init, x_dim))
    Ynp = func(Xnp).reshape(-1, 1)
    best, bsf = -np.inf, []
    for v in Ynp.ravel():
        best = v if v > best else best
        bsf.append(best)

    X = torch.tensor(Xnp, dtype=dtype)
    Y = torch.tensor(Ynp, dtype=dtype)
    state = None
    step = 0
    while len(bsf) < budget:
        model = SingleTaskGP(X, standardize(Y))
        if state is not None and step % refit_every != 0:
            model.load_state_dict(state)          # reuse hypers (amortized fit)
        else:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            state = model.state_dict()
        ei = ExpectedImprovement(model, best_f=standardize(Y).max())
        cand, _ = optimize_acqf(ei, bounds=bounds, q=1,
                                num_restarts=restarts, raw_samples=raw_samples)
        xnew = cand.detach().cpu().numpy().reshape(1, -1)
        ynew = float(func(xnew).ravel()[0])
        best = ynew if ynew > best else best
        bsf.append(best)
        X = torch.cat([X, cand.detach()], dim=0)
        Y = torch.cat([Y, torch.tensor([[ynew]], dtype=dtype)], dim=0)
        step += 1
    return _pad(bsf, budget)


ROLLOUT = {"de": rollout_de, "cmaes": rollout_cmaes, "bo": rollout_bo}


def _pad(bsf, budget):
    arr = np.asarray(bsf[:budget], dtype=float)
    if arr.size < budget:
        arr = np.concatenate([arr, np.full(budget - arr.size, arr[-1] if arr.size else 0.0)])
    return arr


# ----------------------------- driver -----------------------------
def load_case(case_key, n):
    npz_path, kb, kd, ko, ssid, dim, label = CASES[case_key]
    d = np.load(npz_path, allow_pickle=True)
    rb, rd, opt = d[kb][:n], d[kd][:n], d[ko][:n]
    if rb.ndim == 3:
        rb = rb[:, 0, :]
    if rd.ndim == 3:
        rd = rd[:, 0, :]
    return rb, rd, opt, ssid, dim, label


def compute_method(case_key, method, n, ssid, dim, force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{case_key}_{method}_n{n}.npy")
    if os.path.exists(path) and not force:
        return np.load(path)
    fn = ROLLOUT[method]
    out = np.zeros((n, N_STEPS))
    t0 = time.time()
    for i in range(n):
        cid = str(30 + i)
        func = IFCNumpy(ssid, cid, dim=dim)
        out[i] = fn(func, dim, N_STEPS, seed=int(cid) + 2000)
        if (i + 1) % max(1, n // 8) == 0 or method == "bo":
            dt = time.time() - t0
            print(f"    {method} [{i+1}/{n}] ({dt:.0f}s, {dt/(i+1):.1f}s/ch)")
    np.save(path, out)
    return out


def bsf(y):
    return np.maximum.accumulate(y, axis=1)


def gap_curve(curve_bsf, opt):
    return (100.0 * (opt[:, None] - curve_bsf) / opt[:, None]).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["ee_d10"], choices=list(CASES))
    ap.add_argument("--methods", nargs="+", default=["de", "cmaes", "bo"],
                    choices=["de", "cmaes", "bo"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--metric", choices=["gap", "raw"], default="gap",
                    help="gap=optimality gap %%; raw=actual best-so-far reward (+ optimum line)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for case in args.cases:
        rb, rd, opt, ssid, dim, label = load_case(case, args.n)
        n = rb.shape[0]
        print(f"\n=== {label} (D={dim}, n={n}) ===")
        curves = {"ribbo": bsf(rb), "rand": bsf(rd)}
        for m in args.methods:
            curves[m] = compute_method(case, m, n, ssid, dim, force=args.force)
        gaps = {k: gap_curve(v, opt) for k, v in curves.items()}
        finals = {k: gaps[k][-1] for k in gaps}
        # win-rate: RIBBO smaller final gap than each baseline
        g_rb = 100.0 * (opt - curves["ribbo"][:, -1]) / opt
        wins = {}
        for m in args.methods:
            g_m = 100.0 * (opt - curves[m][:, -1]) / opt
            wins[m] = 100.0 * np.mean(g_rb < g_m)
        curves_mean = {k: v.mean(0) for k, v in curves.items()}      # actual best-so-far reward
        finals_raw = {k: float(curves[k][:, -1].mean()) for k in curves}
        results.append(dict(case=case, label=label, n=n, gaps=gaps,
                            finals=finals, wins=wins, methods=args.methods,
                            curves_mean=curves_mean, finals_raw=finals_raw,
                            opt_mean=float(opt.mean()), opt_std=float(opt.std())))
        msg = " | ".join([f"{k} {finals[k]:.1f}" for k in
                          (["ribbo"] + args.methods + ["rand"])])
        print(f"  final gap%: {msg}")
        print("  RIBBO win-rate vs " + ", ".join([f"{m} {wins[m]:.0f}%" for m in args.methods]))

    # ---- plot ----
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    evals = np.arange(1, N_STEPS + 1)
    k = len(results)
    raw = args.metric == "raw"
    fig, axes = plt.subplots(1, k, figsize=(6.2 * k, 4.8), squeeze=False)
    for ax, r in zip(axes[0], results):
        order = ["ribbo"] + r["methods"] + ["rand"]
        if raw:  # actual best-so-far reward + optimum reference
            ax.axhline(r["opt_mean"], color="#111111", lw=1.6, ls=":",
                       label=f"optimum ({r['opt_mean']:.2f})")
            ax.fill_between(evals, r["opt_mean"] - r["opt_std"], r["opt_mean"] + r["opt_std"],
                            color="#111111", alpha=0.07)
        for key in order:
            c, lbl = METHOD_STYLE[key]
            ls = "--" if key == "rand" else "-"
            lw = 2.6 if key == "ribbo" else 2.0
            if raw:
                ax.plot(evals, r["curves_mean"][key], color=c, lw=lw, ls=ls,
                        label=f"{lbl} ({r['finals_raw'][key]:.2f})")
            else:
                ax.plot(evals, r["gaps"][key], color=c, lw=lw, ls=ls,
                        label=f"{lbl} ({r['finals'][key]:.1f}%)")
        ax.set_title(f"IFC {r['label']} ({r['n']} ch)", fontsize=11)
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("best-so-far reward" if raw else "optimality gap (%)")
        ax.legend(fontsize=8.5, loc=("lower right" if raw else "upper right"))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.set_xlim(1, N_STEPS)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    tag = ("_" + args.tag) if args.tag else ""
    out = os.path.join(OUT_DIR, f"baselines{tag}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    print("\n" + "=" * 78)
    hdr = f"{'case':<12}{'RIBBO':>8}" + "".join([f"{m.upper():>8}" for m in args.methods]) + f"{'Random':>8}"
    print(hdr)
    print("-" * 78)
    for r in results:
        row = f"{r['case']:<12}{r['finals']['ribbo']:>7.1f}"
        for m in r["methods"]:
            row += f"{r['finals'][m]:>8.1f}"
        row += f"{r['finals']['rand']:>8.1f}"
        print(row)
    print("=" * 78)


if __name__ == "__main__":
    main()
