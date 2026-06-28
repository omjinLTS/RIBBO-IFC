"""
(A) Sample-efficiency / convergence-speed analysis from existing eval npz.
For each experiment, compute the number of function evaluations the AVERAGE best-so-far
curve needs to reach a target fraction of the optimum, for RIBBO vs Random, and the
speedup factor. RIBBO's cheap inference means "just run longer", so this quantifies how
much faster it gets to a given quality.

Outputs: results/12_sample_efficiency/{sample_efficiency.pdf, README.md}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/12_sample_efficiency"
os.makedirs(OUT, exist_ok=True)


def bsf(a):
    return np.maximum.accumulate(a, axis=-1)


def steps_to(curve, target):
    """first 1-based eval index where mean best-so-far >= target, else np.inf."""
    idx = np.argmax(curve >= target)
    if curve[idx] >= target:
        return idx + 1
    return np.inf


# (label, ribbo_raw, random_raw, opt_per_channel, horizon)
def load():
    bud = np.load("results/08_eval_budget_extension/eval_ifc_budget.npz")
    ee = np.load("results/09_ifc_ee/eval_ifc_ee.npz")
    d20 = np.load("results/11_ifc_d20_scalability/eval_ifc_d20_4ba.npz")
    return [
        ("SE D=3 (3BA, 1000)", bud["r3"], bud["rr3"], bud["w3"]),
        ("SE D=10 (5BA-filter, 1000)", bud["r10"], bud["rr10"], bud["w10"]),
        ("EE D=10 (3BA, 300)", ee["rb10"], ee["rd10"], ee["o10"]),
        ("SE D=20 (4BA-filter, 300)", d20["rb"], d20["rd"], d20["ref"]),
    ]


def main():
    cases = load()
    targets = [0.5, 0.6, 0.7, 0.8, 0.9]
    lines = []
    lines.append("# (A) Sample-efficiency: function evals to reach a target % of optimum\n")
    lines.append("Average best-so-far curve crossing the target. 'inf' = never within budget.\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    panel_cases = {"SE D=10 (5BA-filter, 1000)": axes[0], "EE D=10 (3BA, 300)": axes[1]}
    for label, rb, rd, opt in cases:
        om = opt.mean()
        cr, cd = bsf(rb).mean(0), bsf(rd).mean(0)
        T = cr.shape[0]
        lines.append(f"\n## {label}  (optimum mean={om:.4f}, horizon={T})")
        lines.append(f"{'target':>8}{'RIBBO evals':>14}{'Random evals':>14}{'speedup':>10}")
        for f in targets:
            tv = f * om
            sr, sd = steps_to(cr, tv), steps_to(cd, tv)
            sp = (sd / sr) if (np.isfinite(sr) and np.isfinite(sd)) else np.inf
            sps = f"{sp:.1f}x" if np.isfinite(sp) else ("RIBBO only" if np.isfinite(sr) else "neither")
            srs = str(int(sr)) if np.isfinite(sr) else "inf"
            sds = str(int(sd)) if np.isfinite(sd) else "inf"
            lines.append(f"{int(f*100):>7}%{srs:>14}{sds:>14}{sps:>10}")
        # final % too
        lines.append(f"  final: RIBBO {100*cr[-1]/om:.1f}% of opt, Random {100*cd[-1]/om:.1f}%")

        if label in panel_cases:
            ax = panel_cases[label]
            xs = [int(f * 100) for f in targets]
            yr = [steps_to(cr, f * om) for f in targets]
            yd = [steps_to(cd, f * om) for f in targets]
            yr = [y if np.isfinite(y) else T * 1.6 for y in yr]
            yd = [y if np.isfinite(y) else T * 1.6 for y in yd]
            ax.plot(xs, yr, "o-", color="#7C3AED", lw=2.5, ms=8, label="RIBBO")
            ax.plot(xs, yd, "s--", color="#9CA3AF", lw=2.2, ms=7, label="Random")
            ax.axhline(T, color="k", ls=":", lw=1.3, label=f"budget ({T})")
            ax.set_yscale("log")
            ax.set_xlabel("Target (% of optimum)")
            ax.set_ylabel("Function evals to reach target")
            ax.set_title(label)
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend()
    fig.suptitle("Sample efficiency: evals to reach a quality target (lower = faster)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sample_efficiency.pdf", dpi=150, bbox_inches="tight")
    with open(f"{OUT}/README.md", "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT}/sample_efficiency.pdf")


if __name__ == "__main__":
    main()
