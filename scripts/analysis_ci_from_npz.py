"""
(C, CPU part) 95% confidence intervals on the headline % -of-optimum numbers, computed from
the existing per-channel eval npz (no GPU). Complements the 500-channel re-eval (eval_ci.py)
by covering every result with mean +/- 95% CI and per-channel win-rate.

Output: results/14_error_bars/ci_from_npz.md
"""
import os
import numpy as np

OUT = "results/14_error_bars"
os.makedirs(OUT, exist_ok=True)


def bsf_final(raw):
    if raw.ndim == 3:
        raw = np.squeeze(raw, 1)
    return np.maximum.accumulate(raw, axis=-1)[:, -1]


def ci95(a):
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def report(label, rb_raw, rd_raw, opt):
    fr, fd = bsf_final(rb_raw), bsf_final(rd_raw)
    pr, pd = 100 * fr / opt, 100 * fd / opt
    win = 100 * np.mean(fr > fd)
    n = len(fr)
    return (f"{label:<26} n={n:<4} RIBBO {pr.mean():5.1f}% +/- {ci95(pr):3.1f}   "
            f"Random {pd.mean():5.1f}% +/- {ci95(pd):3.1f}   win-rate {win:5.1f}%")


def main():
    lines = ["# (C) 95% CIs on % of optimum (from existing per-channel npz)\n",
             "mean +/- 1.96*sd/sqrt(n). win-rate = fraction of channels RIBBO_final > Random_final.\n"]

    d3 = np.load("results/07_ifc_5ba_filter_ablation/eval_ifc_d3_3way.npz")
    d10 = np.load("results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz")
    lines.append(report("SE D=3 (3BA)", d3["raw_3ba"], d3["raw_rand"], d3["wmmse"]))
    lines.append(report("SE D=10 (5BA-filter)", d10["raw_5ba"], d10["raw_rand"], d10["wmmse"]))
    lines.append(report("SE D=10 (3BA)", d10["raw_3ba"], d10["raw_rand"], d10["wmmse"]))

    ee = np.load("results/09_ifc_ee/eval_ifc_ee.npz")
    lines.append(report("EE D=3 (3BA)", ee["rb3"], ee["rd3"], ee["o3"]))
    lines.append(report("EE D=10 (3BA)", ee["rb10"], ee["rd10"], ee["o10"]))

    d20 = np.load("results/11_ifc_d20_scalability/eval_ifc_d20_4ba.npz")
    lines.append(report("SE D=20 (4BA-filter)", d20["rb"], d20["rd"], d20["ref"]))

    # EE D=20 if the E-eval has finished
    p = "results/16_ee_d20/eval_ee_d20.npz"
    if os.path.exists(p):
        e = np.load(p)
        lines.append(report("EE D=20 (3BA)", e["rb"], e["rd"], e["opt"]))

    bud = np.load("results/08_eval_budget_extension/eval_ifc_budget.npz")
    lines.append(report("SE D=3 @1000 (3BA)", bud["r3"], bud["rr3"], bud["w3"]))
    lines.append(report("SE D=10 @1000 (5BA)", bud["r10"], bud["rr10"], bud["w10"]))

    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "ci_from_npz.md"), "w") as fp:
        fp.write(text)
    print(text)


if __name__ == "__main__":
    main()
