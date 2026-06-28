"""
(B) Per-channel statistics: go beyond the mean. For each experiment compute, per channel,
the final best-so-far as % of that channel's optimum, then report:
  - win-rate: fraction of channels where RIBBO_final > Random_final
  - median % of optimum (RIBBO vs Random)
  - fraction of channels reaching >= 80% / >= 90% of optimum
and plot the CDF of per-channel % of optimum (RIBBO vs Random) for the key cases.
All at the 300-step operating horizon (matches the main deck).

Outputs: results/13_per_channel_stats/{per_channel_cdf.pdf, README.md}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/13_per_channel_stats"
os.makedirs(OUT, exist_ok=True)


def bsf(a):
    return np.maximum.accumulate(a, axis=-1)


def finals(raw):
    """raw shape (C, T) or (C,1,T) -> per-channel final best-so-far (C,)."""
    if raw.ndim == 3:
        raw = np.squeeze(raw, 1)
    return bsf(raw)[:, -1]


def load():
    d07_3 = np.load("results/07_ifc_5ba_filter_ablation/eval_ifc_d3_3way.npz")
    d07_10 = np.load("results/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz")
    ee = np.load("results/09_ifc_ee/eval_ifc_ee.npz")
    d20 = np.load("results/11_ifc_d20_scalability/eval_ifc_d20_4ba.npz")
    # (label, ribbo_final, random_final, opt_per_channel, is_key_panel)
    return [
        ("SE D=3 (3BA)", finals(d07_3["raw_3ba"]), finals(d07_3["raw_rand"]), d07_3["wmmse"], False),
        ("SE D=10 (5BA-filter)", finals(d07_10["raw_5ba"]), finals(d07_10["raw_rand"]), d07_10["wmmse"], True),
        ("EE D=3 (3BA)", finals(ee["rb3"]), finals(ee["rd3"]), ee["o3"], False),
        ("EE D=10 (3BA)", finals(ee["rb10"]), finals(ee["rd10"]), ee["o10"], True),
        ("SE D=20 (4BA-filter)", finals(d20["rb"]), finals(d20["rd"]), d20["ref"], False),
    ]


def main():
    cases = load()
    lines = ["# (B) Per-channel statistics (300-step horizon)\n",
             "Per channel: final best-so-far as % of that channel's optimum.\n",
             f"{'case':<24}{'win-rate':>9}{'RIBBO med%':>11}{'Rand med%':>11}"
             f"{'R>=80%':>8}{'R>=90%':>8}{'Rnd>=80%':>9}"]
    panels = []
    for label, fr, fd, opt, key in cases:
        pr = 100 * fr / opt
        pd = 100 * fd / opt
        win = 100 * np.mean(fr > fd)
        lines.append(f"{label:<24}{win:>8.1f}%{np.median(pr):>10.1f}%{np.median(pd):>10.1f}%"
                     f"{100*np.mean(pr>=80):>7.0f}%{100*np.mean(pr>=90):>7.0f}%{100*np.mean(pd>=80):>8.0f}%")
        if key:
            panels.append((label, pr, pd))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, pr, pd) in zip(axes, panels):
        for vals, c, lab in [(pr, "#7C3AED", "RIBBO"), (pd, "#9CA3AF", "Random")]:
            xs = np.sort(vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, lw=2.6, color=c, ls="-" if lab == "RIBBO" else "--", label=lab)
        ax.axvline(100, color="k", ls=":", lw=1.3, label="optimum")
        ax.set_xlabel("Per-channel final (% of optimum)")
        ax.set_ylabel("Cumulative fraction of channels")
        ax.set_title(label)
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle("Per-channel CDF: RIBBO distribution is shifted toward the optimum", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/per_channel_cdf.pdf", dpi=150, bbox_inches="tight")
    with open(f"{OUT}/README.md", "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT}/per_channel_cdf.pdf")


if __name__ == "__main__":
    main()
