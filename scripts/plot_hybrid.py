"""(D) Hybrid figure: RIBBO / RIBBO+polish / Random / Random+polish as % of optimum.
Shows a cheap local polish reaches near-optimum from either init (polish dominates)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/15_hybrid"
plt.rcParams.update({"font.size": 15})
cases = [("EE D=10", "ee_d10"), ("SE D=20", "se_d20_4ba")]
fig, ax = plt.subplots(figsize=(9.5, 5.4))
labels = ["RIBBO", "RIBBO\n+polish", "Random", "Random\n+polish"]
keys = ["ribbo", "ribbo_polish", "random", "random_polish"]
case_colors = ["#7C3AED", "#2563EB"]   # one color per case
x = np.arange(len(labels))
w = 0.38
for i, (title, tag) in enumerate(cases):
    d = np.load(f"{OUT}/{tag}.npz")
    vals = [float(d[k].mean()) for k in keys]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=title,
                  color=case_colors[i], edgecolor="k", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
ax.axhline(100, color="k", ls="--", lw=1.6, label="optimum")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("% of optimum"); ax.set_ylim(0, 112)
ax.set_title("Local polish reaches ~optimum from either init")
ax.legend(loc="center right", framealpha=0.9)
ax.grid(True, axis="y", ls="--", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/hybrid.pdf", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/hybrid.pdf")
