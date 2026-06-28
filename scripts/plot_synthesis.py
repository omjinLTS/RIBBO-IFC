"""Synthesis: % of optimum vs dimension for SE and EE, RIBBO vs Random (all 3BA recipe,
% of optimum at 300 steps). Consolidates the scalability + SE-vs-EE story in one figure.

Numbers are the deck headline values (3BA, % of optimum):
  SE  RIBBO  88.9 / 47.8 / 33.0   Random 83.0 / 43.6 / 31.5
  EE  RIBBO  96.5 / 78.1 / 30.2   Random 82.8 / 29.4 / 17.2
Output: results/17_synthesis/synthesis_se_ee_dim.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/17_synthesis"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 15})

D = np.array([3, 10, 20])
se_ribbo = [88.9, 47.8, 33.0]
se_rand = [83.0, 43.6, 31.5]
ee_ribbo = [96.5, 78.1, 30.2]
ee_rand = [82.8, 29.4, 17.2]

C_SE, C_EE = "#2563EB", "#16A34A"
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.axhline(100, color="#111111", ls="--", lw=1.6, label="optimum (WMMSE / global)")
ax.plot(D, se_ribbo, "o-", color=C_SE, lw=3.0, ms=9, label="RIBBO — SE")
ax.plot(D, se_rand, "o:", color=C_SE, lw=1.8, ms=6, alpha=0.7, label="Random — SE")
ax.plot(D, ee_ribbo, "s-", color=C_EE, lw=3.0, ms=9, label="RIBBO — EE")
ax.plot(D, ee_rand, "s:", color=C_EE, lw=1.8, ms=6, alpha=0.7, label="Random — EE")

# shade the RIBBO-Random gap at D=10 for both
ax.annotate("", xy=(10, ee_ribbo[1]), xytext=(10, ee_rand[1]),
            arrowprops=dict(arrowstyle="<->", color=C_EE, lw=2.0))
ax.text(10.4, (ee_ribbo[1] + ee_rand[1]) / 2, "+48.7 pts\n(EE D=10)",
        color=C_EE, fontsize=12, fontweight="bold", va="center")

ax.set_xticks(D)
ax.set_xlabel("Dimension  D")
ax.set_ylabel("% of optimum  (final, 300 steps)")
ax.set_ylim(0, 108)
ax.set_title("RIBBO vs Random across dimension — SE and EE (3BA recipe)")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(loc="lower left", ncol=2, framealpha=0.9, fontsize=12)
fig.tight_layout()
fig.savefig(OUT + "/synthesis_se_ee_dim.pdf", dpi=150, bbox_inches="tight")
print("saved", OUT + "/synthesis_se_ee_dim.pdf")
