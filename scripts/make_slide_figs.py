"""
Regenerate presentation figures from the stored eval npz files: larger fonts,
a light shaded RIBBO-vs-Random gap band, and percentages in the legend.
Deliberately NO in-figure text callouts/arrows (they overlapped) -- the
explanation lives in the slide captions/bullets.

Outputs overwrite presentation/fig_*.pdf (names referenced by progress_0625.tex):
  fig_ablation_2x2.pdf, fig_ee_se.pdf, fig_d20.pdf, fig_transfer.pdf

Run: python scripts/make_slide_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

R = "results"
OUT = "presentation"

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

C_RIBBO = "#7C3AED"   # purple
C_RAND = "#9CA3AF"    # gray
C_OPT = "#111111"     # black
C_D3 = "#2563EB"      # blue
C_D10 = "#16A34A"     # green
C_GAP = "#DC2626"     # red (shaded gap fill only)


def bsf(y):
    """best-so-far along last axis (idempotent if already monotone)."""
    return np.maximum.accumulate(y, axis=-1)


# ----------------------------------------------------------------------
# 1. SE vs EE (results/09) -- 2 panel, clean
# ----------------------------------------------------------------------
def fig_ee():
    d = np.load(f"{R}/09_ifc_ee/eval_ifc_ee.npz", allow_pickle=True)
    evals = np.arange(1, 301)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    panels = [(d["rb3"], d["rd3"], d["o3"], "D=3", axes[0]),
              (d["rb10"], d["rd10"], d["o10"], "D=10", axes[1])]
    out = []
    for rb, rd, o, dl, ax in panels:
        b_rb, b_rd = bsf(rb).mean(0), bsf(rd).mean(0)
        om = o.mean()
        p_rb, p_rd = 100 * b_rb[-1] / om, 100 * b_rd[-1] / om
        out.append((dl, p_rb, p_rd, p_rb - p_rd))
        ax.fill_between(evals, b_rd, b_rb, color=C_GAP, alpha=0.09)
        ax.plot(evals, b_rb, color=C_RIBBO, lw=3.0, label=f"RIBBO  ({p_rb:.0f}% of opt)")
        ax.plot(evals, b_rd, color=C_RAND, lw=2.2, ls="--", label=f"Random  ({p_rd:.0f}% of opt)")
        ax.axhline(om, color=C_OPT, lw=2.0, label="optimum (mean)")
        ax.fill_between(evals, om - o.std(), om + o.std(), color=C_OPT, alpha=0.07)
        ax.set_title(f"IFC EE {dl}")
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Average best EE")
        ax.set_xlim(1, 300); ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.legend(loc="lower right", framealpha=0.9)
    fig.suptitle("RIBBO vs Random vs optimum  (200 unseen channels x 300 steps)", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ee_se.pdf")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 2. BA-pool x dimension 2x2 bar (results/07) -- clean
# ----------------------------------------------------------------------
def fig_2x2():
    d3 = np.load(f"{R}/07_ifc_5ba_filter_ablation/eval_ifc_d3_3way.npz", allow_pickle=True)
    d10 = np.load(f"{R}/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz", allow_pickle=True)

    def fin(arr, wm):
        return 100 * bsf(np.squeeze(arr, 1))[:, -1].mean() / wm.mean()

    w3, w10 = d3["wmmse"], d10["wmmse"]
    v3 = {"3BA": fin(d3["raw_3ba"], w3), "5BA-filter": fin(d3["raw_5ba"], w3),
          "7BA": fin(d3["raw_7ba"], w3)}
    r3 = fin(d3["raw_rand"], w3)
    v10 = {"3BA": fin(d10["raw_3ba"], w10), "5BA-filter": fin(d10["raw_5ba"], w10)}
    r10 = fin(d10["raw_rand"], w10)

    cats = ["3BA", "5BA-filter", "7BA"]
    x = np.arange(len(cats)); w = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    b3 = [v3[c] for c in cats]
    b10 = [v10.get(c, np.nan) for c in cats]
    bars3 = ax.bar(x - w / 2, b3, w, color=C_D3, label="RIBBO D=3")
    bars10 = ax.bar(x + w / 2, b10, w, color=C_D10, label="RIBBO D=10")
    ax.axhline(100, color=C_OPT, ls="--", lw=1.8, label="WMMSE (optimum)")
    ax.axhline(r3, color=C_D3, ls=":", lw=1.8, label=f"Random D=3 ({r3:.0f}%)")
    ax.axhline(r10, color=C_D10, ls=":", lw=1.8, label=f"Random D=10 ({r10:.0f}%)")
    for bars in (bars3, bars10):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.text(x[2] + w / 2, 4, "n/a", ha="center", color=C_RAND, fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Final best SE  (% of WMMSE)")
    ax.set_ylim(0, 112)
    ax.set_title("Best BA pool flips with dimension")
    ax.legend(ncol=2, loc="upper right", framealpha=0.9, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ablation_2x2.pdf")
    plt.close(fig)
    return v3, r3, v10, r10


# ----------------------------------------------------------------------
# 3. D=20 scalability (results/11) -- clean
# ----------------------------------------------------------------------
def fig_d20():
    d = np.load(f"{R}/11_ifc_d20_scalability/eval_ifc_d20_4ba.npz", allow_pickle=True)
    evals = np.arange(1, 301)
    b_rb, b_rd = bsf(d["rb"]).mean(0), bsf(d["rd"]).mean(0)
    wm = d["ref"].mean()
    p_rb, p_rd = 100 * b_rb[-1] / wm, 100 * b_rd[-1] / wm
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.axhline(wm, color=C_OPT, lw=2.0, label="WMMSE (optimum)")
    ax.fill_between(evals, wm - d["ref"].std(), wm + d["ref"].std(), color=C_OPT, alpha=0.07)
    ax.fill_between(evals, b_rd, b_rb, color=C_GAP, alpha=0.09)
    ax.plot(evals, b_rb, color=C_D10, lw=3.0, label=f"RIBBO 4BA-filter  ({p_rb:.0f}% of WMMSE)")
    ax.plot(evals, b_rd, color=C_RAND, lw=2.2, ls="--", label=f"Random  ({p_rd:.0f}%)")
    ax.set_title("IFC SE D=20  (200 unseen channels x 300 steps)")
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("Average best SE")
    ax.set_xlim(1, 300); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_d20.pdf")
    plt.close(fig)
    return p_rb, p_rd


# ----------------------------------------------------------------------
# 4. Cross-domain transfer (results/10) -- clean
# ----------------------------------------------------------------------
def fig_transfer():
    d = np.load(f"{R}/10_rastrigin_transfer_normfix/eval_rastrigin_normfix.npz", allow_pickle=True)
    n = d["A"].shape[1]
    evals = np.arange(1, n + 1)
    wm = d["wm"].mean()
    A, B, C, Dr = (bsf(d[k]).mean(0) for k in ("A", "B", "C", "Dr"))
    pA, pB, pC, pD = (100 * v[-1] / wm for v in (A, B, C, Dr))
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.axhline(wm, color=C_OPT, lw=2.0, label="WMMSE (optimum)")
    ax.fill_between(evals, wm - d["wm"].std(), wm + d["wm"].std(), color=C_OPT, alpha=0.07)
    ax.plot(evals, C, color=C_D10, lw=3.0, label=f"Native IFC D=10  ({pC:.0f}%)")
    ax.plot(evals, Dr, color=C_RAND, lw=2.4, ls="--", label=f"Random  ({pD:.0f}%)")
    ax.plot(evals, A, color=C_GAP, lw=2.4, label=f"Transfer, Rastrigin RTG  ({pA:.0f}%)")
    ax.plot(evals, B, color="#1D4ED8", lw=2.4, ls="-.", label=f"Transfer, IFC RTG fix  ({pB:.0f}%)")
    ax.set_title("Rastrigin -> IFC D=10 zero-shot transfer")
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("Average best SE (nats)")
    ax.set_xlim(1, n); ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_transfer.pdf")
    plt.close(fig)
    return pA, pB, pC, pD


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("EE   :", fig_ee())
    print("2x2  :", fig_2x2())
    print("D20  :", fig_d20())
    print("xfer :", fig_transfer())
    print("done -> presentation/fig_*.pdf")
