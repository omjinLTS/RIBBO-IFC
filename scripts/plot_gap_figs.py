"""
Regenerate ALL presentation figures using "gap to optimum (%)" = 100*(opt - achieved)/opt
(lower = better; optimum = 0), to match Lee's optimality-gap convention. Curves are drawn as
the gap shrinking over iterations; bars/synthesis/CDF use the gap value.

Overwrites presentation/fig_*.pdf. Run: python scripts/plot_gap_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

R, OUT = "results", "presentation"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
C_RIBBO, C_RAND, C_OPT = "#7C3AED", "#9CA3AF", "#111111"
C_D3, C_D10, C_GAP = "#2563EB", "#16A34A", "#DC2626"


def bsf(a):
    return np.maximum.accumulate(a, axis=-1)


def gapcurve(raw, opt):
    om = opt.mean()
    return 100 * (om - bsf(raw).mean(0)) / om


def gapfinal(raw, opt):  # per-channel final gap %
    if raw.ndim == 3:
        raw = np.squeeze(raw, 1)
    return 100 * (opt - bsf(raw)[:, -1]) / opt


# ---------- 1. synthesis: gap vs dimension ----------
def synthesis():
    D = np.array([3, 10, 20])
    g = lambda v: [100 - x for x in v]
    se_r, se_d = g([88.9, 47.8, 33.0]), g([83.0, 43.6, 31.5])
    ee_r, ee_d = g([96.5, 78.1, 30.2]), g([82.8, 29.4, 17.2])
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.axhline(0, color=C_OPT, ls="--", lw=1.6, label="reference (gap = 0)")
    ax.plot(D, se_r, "o-", color=C_D3, lw=3, ms=9, label="RIBBO — SE")
    ax.plot(D, se_d, "o:", color=C_D3, lw=1.8, ms=6, alpha=.7, label="Random — SE")
    ax.plot(D, ee_r, "s-", color=C_D10, lw=3, ms=9, label="RIBBO — EE")
    ax.plot(D, ee_d, "s:", color=C_D10, lw=1.8, ms=6, alpha=.7, label="Random — EE")
    ax.annotate("", xy=(10, ee_d[1]), xytext=(10, ee_r[1]),
                arrowprops=dict(arrowstyle="<->", color=C_D10, lw=2))
    ax.text(10.4, (ee_r[1] + ee_d[1]) / 2, "48.7 pts\nsmaller gap\n(EE D=10)",
            color=C_D10, fontsize=11.5, fontweight="bold", va="center")
    ax.set_xticks(D); ax.set_xlabel("Dimension  D")
    ax.set_ylabel("optimality gap (%)  — lower is better")
    ax.set_ylim(-4, 100)
    ax.legend(loc="upper left", ncol=2, fontsize=11.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_synthesis.pdf"); plt.close(fig)


# ---------- 2. BA-pool x dim bars (gap to WMMSE) ----------
def ablation():
    d3 = np.load(f"{R}/07_ifc_5ba_filter_ablation/eval_ifc_d3_3way.npz")
    d10 = np.load(f"{R}/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz")
    fin = lambda a, w: 100 - 100 * bsf(np.squeeze(a, 1))[:, -1].mean() / w.mean()
    w3, w10 = d3["wmmse"], d10["wmmse"]
    cats = ["3BA", "5BA-filter", "7BA"]
    b3 = [fin(d3["raw_3ba"], w3), fin(d3["raw_5ba"], w3), fin(d3["raw_7ba"], w3)]
    b10 = [fin(d10["raw_3ba"], w10), fin(d10["raw_5ba"], w10), np.nan]
    r3 = fin(d3["raw_rand"], w3); r10 = fin(d10["raw_rand"], w10)
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bars3 = ax.bar(x - w/2, b3, w, color=C_D3, label="RIBBO D=3")
    bars10 = ax.bar(x + w/2, b10, w, color=C_D10, label="RIBBO D=10")
    ax.axhline(r3, color=C_D3, ls=":", lw=1.8, label=f"Random D=3 ({r3:.0f})")
    ax.axhline(r10, color=C_D10, ls=":", lw=1.8, label=f"Random D=10 ({r10:.0f})")
    for bars in (bars3, bars10):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.1f}", ha="center", va="bottom",
                        fontsize=12.5, fontweight="bold")
    ax.text(x[2]+w/2, 2, "n/a", ha="center", color=C_RAND, fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("optimality gap vs WMMSE (%)  — lower is better")
    ax.set_ylim(0, 75)
    ax.legend(ncol=2, loc="upper left", fontsize=11.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation_2x2.pdf"); plt.close(fig)


# ---------- 3. EE SE vs EE: gap curves (2 panel) ----------
def ee_se():
    d = np.load(f"{R}/09_ifc_ee/eval_ifc_ee.npz")
    ev = np.arange(1, 301)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, (rb, rd, o, dl) in zip(axes, [(d["rb3"], d["rd3"], d["o3"], "D=3"),
                                          (d["rb10"], d["rd10"], d["o10"], "D=10")]):
        gr, gd = gapcurve(rb, o), gapcurve(rd, o)
        ax.fill_between(ev, gr, gd, color=C_GAP, alpha=.09)
        ax.plot(ev, gr, color=C_RIBBO, lw=3, label=f"RIBBO (gap {gr[-1]:.0f}%)")
        ax.plot(ev, gd, color=C_RAND, lw=2.2, ls="--", label=f"Random (gap {gd[-1]:.0f}%)")
        ax.axhline(0, color=C_OPT, lw=2, label="reference (gap 0)")
        ax.set_title(f"EE {dl}"); ax.set_xlabel("Function evaluations")
        ax.set_ylabel("optimality gap (%)"); ax.set_xlim(1, 300); ax.set_ylim(-3, 100)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50)); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ee_se.pdf"); plt.close(fig)


# ---------- 4. per-channel gap CDF ----------
def per_channel():
    d07 = np.load(f"{R}/07_ifc_5ba_filter_ablation/eval_ifc_d10_3way.npz")
    ee = np.load(f"{R}/09_ifc_ee/eval_ifc_ee.npz")
    panels = [("SE D=10 (5BA-filter)", gapfinal(d07["raw_5ba"], d07["wmmse"]), gapfinal(d07["raw_rand"], d07["wmmse"])),
              ("EE D=10 (3BA)", gapfinal(ee["rb10"], ee["o10"]), gapfinal(ee["rd10"], ee["o10"]))]
    # Stacked vertically so each panel fills the slide column (readable when scaled down).
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.6))
    for ax, (lab, pr, pd) in zip(axes, panels):
        for v, c, l, ls in [(pr, C_RIBBO, "RIBBO", "-"), (pd, C_RAND, "Random", "--")]:
            xs = np.sort(v); ys = np.arange(1, len(xs)+1)/len(xs)
            ax.plot(xs, ys, lw=3.6, color=c, ls=ls, label=l)
        ax.axvline(0, color=C_OPT, ls=":", lw=1.8, label="reference (gap 0)")
        ax.set_ylabel("frac. of channels", fontsize=18)
        ax.set_title(lab, fontsize=20)
        ax.tick_params(labelsize=16)
        ax.legend(loc="lower right", fontsize=17)
    axes[-1].set_xlabel("per-channel optimality gap (%)", fontsize=18)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_per_channel.pdf"); plt.close(fig)


# ---------- 5. transfer gap curves ----------
def transfer():
    d = np.load(f"{R}/10_rastrigin_transfer_normfix/eval_rastrigin_normfix.npz")
    n = d["A"].shape[1]; ev = np.arange(1, n+1); wm = d["wm"].mean()
    gc = lambda k: 100*(wm - bsf(d[k]).mean(0))/wm
    A, B, C, Dr = gc("A"), gc("B"), gc("C"), gc("Dr")
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.axhline(0, color=C_OPT, lw=2, label="reference (gap 0)")
    ax.plot(ev, C, color=C_D10, lw=3, label=f"Native IFC D=10 (gap {C[-1]:.0f}%)")
    ax.plot(ev, Dr, color=C_RAND, lw=2.4, ls="--", label=f"Random (gap {Dr[-1]:.0f}%)")
    ax.plot(ev, A, color=C_GAP, lw=2.4, label=f"Transfer, unchanged (gap {A[-1]:.0f}%)")
    ax.plot(ev, B, color="#1D4ED8", lw=2.4, ls="-.", label=f"Transfer, rescaled (gap {B[-1]:.0f}%)")
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("optimality gap (%)")
    ax.set_xlim(1, n); ax.set_ylim(0, 100); ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_transfer.pdf"); plt.close(fig)


# ---------- 6. sample efficiency: steps to reach gap target ----------
def sample_eff():
    bud = np.load(f"{R}/08_eval_budget_extension/eval_ifc_budget.npz")
    ee = np.load(f"{R}/09_ifc_ee/eval_ifc_ee.npz")
    cases = [("SE D=10 (5BA-filter, 1000)", bud["r10"], bud["w10"]),
             ("EE D=10 (3BA, 300)", ee["rb10"], ee["o10"])]
    casesd = [bud["rr10"], ee["rd10"]]
    targets = [50, 40, 30, 20, 10]  # gap % targets
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, (lab, rb, opt), rd in zip(axes, cases, casesd):
        T = rb.shape[1]
        gr = 100*(opt.mean() - bsf(rb).mean(0))/opt.mean()
        gd = 100*(opt.mean() - bsf(rd).mean(0))/opt.mean()
        def steps(g, tgt):
            idx = np.argmax(g <= tgt)
            return idx+1 if g[idx] <= tgt else T*1.6
        yr = [steps(gr, t) for t in targets]; yd = [steps(gd, t) for t in targets]
        ax.plot(targets, yr, "o-", color=C_RIBBO, lw=2.5, ms=8, label="RIBBO")
        ax.plot(targets, yd, "s--", color=C_RAND, lw=2.2, ms=7, label="Random")
        ax.axhline(T, color="k", ls=":", lw=1.3, label=f"budget ({T})")
        ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("target gap to optimum (%)  — smaller = harder")
        ax.set_ylabel("function evals to reach target"); ax.set_title(lab); ax.legend()
    fig.suptitle("Steps to reach a target gap (Random stays above it)", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sample_eff.pdf"); plt.close(fig)


# ---------- 7. EE pool ablation: gap bars ----------
def ee_pool():
    ee = np.load(f"{R}/09_ifc_ee/eval_ifc_ee.npz")
    f4 = np.load(f"{R}/09_ifc_ee/eval_ifc_ee_d10_4ba.npz")
    g3 = 100 - 100*bsf(ee["rb10"])[:, -1].mean()/ee["o10"].mean()
    g4 = 100 - 100*bsf(f4["rb"])[:, -1].mean()/f4["ref"].mean()
    gr = 100 - 100*bsf(ee["rd10"])[:, -1].mean()/ee["o10"].mean()
    labs = ["3BA\n(incl. Random)", "4BA-filter\n(no Random)", "Random"]
    vals = [g3, g4, gr]; cols = [C_RIBBO, "#A78BDA", C_RAND]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bars = ax.bar(labs, vals, color=cols, edgecolor="k", lw=.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.axhline(0, color=C_OPT, ls="--", lw=1.5, label="reference (gap 0)")
    ax.set_ylabel("gap to optimum (%)  — lower better"); ax.set_ylim(0, 80)
    ax.set_title("EE D=10: dropping Random increases the gap")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ee_pool.pdf"); plt.close(fig)


# ---------- 8. SE budget gap curves ----------
def budget():
    b = np.load(f"{R}/08_eval_budget_extension/eval_ifc_budget.npz")
    ev = np.arange(1, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, (rb, rd, w, dl) in zip(axes, [(b["r3"], b["rr3"], b["w3"], "D=3 (3BA)"),
                                          (b["r10"], b["rr10"], b["w10"], "D=10 (5BA-filter)")]):
        gr = 100*(w.mean()-bsf(rb).mean(0))/w.mean(); gd = 100*(w.mean()-bsf(rd).mean(0))/w.mean()
        ax.plot(ev, gr, color=C_RIBBO, lw=3, label=f"RIBBO (gap @300 {gr[299]:.0f}%, @1000 {gr[-1]:.0f}%)")
        ax.plot(ev, gd, color=C_RAND, lw=2.2, ls="--", label="Random")
        ax.axhline(0, color=C_OPT, lw=2, label="reference (gap 0)")
        ax.axvline(300, color=C_GAP, ls=":", lw=1.3, label="train horizon")
        ax.set_xscale("log"); ax.set_title(f"SE {dl} budget"); ax.set_xlabel("Function evaluations")
        ax.set_ylabel("gap to optimum (%)"); ax.set_ylim(0, 100); ax.legend(loc="upper right", fontsize=10)
    fig.suptitle("SE budget extension (gap shrinks with more steps)", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_budget.pdf"); plt.close(fig)


# ---------- 9. EE D=20 gap curve ----------
def ee_d20():
    d = np.load(f"{R}/16_ee_d20/eval_ee_d20.npz")
    ev = np.arange(1, 301); gr = gapcurve(d["rb"], d["opt"]); gd = gapcurve(d["rd"], d["opt"])
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.fill_between(ev, gr, gd, color=C_GAP, alpha=.09)
    ax.plot(ev, gr, color=C_RIBBO, lw=3, label=f"RIBBO (gap {gr[-1]:.0f}%)")
    ax.plot(ev, gd, color=C_RAND, lw=2.2, ls="--", label=f"Random (gap {gd[-1]:.0f}%)")
    ax.axhline(0, color=C_OPT, lw=2, label="reference (gap 0)")
    ax.set_title("IFC EE D=20 (200 channels x 300 steps)"); ax.set_xlabel("Function evaluations")
    ax.set_ylabel("gap to optimum (%)"); ax.set_xlim(1, 300); ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50)); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ee_d20.pdf"); plt.close(fig)


# ---------- 10. EE D=20 budget gap curve ----------
def ee_d20_budget():
    d = np.load(f"{R}/16_ee_d20/eval_ee_d20_budget.npz")
    ev = np.arange(1, 1001); gr = gapcurve(d["rb"], d["opt"]); gd = gapcurve(d["rd"], d["opt"])
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.plot(ev, gr, color=C_RIBBO, lw=3, label=f"RIBBO (gap @300 {gr[299]:.0f}%, @1000 {gr[-1]:.0f}%)")
    ax.plot(ev, gd, color=C_RAND, lw=2.2, ls="--", label=f"Random (gap {gd[-1]:.0f}%)")
    ax.axhline(0, color=C_OPT, lw=2, label="reference (gap 0)")
    ax.axvline(300, color=C_GAP, ls=":", lw=1.4, label="train horizon (300)")
    ax.set_xscale("log"); ax.set_title("IFC EE D=20 budget extension (150 ch)")
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("gap to optimum (%)")
    ax.set_ylim(0, 100); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ee_d20_budget.pdf"); plt.close(fig)


# ---------- 11. hybrid gap bars ----------
def hybrid():
    OUTk = ["ribbo", "ribbo_polish", "random", "random_polish"]
    labs = ["RIBBO", "RIBBO\n+polish", "Random", "Random\n+polish"]
    cases = [("EE D=10", "ee_d10", C_RIBBO), ("SE D=20", "se_d20_4ba", C_D3)]
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for i, (title, tag, col) in enumerate(cases):
        d = np.load(f"{R}/15_hybrid/{tag}.npz")
        vals = [100 - float(d[k].mean()) for k in OUTk]
        bars = ax.bar(x + (i-0.5)*w, vals, w, color=col, edgecolor="k", lw=.5, label=title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0, color=C_OPT, ls="--", lw=1.5, label="reference (gap 0)")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("gap to optimum (%)  — lower better")
    ax.set_ylim(0, 80); ax.set_title("Local search closes the gap to ~0 from either start")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hybrid.pdf"); plt.close(fig)


if __name__ == "__main__":
    for fn in [synthesis, ablation, ee_se, per_channel, transfer, sample_eff,
               ee_pool, budget, ee_d20, ee_d20_budget, hybrid]:
        fn(); print("ok:", fn.__name__)
    print("done -> presentation/fig_*.pdf (gap to optimum)")
