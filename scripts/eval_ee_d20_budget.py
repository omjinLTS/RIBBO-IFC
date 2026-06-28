"""EE D=20 budget extension: does running past the 300-step train horizon (to 1000) recover
the low EE D=20 score? Budget is the right recovery lever for EE (pool curation hurts EE).
RIBBO + Random vs near-global optimum.

Outputs: results/16_ee_d20/eval_ee_d20_budget.{npz,pdf}, appends README.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scripts.eval_ifc_ee as E
from problems.ifc import IFCNumpy, ifc_optimum_multistart

N_STEPS, N_CH = 1000, 150
OUT = "results/16_ee_d20"
RUN = "./log/ifc/dt-ifc-ee-d20-3ba"
SSID, DIM = "IFCEED20", 20
DATA, CACHE = "./data/generated_data/ifc_d20_ee", "./cache/ifc_d20_ee/"


def main():
    os.makedirs(OUT, exist_ok=True)
    ck = E.latest_ckpt(RUN)
    ymax, ymin = E.stats(DATA, CACHE, SSID)
    d = E.build_designer(ck, DIM)
    print(f"[EE D=20 budget] ckpt={ck} n={N_CH} steps={N_STEPS}", flush=True)

    rb = np.zeros((N_CH, N_STEPS))
    rd = np.zeros((N_CH, N_STEPS))
    opt = np.zeros(N_CH)
    for i in range(N_CH):
        cid = str(30 + i)
        f = IFCNumpy(SSID, cid, dim=DIM)
        rb[i] = E.rollout_ribbo(d, f, DIM, ymax, ymin, N_STEPS)
        rd[i] = E.rollout_random(f, DIM, N_STEPS, seed=int(cid) + 1000)
        opt[i], _ = ifc_optimum_multistart(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix,
                                           reward_type="EE", n_restart=40)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{N_CH}]", flush=True)

    np.savez(OUT + "/eval_ee_d20_budget.npz", rb=rb, rd=rd, opt=opt, ckpt=ck)
    b_rb = np.maximum.accumulate(rb, 1).mean(0)
    b_rd = np.maximum.accumulate(rd, 1).mean(0)
    om = opt.mean()
    ev = np.arange(1, N_STEPS + 1)
    p300, p1000 = 100 * b_rb[299] / om, 100 * b_rb[-1] / om
    pr300 = 100 * b_rd[299] / om

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.axhline(om, color="#111111", lw=2.0, label="optimum (mean)")
    ax.fill_between(ev, om - opt.std(), om + opt.std(), color="#111111", alpha=0.07)
    ax.fill_between(ev, b_rd, b_rb, color="#DC2626", alpha=0.09)
    ax.plot(ev, b_rb, color="#7C3AED", lw=3.0,
            label=f"RIBBO (@300 {p300:.0f}%, @1000 {p1000:.0f}%)")
    ax.plot(ev, b_rd, color="#9CA3AF", lw=2.2, ls="--", label=f"Random (@300 {pr300:.0f}%)")
    ax.axvline(300, color="#DC2626", ls=":", lw=1.4, label="train horizon (300)")
    ax.set_xscale("log")
    ax.set_title(f"IFC EE D=20 budget extension ({N_CH} ch)")
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("Average best EE")
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT + "/eval_ee_d20_budget.pdf", dpi=150, bbox_inches="tight")

    line = f"EE D=20 budget: RIBBO @300={p300:.1f}%  @1000={p1000:.1f}%  (+{p1000-p300:.1f}p), Random @300={pr300:.1f}%"
    print("\n" + line, flush=True)
    with open(OUT + "/README.md", "a") as fp:
        fp.write("\n## budget extension\n" + line + "\n")


if __name__ == "__main__":
    main()
