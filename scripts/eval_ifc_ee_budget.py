"""EE budget extension: does EE also close the gap to optimum with more eval budget?
Extend EE D=3 (3BA) and D=10 (3BA) models to 1000 steps vs the global EE optimum."""
import os, numpy as np, torch
import matplotlib.pyplot as plt
import scripts.eval_ifc_ee as E
from problems.ifc import IFCNumpy, ifc_optimum_bruteforce, ifc_optimum_multistart

N_STEPS, N_CH = 1000, 100
OUT = "results/09_ifc_ee/eval_ifc_ee_budget"


def run(run_dir, dim, ssid, data_dir, cache_dir):
    ck = E.latest_ckpt(run_dir)
    ymax, ymin = E.stats(data_dir, cache_dir, ssid)
    d = E.build_designer(ck, dim)
    rb = np.zeros((N_CH, N_STEPS)); opt = np.zeros(N_CH)
    for i in range(N_CH):
        cid = str(30 + i); f = IFCNumpy(ssid, cid, dim=dim)
        rb[i] = E.rollout_ribbo(d, f, dim, ymax, ymin, N_STEPS)
        if dim <= 3:
            opt[i], _ = ifc_optimum_bruteforce(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", grid=51)
        else:
            opt[i], _ = ifc_optimum_multistart(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", n_restart=40)
        if (i+1) % 25 == 0: print(f"  D={dim} [{i+1}/{N_CH}]", flush=True)
    return rb, opt


def main():
    os.makedirs("results/09_ifc_ee", exist_ok=True)
    rb3, o3 = run("./log/ifc/dt-ifc-ee-d3-3ba", 3, "IFCEE", "./data/generated_data/ifc_ee", "./cache/ifc_ee/")
    rb10, o10 = run("./log/ifc/dt-ifc-ee-d10-3ba", 10, "IFCEED10", "./data/generated_data/ifc_d10_ee", "./cache/ifc_d10_ee/")
    np.savez(OUT + ".npz", rb3=rb3, o3=o3, rb10=rb10, o10=o10)
    ev = np.arange(1, N_STEPS+1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (rb, o, dl) in zip(axes, [(rb3, o3, "D=3"), (rb10, o10, "D=10")]):
        b = np.maximum.accumulate(rb, 1); om = o.mean()
        p300, p1000 = 100*b[:, 299].mean()/om, 100*b[:, -1].mean()/om
        ax.plot(ev, b.mean(0), color="#7C3AED", lw=2.5, label=f"RIBBO-EE (@300 {p300:.0f}%, @1000 {p1000:.0f}%)")
        ax.axhline(om, color="#000", lw=1.6, label="EE optimum")
        ax.axvline(300, color="#DC2626", ls=":", lw=1.2, label="train horizon")
        ax.set_xscale("log"); ax.set_title(f"EE {dl} budget extension", fontsize=10)
        ax.set_xlabel("Function Evaluations"); ax.set_ylabel("Avg Best EE")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, ls="--", alpha=0.3); ax.set_ylim(bottom=0)
        print(f"EE {dl}: @300={p300:.1f}%  @1000={p1000:.1f}%  (+{p1000-p300:.1f}p)")
    fig.tight_layout(); fig.savefig(OUT + ".pdf", dpi=150, bbox_inches="tight")
    print("saved", OUT + ".pdf")


if __name__ == "__main__":
    main()
