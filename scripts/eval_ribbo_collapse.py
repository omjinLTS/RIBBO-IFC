"""
Visualize RIBBO 'policy collapse' in high dimension.
Hypothesis: at D=20 RIBBO's learned search policy converges early to a narrow region
(exploration radius shrinks) and stops finding improvements -> best-so-far plateaus
far from optimum. At D=10 the policy keeps exploring -> best-so-far keeps improving.

We re-roll RIBBO recording the proposed action x_t each step, then measure the
exploration radius = mean over dims of the std of x within a sliding window, averaged
over channels. Uniform-random search has per-dim std = 1/sqrt(12) ~ 0.289 (reference).

Outputs: results/19_budget_baselines/ribbo_collapse.{pdf,png}
"""
import os
import numpy as np
import torch

from problems.ifc import IFCNumpy
from scripts.eval_ifc_ee import build_designer, latest_ckpt, stats

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_STEPS = 300
WIN = 15
OUT = "results/19_budget_baselines"

CASES = [
    dict(run="./log/ifc/dt-ifc-ee-d10-3ba", ssid="IFCEED10", dim=10,
         data="./data/generated_data/ifc_d10_ee", cache="./cache/ifc_d10_ee/",
         color="#7C3AED", label="EE D=10 (keeps improving)"),
    dict(run="./log/ifc/dt-ifc-ee-d20-3ba", ssid="IFCEED20", dim=20,
         data="./data/generated_data/ifc_d20_ee", cache="./cache/ifc_d20_ee/",
         color="#B91C1C", label="EE D=20 (collapses)"),
]


def rollout_record_x(designer, func, x_dim, ymax, ymin, n_steps):
    """RIBBO rollout (same as eval) but record proposed action x_real each step."""
    B = 1
    designer.past_x = torch.zeros([B, n_steps + 1, x_dim], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = 0
    ts = torch.arange(n_steps + 1).clamp(max=299).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps + 1).repeat(B, 1)
    designer.step_count = 0
    scale = max(ymax - ymin, 1e-6)
    last_x = last_y = last_regret = None
    X = np.zeros((n_steps, x_dim))
    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(last_x=last_x, last_y=last_y,
                                      last_onestep_regret=last_regret,
                                      deterministic=False, regret_strategy="none")
        x_real = last_x.cpu().numpy() / 2 + 0.5          # [1, D] in [0,1]
        X[i] = x_real[0]
        y = float(func(x_real)[0, 0])
        last_y = torch.tensor([[(y - ymin) / scale]], dtype=torch.float).to(DEVICE)
        last_regret = torch.tensor([[(ymax - y) / scale]], dtype=torch.float).to(DEVICE)
    return X


def radius_curve(X):
    """X: [n_ch, T, D] -> per-step mean(over dims) std within sliding window WIN, mean over ch."""
    n, T, D = X.shape
    rad = np.zeros(T)
    for t in range(T):
        lo = max(0, t - WIN + 1)
        w = X[:, lo:t + 1, :]                    # [n, win, D]
        rad[t] = w.std(axis=1).mean()            # std over time, mean over dims & channels
    return rad


def main():
    os.makedirs(OUT, exist_ok=True)
    n_ch = 20
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for c in CASES:
        ck = latest_ckpt(c["run"])
        ymax, ymin = stats(c["data"], c["cache"], c["ssid"])
        d = build_designer(ck, c["dim"])
        Xall = np.zeros((n_ch, N_STEPS, c["dim"]))
        for i in range(n_ch):
            func = IFCNumpy(c["ssid"], str(30 + i), dim=c["dim"])
            Xall[i] = rollout_record_x(d, func, c["dim"], ymax, ymin, N_STEPS)
        rad = radius_curve(Xall)
        ax.plot(np.arange(1, N_STEPS + 1), rad, color=c["color"], lw=2.6, label=c["label"])
        print(f"{c['label']}: radius @1/50/100/300 = "
              f"{rad[0]:.3f}/{rad[49]:.3f}/{rad[99]:.3f}/{rad[-1]:.3f}")
    ax.axhline(1 / np.sqrt(12), color="#888", ls="--", lw=1.2,
               label="uniform-random search (0.289)")
    ax.set_title("RIBBO exploration radius vs step (EE, 3BA, 20 channels)\n"
                 "std of proposed action within a 15-step window — collapse = radius -> 0",
                 fontsize=10)
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("exploration radius (mean per-dim std of proposed x)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(1, N_STEPS)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ribbo_collapse.pdf"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(OUT, "ribbo_collapse.pdf"))


if __name__ == "__main__":
    main()
