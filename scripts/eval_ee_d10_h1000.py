"""
Fair extended-budget eval: RIBBO retrained at HORIZON 1000 (EE D=10, 3BA) vs
the horizon-300 RIBBO, CMA-ES@1000, DE@1000, Random@1000 — on the SAME 100 unseen
channels (cid 30+) and SAME optimum used in results/19.

Key: this model was trained with seq_len=1000 (timestep embeddings 0..999), so we
build the designer with seq_len=1000 and DO NOT clamp timesteps at 299.

Outputs: results/19_budget_baselines/ee_d10_h1000.{pdf,png,npz}
"""
import os
import numpy as np
import torch

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCNumpy, IFCMetaProblem
from scripts.eval_ifc_ee import latest_ckpt


def stats_at(data_dir, cache_dir, ssid, max_input_seq_len):
    """RTG normalization stats computed at the SAME horizon the model trained on.
    The shared eval_ifc_ee.stats() hardcodes max_input_seq_len=300, which mismatches
    this horizon-1000 model (trained with max_input_seq_len=1000) and corrupts the
    regret signal fed during rollout. Here we match the training horizon."""
    prob = IFCMetaProblem(search_space_id=ssid, root_dir=None, data_dir=data_dir,
                          cache_dir=cache_dir, input_seq_len=INPUT_SEQ_LEN,
                          max_input_seq_len=max_input_seq_len,
                          normalize_method="random", n_block=100)
    info = prob.dataset.sp_id2info[ssid]
    return info["y_max_mean"], info["y_min_mean"]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8
INPUT_SEQ_LEN = 50
OUT = "results/19_budget_baselines"
N_STEPS = 1000
SEQ_LEN = 1000           # model horizon (must match training max_input_seq_len)


def build_designer(ckpt_path, x_dim, seq_len):
    transformer = DecisionTransformer(
        x_dim=x_dim, y_dim=1, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=seq_len, num_heads=NUM_HEADS, add_bos=True, mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1,
        pos_encoding="embed",
    )
    d = DecisionTransformerDesigner(
        transformer=transformer, x_dim=x_dim, y_dim=1, embed_dim=EMBED_DIM,
        seq_len=seq_len, input_seq_len=INPUT_SEQ_LEN, x_type="stochastic",
        y_loss_coeff=0.0, use_abs_timestep=True, device=DEVICE,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    d.load_state_dict(ckpt)
    d.eval()
    return d


def rollout_ribbo(designer, func, x_dim, ymax, ymin, n_steps, seq_len):
    B = 1
    designer.past_x = torch.zeros([B, n_steps + 1, x_dim], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = 0
    ts = torch.arange(n_steps + 1).clamp(max=seq_len - 1).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps + 1).repeat(B, 1)
    designer.step_count = 0
    scale = max(ymax - ymin, 1e-6)
    raw = np.zeros(n_steps)
    last_x = last_y = last_regret = None
    for i in range(n_steps):
        with torch.no_grad():
            last_x = designer.suggest(last_x=last_x, last_y=last_y,
                                      last_onestep_regret=last_regret,
                                      deterministic=False, regret_strategy="none")
        x_real = last_x.cpu().numpy() / 2 + 0.5
        y = float(func(x_real)[0, 0])
        last_y = torch.tensor([[(y - ymin) / scale]], dtype=torch.float).to(DEVICE)
        last_regret = torch.tensor([[(ymax - y) / scale]], dtype=torch.float).to(DEVICE)
        raw[i] = y
    return raw


def main():
    run = "./log/ifc/dt-ifc-ee-d10-3ba-h1000"
    ck = latest_ckpt(run)
    print("h1000 ckpt:", ck)
    # Match the training horizon (max_input_seq_len=1000) for RTG normalization.
    ymax, ymin = stats_at("./data/generated_data/ifc_d10_ee_h1000",
                          "./cache/ifc_d10_ee_h1000/", "IFCEED10",
                          max_input_seq_len=SEQ_LEN)
    print(f"  RTG norm (horizon={SEQ_LEN}): y_max={ymax:.4f} y_min={ymin:.4f}")
    d = build_designer(ck, 10, SEQ_LEN)

    bud = np.load("results/09_ifc_ee/eval_ifc_ee_budget.npz", allow_pickle=True)
    opt = bud["o10"][:100]
    rb_h300 = np.maximum.accumulate(bud["rb10"][:100], 1)             # horizon-300 RIBBO
    n = 100

    rb_h1000 = np.zeros((n, N_STEPS))
    for i in range(n):
        func = IFCNumpy("IFCEED10", str(30 + i), dim=10)
        rb_h1000[i] = rollout_ribbo(d, func, 10, ymax, ymin, N_STEPS, SEQ_LEN)
        if (i + 1) % 20 == 0:
            print(f"  RIBBO-h1000 [{i+1}/{n}]")
    rb_h1000_bsf = np.maximum.accumulate(rb_h1000, 1)

    cache = "results/19_budget_baselines/cache"
    cm = np.maximum.accumulate(np.load(f"{cache}/ee_d10_cmaes_n100_b1000.npy"), 1)
    de = np.maximum.accumulate(np.load(f"{cache}/ee_d10_de_n100_b1000.npy"), 1)
    rd = np.maximum.accumulate(np.load(f"{cache}/ee_d10_rand_n100_b1000.npy"), 1)

    def g(b):
        return (100 * (opt[:, None] - b) / opt[:, None]).mean(0)
    curves = {"RIBBO-h1000": (g(rb_h1000_bsf), "#7C3AED", "-"),
              "RIBBO-h300": (g(rb_h300), "#C4B5FD", "--"),
              "CMA-ES": (g(cm), "#059669", "-"),
              "DE": (g(de), "#D97706", "-"),
              "Random": (g(rd), "#9CA3AF", "--")}

    np.savez(os.path.join(OUT, "ee_d10_h1000.npz"),
             rb_h1000=rb_h1000, opt=opt, ckpt=ck)

    print("\nEE D=10 final gap%% (@300 -> @1000):")
    for k, (gc, _, _) in curves.items():
        print(f"  {k:12s}: {gc[299]:5.1f} -> {gc[-1]:5.1f}")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    ev = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k, (gc, col, ls) in curves.items():
        lw = 2.8 if k == "RIBBO-h1000" else (1.8 if "h300" in k else 2.0)
        ax.plot(ev, gc, color=col, ls=ls, lw=lw, label=f"{k} ({gc[-1]:.1f}%)")
    ax.axvline(300, color="#888", ls=":", lw=1, alpha=.7)
    ax.text(305, ax.get_ylim()[1] * 0.96, "old budget / old horizon (300)", fontsize=8, color="#666")
    ax.set_title("IFC EE D=10 (100 ch) — RIBBO retrained at horizon 1000 (fair) vs CMA-ES@1000",
                 fontsize=10)
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("optimality gap (%)  [lower = better]")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(1, N_STEPS); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ee_d10_h1000.pdf"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(OUT, "ee_d10_h1000.pdf"))


if __name__ == "__main__":
    main()
