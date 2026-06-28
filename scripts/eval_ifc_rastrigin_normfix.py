"""
Task #3 follow-up: is the failed Rastrigin->IFC D=10 zero-shot transfer caused by
RTG-normalization mismatch? Re-run the same Rastrigin-trained model but feed IFC's
OWN normalization stats (instead of Rastrigin's). If that rescues it, normalization
is the culprit; if not, the model didn't learn transferable optimization behavior.

Variants (IFC SE D=10, dimension-matched, N_STEPS=150 = Rastrigin pos-emb limit):
  A. Rastrigin model + Rastrigin RTG stats   (original mismatch, reproduces failure)
  B. Rastrigin model + IFC RTG stats          (normalization fix)
  C. Native IFC D=10 3BA model (reference, capped to 150 steps)
  D. Random
  + WMMSE band

Outputs:
  results/10_rastrigin_transfer_normfix/eval_rastrigin_normfix.pdf / .npz
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCNumpy, IFCMetaProblem, wmmse_sum_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RAST_CKPT = "./log/synthetic/dt-convergence-check/Rastrigin-seed0-05-18-10-28-26005/ckpt/1000.ckpt"
NATIVE_CKPT = "./log/ifc/dt-ifc-d10-run1/IFCSED10-seed0-05-21-10-34-541982/ckpt/3000.ckpt"
X_DIM = 10
N_CHANNELS = 150
N_STEPS = 150
INPUT_SEQ_LEN = 50
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8
OUT_DIR = "./results/10_rastrigin_transfer_normfix"


def build(ckpt, seq_len):
    tr = DecisionTransformer(
        x_dim=X_DIM, y_dim=1, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=seq_len, num_heads=NUM_HEADS, add_bos=True, mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1, pos_encoding="embed")
    d = DecisionTransformerDesigner(
        transformer=tr, x_dim=X_DIM, y_dim=1, embed_dim=EMBED_DIM, seq_len=seq_len,
        input_seq_len=INPUT_SEQ_LEN, x_type="stochastic", y_loss_coeff=0.0,
        use_abs_timestep=True, device=DEVICE)
    d.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    d.eval()
    return d


def rollout(designer, func, ymax, ymin, n_steps, seq_len, init_regret=0):
    B = 1
    designer.past_x = torch.zeros([B, n_steps + 1, X_DIM], dtype=torch.float).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps + 1, 1], dtype=torch.float).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    ts = torch.arange(n_steps + 1).clamp(max=seq_len - 1).long()
    designer.timesteps = ts.to(DEVICE).reshape(1, n_steps + 1).repeat(B, 1)
    designer.step_count = 0
    raw = np.zeros(n_steps)
    lx = ly = lr = None
    scale = max(ymax - ymin, 1e-6)
    for i in range(n_steps):
        with torch.no_grad():
            lx = designer.suggest(last_x=lx, last_y=ly, last_onestep_regret=lr,
                                  deterministic=False, regret_strategy="none")
        x_real = lx.cpu().numpy() / 2 + 0.5
        y = float(func(x_real)[0, 0])
        ly = torch.tensor([[(y - ymin) / scale]], dtype=torch.float).to(DEVICE)
        lr = torch.tensor([[(ymax - y) / scale]], dtype=torch.float).to(DEVICE)
        raw[i] = y
    return raw


def rand_rollout(func, n_steps, seed):
    rng = np.random.default_rng(seed)
    return func(rng.uniform(0, 1, (n_steps, X_DIM))).ravel()


def get_stats(ssid, data_dir, cache_dir):
    p = IFCMetaProblem(search_space_id=ssid, root_dir=None, data_dir=data_dir,
                       cache_dir=cache_dir, input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=300,
                       normalize_method="random", n_block=100)
    info = p.dataset.sp_id2info[ssid]
    return info["y_max_mean"], info["y_min_mean"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Rastrigin training stats (its own normalization range)
    from problems.synthetic import SyntheticMetaProblem
    rp = SyntheticMetaProblem(search_space_id="Rastrigin", root_dir=None,
                              data_dir="./data/generated_data/synthetic", cache_dir="./cache/synthetic/",
                              input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=150,
                              normalize_method="random", n_block=100)
    rmax, rmin = rp.dataset.sp_id2info["Rastrigin"]["y_max_mean"], rp.dataset.sp_id2info["Rastrigin"]["y_min_mean"]
    imax, imin = get_stats("IFCSED10", "./data/generated_data/ifc_d10", "./cache/ifc_d10/")
    print(f"Rastrigin stats: y_max={rmax:.3f} y_min={rmin:.3f}")
    print(f"IFC D=10 stats : y_max={imax:.3f} y_min={imin:.3f}")

    d_rast = build(RAST_CKPT, seq_len=150)
    d_nat = build(NATIVE_CKPT, seq_len=300)

    A = np.zeros((N_CHANNELS, N_STEPS))  # rast model + rast stats
    Bv = np.zeros((N_CHANNELS, N_STEPS))  # rast model + IFC stats (fix)
    C = np.zeros((N_CHANNELS, N_STEPS))  # native IFC D=10
    Dr = np.zeros((N_CHANNELS, N_STEPS))  # random
    wm = np.zeros(N_CHANNELS)
    for i in range(N_CHANNELS):
        cid = str(30 + i)
        func = IFCNumpy("IFCSED10", cid, dim=X_DIM)
        A[i] = rollout(d_rast, func, rmax, rmin, N_STEPS, 150)
        Bv[i] = rollout(d_rast, func, imax, imin, N_STEPS, 150)
        C[i] = rollout(d_nat, func, imax, imin, N_STEPS, 300)
        Dr[i] = rand_rollout(func, N_STEPS, seed=int(cid) + 1000)
        wm[i], _ = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{N_CHANNELS}]")

    np.savez(os.path.join(OUT_DIR, "eval_rastrigin_normfix.npz"),
             A=A, B=Bv, C=C, Dr=Dr, wm=wm)

    def bsf(a):
        return np.maximum.accumulate(a, axis=1)
    evals = np.arange(1, N_STEPS + 1)
    wmean, wstd = wm.mean(), wm.std()
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        (A, "#DC2626", "-", "Rastrigin model + Rastrigin RTG (mismatch)"),
        (Bv, "#2563EB", "-", "Rastrigin model + IFC RTG (norm. fix)"),
        (C, "#16A34A", "-", "Native IFC D=10 (3BA, reference)"),
        (Dr, "#9CA3AF", "--", "Random"),
    ]
    print("\n" + "=" * 64)
    print(f"{'Variant':<42}{'Final':>9}{'%WMMSE':>10}")
    for arr, col, ls, lab in series:
        b = bsf(arr).mean(0)
        ax.plot(evals, b, color=col, lw=2.2, ls=ls, label=lab)
        print(f"{lab:<42}{b[-1]:>9.3f}{100*b[-1]/wmean:>9.1f}%")
    ax.axhline(wmean, color="#000000", lw=1.6, label="WMMSE (mean)")
    ax.fill_between(evals, wmean - wstd, wmean + wstd, color="#000000", alpha=0.10)
    print(f"{'WMMSE':<42}{wmean:>9.3f}{100:>9.1f}%")
    print("=" * 64)
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Average Best SE (nats)")
    ax.set_title(f"Rastrigin->IFC D=10 transfer: does fixing RTG normalization help?\n"
                 f"{N_CHANNELS} unseen channels × {N_STEPS} steps", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_xlim(1, N_STEPS)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eval_rastrigin_normfix.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
