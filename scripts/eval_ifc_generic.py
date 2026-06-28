"""
Generic IFC evaluator for any (dimension, pool, reward). RIBBO vs Random vs reference.
Reference: SE -> WMMSE (Shi 2011); EE -> brute-force grid (dim<=3) or multistart gradient.

Example:
  python scripts/eval_ifc_generic.py --ssid IFCSED20 --dim 20 --reward SE \
     --data_dir ./data/generated_data/ifc_d20_5bafilter --cache_dir ./cache/ifc_d20_5bafilter/ \
     --run ./log/ifc/dt-ifc-d20-5ba --n 200 \
     --out results/11_ifc_d20_scalability/eval_ifc_d20_5ba --title "IFC SE D=20 5BA-filter"
"""
import os, glob, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from algorithms.designers.dt_designer import DecisionTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import (IFCNumpy, IFCMetaProblem, wmmse_sum_rate,
                          ifc_optimum_bruteforce, ifc_optimum_multistart)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
INPUT_SEQ_LEN = 50
EMBED_DIM, NUM_LAYERS, NUM_HEADS = 256, 12, 8
N_STEPS = 300


def latest_ckpt(run_dir):
    cks = glob.glob(os.path.join(run_dir, "*", "ckpt", "*.ckpt"))
    return max(cks, key=lambda p: int(os.path.basename(p).split(".")[0]))


def build(ckpt, x_dim):
    tr = DecisionTransformer(x_dim=x_dim, y_dim=1, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=300, num_heads=NUM_HEADS, add_bos=True, mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1, pos_encoding="embed")
    d = DecisionTransformerDesigner(transformer=tr, x_dim=x_dim, y_dim=1, embed_dim=EMBED_DIM,
        seq_len=300, input_seq_len=INPUT_SEQ_LEN, x_type="stochastic", y_loss_coeff=0.0,
        use_abs_timestep=True, device=DEVICE)
    d.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    d.eval(); return d


def rollout(designer, func, x_dim, ymax, ymin, n_steps, init_regret=0):
    B = 1
    designer.past_x = torch.zeros([B, n_steps+1, x_dim]).to(DEVICE)
    designer.past_y = torch.zeros([B, n_steps+1, 1]).to(DEVICE)
    designer.past_regrets = torch.zeros([B, n_steps+1, 1]).to(DEVICE)
    designer.past_regrets[:, 0] = init_regret
    designer.timesteps = torch.arange(n_steps+1).clamp(max=299).long().to(DEVICE).reshape(1, n_steps+1)
    designer.step_count = 0
    raw = np.zeros(n_steps); lx = ly = lr = None
    scale = max(ymax - ymin, 1e-6)
    for i in range(n_steps):
        with torch.no_grad():
            lx = designer.suggest(last_x=lx, last_y=ly, last_onestep_regret=lr,
                                  deterministic=False, regret_strategy="none")
        y = float(func(lx.cpu().numpy()/2+0.5)[0, 0])
        ly = torch.tensor([[(y-ymin)/scale]]).to(DEVICE)
        lr = torch.tensor([[(ymax-y)/scale]]).to(DEVICE)
        raw[i] = y
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssid", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--reward", choices=["SE", "EE"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    ck = latest_ckpt(a.run); print("ckpt:", ck)
    prob = IFCMetaProblem(search_space_id=a.ssid, root_dir=None, data_dir=a.data_dir,
        cache_dir=a.cache_dir, input_seq_len=INPUT_SEQ_LEN, max_input_seq_len=300,
        normalize_method="random", n_block=100)
    info = prob.dataset.sp_id2info[a.ssid]
    ymax, ymin = info["y_max_mean"], info["y_min_mean"]
    print(f"stats y_max={ymax:.4f} y_min={ymin:.4f}")
    d = build(ck, a.dim)

    rb = np.zeros((a.n, N_STEPS)); rd = np.zeros((a.n, N_STEPS)); ref = np.zeros(a.n)
    for i in range(a.n):
        cid = str(30+i)
        f = IFCNumpy(a.ssid, cid, dim=a.dim)
        rb[i] = rollout(d, f, a.dim, ymax, ymin, N_STEPS)
        rng = np.random.default_rng(int(cid)+1000)
        rd[i] = f(rng.uniform(0, 1, (N_STEPS, a.dim))).ravel()
        if a.reward == "SE":
            ref[i], _ = wmmse_sum_rate(f.H_sq, Ptx=f.Ptx, n_restart=10)
        elif a.dim <= 3:
            ref[i], _ = ifc_optimum_bruteforce(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", grid=51)
        else:
            ref[i], _ = ifc_optimum_multistart(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", n_restart=40)
        if (i+1) % 25 == 0: print(f"  [{i+1}/{a.n}]")

    np.savez(a.out + ".npz", rb=rb, rd=rd, ref=ref, ckpt=ck)
    brb = np.maximum.accumulate(rb, 1); brd = np.maximum.accumulate(rd, 1)
    rm = ref.mean(); refname = "WMMSE" if a.reward == "SE" else "optimum"
    prb, prd = 100*brb[:, -1].mean()/rm, 100*brd[:, -1].mean()/rm

    evals = np.arange(1, N_STEPS+1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evals, brb.mean(0), color="#16A34A", lw=2.5, label=f"RIBBO ({prb:.0f}% of {refname})")
    ax.plot(evals, brd.mean(0), color="#9CA3AF", lw=1.8, ls="--", label=f"Random ({prd:.0f}%)")
    ax.axhline(rm, color="#000", lw=1.8, label=f"{refname} (mean)")
    ax.fill_between(evals, rm-ref.std(), rm+ref.std(), color="#000", alpha=0.10)
    ax.set_xlabel("Function Evaluations"); ax.set_ylabel(f"Average Best {a.reward}")
    ax.set_title(f"{a.title}\n{a.n} unseen channels × {N_STEPS} steps", fontsize=10)
    ax.legend(fontsize=9, loc="lower right"); ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(1, N_STEPS); ax.set_ylim(bottom=0); ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out + ".pdf", dpi=150, bbox_inches="tight")
    print("="*60)
    print(f"{a.title}")
    print(f"  RIBBO   {brb[:,-1].mean():.4f}  ({prb:.1f}% of {refname})")
    print(f"  Random  {brd[:,-1].mean():.4f}  ({prd:.1f}%)")
    print(f"  {refname}  {rm:.4f}")
    print(f"saved {a.out}.pdf/.npz")
    print("="*60)


if __name__ == "__main__":
    main()
