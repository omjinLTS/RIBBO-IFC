"""Evaluate trained RIBBO on IFC SE test channels and compare with Random baseline."""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import sqrt

# ── model imports ────────────────────────────────────────────────────────────
from algorithms.designers.dt_designer import (
    DecisionTransformerDesigner,
    evaluate_decision_transformer_designer,
)
from algorithms.modules.dt import DecisionTransformer
from problems.ifc import IFCMetaProblem, IFCNumpy

# ── config ───────────────────────────────────────────────────────────────────
CKPT = "./log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-12-50-15740/ckpt/3000.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TEST_DATASETS  = [str(i) for i in range(20, 70)]   # 50 unseen channels (Lee+25: many random realizations)
TRAIN_DATASETS = [str(i) for i in range(15)]        # 15 train channels

EVAL_EPISODES     = 3  # 50 channels × 3 episodes는 충분한 통계
INIT_REGRET       = 0
REGRET_STRATEGY   = "none"
DETERMINISTIC_EVAL = False

# model arch (must match training)
X_DIM = 3; Y_DIM = 1
EMBED_DIM = 256; NUM_LAYERS = 12; NUM_HEADS = 8
INPUT_SEQ_LEN = 50; MAX_INPUT_SEQ_LEN = 300


# ── helpers ──────────────────────────────────────────────────────────────────

def build_model():
    transformer = DecisionTransformer(
        x_dim=X_DIM, y_dim=Y_DIM,
        embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
        seq_len=300,          # dataset seq_len
        num_heads=NUM_HEADS,
        add_bos=True,
        mix_method="concat",
        attention_dropout=0.1, residual_dropout=0.1, embed_dropout=0.1,
        pos_encoding="embed",
    )
    designer = DecisionTransformerDesigner(
        transformer=transformer,
        x_dim=X_DIM, y_dim=Y_DIM,
        embed_dim=EMBED_DIM, seq_len=300,
        input_seq_len=INPUT_SEQ_LEN,
        x_type="stochastic",
        y_loss_coeff=0.0,
        use_abs_timestep=True,
        device=DEVICE,
    )
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    designer.load_state_dict(ckpt)
    designer.eval()
    print(f"Loaded checkpoint: {CKPT}")
    return designer


def build_problem():
    return IFCMetaProblem(
        search_space_id="IFCSE",
        root_dir=None,
        data_dir="./data/generated_data/ifc",
        cache_dir="./cache/ifc/",
        input_seq_len=INPUT_SEQ_LEN,
        max_input_seq_len=MAX_INPUT_SEQ_LEN,
        normalize_method="random",
        n_block=100,
    )


def random_baseline(channel_ids, seq_len=300, n_episodes=5, seed=42):
    """Pure random search: uniform x ~ [0,1]^3 per step."""
    rng = np.random.default_rng(seed)
    results = {}   # id -> [n_episodes, seq_len]
    for cid in channel_ids:
        func = IFCNumpy("IFCSE", cid, dim=3, lb=0.0, ub=1.0)
        ep_ys = []
        for _ in range(n_episodes):
            xs = rng.uniform(0, 1, size=(seq_len, 3))
            ys = np.array([func(xs[[i], :]).item() for i in range(seq_len)])
            ep_ys.append(ys)
        results[cid] = np.array(ep_ys)   # [n_ep, seq_len]
    return results


def best_so_far(y_arr):
    """y_arr: [n_ep, seq_len] → [seq_len] cumulative best mean over episodes."""
    cum_best = np.maximum.accumulate(y_arr, axis=1)   # [n_ep, seq_len]
    return cum_best.mean(axis=0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    problem = build_problem()
    designer = build_model()

    seq_len = problem.seq_len   # 300

    print(f"\n=== Evaluating RIBBO on {len(TEST_DATASETS)} test channels ===")
    _, record = evaluate_decision_transformer_designer(
        problem, designer, TEST_DATASETS,
        EVAL_EPISODES, "stochastic" if not DETERMINISTIC_EVAL else "deterministic",
        INIT_REGRET, REGRET_STRATEGY,
    )

    # raw r_SE [n_ep, seq_len] per channel
    ribbo_bsf = []
    ribbo_inst = []
    for cid in TEST_DATASETS:
        raw_y = record[cid]['y']          # [n_ep, seq_len]
        ribbo_bsf.append(best_so_far(raw_y))
        ribbo_inst.append(raw_y.mean(axis=0))
    ribbo_bsf = np.stack(ribbo_bsf)       # [n_ch, seq_len]
    ribbo_inst = np.stack(ribbo_inst)

    print(f"\n=== Random baseline on {len(TEST_DATASETS)} test channels ===")
    rand_record = random_baseline(TEST_DATASETS, seq_len, EVAL_EPISODES)
    rand_bsf = np.stack([best_so_far(rand_record[cid]) for cid in TEST_DATASETS])
    rand_inst = np.stack([rand_record[cid].mean(axis=0) for cid in TEST_DATASETS])

    # ── print final numbers ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Method':<18} {'Final r_SE (nats)':<20} {'Step 50':<12}")
    print("-"*55)
    r_ribbo_final = ribbo_bsf[:, -1].mean()
    r_ribbo_s50   = ribbo_bsf[:, 49].mean()
    r_rand_final  = rand_bsf[:, -1].mean()
    r_rand_s50    = rand_bsf[:, 49].mean()
    print(f"{'RIBBO (ours)':<18} {r_ribbo_final:<20.4f} {r_ribbo_s50:<12.4f}")
    print(f"{'Random':<18} {r_rand_final:<20.4f} {r_rand_s50:<12.4f}")
    print("="*55)

    print(f"\n[Note] Values in nats (ln). Divide by ln2={np.log(2):.4f} for bits.")
    print(f"  RIBBO final (bits): {r_ribbo_final/np.log(2):.4f}")
    print(f"  Random final (bits): {r_rand_final/np.log(2):.4f}")

    # ── per-channel breakdown ────────────────────────────────────────────────
    print("\nPer-channel final r_SE (nats) — RIBBO vs Random:")
    print(f"  {'ch':<6} {'RIBBO':>10} {'Random':>10}")
    for i, cid in enumerate(TEST_DATASETS):
        print(f"  {cid:<6} {ribbo_bsf[i,-1]:>10.4f} {rand_bsf[i,-1]:>10.4f}")

    # ── plot ─────────────────────────────────────────────────────────────────
    # Lee et al. (2025) Fig. 7(b) reference values (read from graph, L=1, IFC SE)
    # Their x-axis = iterations, P=5 actions/iter → our step 300 ≈ their iter 60
    ref = {
        'GNN [Lee+25]':          2.70,
        'WMMSE (local opt) [Lee+25]': 2.50,
        'LLMO-E [Lee+25]':       2.50,
        'GA [Lee+25]':           2.40,
        'BO [Lee+25]':           2.20,
        'Brute-force [Lee+25]':  1.50,
    }
    ref_colors = {
        'GNN [Lee+25]':               '#15803D',
        'WMMSE (local opt) [Lee+25]': '#000000',
        'LLMO-E [Lee+25]':            '#DC2626',
        'GA [Lee+25]':                '#D97706',
        'BO [Lee+25]':                '#7C3AED',
        'Brute-force [Lee+25]':       '#6B7280',
    }

    steps = np.arange(1, seq_len + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    # our methods
    ax.plot(steps, ribbo_bsf.mean(0), color='#2563EB', lw=2.5, label='RIBBO best-so-far', zorder=5)
    ax.fill_between(steps,
                    ribbo_bsf.mean(0) - ribbo_bsf.std(0),
                    ribbo_bsf.mean(0) + ribbo_bsf.std(0),
                    color='#2563EB', alpha=0.15)
    ax.plot(steps, rand_bsf.mean(0), color='#93C5FD', lw=1.8, linestyle='-.', label='Random best-so-far', zorder=4)
    ax.fill_between(steps,
                    rand_bsf.mean(0) - rand_bsf.std(0),
                    rand_bsf.mean(0) + rand_bsf.std(0),
                    color='#93C5FD', alpha=0.12)
    # instantaneous (current-step) reward overlay
    ax.plot(steps, ribbo_inst.mean(0), color='#2563EB', lw=0.9, linestyle=':', alpha=0.6, label='RIBBO current', zorder=3)
    ax.plot(steps, rand_inst.mean(0),  color='#93C5FD', lw=0.9, linestyle=':', alpha=0.6, label='Random current', zorder=2)

    # Lee et al. reference lines
    for label, val in ref.items():
        ax.axhline(val, color=ref_colors[label], lw=1.2, linestyle='--', alpha=0.75, label=label)

    ax.set_xlabel('Step (function evaluations)', fontsize=12)
    ax.set_ylabel('Best r_SE found (nats)', fontsize=12)
    ax.set_title('IFC SE Power Control — RIBBO vs Baselines\n'
                 '(10 unseen channels, avg ± std;  dashed = Lee et al. 2025 Fig.7b asymptotic values)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(1, seq_len)

    # annotation: step 300 ≈ Lee et al. iter 60
    ax.axvline(300, color='gray', lw=0.8, linestyle=':')
    ax.text(295, ax.get_ylim()[0] + 0.05, '≈iter 60\n(Lee+25)', fontsize=8,
            ha='right', color='gray')

    out = './presentation/eval_ifc_ribbo_vs_baselines_with_instant.pdf'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
