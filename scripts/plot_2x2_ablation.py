"""2x2 ablation summary: (BA pool: 3BA / 5BA-filter / 7BA) × (dim: 3 / 10).

Generates:
  results/07_ifc_5ba_filter_ablation/ablation_2x2_summary.pdf  (bar chart)
  results/07_ifc_5ba_filter_ablation/ablation_curves_combined.pdf  (4 panels)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR = "./results/07_ifc_5ba_filter_ablation"

# Load all eval data
d3 = np.load(f"{OUT_DIR}/eval_ifc_d3_3way.npz")
d10 = np.load(f"{OUT_DIR}/eval_ifc_d10_3way.npz")

def bsf_mean(raw):
    # raw: [N_CH, N_EP, N_STEPS]
    bsf = np.maximum.accumulate(raw.mean(axis=1), axis=1)
    return bsf  # [N_CH, N_STEPS]

bsf_d3_3   = bsf_mean(d3['raw_3ba']);   final_d3_3 = bsf_d3_3[:, -1].mean()
bsf_d3_5   = bsf_mean(d3['raw_5ba']);   final_d3_5 = bsf_d3_5[:, -1].mean()
bsf_d3_7   = bsf_mean(d3['raw_7ba']);   final_d3_7 = bsf_d3_7[:, -1].mean()
bsf_d3_r   = bsf_mean(d3['raw_rand']);  final_d3_r = bsf_d3_r[:, -1].mean()
wmm_d3     = d3['wmmse'].mean();        wmm_d3_std = d3['wmmse'].std()

bsf_d10_3  = bsf_mean(d10['raw_3ba']);  final_d10_3 = bsf_d10_3[:, -1].mean()
bsf_d10_5  = bsf_mean(d10['raw_5ba']);  final_d10_5 = bsf_d10_5[:, -1].mean()
bsf_d10_r  = bsf_mean(d10['raw_rand']); final_d10_r = bsf_d10_r[:, -1].mean()
wmm_d10    = d10['wmmse'].mean();       wmm_d10_std = d10['wmmse'].std()

# ── Bar chart summary ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['3BA', '5BA-filter', '7BA']
d3_vals = [100 * final_d3_3 / wmm_d3,
           100 * final_d3_5 / wmm_d3,
           100 * final_d3_7 / wmm_d3]
d10_vals = [100 * final_d10_3 / wmm_d10,
            100 * final_d10_5 / wmm_d10,
            np.nan]

x = np.arange(len(labels)); w = 0.35
b1 = ax.bar(x - w/2, d3_vals,  w, label='D=3',  color='#2563EB')
b2 = ax.bar(x + w/2, d10_vals, w, label='D=10', color='#16A34A')

# Random baselines (horizontal lines per dim)
ax.axhline(100 * final_d3_r / wmm_d3,  color='#2563EB', ls=':', alpha=0.6, label='Random (D=3)')
ax.axhline(100 * final_d10_r / wmm_d10, color='#16A34A', ls=':', alpha=0.6, label='Random (D=10)')
ax.axhline(100, color='black', lw=1.2, ls='--', alpha=0.7, label='WMMSE')

for i, v in enumerate(d3_vals):
    ax.text(x[i] - w/2, v + 1.5, f"{v:.1f}%", ha='center', fontsize=9, color='#1E40AF')
for i, v in enumerate(d10_vals):
    if not np.isnan(v):
        ax.text(x[i] + w/2, v + 1.5, f"{v:.1f}%", ha='center', fontsize=9, color='#15803D')
    else:
        ax.text(x[i] + w/2, 5, "n/a", ha='center', fontsize=9, color='#9CA3AF')

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Final Best SE  /  WMMSE  (%)', fontsize=12)
ax.set_title('BA-pool × Dimension ablation — IFC SE\n3BA = {Random, HC, CMA}  •  5BA-filter drops Random+SGS  •  7BA = all', fontsize=10)
ax.set_ylim(0, 110)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.grid(True, axis='y', ls='--', alpha=0.3)
ax.legend(fontsize=9, loc='lower left', ncol=2)
fig.tight_layout()
out_bar = f"{OUT_DIR}/ablation_2x2_summary.pdf"
fig.savefig(out_bar, dpi=150, bbox_inches='tight')
print(f"Saved: {out_bar}")

# ── 4-panel curves (cleaner than the previous overlay plots) ─────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

def panel(ax, curves, wmm, wmm_std, n_steps, title, ylim_top=None):
    steps = np.arange(1, n_steps + 1)
    for name, arr, color, ls in curves:
        ax.plot(steps, arr.mean(0), color=color, lw=2.2, ls=ls, label=name)
    ax.axhline(wmm, color='black', lw=1.5, label='WMMSE (mean)')
    ax.fill_between(steps, wmm - wmm_std, wmm + wmm_std, color='black', alpha=0.10, label='WMMSE ±1σ')
    ax.set_xlabel('Function Evaluations', fontsize=11)
    ax.set_ylabel('Average Best SE (nats)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(1, n_steps)
    ax.set_ylim(bottom=0, top=ylim_top)
    ax.grid(True, ls='--', alpha=0.3)

panel(axes[0],
      [('RIBBO 3BA (3000ep)', bsf_d3_3, '#2563EB', '-'),
       ('RIBBO 5BA-filter (5000ep)', bsf_d3_5, '#16A34A', '-'),
       ('RIBBO 7BA (5000ep)', bsf_d3_7, '#DC2626', '-'),
       ('Random', bsf_d3_r, '#9CA3AF', '--')],
      wmm_d3, wmm_d3_std, bsf_d3_3.shape[1], 'IFC SE  D=3  (500 ch × 1 ep × 300 step)')

panel(axes[1],
      [('RIBBO 3BA (3000ep)', bsf_d10_3, '#2563EB', '-'),
       ('RIBBO 5BA-filter (5000ep)', bsf_d10_5, '#16A34A', '-'),
       ('Random', bsf_d10_r, '#9CA3AF', '--')],
      wmm_d10, wmm_d10_std, bsf_d10_3.shape[1], 'IFC SE  D=10  (200 ch × 1 ep × 300 step)')

fig.tight_layout()
out_curves = f"{OUT_DIR}/ablation_curves_combined.pdf"
fig.savefig(out_curves, dpi=150, bbox_inches='tight')
print(f"Saved: {out_curves}")

# ── Print final numbers table ────────────────────────────────────
print("\n" + "="*78)
print(f"{'Setup':<40} {'Final (nats)':<14} {'% WMMSE':<10}")
print("-"*78)
print(f"{'WMMSE D=3 (mean)':<40} {wmm_d3:<14.4f} 100.0")
print(f"{'RIBBO 3BA  D=3':<40} {final_d3_3:<14.4f} {100*final_d3_3/wmm_d3:.1f}")
print(f"{'RIBBO 5BA-filter D=3':<40} {final_d3_5:<14.4f} {100*final_d3_5/wmm_d3:.1f}")
print(f"{'RIBBO 7BA  D=3':<40} {final_d3_7:<14.4f} {100*final_d3_7/wmm_d3:.1f}")
print(f"{'Random  D=3':<40} {final_d3_r:<14.4f} {100*final_d3_r/wmm_d3:.1f}")
print("-"*78)
print(f"{'WMMSE D=10 (mean)':<40} {wmm_d10:<14.4f} 100.0")
print(f"{'RIBBO 3BA  D=10':<40} {final_d10_3:<14.4f} {100*final_d10_3/wmm_d10:.1f}")
print(f"{'RIBBO 5BA-filter D=10':<40} {final_d10_5:<14.4f} {100*final_d10_5/wmm_d10:.1f}")
print(f"{'Random  D=10':<40} {final_d10_r:<14.4f} {100*final_d10_r/wmm_d10:.1f}")
print("="*78)
