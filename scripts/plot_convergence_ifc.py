"""Plot IFC training convergence (x_loss) combining both runs."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN1_TB = './log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-11-31-94205/tb'
RUN2_TB = './log/ifc/dt-ifc-run1/IFCSE-seed0-05-18-12-50-15740/tb'
OUT     = './presentation/convergence_ifc.pdf'

def load(path, tag='loss/x_loss'):
    ea = EventAccumulator(path); ea.Reload()
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])

def smooth(v, w=0.9):
    s, last = [], v[0]
    for x in v:
        last = w*last + (1-w)*x; s.append(last)
    return np.array(s)

steps1, loss1 = load(RUN1_TB)
steps2, loss2 = load(RUN2_TB)

# run1: 1~1000, run2: 1001~3000
mask1 = steps1 <= 1000
steps = np.concatenate([steps1[mask1], steps2])
loss  = np.concatenate([loss1[mask1],  loss2])

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(steps, loss,         color='#d1d5db', lw=0.7, alpha=0.7, label='Raw')
ax.plot(steps, smooth(loss), color='#2563EB', lw=2.0,              label='Smoothed (EMA)')
ax.axhline(loss[-50:].mean(), color='#DC2626', lw=1.2, ls='--',
           label=f'Converged mean ({loss[-50:].mean():.1f})')

ax.axvline(1000, color='#9CA3AF', lw=1.0, ls=':')
ax.text(1005, ax.get_ylim()[0] if len(ax.get_lines()) > 0 else -30,
        'Resume\n(epoch 1000)', fontsize=8, color='#6B7280', va='bottom')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('x loss (NLL)', fontsize=12)
ax.set_title('RIBBO on IFC SE — Training Convergence (3000 epochs)', fontsize=13, pad=8)
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.grid(True, ls='--', alpha=0.4)
ax.set_xlim(1, steps[-1])

# fix resume annotation y position after ylim is set
ybot = ax.get_ylim()[0]
for txt in ax.texts:
    txt.set_y(ybot + 0.5)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved: {OUT}')

print('\nConvergence summary:')
print(f'  Epoch    1: {loss[0]:.2f}')
for ep in [100, 500, 1000, 1500, 2000, 2500, 3000]:
    idx = np.searchsorted(steps, ep)
    if idx < len(loss):
        print(f'  Epoch {ep:4d}: {loss[idx]:.2f}')
