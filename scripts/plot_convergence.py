"""Plot x_loss convergence curve from TensorBoard logs."""
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth(values, weight=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)

def load_scalars(tb_path, tag):
    ea = EventAccumulator(tb_path)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb_path', default='./log/synthetic/dt-convergence-check/Rastrigin-seed0-05-18-10-28-26005/tb')
    parser.add_argument('--out', default='./presentation/convergence_rastrigin.pdf')
    parser.add_argument('--smooth_weight', type=float, default=0.85)
    args = parser.parse_args()

    steps, x_loss = load_scalars(args.tb_path, 'loss/x_loss')

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(steps, x_loss, color='#cccccc', linewidth=0.8, alpha=0.7, label='Raw')
    ax.plot(steps, smooth(x_loss, args.smooth_weight), color='#2563EB', linewidth=2.0, label='Smoothed (EMA)')

    ax.axhline(y=x_loss[-50:].mean(), color='#DC2626', linewidth=1.2, linestyle='--', label=f'Converged mean ({x_loss[-50:].mean():.1f})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('x loss (NLL)', fontsize=12)
    ax.set_title('RIBBO on BBOB Rastrigin — Training Convergence', fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(1, steps[-1])

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved: {args.out}')

    print(f'\nConvergence summary:')
    print(f'  Epoch 1:    {x_loss[0]:.2f}')
    for ep in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        idx = ep - 1
        if idx < len(x_loss):
            print(f'  Epoch {ep}: {x_loss[idx]:.2f}')

if __name__ == '__main__':
    main()
