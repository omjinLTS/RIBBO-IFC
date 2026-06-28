"""Print loss/x_loss convergence summary from a run's tensorboard events.
Usage: python scripts/tb_loss.py <run_dir>"""
import sys
import glob
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def main(run_dir):
    ev = glob.glob(os.path.join(run_dir, "*", "tb", "events.*"))
    if not ev:
        ev = glob.glob(os.path.join(run_dir, "tb", "events.*"))
    if not ev:
        print(f"no events under {run_dir}")
        return
    ev = max(ev, key=os.path.getmtime)
    acc = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
    acc.Reload()
    tags = [t for t in acc.Tags()["scalars"] if "x_loss" in t]
    if not tags:
        print("tags:", acc.Tags()["scalars"][:20])
        return
    tag = tags[0]
    s = acc.Scalars(tag)
    steps = np.array([x.step for x in s])
    vals = np.array([x.value for x in s])
    print(f"event file: {ev}")
    print(f"tag: {tag}, n points: {len(vals)}, last epoch: {steps[-1]}")
    for q in [1, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
        idx = np.where(steps <= q)[0]
        if len(idx):
            print(f"  epoch <= {q:>5}: x_loss = {vals[idx[-1]]:>9.4f}")
    tail = vals[steps >= steps[-1] - 500]
    if len(tail) > 1:
        slope = np.polyfit(steps[steps >= steps[-1] - 500], tail, 1)[0]
        print(f"  last-500-epoch mean = {tail.mean():.4f}, slope = {slope:.6f}/ep")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
