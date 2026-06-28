"""Sanity check: WMMSE vs brute-force grid on IFC SE (D=3).

For 10 random Rayleigh channels, compute:
  (a) WMMSE rate (with multi-restart)
  (b) Brute-force best over 20^3 = 8000 grid points in [0,1]^3
  (c) Brute-force best over 40^3 = 64000 grid points (finer)

WMMSE should be >= grid best (or within discretization noise).
"""
import numpy as np
from problems.ifc import IFCNumpy, wmmse_sum_rate, _ifc_sum_rate


def brute_force_best(H_sq, Ptx=10.0, resolution=20):
    g = np.linspace(0, 1, resolution + 1)
    X = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)  # [R^3, 3]
    rates = np.array([_ifc_sum_rate(H_sq, x * Ptx) for x in X])
    best_idx = np.argmax(rates)
    return rates[best_idx], X[best_idx]


def main():
    print(f"{'cid':>4} | {'WMMSE':>9} | {'BF20':>9} {'gap':>7} | {'BF40':>9} {'gap':>7}")
    print("-" * 60)

    wmmse_rates, bf20_rates, bf40_rates = [], [], []
    for cid in range(100, 110):
        func = IFCNumpy("IFCSE", str(cid), dim=3)
        r_wmmse, x_wmmse = wmmse_sum_rate(func.H_sq, Ptx=func.Ptx, n_restart=10)
        r_bf20, _ = brute_force_best(func.H_sq, Ptx=func.Ptx, resolution=20)
        r_bf40, _ = brute_force_best(func.H_sq, Ptx=func.Ptx, resolution=40)
        gap20 = r_wmmse - r_bf20
        gap40 = r_wmmse - r_bf40
        print(f"{cid:>4} | {r_wmmse:>9.4f} | {r_bf20:>9.4f} {gap20:>+7.4f} | {r_bf40:>9.4f} {gap40:>+7.4f}")
        wmmse_rates.append(r_wmmse)
        bf20_rates.append(r_bf20)
        bf40_rates.append(r_bf40)

    print("-" * 60)
    print(f"{'mean':>4} | {np.mean(wmmse_rates):>9.4f} | "
          f"{np.mean(bf20_rates):>9.4f} {np.mean(wmmse_rates)-np.mean(bf20_rates):>+7.4f} | "
          f"{np.mean(bf40_rates):>9.4f} {np.mean(wmmse_rates)-np.mean(bf40_rates):>+7.4f}")

    n_above_bf40 = sum(w >= b for w, b in zip(wmmse_rates, bf40_rates))
    print(f"\nWMMSE >= BF40 in {n_above_bf40}/10 channels")
    print("Note: gap > 0 means WMMSE found higher rate; gap < 0 means grid found higher.")


if __name__ == "__main__":
    main()
