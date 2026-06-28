"""EE sanity check: SE regression, multistart-vs-bruteforce, interior-vs-corner."""
import numpy as np
from problems.ifc import (IFCNumpy, ifc_rates, ifc_optimum_bruteforce,
                          ifc_optimum_multistart, wmmse_sum_rate)

np.set_printoptions(precision=4, suppress=True)


def old_se_scalar(H_sq, x, Ptx=10.0, sigma2=1.0):
    se = 0.0
    D = H_sq.shape[0]
    for d in range(D):
        signal = Ptx * H_sq[d, d] * x[d]
        interf = sigma2 + Ptx * sum(H_sq[dp, d] * x[dp] for dp in range(D) if dp != d)
        se += np.log(1.0 + signal / interf)
    return se


print("=" * 70)
print("1) SE regression: vectorized ifc_rates vs old scalar loop (D=3)")
rng = np.random.default_rng(123)
maxdiff = 0.0
for cid in range(5):
    f = IFCNumpy("IFCSE", cid, dim=3)
    X = rng.uniform(0, 1, (20, 3))
    y_vec = ifc_rates(f.H_sq, X, reward_type="SE").ravel()
    y_old = np.array([old_se_scalar(f.H_sq, x) for x in X])
    maxdiff = max(maxdiff, np.abs(y_vec - y_old).max())
print(f"   max |vectorized - scalar| over 100 points = {maxdiff:.2e}  ({'OK' if maxdiff < 1e-9 else 'MISMATCH'})")

print("\n2) WMMSE sanity still passes (SE, D=3, vs brute force 41^3), 10 channels")
gap = 0
for cid in range(10):
    f = IFCNumpy("IFCSE", cid, dim=3)
    wr, _ = wmmse_sum_rate(f.H_sq, Ptx=f.Ptx, n_restart=10)
    bv, _ = ifc_optimum_bruteforce(f.H_sq, reward_type="SE", grid=41)
    if wr < bv - 1e-3:
        gap += 1
print(f"   WMMSE >= bruteforce on {10 - gap}/10 channels (expect 10/10 up to grid res)")

print("\n3) EE multistart vs brute force (D=3, IFCEE), 8 channels")
print(f"   {'ch':>3} {'bruteEE':>9} {'multiEE':>9} {'diff':>9}  {'best_x (multistart)':>28}")
for cid in range(8):
    f = IFCNumpy("IFCEE", cid, dim=3)
    bv, bx = ifc_optimum_bruteforce(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", grid=51)
    mv, mx = ifc_optimum_multistart(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", n_restart=40)
    print(f"   {cid:>3} {bv:>9.4f} {mv:>9.4f} {mv-bv:>9.2e}  {np.array2string(mx, precision=3):>28}")

print("\n4) Interior vs corner: optimum x for SE vs EE (D=3, 6 channels)")
print("   (SE optima cluster at 0/1 corners; EE optima are interior)")
print(f"   {'ch':>3}  {'SE x*':>22}  {'EE x*':>22}")
for cid in range(6):
    f = IFCNumpy("IFCSE", cid, dim=3)
    _, sx = ifc_optimum_bruteforce(f.H_sq, reward_type="SE", grid=51)
    _, ex = ifc_optimum_bruteforce(f.H_sq, Pfix=1.0, reward_type="EE", grid=51)
    print(f"   {cid:>3}  {np.array2string(sx, precision=2):>22}  {np.array2string(ex, precision=2):>22}")

print("\n5) Random-action EE reward distribution (D=3, channel 0, 5000 samples)")
f = IFCNumpy("IFCEE", 0, dim=3)
X = np.random.default_rng(0).uniform(0, 1, (5000, 3))
y = ifc_rates(f.H_sq, X, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE").ravel()
bv, _ = ifc_optimum_bruteforce(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", grid=51)
print(f"   random EE: min={y.min():.3f} mean={y.mean():.3f} max={y.max():.3f} | optimum={bv:.3f}")
print(f"   random reaches {100*y.max()/bv:.1f}% of optimum with 5000 random samples")

print("\n6) EE D=10 multistart (IFCEED10), 3 channels (no brute force, sanity only)")
for cid in range(3):
    f = IFCNumpy("IFCEED10", cid, dim=10)
    mv, mx = ifc_optimum_multistart(f.H_sq, Ptx=f.Ptx, Pfix=f.Pfix, reward_type="EE", n_restart=40)
    n_interior = int(((mx > 0.02) & (mx < 0.98)).sum())
    print(f"   ch{cid}: EE*={mv:.4f}, interior coords={n_interior}/10, x*={np.array2string(mx, precision=2)}")
print("=" * 70)
