# (C) 95% CIs on % of optimum (from existing per-channel npz)

mean +/- 1.96*sd/sqrt(n). win-rate = fraction of channels RIBBO_final > Random_final.

SE D=3 (3BA)               n=500  RIBBO  89.3% +/- 1.3   Random  82.2% +/- 1.0   win-rate  73.6%
SE D=10 (5BA-filter)       n=200  RIBBO  71.6% +/- 1.5   Random  43.3% +/- 1.2   win-rate 100.0%
SE D=10 (3BA)              n=200  RIBBO  47.8% +/- 1.8   Random  43.3% +/- 1.2   win-rate  64.0%
EE D=3 (3BA)               n=200  RIBBO  96.8% +/- 0.5   Random  82.4% +/- 1.7   win-rate  88.5%
EE D=10 (3BA)              n=200  RIBBO  78.5% +/- 1.0   Random  29.4% +/- 0.7   win-rate 100.0%
SE D=20 (4BA-filter)       n=200  RIBBO  42.0% +/- 1.0   Random  31.5% +/- 0.7   win-rate  99.5%
EE D=20 (3BA)              n=200  RIBBO  30.2% +/- 0.8   Random  17.2% +/- 0.3   win-rate 100.0%
SE D=3 @1000 (3BA)         n=100  RIBBO  92.2% +/- 2.6   Random  86.4% +/- 1.8   win-rate  78.0%
SE D=10 @1000 (5BA)        n=100  RIBBO  81.1% +/- 1.8   Random  46.5% +/- 1.8   win-rate 100.0%
