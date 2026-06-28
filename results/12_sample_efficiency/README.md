# (A) Sample-efficiency: function evals to reach a target % of optimum

Average best-so-far curve crossing the target. 'inf' = never within budget.


## SE D=3 (3BA, 1000)  (optimum mean=2.8245, horizon=1000)
  target   RIBBO evals  Random evals   speedup
     50%             1             2      2.0x
     60%             2             6      3.0x
     70%             3            27      9.0x
     80%            94           171      1.8x
     90%           486           infRIBBO only
  final: RIBBO 91.7% of opt, Random 86.7%

## SE D=10 (5BA-filter, 1000)  (optimum mean=3.8866, horizon=1000)
  target   RIBBO evals  Random evals   speedup
     50%            60           infRIBBO only
     60%           109           infRIBBO only
     70%           239           infRIBBO only
     80%           815           infRIBBO only
     90%           inf           inf   neither
  final: RIBBO 81.0% of opt, Random 46.9%

## EE D=10 (3BA, 300)  (optimum mean=1.0067, horizon=300)
  target   RIBBO evals  Random evals   speedup
     50%            47           infRIBBO only
     60%            55           infRIBBO only
     70%            82           infRIBBO only
     80%           inf           inf   neither
     90%           inf           inf   neither
  final: RIBBO 78.1% of opt, Random 29.4%

## SE D=20 (4BA-filter, 300)  (optimum mean=4.7640, horizon=300)
  target   RIBBO evals  Random evals   speedup
     50%           inf           inf   neither
     60%           inf           inf   neither
     70%           inf           inf   neither
     80%           inf           inf   neither
     90%           inf           inf   neither
  final: RIBBO 41.9% of opt, Random 31.5%
