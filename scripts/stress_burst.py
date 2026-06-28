"""One GPU stress burst in a FRESH process: create CUDA context -> sustained matmul
load -> memory alloc/free churn -> exit (destroy context). Run repeatedly to compress
the conditions that triggered the freezes (context create/destroy + load + idle)."""
import sys
import time
import torch

d = "cuda:0"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

a = torch.randn(n, n, device=d)
b = torch.randn(n, n, device=d)
t = time.time()
it = 0
while time.time() - t < dur:
    c = a @ b
    a = torch.tanh(c) * 1e-3 + a * 0.999
    it += 1
torch.cuda.synchronize()

# memory alloc/free churn (allocator + GSP-less path stress)
for _ in range(4):
    x = torch.randn(8192, 8192, device=d)
    y = x @ x
    del x, y
    torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"ok iters={it} checksum={float(a.float().abs().mean()):.5f}", flush=True)
