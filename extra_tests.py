"""Extra correctness checks beyond benchmark.py (not submitted).

    python extra_tests.py
"""
import math
import torch
from benchmark import build, linear_error

torch.manual_seed(0)
ext = build()


def offset_tensor(shape, offset):
    s = torch.randn(math.prod(shape) + offset, device="cuda")
    return s[offset:].view(shape)


fails = 0
linear_cases = [
    (1, 1, 1), (1, 7, 1), (127, 9, 129), (128, 8, 128), (129, 16, 127), (5, 4, 3),
    (300, 1, 300), (1, 4096, 1), (257, 1023, 255), (4096, 13, 17), (77, 0, 33),
    (512, 512, 512), (130, 260, 70),
]
for m, k, n in linear_cases:
    for ox, ow, ob in [(0, 0, 0), (1, 1, 1), (1, 0, 2), (0, 3, 0), (2, 2, 2)]:
        x, w, b = offset_tensor((m, k), ox), offset_tensor((n, k), ow), offset_tensor((n,), ob)
        out = ext.linear_forward(x, w, b)
        torch.cuda.synchronize()
        err, allowed = linear_error(out, x, w, b)
        if not err <= allowed:
            fails += 1
            print(f"linear WRONG M={m} K={k} N={n} offsets={ox, ow, ob}: err {err:.2e} > {allowed:.1e}")

# non-contiguous inputs
x = torch.randn(100, 70, device="cuda")[:, :64]
w = torch.randn(80, 64, device="cuda")[::2]
b = torch.randn(80, device="cuda")[::2]
err, allowed = linear_error(ext.linear_forward(x, w, b), x, w, b)
if not err <= allowed:
    fails += 1
    print("linear WRONG non-contiguous")

for size in [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 1023, 1024, 1025, 4097, 16383, 16385, (1 << 22) + 3]:
    for og, ox in [(0, 0), (1, 1), (2, 2), (3, 3), (1, 0), (0, 2), (3, 1)]:
        x, g = offset_tensor((size,), ox), offset_tensor((size,), og)
        x[::5] = 0.0
        out = ext.relu_backward(g, x)
        torch.cuda.synchronize()
        if not torch.equal(out, torch.where(x > 0, g, torch.zeros_like(g))):
            fails += 1
            print(f"relu WRONG size={size} offsets={og, ox}")

# multi-dimensional and empty
x, g = torch.randn(33, 17, 5, device="cuda"), torch.randn(33, 17, 5, device="cuda")
if not torch.equal(ext.relu_backward(g, x), torch.where(x > 0, g, torch.zeros_like(g))):
    fails += 1
    print("relu WRONG 3-D")
e = torch.empty(0, device="cuda")
assert ext.relu_backward(e, e).numel() == 0

torch.cuda.synchronize()
print("ALL EXTRA TESTS PASSED" if fails == 0 else f"{fails} FAILURES")
