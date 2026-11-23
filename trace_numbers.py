"""Compute the report numbers from trace_naive.json and trace.json (not submitted).

    python trace_numbers.py            # runs anywhere, no GPU needed

For each label it averages (over the 3 calls) the GPU kernel time inside that
label's GPU annotation. Cross-check a few against Perfetto, as the report asks.
"""
import json
import sys

LABELS = ["student_linear_4096", "cublas_linear_4096", "student_relu_backward", "pytorch_relu_backward"]
CALLS = 3


def per_call_ms(path):
    events = json.load(open(path))["traceEvents"]
    events = [e for e in events if e.get("ph") == "X"]
    ann = {e["name"]: e for e in events if e.get("cat") == "gpu_user_annotation" and e["name"] in LABELS}
    kernels = [e for e in events if e.get("cat") == "kernel"]
    out = {}
    for label, a in ann.items():
        lo, hi = a["ts"], a["ts"] + a["dur"]
        ks = [k for k in kernels if lo <= k["ts"] and k["ts"] + k["dur"] <= hi + 1]
        out[label] = (sum(k["dur"] for k in ks) / CALLS / 1000.0, len(ks), [k["dur"] / 1000 for k in ks])
    return out


naive = per_call_ms(sys.argv[1] if len(sys.argv) > 1 else "trace_naive.json")
final = per_call_ms(sys.argv[2] if len(sys.argv) > 2 else "trace.json")
for name, t in (("trace_naive.json", naive), ("trace.json", final)):
    print(name)
    for label in LABELS:
        if label in t:
            ms, n, each = t[label]
            print(f"  {label:24s} {ms:9.3f} ms/call   ({n} kernels: {', '.join(f'{d:.3f}' for d in each)})")

b, a, c = naive["student_linear_4096"][0], final["student_linear_4096"][0], final["cublas_linear_4096"][0]
print(f"\nlinear 4096: before {b:.3f} ms, after {a:.3f} ms, cuBLAS {c:.3f} ms, "
      f"speedup {b / a:.1f}x, {100 * c / a:.1f}% of cuBLAS")
b, a, c = naive["student_relu_backward"][0], final["student_relu_backward"][0], final["pytorch_relu_backward"][0]
print(f"relu 64M:    before {b:.3f} ms, after {a:.3f} ms, PyTorch {c:.3f} ms, "
      f"speedup {b / a:.2f}x, {100 * c / a:.1f}% of PyTorch")
