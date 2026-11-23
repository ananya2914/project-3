"""Record a timeline of your kernels and the library kernels, for Perfetto.

Runs each kernel three times next to its library counterpart under the
PyTorch profiler and writes a Chrome trace JSON file. Open it at
https://ui.perfetto.dev (Open trace file, or drag the file onto the page).

    python trace_kernels.py                            # writes trace.json
    python trace_kernels.py --output trace_naive.json  # another name

Each group of calls is labeled in the trace (on the CPU row and on the GPU
row) with one of these names:

    student_linear_4096    your linear_forward,  M = K = N = 4096
    cublas_linear_4096     torch F.linear (cuBLAS, TF32 off), same shape
    student_relu_backward  your relu_backward,   64M elements
    pytorch_relu_backward  PyTorch's ReLU backward (threshold_backward), same size
"""

import argparse
import os
import sys

os.environ.setdefault("KINETO_LOG_LEVEL", "5")  # quiet profiler

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from benchmark import build

CALLS = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="trace.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU is visible. Run on a Sol GPU node (an A100).")

    print("Building linear_forward.cu and relu_backward.cu...")
    ext = build()
    torch.backends.cuda.matmul.allow_tf32 = False  # cuBLAS in plain FP32, like you

    def linear_inputs(m, k, n):
        return torch.randn(m, k, device="cuda"), torch.randn(n, k, device="cuda"), torch.randn(n, device="cuda")

    big = linear_inputs(4096, 4096, 4096)
    g, x = torch.randn(1 << 26, device="cuda"), torch.randn(1 << 26, device="cuda")
    work = [
        ("student_linear_4096", lambda: ext.linear_forward(*big)),
        ("cublas_linear_4096", lambda: F.linear(*big)),
        ("student_relu_backward", lambda: ext.relu_backward(g, x)),
        ("pytorch_relu_backward", lambda: torch.ops.aten.threshold_backward(g, x, 0.0)),
    ]

    # Warm up: the first call of each kernel includes one-time setup
    for _label, fn in work:
        fn()
    torch.cuda.synchronize()

    print("Profiling...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for label, fn in work:
            with record_function(label):
                for _ in range(CALLS):
                    fn()
                torch.cuda.synchronize()
    prof.export_chrome_trace(args.output)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1e3:.0f} kB). "
          "Open it at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
