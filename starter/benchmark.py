"""Check and time your kernels against cuBLAS and PyTorch.

Builds linear_forward.cu and relu_backward.cu (with bindings.cpp) into a
PyTorch extension, checks both kernels for correctness, times them against
the library kernels on the same GPU, and prints your speed as a percentage
of the library's.

The correctness checks below are a sample. Your kernels are also checked on
other shapes and sizes, so make them correct for any M, K, N and any size,
including tensors that do not start on a 16-byte boundary.

    python benchmark.py                 # both kernels
    python benchmark.py --only linear   # just linear_forward
    python benchmark.py --only relu     # just relu_backward
    python benchmark.py --no-perf       # correctness only (fast)
"""

import argparse
import json
import math
import os
import re
import sys
import traceback

os.environ.setdefault("KINETO_LOG_LEVEL", "5")  # quiet profiler

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["bindings.cpp", "linear_forward.cu", "relu_backward.cu"]

# Shapes are (M, K, N): input (M, K), weight (N, K), output (M, N).
# The correctness shapes include sizes that are not multiples of any tile
# size, input sizes (K) that are not multiples of 4, and one case where the
# tensors start 4 bytes past a 16-byte boundary ("offset").
LINEAR_CORRECTNESS = [
    (1, 1, 1), (3, 5, 2), (64, 784, 128), (33, 100, 65), (200, 131, 300),
    (65, 48, 129), (100, 2051, 70), (2, 3000, 2000), (1000, 2, 5), (48, 192, 96, "offset"),
]
LINEAR_PERF = [(4096, 4096, 4096), (8192, 1024, 4096), (16384, 784, 1024)]

RELU_CORRECTNESS = [1, 5, 64, 999, (2000, "offset"), 50001, (1 << 20) + 1]
RELU_PERF = 1 << 26  # 64M elements: 768 MB of traffic per call

# Library kernels your wrappers may not launch, and source code they may not use
LIBRARY_KERNEL = re.compile(r"gemm|cublas|cutlass|cudnn|xmma|ampere_|sm80_|^void at::|at::native", re.I)
ALLOWED_LIBRARY_KERNEL = re.compile(r"FillFunctor")  # torch::zeros / .zero_() is fine
FORBIDDEN_SOURCE = re.compile(
    r"\bcublas\w*|\bcudnn\w*|\bcutlass\b|\bwmma\b|\bmma\b|mma\.sync|\btf32\b|__half\b|__nv_bfloat16"
    r"|\b(?:torch|at)::(?:matmul|mm|addmm|linear|bmm|einsum|where|relu|threshold\w*|clamp\w*|mul|add|sum)\b"
    r"|\.(?:matmul|mm|addmm|bmm|relu|where|clamp|clamp_min|mul|mul_|add|add_|sum|t)\s*\(",
    re.I,
)


def build():
    from torch.utils.cpp_extension import load

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    build_dir = os.path.join(HERE, "build")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="project3_kernels",
        sources=[os.path.join(HERE, s) for s in SOURCES],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=build_dir,
        verbose=False,
    )


def forbidden_code(filename):
    text = open(os.path.join(HERE, filename), encoding="utf-8", errors="replace").read()
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)  # ignore comments
    return sorted({m.group(0) for m in FORBIDDEN_SOURCE.finditer(text)})


def library_kernels(fn):
    """Names of library kernels (cuBLAS, PyTorch, ...) that one call of fn launches"""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
    return sorted(n for n in names if LIBRARY_KERNEL.search(n) and not ALLOWED_LIBRARY_KERNEL.search(n))


def time_ms(fns, calls, reps=7):
    """Median time per call of each function in fns, measured with CUDA events.

    Each rep times `calls` back-to-back calls; the functions take turns so
    that clock changes affect all of them alike.
    """
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    times = [[] for _ in fns]
    for _ in range(reps):
        for i, fn in enumerate(fns):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(calls):
                fn()
            end.record()
            end.synchronize()
            times[i].append(start.elapsed_time(end) / calls)
    return [sorted(t)[len(t) // 2] for t in times]


def offset_tensor(shape, offset):
    """A contiguous random tensor whose data starts `offset` floats into its storage"""
    storage = torch.randn(math.prod(shape) + offset, device="cuda")
    return storage[offset:].view(shape)


class KernelError(Exception):
    pass


def call(fn, label, *args):
    """One call of your wrapper, synchronized so that a CUDA error in your
    kernel is reported here (and not by a later, unrelated operation)"""
    try:
        out = fn(*args)
        torch.cuda.synchronize()
        return out
    except RuntimeError as e:
        message = str(e).strip().splitlines()[0]
        print(f"  {label:32s} CRASHED: {message}")
        print("  After a CUDA error the GPU context is unusable, so the remaining checks are skipped.")
        raise KernelError(message) from None


def linear_error(out, x, w, b):
    """Largest error against a float64 reference, and the allowed error.

    FP32 accumulation over K terms gives errors that grow like sqrt(K); TF32
    or half precision would give errors about 10x above the allowance.
    """
    expected = x.double() @ w.double().T + b.double()
    if out.shape != expected.shape or out.dtype != torch.float32:
        return math.inf, 0.0
    error = (out.double() - expected).abs().max().item()
    return error, 2e-4 * math.sqrt(x.shape[1]) + 1e-5


def check_linear(ext, report):
    ok = True
    for case in LINEAR_CORRECTNESS:
        m, k, n = case[:3]
        offset = 1 if "offset" in case else 0
        x, w, b = offset_tensor((m, k), offset), offset_tensor((n, k), offset), offset_tensor((n,), offset)
        label = f"M={m} K={k} N={n}" + (" (offset)" if offset else "")
        error, allowed = linear_error(call(ext.linear_forward, label, x, w, b), x, w, b)
        passed = error <= allowed
        ok &= passed
        print(f"  {label:32s} {'ok' if passed else 'WRONG'}   max error {error:.2e} (allowed {allowed:.1e})")
        report["cases"].append({"case": label, "passed": passed, "error": error})
    return ok


def perf_linear(ext, report):
    torch.backends.cuda.matmul.allow_tf32 = False  # cuBLAS in plain FP32, like you
    print(f"  {'M':>6} {'K':>6} {'N':>6} {'yours (ms)':>11} {'cuBLAS (ms)':>12} {'GFLOP/s':>9} {'% of cuBLAS':>12}")
    ratios = []
    for m, k, n in LINEAR_PERF:
        x, w, b = torch.randn(m, k, device="cuda"), torch.randn(n, k, device="cuda"), torch.randn(n, device="cuda")
        call(ext.linear_forward, f"M={m} K={k} N={n}", x, w, b)  # warm up
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        ext.linear_forward(x, w, b)
        end.record()
        end.synchronize()
        slow = start.elapsed_time(end) > 100
        yours, cublas = time_ms(
            [lambda: ext.linear_forward(x, w, b), lambda: F.linear(x, w, b)],
            calls=1 if slow else 5, reps=3 if slow else 7,
        )
        # The result must depend on the current input values
        x.neg_()
        error, allowed = linear_error(ext.linear_forward(x, w, b), x, w, b)
        if not error <= allowed:
            print(f"  M={m} K={k} N={n}: WRONG after timing (max error {error:.2e})")
            report["perf_wrong"] = True
            return None
        ratio = 100.0 * cublas / yours
        ratios.append(ratio)
        gflops = 2.0 * m * n * k / (yours * 1e6)
        print(f"  {m:6d} {k:6d} {n:6d} {yours:11.3f} {cublas:12.3f} {gflops:9.0f} {ratio:11.1f}%")
        report["perf"].append({"shape": [m, k, n], "yours_ms": yours, "cublas_ms": cublas, "percent": ratio})

    x = torch.randn(LINEAR_PERF[0][:2], device="cuda")
    w = torch.randn(LINEAR_PERF[0][2], LINEAR_PERF[0][1], device="cuda")
    b = torch.randn(LINEAR_PERF[0][2], device="cuda")
    used = library_kernels(lambda: ext.linear_forward(x, w, b))
    if used:
        print(f"  NOT ALLOWED: linear_forward launched library kernels: {used}")
        report["library_kernels"] = used
        return None
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def check_relu(ext, report):
    ok = True
    for case in RELU_CORRECTNESS:
        size, offset = (case[0], 1) if isinstance(case, tuple) else (case, 0)
        x, g = offset_tensor((size,), offset), offset_tensor((size,), offset)
        x[::7] = 0.0  # zero input must give zero gradient
        label = f"size={size}" + (" (offset)" if offset else "")
        out = call(ext.relu_backward, label, g, x)
        expected = torch.where(x > 0, g, torch.zeros_like(g))
        passed = out.shape == expected.shape and out.dtype == torch.float32 and torch.equal(out, expected)
        ok &= passed
        print(f"  {label:32s} {'ok' if passed else 'WRONG'}")
        report["cases"].append({"case": label, "passed": passed})
    return ok


def perf_relu(ext, report):
    n = RELU_PERF
    g, x = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    call(ext.relu_backward, f"size={n}", g, x)  # warm up
    yours, pytorch = time_ms(
        [lambda: ext.relu_backward(g, x), lambda: torch.ops.aten.threshold_backward(g, x, 0.0)], calls=20
    )
    x.neg_()
    if not torch.equal(ext.relu_backward(g, x), torch.where(x > 0, g, torch.zeros_like(g))):
        print("  WRONG after timing")
        report["perf_wrong"] = True
        return None
    used = library_kernels(lambda: ext.relu_backward(g, x))
    if used:
        print(f"  NOT ALLOWED: relu_backward launched library kernels: {used}")
        report["library_kernels"] = used
        return None
    ratio = 100.0 * pytorch / yours
    bytes_moved = 3 * 4 * n
    print(f"  {'size':>10} {'yours (ms)':>11} {'PyTorch (ms)':>13} {'GB/s':>7} {'% of PyTorch':>13}")
    print(f"  {n:10d} {yours:11.3f} {pytorch:13.3f} {bytes_moved / (yours * 1e6):7.0f} {ratio:12.1f}%")
    report["perf"].append({"size": n, "yours_ms": yours, "pytorch_ms": pytorch, "percent": ratio})
    return ratio


def run(ext, only, perf):
    results = {}
    if only in (None, "linear"):
        report = results["linear"] = {"cases": [], "perf": [], "correct": False, "percent": None,
                                      "forbidden": forbidden_code("linear_forward.cu")}
        print("\nlinear_forward: correctness")
        try:
            report["correct"] = check_linear(ext, report)
            if report["forbidden"]:
                print(f"  NOT ALLOWED in linear_forward.cu: {report['forbidden']}")
            elif perf and report["correct"]:
                print("\nlinear_forward: performance (FP32, cuBLAS with TF32 off)")
                report["percent"] = perf_linear(ext, report)
        except Exception as e:
            if not isinstance(e, KernelError):
                traceback.print_exc(limit=3)
            report["correct"] = False
            report["percent"] = None
            report["crashed"] = True
        if report["percent"] is not None:
            print(f"  geometric mean: {report['percent']:.1f}% of cuBLAS")

    if only in (None, "relu"):
        report = results["relu"] = {"cases": [], "perf": [], "correct": False, "percent": None,
                                    "forbidden": forbidden_code("relu_backward.cu")}
        print("\nrelu_backward: correctness")
        if only is None and results["linear"].get("crashed"):
            print("  skipped: the GPU context was lost when linear_forward crashed. "
                  "Run: python benchmark.py --only relu")
            return results
        try:
            report["correct"] = check_relu(ext, report)
            if report["forbidden"]:
                print(f"  NOT ALLOWED in relu_backward.cu: {report['forbidden']}")
            elif perf and report["correct"]:
                print("\nrelu_backward: performance")
                report["percent"] = perf_relu(ext, report)
        except Exception as e:
            if not isinstance(e, KernelError):
                traceback.print_exc(limit=3)
            report["correct"] = False
            report["percent"] = None
    return results


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["linear", "relu"])
    parser.add_argument("--no-perf", action="store_true", help="check correctness only")
    parser.add_argument("--json", help=argparse.SUPPRESS)  # write the results as JSON
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU is visible. Run on a Sol GPU node (an A100).")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
    if "A100" not in props.name or props.multi_processor_count != 108:
        print("WARNING: the targets assume a full A100 (108 SMs). Percentages on other GPUs will differ.")

    print("Building linear_forward.cu and relu_backward.cu...")
    try:
        ext = build()
    except Exception as e:
        print(f"Build failed:\n{e}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"build_failed": True}, f)
        sys.exit(1)

    results = run(ext, args.only, not args.no_perf)
    results["gpu"] = props.name
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)

    if args.only is None:
        print("\nSummary:")
        for key, name, library in (("linear", "linear_forward", "cuBLAS"), ("relu", "relu_backward", "PyTorch")):
            r = results.get(key)
            if r is None:
                continue
            speed = "" if r["percent"] is None else f", {r['percent']:.1f}% of {library}"
            print(f"  {name}: {'correct' if r['correct'] else 'NOT correct'}{speed}")


if __name__ == "__main__":
    main()
