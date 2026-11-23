# project-3

CSE 520 Project 3: optimizing CUDA kernels (FP32 linear forward and ReLU backward) on an NVIDIA A100.

| Kernel | Before (starter) | After | Library | Result |
|---|---|---|---|---|
| `linear_forward`, M=K=N=4096 | 517.009 ms | 8.913 ms | cuBLAS 8.036 ms | 58.0x faster, 90.2% of cuBLAS |
| `relu_backward`, 64M elements | 0.563 ms | 0.473 ms | PyTorch 0.448 ms | 1.19x faster, 94.6% of PyTorch |

`benchmark.py` geometric mean over the three timed shapes: 91.8% of cuBLAS; ReLU backward 99.7% of PyTorch.

## What changed

- **linear_forward.cu**: tiled SGEMM. Each 256-thread block computes a 128x128 tile and each thread an 8x8 register tile. Shared memory is double-buffered (BK = 8, transposed, float4 reads) with register prefetching, and every edge in M, N and K is predicated.
- **relu_backward.cu**: float4 vectorized streaming loads and stores (`__ldcs`/`__stcs`), 16 elements per thread, a scalar prologue and epilogue for unaligned tensors, and no `cudaDeviceSynchronize`.

## Files

- `linear_forward.cu`, `relu_backward.cu`: the optimized kernels
- `trace_naive.json`, `trace.json`: PyTorch profiler traces (before and after), for https://ui.perfetto.dev
- `perfetto.png`, `perfetto_timeline.png`: Perfetto screenshots of the linear kernel
- `starter/`: the untouched starter code
- `benchmark.py`, `trace_kernels.py`, `bindings.cpp`: course-provided harness
- `extra_tests.py`: extra edge-case correctness tests
- `trace_numbers.py`: computes report numbers from the two traces

## Running (Sol, full A100)

```bash
module load cuda-13.2.1-gcc-13.2.0 gcc/15.2.0 mamba/latest
source activate pytorch-gpu-2.3.1-cuda-12.1
python benchmark.py
python trace_kernels.py
```
