// ReLU backward: grad_input[i] = grad_output[i] if input[i] > 0, else 0
//
// This operation is limited by memory bandwidth (8 bytes read + 4 written per
// element, one compare). Optimizations:
//   * 128-bit (float4) loads and stores: 4x fewer memory instructions.
//   * Each thread handles UNROLL float4s and issues all its loads before any
//     store, so many independent memory requests are in flight per thread.
//   * Streaming cache hints (__ldcs / __stcs): every byte is touched once, so
//     it is marked evict-first instead of polluting L1/L2.
//   * Misaligned tensors: if all three pointers share the same offset within a
//     16-byte boundary, a few scalar "head" elements are peeled off so the rest
//     is aligned; the < 4 leftover "tail" elements are done with scalar code.
//     If the offsets differ, a plain scalar kernel is used (correct for any input).
//   * No cudaDeviceSynchronize: launched asynchronously on PyTorch's stream.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

constexpr int THREADS = 256;
constexpr int UNROLL = 4; // float4s per thread -> 16 elements per thread

__device__ __forceinline__ float relu_grad(float g, float x)
{
    return x > 0.0f ? g : 0.0f;
}

__global__ void __launch_bounds__(THREADS)
relu_backward_vec4_kernel(const float *__restrict__ g, const float *__restrict__ x,
                          float *__restrict__ out, long long n, int head)
{
    // Elements [0, head) are scalar, [head, head + 4*n4) are float4, the rest (< 4) scalar.
    const long long n4 = (n - head) >> 2;
    const float4 *__restrict__ g4 = reinterpret_cast<const float4 *>(g + head);
    const float4 *__restrict__ x4 = reinterpret_cast<const float4 *>(x + head);
    float4 *__restrict__ o4 = reinterpret_cast<float4 *>(out + head);

    const long long base = (long long)blockIdx.x * (THREADS * UNROLL) + threadIdx.x;

    float4 gv[UNROLL], xv[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
    {
        const long long i = base + (long long)u * THREADS;
        if (i < n4)
        {
            gv[u] = __ldcs(g4 + i);
            xv[u] = __ldcs(x4 + i);
        }
        else
        {
            gv[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            xv[u] = gv[u];
        }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
    {
        const long long i = base + (long long)u * THREADS;
        if (i < n4)
        {
            float4 r;
            r.x = relu_grad(gv[u].x, xv[u].x);
            r.y = relu_grad(gv[u].y, xv[u].y);
            r.z = relu_grad(gv[u].z, xv[u].z);
            r.w = relu_grad(gv[u].w, xv[u].w);
            __stcs(o4 + i, r);
        }
    }

    // Scalar head (threads 0..3) and tail (threads 4..7) in block 0
    if (blockIdx.x == 0 && threadIdx.x < 8)
    {
        const int t = threadIdx.x;
        long long idx;
        bool valid;
        if (t < 4)
        {
            idx = t;
            valid = t < head;
        }
        else
        {
            idx = head + n4 * 4 + (t - 4);
            valid = idx < n;
        }
        if (valid)
            out[idx] = relu_grad(g[idx], x[idx]);
    }
}

// Fallback for pointers with different alignments: grid-stride scalar loop.
__global__ void __launch_bounds__(THREADS)
relu_backward_scalar_kernel(const float *__restrict__ g, const float *__restrict__ x,
                            float *__restrict__ out, long long n)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = relu_grad(__ldg(g + i), __ldg(x + i));
}

} // namespace

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input)
{
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda(), "tensors must be on the GPU");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && grad_output.scalar_type() == torch::kFloat32,
                "tensors must be float32");
    TORCH_CHECK(grad_output.numel() == input.numel(), "grad_output and input must have the same size");

    const c10::cuda::CUDAGuard device_guard(input.device());

    const long long size = input.numel();
    auto grad_output_contig = grad_output.contiguous();
    auto input_contig = input.contiguous();
    auto grad_input = torch::empty_like(input_contig);
    if (size == 0)
        return grad_input;

    const float *g = grad_output_contig.data_ptr<float>();
    const float *x = input_contig.data_ptr<float>();
    float *o = grad_input.data_ptr<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto ga = reinterpret_cast<std::uintptr_t>(g);
    const auto xa = reinterpret_cast<std::uintptr_t>(x);
    const auto oa = reinterpret_cast<std::uintptr_t>(o);
    const bool same_offset = ((ga & 15) == (xa & 15)) && ((ga & 15) == (oa & 15)) && ((ga & 3) == 0);

    if (same_offset)
    {
        long long head = (long long)(((16 - (ga & 15)) & 15) >> 2); // scalars until 16-byte aligned
        if (head > size)
            head = size;
        const long long n4 = (size - head) >> 2;
        const long long per_block = (long long)THREADS * UNROLL;
        long long blocks = (n4 + per_block - 1) / per_block;
        if (blocks < 1)
            blocks = 1; // block 0 still handles the scalar head/tail
        relu_backward_vec4_kernel<<<(unsigned int)blocks, THREADS, 0, stream>>>(g, x, o, size, (int)head);
    }
    else
    {
        long long blocks = (size + THREADS - 1) / THREADS;
        if (blocks > 65535LL * 32)
            blocks = 65535LL * 32;
        relu_backward_scalar_kernel<<<(unsigned int)blocks, THREADS, 0, stream>>>(g, x, o, size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_input;
}
