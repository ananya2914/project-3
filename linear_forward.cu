// Linear forward: output[b, o] = sum_i input[b, i] * weight[o, i] + bias[o]
//
//   input  (batch_size, input_size)   row-major      -> A (M x K)
//   weight (output_size, input_size)  row-major      -> B (N x K)
//   bias   (output_size)
//   output (batch_size, output_size)  row-major      -> C (M x N)
//
// Optimized FP32 SGEMM on the CUDA cores (C = A * B^T + bias):
//   * Each 256-thread block computes a 128 x 128 tile of C; each thread
//     computes an 8 x 8 sub-tile held in 64 registers (register blocking).
//   * A and B tiles of depth BK = 8 are staged through shared memory, stored
//     transposed (k-major) so the inner loop reads them with float4 loads.
//     Every value loaded from global memory is reused 128 times.
//   * Shared memory is double buffered and the next tile is prefetched into
//     registers while the current tile is being multiplied, so global-memory
//     latency overlaps with the FMAs and there is only one barrier per tile.
//   * Global loads are 128-bit (float4) when K % 4 == 0 and the pointers are
//     16-byte aligned; otherwise the same kernel uses guarded scalar loads.
//     All edges in M, N and K are predicated, so any shape is correct.
//   * No cudaDeviceSynchronize: the kernel is launched asynchronously on
//     PyTorch's current stream, like the library kernels.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

constexpr int BM = 128;      // tile rows of C (batch dimension)
constexpr int BN = 128;      // tile cols of C (output features)
constexpr int BK = 8;        // depth of each K slice
constexpr int PAD = 4;       // smem padding: removes bank conflicts on the transposed stores
constexpr int THREADS = 256; // 16 x 16 threads, 8 x 8 outputs each

// Load 4 consecutive K values of one row starting at p (= row + k).
// VEC: K % 4 == 0 and the row is 16-byte aligned, so k < K means all 4 are in range.
template <bool VEC>
__device__ __forceinline__ float4 load_k4(const float *__restrict__ p, int k, int K, bool row_ok)
{
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (!row_ok)
        return v;
    if (VEC)
    {
        if (k < K)
            v = __ldg(reinterpret_cast<const float4 *>(p));
    }
    else
    {
        if (k + 0 < K) v.x = __ldg(p + 0);
        if (k + 1 < K) v.y = __ldg(p + 1);
        if (k + 2 < K) v.z = __ldg(p + 2);
        if (k + 3 < K) v.w = __ldg(p + 3);
    }
    return v;
}

// Store 4 K values of one row into a k-major (transposed) shared tile.
__device__ __forceinline__ void store_transposed(float (*tile)[BM + PAD], int k, int row, float4 v)
{
    tile[k + 0][row] = v.x;
    tile[k + 1][row] = v.y;
    tile[k + 2][row] = v.z;
    tile[k + 3][row] = v.w;
}

template <bool VEC>
__global__ void __launch_bounds__(THREADS, 2)
linear_forward_tiled_kernel(const float *__restrict__ X, const float *__restrict__ W,
                            const float *__restrict__ bias, float *__restrict__ Y,
                            int M, int K, int N)
{
    __shared__ __align__(16) float As[2][BK][BM + PAD];
    __shared__ __align__(16) float Bs[2][BK][BN + PAD];

    const int tid = threadIdx.x;
    const int tx = tid & 15; // column group of this thread's 8 x 8 sub-tile
    const int ty = tid >> 4; // row group
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    // Global -> shared mapping: each thread loads one float4 of A and one of B per tile.
    const int ld_row = tid >> 1;      // 0..127
    const int ld_k = (tid & 1) << 2;  // 0 or 4
    const bool a_ok = m0 + ld_row < M;
    const bool b_ok = n0 + ld_row < N;
    const float *a_src = X + (size_t)(a_ok ? m0 + ld_row : 0) * K;
    const float *b_src = W + (size_t)(b_ok ? n0 + ld_row : 0) * K;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.f;

    const int num_tiles = (K + BK - 1) / BK;

    // Prologue: tile 0 into buffer 0
    float4 ra = load_k4<VEC>(a_src + ld_k, ld_k, K, a_ok);
    float4 rb = load_k4<VEC>(b_src + ld_k, ld_k, K, b_ok);
    store_transposed(As[0], ld_k, ld_row, ra);
    store_transposed(Bs[0], ld_k, ld_row, rb);
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t)
    {
        const int cur = t & 1;
        const bool has_next = t + 1 < num_tiles;

        // Prefetch the next tile into registers; these loads overlap with the FMAs below.
        if (has_next)
        {
            const int k = (t + 1) * BK + ld_k;
            ra = load_k4<VEC>(a_src + k, k, K, a_ok);
            rb = load_k4<VEC>(b_src + k, k, K, b_ok);
        }

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            const float4 a0 = *reinterpret_cast<const float4 *>(&As[cur][kk][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4 *>(&As[cur][kk][ty * 4 + 64]);
            const float4 b0 = *reinterpret_cast<const float4 *>(&Bs[cur][kk][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4 *>(&Bs[cur][kk][tx * 4 + 64]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        // Write the prefetched tile into the other buffer. Nobody reads that buffer now:
        // its last readers finished before the barrier at the end of the previous iteration.
        if (has_next)
        {
            store_transposed(As[cur ^ 1], ld_k, ld_row, ra);
            store_transposed(Bs[cur ^ 1], ld_k, ld_row, rb);
        }
        __syncthreads();
    }

    // Epilogue: add bias and write the 8 x 8 sub-tile (two 4-wide column groups).
    const bool vec_out = (N & 3) == 0; // Y comes from torch::empty, so rows are 16-byte aligned
#pragma unroll
    for (int jh = 0; jh < 2; ++jh)
    {
        const int c = n0 + jh * 64 + tx * 4;
        float bv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            bv[j] = (c + j < N) ? __ldg(bias + c + j) : 0.f;

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int r = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
            if (r >= M)
                continue;
            float *dst = Y + (size_t)r * N + c;
            const float v0 = acc[i][jh * 4 + 0] + bv[0];
            const float v1 = acc[i][jh * 4 + 1] + bv[1];
            const float v2 = acc[i][jh * 4 + 2] + bv[2];
            const float v3 = acc[i][jh * 4 + 3] + bv[3];
            if (vec_out && c + 3 < N)
            {
                *reinterpret_cast<float4 *>(dst) = make_float4(v0, v1, v2, v3);
            }
            else
            {
                if (c + 0 < N) dst[0] = v0;
                if (c + 1 < N) dst[1] = v1;
                if (c + 2 < N) dst[2] = v2;
                if (c + 3 < N) dst[3] = v3;
            }
        }
    }
}

inline bool aligned16(const void *p)
{
    return (reinterpret_cast<std::uintptr_t>(p) & 15) == 0;
}

} // namespace

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "tensors must be on the GPU");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 &&
                    bias.scalar_type() == torch::kFloat32,
                "tensors must be float32");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && bias.dim() == 1, "expected 2-D input/weight, 1-D bias");
    TORCH_CHECK(input.size(1) == weight.size(1), "input_size mismatch");
    TORCH_CHECK(bias.size(0) == weight.size(0), "bias size mismatch");

    const c10::cuda::CUDAGuard device_guard(input.device());

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = weight.size(0);

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto output = torch::empty({batch_size, output_size}, input.options());
    if (batch_size == 0 || output_size == 0)
        return output;

    const float *x = input_contig.data_ptr<float>();
    const float *w = weight_contig.data_ptr<float>();
    const float *b = bias_contig.data_ptr<float>();
    float *y = output.data_ptr<float>();

    const bool vec = (input_size % 4 == 0) && aligned16(x) && aligned16(w);
    const dim3 grid((output_size + BN - 1) / BN, (batch_size + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (vec)
        linear_forward_tiled_kernel<true><<<grid, THREADS, 0, stream>>>(x, w, b, y, batch_size, input_size, output_size);
    else
        linear_forward_tiled_kernel<false><<<grid, THREADS, 0, stream>>>(x, w, b, y, batch_size, input_size, output_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
