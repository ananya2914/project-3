// Linear forward: output[b, o] = sum_i input[b, i] * weight[o, i] + bias[o]
//
//   input  (batch_size, input_size)   row-major
//   weight (output_size, input_size)  row-major, as in torch.nn.Linear
//   bias   (output_size)
//   output (batch_size, output_size)  row-major
//
// This is the naive kernel from Project 1. It is correct but slow. Make it as
// fast as you can: you may change the kernel, add kernels and helper
// functions, and change how linear_forward launches them. Keep the signature
// of linear_forward: bindings.cpp and benchmark.py call it.
//
// Rules: FP32 arithmetic on CUDA cores only. No cuBLAS, cuDNN, CUTLASS,
// tensor cores (wmma/mma), TF32, or PyTorch math (torch::matmul, at::addmm, ...).

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_forward_kernel(const float *input, const float *weight, const float *bias,
                                      float *output, int batch_size, int input_size, int output_size)
{
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < output_size)
    {
        float sum = bias[out_idx];
        for (int i = 0; i < input_size; i++)
        {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }
        output[batch_idx * output_size + out_idx] = sum;
    }
}

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "tensors must be on the GPU");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "tensors must be float32");

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = weight.size(0);

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto output = torch::empty({batch_size, output_size}, input.options());

    dim3 threads_per_block(256);
    dim3 number_of_blocks((output_size + threads_per_block.x - 1) / threads_per_block.x, batch_size);

    linear_forward_kernel<<<number_of_blocks, threads_per_block>>>(
        input_contig.data_ptr<float>(), weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, input_size, output_size);

    cudaDeviceSynchronize();
    return output;
}
