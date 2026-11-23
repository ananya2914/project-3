// ReLU backward: grad_input[i] = grad_output[i] if input[i] > 0, else 0
//
// This is the naive kernel from Project 1. It is correct, and this operation
// is limited by memory bandwidth, not arithmetic. Make it move bytes as fast
// as the GPU allows: you may change the kernel, add kernels and helper
// functions, and change how relu_backward launches them. Keep the signature
// of relu_backward: bindings.cpp and benchmark.py call it.
//
// Rules: no PyTorch math (torch::where, torch::relu, at::threshold_backward, ...).

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_backward_kernel(const float *grad_output, const float *input, float *grad_input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input)
{
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda(), "tensors must be on the GPU");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "tensors must be float32");

    const int size = input.numel();
    auto grad_output_contig = grad_output.contiguous();
    auto input_contig = input.contiguous();
    auto grad_input = torch::empty_like(input_contig);

    dim3 threads_per_block(256);
    dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);

    relu_backward_kernel<<<number_of_blocks, threads_per_block>>>(
        grad_output_contig.data_ptr<float>(), input_contig.data_ptr<float>(),
        grad_input.data_ptr<float>(), size);

    cudaDeviceSynchronize();
    return grad_input;
}
