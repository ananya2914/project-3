// Python bindings for the two kernels. DO NOT EDIT: you do not submit this
// file, so changes to it are not used.
#include <torch/extension.h>

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("linear_forward", &linear_forward, "output = input @ weight.T + bias");
    m.def("relu_backward", &relu_backward, "grad_input = grad_output where input > 0, else 0");
}
