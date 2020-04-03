#pragma once

//#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

/*
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros') -> Tensor
*/
at::Tensor DepthWiseConv2d_forward(const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                const int stride,
                                const int padding,
                                const int dilation,
                                const int groups) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return DepthWiseConv2d_forward_cuda(input, weight, bias, stride, padding, dilation, groups);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> DepthWiseConv2d_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& weight,
                                 const at::Tensor& bias,
                                 const int stride,
                                 const int padding,
                                 const int dilation,
                                 const int groups) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return DepthWiseConv2d_backward_cuda(grad, input, weight, bias, stride, padding, dilation, groups);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}



