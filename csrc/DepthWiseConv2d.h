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
/*
at::Tensor DepthWiseConv2d_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return DepthWiseConv2d_backward_cuda(grad, input, rois, argmax, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
*/


