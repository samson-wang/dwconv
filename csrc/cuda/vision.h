// Copyright (c) Samson. All Rights Reserved.
#pragma once
#include <torch/extension.h>


at::Tensor DepthWiseConv2d_forward_cuda(const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                const int stride,
                                const int padding,
                                const int dilation,
                                const int groups);

std::vector<at::Tensor> DepthWiseConv2d_backward_cuda(const at::Tensor& grad,
                                const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                const int stride,
                                const int padding,
                                const int dilation,
                                const int groups);
