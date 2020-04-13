// Copyright (c) Samson Wang. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// For small input computation
template <typename T, int FixedKernelSize>
__global__ void DepthWiseConv2dSmallFForward(const T* bottom_data,
    const T* weight_data,
    const T* bias_data,
    const int channels, const int padding, const int height,
    const int width, const int in_kernel_size,
    const int out_height, const int out_width, const int batch_size, const bool forward,
    T* top_data) {
    const int in_num = height * width;
    const int out_num = out_height * out_width;
    const int pad_height = height + padding*2;
    const int pad_width = width + padding*2;
    const int pad_num = pad_height * pad_width;
    const int kernel_size = FixedKernelSize > 0 ? FixedKernelSize : in_kernel_size;
    const int kernel_num = kernel_size * kernel_size;
    const int thread_num = blockDim.x * blockDim.y;
    const int n_steps = blockDim.z * gridDim.z;
    const int out_num_total = n_steps * out_num;
    const int in_num_total = n_steps * in_num;

    const int tidz = threadIdx.z + blockDim.z * blockIdx.z;
    const int tidx = blockDim.x * threadIdx.y + threadIdx.x;
    __shared__ T w_shared[16*16];
    __shared__ T tmp_shared[4*16*16];

    // Initialize tmp shared for input data
    for (int off = threadIdx.z * thread_num + tidx; off < 4 * 256; off += thread_num) {
        tmp_shared[off] = T(0);
    }

        T bias = T(0);
//        if (bias_data != NULL) bias = bias_data[c];

    __syncthreads();
    const int bound = batch_size * channels;
    const int pidx = pad_width * (threadIdx.y + padding) + threadIdx.x + padding;
    const int opidx = pad_width * threadIdx.y + threadIdx.x;
    int tmp_p_off = threadIdx.z * pad_num;
    int tmp_w_off = threadIdx.z * kernel_num;
    int tmp_off = width * threadIdx.y + threadIdx.x + tidz * in_num;
    int tmp_out_off = threadIdx.y * out_width + threadIdx.x + tidz * out_num;
    int half_pad_off = pad_width * blockDim.y;
    int half_in_off = width * blockDim.y;
    int half_out_off = out_width * blockDim.y;
    for (int n_off = 0; n_off < bound; n_off += n_steps) {
        int n_z = n_off + tidz;
        int c = n_z % channels;
        int c_off = c * kernel_num;
        if (n_z < bound) {
        // Load kernels into shared memory
            for (int off = tidx; off < kernel_num; off += thread_num) {
                if (forward) {
                    w_shared[tmp_w_off + off] = weight_data[c_off + off];
                } else {
                    w_shared[tmp_w_off + off] = weight_data[c_off - off + kernel_num - 1];
                }
            }

        // Load input data input shared memory, pay attention to the padding.
            if (threadIdx.x < width && threadIdx.y < height) {
            tmp_shared[tmp_p_off + pidx] = bottom_data[tmp_off];
            if ((threadIdx.y + blockDim.y < height)) {
                tmp_shared[tmp_p_off + pidx + half_pad_off] = bottom_data[tmp_off + half_in_off];
            }
            }
        }

        __syncthreads();
/*
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    printf("%f ", w_shared[i * kernel_size + j]);
                }
                printf("\n");
            }
            for (int i = 0; i < pad_height; i++) {
                for (int j = 0; j < pad_width; j++) {
                    printf("%f ", tmp_shared[i * pad_width + j]);
                }
                printf("\n");
            }
            printf("blockdim %d, %d, %d", blockDim.x, blockDim.y, blockDim.z);
        }
*/
        if (n_z < bound && threadIdx.x < out_width && threadIdx.y < out_height) {
            // To do the math
            T sum = T(0);
            T sum1 = T(0);
            int i_poff = tmp_p_off + opidx;
            #pragma unroll
            for (int i = 0; i < kernel_num; i+= kernel_size) {
                #pragma unroll
                for (int j = 0; j < kernel_size; j++) {
                    const T f = w_shared[i + tmp_w_off + j];
                    sum += tmp_shared[i_poff + j] * f;
                    if ((threadIdx.y + blockDim.y < out_height)) {
                        sum1 += tmp_shared[i_poff + j + half_pad_off] * f;
                    }
                }
                i_poff += pad_width;
            }
//            sum += bias;
            top_data[tmp_out_off] = sum;
            if ((threadIdx.y + blockDim.y < out_height)) {
                top_data[tmp_out_off + half_out_off] = sum1;
                //printf("top data %d, %d, %d, %d\n", threadIdx.x, threadIdx.y, tmp_out_off, half_out_off);
            }
        }
        tmp_off += in_num_total;
        tmp_out_off += out_num_total;
        __syncthreads();
    }
}

template <typename T>
__global__ void DepthWiseConv2dFForward(const T* bottom_data,
    const T* weight_data,
    const T* bias_data,
    const int channels, const int padding, const int height,
    const int width, const int kernel_size,
    const int out_height, const int out_width, const int output_size, const bool forward,
    T* top_data) {
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int o_idx = blockIdx.x * (blockDim.x - kernel_size + 1) + threadIdx.x;
    int o_idy = blockIdx.y * (blockDim.y - kernel_size + 1) + threadIdx.y;
    int c = (blockIdx.z) % channels;
    T bias = 0;
    if (bias_data != NULL) {
        bias = bias_data[c];
    }

    int kernel_num = kernel_size * kernel_size;
    __shared__ T w_shared[32];
    if (tidx < kernel_num) {
        if (forward) {
            w_shared[tidx] = weight_data[c * kernel_num + tidx];
        } else {
            w_shared[tidx] = weight_data[c * kernel_num + kernel_num - 1  - tidx];
        }
    }
    __syncthreads();

    __shared__ T tmp_shared[32*32];
for (int n_off = 0; n_off < output_size; n_off += gridDim.z) {
  if (blockIdx.z + n_off < output_size) {
    T sum = 0;
    //int n = blockIdx.z / channels;
//    int i_off_x = threadIdx.x - padding;
//    int i_off_y = threadIdx.y - padding;


    if (o_idx - padding >= 0 && o_idx - padding < width && o_idy - padding >=0 && o_idy - padding < height) {
        tmp_shared[threadIdx.y * blockDim.x + threadIdx.x] = bottom_data[(blockIdx.z + n_off) * width * height + (o_idy - padding) * width + o_idx - padding];
//        printf("tids %d, %d, oid %d, %d, padding %d, width %d, height %d, block %d, %d\n", tidx, tidy, o_idx, o_idy, padding, width, height, blockDim.x, blockDim.y);
    } else {
        tmp_shared[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    }
    __syncthreads();
//    std::cout << tidx << " " << tidy << " " << " o " << o_idx << "  " << o_idy << " padding " << padding << " " << width << std::endl;
    if (o_idx >= 0 && o_idx < out_width && o_idy >=0 && o_idy < out_height && threadIdx.x < blockDim.x - kernel_size + 1 && threadIdx.y < blockDim.y - kernel_size + 1) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += tmp_shared[(threadIdx.y + i) * blockDim.x + threadIdx.x + j] * w_shared[i * kernel_size + j];
            }
        }
        top_data[(n_off + blockIdx.z) * out_width * out_height + (o_idy ) * out_width + o_idx ] = sum + bias;
    }
  } else {
//    printf("blockDim %d, %d, %d. gridDim %d, %d, %d os %d z %d off %d ch %d\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, output_size, blockIdx.z, n_off, channels);
  }
  __syncthreads();
}
}

template <typename T>
__global__ void DepthWiseConv2dLargeFForward(const T* bottom_data,
    const T* weight_data,
    const T* bias_data,
    const int channels, const int padding, const int height,
    const int width, const int kernel_size,
    const int out_height, const int out_width, const int batch_size,
    T* top_data) {
    __shared__ T true_r_shared[32*32];
    int n_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int n_num = gridDim.x * blockDim.y;

while (n_idx < channels * batch_size) {
    T* r_shared = true_r_shared;
    T sum[8] = {0};
    T tmp = 0;
    const int c = n_idx % channels;
    int valid_kernel_w = kernel_size;
    int valid_kernel_h = kernel_size;
    T* data = const_cast<T*> (bottom_data);
    data = data + n_idx * width * height;
    T* weight = const_cast<T*> (weight_data);
    weight = weight + c * kernel_size * kernel_size;
    const int y_shift = blockIdx.y - padding;
    const int x_shift = blockIdx.z - padding;
    if (blockIdx.y < padding) {
        valid_kernel_h = valid_kernel_h + y_shift;
        weight = weight - y_shift * kernel_size;
    } else if (blockIdx.y >= out_height - padding) {
        valid_kernel_h = valid_kernel_h - (blockIdx.y - out_height + padding + 1);
        data = data + y_shift * width;
    } else {
        data = data + y_shift * width;
    }
    if (blockIdx.z < padding) {
        valid_kernel_w = valid_kernel_w + x_shift;
        weight = weight - x_shift;
    } else if (blockIdx.z >= out_width - padding) {
        valid_kernel_w = valid_kernel_w - (blockIdx.z - out_width + padding + 1);
        data = data + x_shift;
    } else {
        data = data + x_shift;
    }

    const int y_num = (valid_kernel_h / 8) * 8;
    r_shared = r_shared + threadIdx.y * blockDim.x;
    for (int tidx = threadIdx.x; tidx < valid_kernel_w; tidx += blockDim.x) {
        int tmp_tidx_d = tidx;
        int tmp_tidx_w = tidx;
        for (int tidy = 0; tidy < y_num; tidy += 8) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                sum[j] += data[j * width + tmp_tidx_d] * weight[j * kernel_size + tmp_tidx_w];
            }
            tmp_tidx_d = tmp_tidx_d + 8 * width;
            tmp_tidx_w = tmp_tidx_w + 8 * kernel_size;
        }
        for (int j = 0; j < valid_kernel_h - y_num; j++) {
            sum[j] += data[j * width + tmp_tidx_d] * weight[j * kernel_size + tmp_tidx_w];
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        tmp += sum[j];
    }
    r_shared[threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.x < 32) {
        for (int j = 32 + threadIdx.x; j < blockDim.x; j += 32) {
            tmp += r_shared[j];
        }
        r_shared[threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        tmp = r_shared[0];
        for (int j = 1; j < 32; j++) {
            tmp += r_shared[j];
        }
        top_data[n_idx * out_width * out_height + blockIdx.y * out_width + blockIdx.z] = tmp;
    }
    __syncthreads();
    n_idx += n_num;
}
}

at::Tensor DepthWiseConv2d_forward_cuda(const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                const int stride,
                                const int padding,
                                const int dilation,
                                const int groups) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");

  auto batch_size = input.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto kernel_size = weight.size(2);

  auto out_height = (height - kernel_size + 1 + padding * 2) / stride;
  auto out_width = (width - kernel_size + 1 + padding * 2) / stride;
  AT_ASSERTM(weight.size(0) == channels, "Weight output channel must be equal to Input channel");

  auto output = at::empty({batch_size, channels, out_height, out_width}, input.options());
  auto blockdim = 32;
  if (out_width < kernel_size && out_width + kernel_size - 1 < 32) {
    blockdim = kernel_size;
  } else if (out_width + kernel_size - 1 < 32) {
    blockdim = out_width + kernel_size - 1;
  }
  auto blocks_x = THCCeilDiv((long)out_width, blockdim-kernel_size+1L);
  auto blocks_y = THCCeilDiv((long)out_height, blockdim-kernel_size+1L);

  auto output_size = batch_size * channels;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto znum = output_size;
  if (znum > 2048) {
    znum = std::max((2048 / channels) * channels, channels);
  }
if (kernel_size > 16) {
    int blocks_x = kernel_size <= 1024 ? kernel_size : 1024;
    int blocks_y = (1024) / blocks_x;
    //dim3 grid((channels * batch_size + blocks_y - 1) / blocks_y, out_height, out_width);
    dim3 grid((channels * batch_size) / blocks_y / 2, out_height, out_width);
    dim3 block(blocks_x, blocks_y);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2d_forward", [&] {
    DepthWiseConv2dLargeFForward<scalar_t><<<grid, block, 0, stream>>>(
         input.contiguous().data<scalar_t>(),
         weight.contiguous().data<scalar_t>(),
         bias.contiguous().data<scalar_t>(),
         channels,
         padding,
         height,
         width,
         kernel_size,
         out_height,
         out_width,
         batch_size,
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());

} else if (width + 2*padding > 16 || height + 2 * padding> 16) {
  dim3 grid(blocks_x, blocks_y, znum);
  dim3 block(blockdim, blockdim);

//  std::cout << "SHAPE dim x " << blocks_x << " dim y " << blocks_y << " nc " << batch_size * channels << std::endl;

//  std::cout << channels << " " << padding << " " << height << " " << width << " " << kernel_size << std::endl;
  //printf("blockdim %d, %d, %d, griddim %d, %d, %d outputsize %d\n", block.x, block.y, block.z, grid.x, grid.y, grid.z, output_size);

  //if (output.numel() == 0) {
  //  THCudaCheck(cudaGetLastError());
  //  return output;
  //}
//niu
//  printf("blockdim %d, %d, %d, griddim %d, %d, %d outputsize %d, channels %d, width %d, height %d, padding %d, stride %d, bias %s, kernel_size %d\n", block.x, block.y, block.z, grid.x, grid.y, grid.z, batch_size, channels, width, height, padding, stride, bias.size(0), kernel_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2d_forward", [&] {
    DepthWiseConv2dFForward<scalar_t><<<grid, block, 0, stream>>>(
         input.contiguous().data<scalar_t>(),
         weight.contiguous().data<scalar_t>(),
         bias.contiguous().data<scalar_t>(),
         channels,
         padding,
         height,
         width,
         kernel_size,
         out_height,
         out_width,
         output_size,
         true,
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
} else {
  auto dimy = THCCeilDiv((long)(height > out_height ? height : out_height), 2L);
  auto blocks_x = 1;
  auto blocks_y = 1;
 
  dim3 grid(blocks_x, blocks_y, THCCeilDiv((long)channels*batch_size, 64L));
  dim3 block(width > out_width ? width : out_width, dimy, 8);
//  printf("Small blockdim %d, %d, %d, griddim %d, %d, %d outputsize %d, channels %d, width %d, height %d, padding %d, stride %d, bias %s, kernel_size %d\n", block.x, block.y, block.z, grid.x, grid.y, grid.z, batch_size, channels, width, height, padding, stride, bias.size(0), kernel_size);
if (kernel_size == 3) {
  AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2dSmall_forward", [&] {
    DepthWiseConv2dSmallFForward<scalar_t, 3><<<grid, block, 0, stream>>>(
         input.contiguous().data<scalar_t>(),
         weight.contiguous().data<scalar_t>(),
         bias.contiguous().data<scalar_t>(),
         channels,
         padding,
         height,
         width,
         kernel_size,
         out_height,
         out_width,
         batch_size,
         true,
         output.data<scalar_t>());
  });
} else if (kernel_size == 5) {
  AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2dSmall_forward", [&] {
    DepthWiseConv2dSmallFForward<scalar_t, 5><<<grid, block, 0, stream>>>(
         input.contiguous().data<scalar_t>(),
         weight.contiguous().data<scalar_t>(),
         bias.contiguous().data<scalar_t>(),
         channels,
         padding,
         height,
         width,
         kernel_size,
         out_height,
         out_width,
         batch_size,
         true,
         output.data<scalar_t>());
  });
} else {
  AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2dSmall_forward", [&] {
    DepthWiseConv2dSmallFForward<scalar_t, 0><<<grid, block, 0, stream>>>(
         input.contiguous().data<scalar_t>(),
         weight.contiguous().data<scalar_t>(),
         bias.contiguous().data<scalar_t>(),
         channels,
         padding,
         height,
         width,
         kernel_size,
         out_height,
         out_width,
         batch_size,
         true,
         output.data<scalar_t>());
  });

}
  THCudaCheck(cudaGetLastError());

}
  return output;
}

std::vector<at::Tensor> DepthWiseConv2d_backward_cuda(const at::Tensor& grad,
                                const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                const int stride,
                                const int padding,
                                const int dilation,
                                const int groups) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");

  auto batch_size = input.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto kernel_size = weight.size(2);

  auto out_height = (height - kernel_size + 1 + padding * 2) / stride;
  auto out_width = (width - kernel_size + 1 + padding * 2) / stride;
  AT_ASSERTM(weight.size(0) == channels, "Weight output channel must be equal to Input channel");

  // To deal with input grad computation.
  auto grad_input = at::empty({batch_size, channels, height, width}, grad.options());
  auto grad_weight = at::empty({channels, 1, kernel_size, kernel_size}, grad.options());
  auto grad_bias = at::empty({bias.size(0)}, grad.options());
  auto blockdim = 32;

  auto bwd_padding = kernel_size - 1 - padding;
  auto bwd_s = 1;
    std::cout << out_width << "x" << out_height << " Grad " << grad.size(2) << "x" << grad.size(3) << std::endl;
    std::cout << grad.size(3) - kernel_size + 1 + bwd_padding * 2 << " bwd " << bwd_padding << std::endl;
  AT_ASSERTM(width == (grad.size(3) - kernel_size + 1 + bwd_padding * 2), "grad_input computed size should be equal to input size")

  if (width < kernel_size && width + kernel_size - 1 < 32) {
    blockdim = kernel_size;
  } else if (width + kernel_size - 1 < 32) {
    blockdim = width + kernel_size - 1;
  }
  auto blocks_x = THCCeilDiv((long)width, blockdim-kernel_size+1L);
  auto blocks_y = THCCeilDiv((long)height, blockdim-kernel_size+1L);

  auto output_size = batch_size * channels;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto znum = output_size;
  if (znum > 2048) {
    znum = std::max((2048 / channels) * channels, channels);
  }
  if (out_width + 2*padding > 16 || out_height + 2 * padding> 16) {
    dim3 grid(blocks_x, blocks_y, znum);
    dim3 block(blockdim, blockdim);
  
    AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2d_forward", [&] {
      DepthWiseConv2dFForward<scalar_t><<<grid, block, 0, stream>>>(
           grad.contiguous().data<scalar_t>(),
           weight.contiguous().data<scalar_t>(),
           bias.contiguous().data<scalar_t>(),
           channels,
           bwd_padding,
           out_height,
           out_width,
           kernel_size,
           height,
           width,
           output_size,
           false,
           grad_input.data<scalar_t>());
    });
    THCudaCheck(cudaGetLastError());
  } else {
    auto dimy = THCCeilDiv((long)(out_height > height ? out_height : height), 2L);
    auto blocks_x = 1;
    auto blocks_y = 1;
 
    dim3 grid(blocks_x, blocks_y, THCCeilDiv((long)channels*batch_size, 64L));
    dim3 block(out_width > width ? out_width : width, dimy, 8);

    if (kernel_size == 3) {
      AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2dSmall_forward", [&] {
        DepthWiseConv2dSmallFForward<scalar_t, 3><<<grid, block, 0, stream>>>(
             grad.contiguous().data<scalar_t>(),
             weight.contiguous().data<scalar_t>(),
             bias.contiguous().data<scalar_t>(),
             channels,
             bwd_padding,
             out_height,
             out_width,
             kernel_size,
             height,
             width,
             batch_size,
             false,
             grad_input.data<scalar_t>());
      });
        std::cout << "3 small" << std::endl;
    } else if (kernel_size == 5) {
      AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2dSmall_forward", [&] {
        DepthWiseConv2dSmallFForward<scalar_t, 5><<<grid, block, 0, stream>>>(
             grad.contiguous().data<scalar_t>(),
             weight.contiguous().data<scalar_t>(),
             bias.contiguous().data<scalar_t>(),
             channels,
             bwd_padding,
             out_height,
             out_width,
             kernel_size,
             height,
             width,
             batch_size,
             false,
             grad_input.data<scalar_t>());
      });
        std::cout << "5 small" << std::endl;
    } else {
      AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2dSmall_forward", [&] {
        DepthWiseConv2dSmallFForward<scalar_t, 0><<<grid, block, 0, stream>>>(
             grad.contiguous().data<scalar_t>(),
             weight.contiguous().data<scalar_t>(),
             bias.contiguous().data<scalar_t>(),
             channels,
             bwd_padding,
             out_height,
             out_width,
             kernel_size,
             height,
             width,
             batch_size,
             false,
             grad_input.data<scalar_t>());
      });
        std::cout << "Common small" << std::endl;
        //printf("<%d, %d, %d>\nGrid <%d, %d, %d>\nshape %d, %d, %d, %d\n", block.x, block.y, block.z, grid.x, grid.y, grid.z, width, height, out_width, out_height);
    }
    THCudaCheck(cudaGetLastError());
  
  }

   //std::cout << "before return" << std::endl << out_width << std::endl << padding << std::endl << out_height << std::endl << width << std::endl;
   return std::vector<at::Tensor> {grad_input, grad_weight, grad_bias};
}

