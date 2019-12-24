// Copyright (c) Samson Wang. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void DepthWiseConv2dFForward(const T* bottom_data,
    const T* weight_data,
    const T* bias_data,
    const int channels, const int padding, const int height,
    const int width, const int kernel_size,
    const int out_height, const int out_width,
    T* top_data) {

    T sum = 0;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    //int n = blockIdx.z / channels;
    int o_idx = blockIdx.x * (blockDim.x - kernel_size + 1) + threadIdx.x;
    int o_idy = blockIdx.y * (blockDim.y - kernel_size + 1) + threadIdx.y;
    T bias = 0;
    if (bias_data != NULL) {
        bias = bias_data[c];
    }

//    int i_off_x = threadIdx.x - padding;
//    int i_off_y = threadIdx.y - padding;

    __shared__ T tmp_shared[32][32];
    __shared__ T w_shared[32];

    if (o_idx - padding >= 0 && o_idx - padding < width && o_idy - padding >=0 && o_idy - padding < height) {
        tmp_shared[threadIdx.y][threadIdx.x] = bottom_data[blockIdx.z * width * height + (o_idy - padding) * width + o_idx - padding];
//        printf("tids %d, %d, oid %d, %d, padding %d, width %d, height %d, block %d, %d\n", tidx, tidy, o_idx, o_idy, padding, width, height, blockDim.x, blockDim.y);
    } else {
        tmp_shared[threadIdx.y][threadIdx.x] = 0;
    }
    if (threadIdx.x < kernel_size * kernel_size) {
        w_shared[threadIdx.x] = weight_data[c * kernel_size * kernel_size + threadIdx.x];
    }
    __syncthreads();
//    std::cout << tidx << " " << tidy << " " << " o " << o_idx << "  " << o_idy << " padding " << padding << " " << width << std::endl;
    if (o_idx >= 0 && o_idx < out_width && o_idy >=0 && o_idy < out_height && threadIdx.x < blockDim.x - kernel_size + 1 && threadIdx.y < blockDim.y - kernel_size + 1) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += tmp_shared[threadIdx.y + i][threadIdx.x + j] * w_shared[i * kernel_size + j];
            }
        }
        top_data[blockIdx.z * out_width * out_height + (o_idy ) * out_width + o_idx ] = sum + bias;
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
  AT_ASSERTM(weight.size(0) == channels, "Weight input channel must be equal to Input channel");

  auto output = at::empty({batch_size, channels, out_height, out_width}, input.options());
  auto blocks_x = THCCeilDiv((long)out_width, 32L-kernel_size+1);
  auto blocks_y = THCCeilDiv((long)out_height, 32L-kernel_size+1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(blocks_x, blocks_y, batch_size * channels);
  dim3 block(32, 32);

//  std::cout << "SHAPE dim x " << blocks_x << " dim y " << blocks_y << " nc " << batch_size * channels << std::endl;

//  std::cout << channels << " " << padding << " " << height << " " << width << " " << kernel_size << std::endl;
  //printf("blockdim %d, %d, %d, griddim %d, %d, %d \n", block.x, block.y, block.z, grid.x, grid.y, grid.z);

  //if (output.numel() == 0) {
  //  THCudaCheck(cudaGetLastError());
  //  return output;
  //}
//niu
//AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ROIPool_forward", [&] {
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
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}


