// Copyright (c) Samson Wang. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// For small input computation
template <typename T>
__global__ void XDepthWiseConv2dSmallFForward(const T* bottom_data,
    const T* weight_data,
    const T* bias_data,
    const int channels, const int padding, const int height,
    const int width, const int kernel_size,
    const int out_height, const int out_width, const int batch_size,
    T* top_data) {
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int o_idx = blockIdx.x * (blockDim.x) + threadIdx.x;
    int o_idy = blockIdx.y * (blockDim.y) + threadIdx.y;
    int c = (blockIdx.z * blockDim.z + threadIdx.z) % channels;
    int dim_in_x = blockDim.x + 2*padding;
    int dim_in_y = blockDim.y + 2*padding;
    T bias = 0;
    if (bias_data != NULL) {
        bias = bias_data[c];
    }

    int z_off = threadIdx.z * dim_in_x * dim_in_y;
    int shared_z_off = threadIdx.z * kernel_size * kernel_size;

    __shared__ T tmp_shared[8*16*16];
    __shared__ T w_shared[8 * 32];
    if (tidx < kernel_size * kernel_size && c < channels) {
        w_shared[shared_z_off + tidx] = weight_data[c * kernel_size * kernel_size + tidx];
    }
    for (int i = 0; tidx + i < 16*16; i += blockDim.x * blockDim.y) {
        tmp_shared[threadIdx.z * 16 * 16 + tidx + i] = 0;
    }
    __syncthreads();

for (int n_off = 0; n_off < batch_size; n_off++) {
    T sum = 0;
    if (c < channels) {
        tmp_shared[z_off + (threadIdx.y + padding)* dim_in_x + threadIdx.x + padding] = bottom_data[(c + n_off * channels) * width * height + (o_idy) * width + o_idx];
//        printf("tids %d, %d, oid %d, %d, padding %d, width %d, height %d, block %d, %d\n", tidx, tidy, o_idx, o_idy, padding, width, height, blockDim.x, blockDim.y);
    } else {
//        tmp_shared[threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x] = 0;
    }
    __syncthreads();
//    std::cout << tidx << " " << tidy << " " << " o " << o_idx << "  " << o_idy << " padding " << padding << " " << width << std::endl;
    if (c < channels) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += tmp_shared[z_off + (threadIdx.y + i) * dim_in_x + threadIdx.x + j] * w_shared[shared_z_off + i * kernel_size + j];
            }
        }
        sum += bias;
        //top_data[(c + n_off * channels) * out_width * out_height + (o_idy ) * out_width + o_idx ] = sum + bias;
    }
    __syncthreads();
    top_data[(c + n_off * channels) * out_width * out_height + (threadIdx.y) * out_width + threadIdx.x] = sum;
}
}

template <typename T>
__global__ void DepthWiseConv2dSmallFForward(const T* bottom_data,
    const T* weight_data,
    const T* bias_data,
    const int channels, const int padding, const int height,
    const int width, const int kernel_size,
    const int out_height, const int out_width, const int batch_size,
    T* top_data) {
    __shared__ T tmp_shared[8*16*16];
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int c = (blockIdx.z * blockDim.z + threadIdx.z) % channels;

    int z_off = threadIdx.z * blockDim.x * blockDim.y;
    for (int n_off = 0; n_off < batch_size; n_off++) {
    if (blockIdx.z * blockDim.z + threadIdx.z + n_off * channels < batch_size * channels) {
        tmp_shared[z_off + threadIdx.y * blockDim.x + threadIdx.x] = bottom_data[(c+n_off*channels) * width * height + threadIdx.y * width + threadIdx.x];
        __syncthreads();
        top_data[(c+n_off*channels) * width * height + threadIdx.y * width + threadIdx.x] = tmp_shared[z_off + threadIdx.y * blockDim.x + threadIdx.x];
    }
        __syncthreads();
    }
}

template <typename T>
__global__ void DepthWiseConv2dFForward(const T* bottom_data,
    const T* weight_data,
    const T* bias_data,
    const int channels, const int padding, const int height,
    const int width, const int kernel_size,
    const int out_height, const int out_width, const int output_size,
    T* top_data) {
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int o_idx = blockIdx.x * (blockDim.x - kernel_size + 1) + threadIdx.x;
    int o_idy = blockIdx.y * (blockDim.y - kernel_size + 1) + threadIdx.y;
    int c = (blockIdx.z) % channels;
    T bias = 0;
    if (bias_data != NULL) {
        bias = bias_data[c];
    }


    __shared__ T w_shared[32];
    if (tidx < kernel_size * kernel_size) {
        w_shared[tidx] = weight_data[c * kernel_size * kernel_size + tidx];
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
  auto blockdim = 32;
  if (out_width < kernel_size && out_width < 32) {
    blockdim = kernel_size;
  } else if (out_width < 32) {
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
if (out_width > 16) {
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
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
} else {
  blockdim = out_width;
  auto blocks_x = THCCeilDiv((long)out_width, (long)blockdim);
  auto blocks_y = THCCeilDiv((long)out_height, (long)blockdim);
 
  dim3 grid(blocks_x, blocks_y, THCCeilDiv((long)channels, 8L));
  dim3 block(blockdim, blockdim, 8);
  printf("blockdim %d, %d, %d, griddim %d, %d, %d outputsize %d\n", block.x, block.y, block.z, grid.x, grid.y, grid.z, batch_size);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "DepthWiseConv2dSmall_forward", [&] {
    DepthWiseConv2dSmallFForward<scalar_t><<<grid, block, 0, stream>>>(
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

}
  return output;
}


