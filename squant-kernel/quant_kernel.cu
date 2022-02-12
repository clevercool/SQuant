// *
// @file Different utility functions
// Copyright (c) Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao, 
// Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, Minyi Guo
// All rights reserved.
// This file is part of SQuant repository.
//
// SQuant is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SQuant is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with SQuant repository.  If not, see <http://www.gnu.org/licenses/>.
// *

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
using namespace std;
namespace {

template <typename scalar_t>
__global__ void quant_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> y,
    size_t x_size,
    size_t y_size,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> z,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> tensor_idx)
{   
    __shared__ float y_shared[256];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < y_size) y_shared[threadIdx.x] = y[threadIdx.x];
    __syncthreads();
    float sub_min = 102400.0;
    float z_min = 0.0;
    float idx_min = 0.0;
    if(idx < x_size) {
        float x_v = x[idx];
        for(int i = 0; i < y_size; i++){
            float sub_v = fabsf(x_v - y_shared[i]);
            if(sub_v <= sub_min)
            {
                sub_min = sub_v;
                z_min = y_shared[i];
                idx_min = i;
            }
            else
                break;
        }
        z[idx] = z_min;
        tensor_idx[idx] = idx_min;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void rounding_forward_cuda_kernel( 
    scalar_t delta,   
    scalar_t rounding_error_sum,
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> rounding_number_,
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> rounding_error_,

    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> number_,
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> error_,
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> priority_,
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> order_,

    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> error_1,
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> priority_1,

    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_number_,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_number,
    
    const size_t en,
    const size_t oc,
    const size_t ic
)
{
    rounding_error_sum = fabsf(rounding_error_sum);
    auto idx = order_;
    size_t topk = __float2int_rn(fabsf(rounding_error_sum));
    bool over_calibration = (topk >= fabsf(rounding_error_sum));

    for(size_t i = 0; i < topk; i++)
    {   
        size_t idx_ = idx[i];
        rounding_error_[idx_] =  error_[idx_];
        rounding_number_[idx_] = number_[idx_];
    }
    if(over_calibration)
    {
        size_t idx_c = idx[topk - 1];
        priority_1[idx_c] = fabsf(rounding_error_[idx_c]);
    }
    else
    {
        size_t idx_c = idx[topk];
        priority_[idx_c] = fabsf(rounding_error_[idx_c]);
    }
}

template <typename scalar_t>
__global__ void rounding_loop_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_number,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_up_number,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_down_number,
    
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> rounding_error_sum,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> rounding_number_,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> rounding_error_,

    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_number_,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_error_,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_priority_,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_order_,

    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_number_,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_error_,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_priority_,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_order_,

    const size_t input_channel,
    const size_t element_number
)
{  
    const int oc = blockIdx.y;
    const int ic = blockIdx.x * blockDim.x + threadIdx.x;
    if (ic >= input_channel) return;

    if(rounding_error_sum[oc][ic] < 0)
    {
        // UP
        scalar_t delta = 1.0;
        rounding_forward_cuda_kernel(
            delta,
            rounding_error_sum[oc][ic],
            rounding_number_[oc][ic],
            rounding_error_[oc][ic],

            up_number_[oc][ic],
            up_error_[oc][ic],
            up_priority_[oc][ic],
            up_order_[oc][ic],

            down_error_[oc][ic],
            down_priority_[oc][ic],

            flip_up_number,
            flip_number,
            
            element_number,
            oc,
            ic
        );
    }
    else
    {
        // Down
        scalar_t delta = -1.0;
        rounding_forward_cuda_kernel(
            delta,
            rounding_error_sum[oc][ic],
            rounding_number_[oc][ic],
            rounding_error_[oc][ic],

            down_number_[oc][ic],
            down_error_[oc][ic],
            down_priority_[oc][ic],
            down_order_[oc][ic],

            up_error_[oc][ic],
            up_priority_[oc][ic],

            flip_down_number,
            flip_number,
            
            element_number,
            oc,
            ic
        );
    }

    return;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor>  quant_forward_cuda(
    torch::Tensor x,
    torch::Tensor y)
{
    const int threads = 1024;
    const dim3 blocks((x.size(0) + threads - 1) / threads);
    auto z   = torch::zeros_like(x);
    auto idx = torch::zeros_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "quant_forward_cuda", ([&] {
        quant_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            y.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            x.size(0),
            y.size(0),
            z.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            idx.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));

    return std::make_tuple(z,idx);
}



void rounding_loop_forward_cuda(
    torch::Tensor flip_number,
    torch::Tensor flip_up_number,
    torch::Tensor flip_down_number,

    torch::Tensor rounding_error_sum,
    torch::Tensor rounding_number_, 
    torch::Tensor rounding_error_, 

    torch::Tensor up_number_, 
    torch::Tensor up_error_, 
    torch::Tensor up_priority_,
    torch::Tensor up_order_, 
    
    torch::Tensor down_number_, 
    torch::Tensor down_error_, 
    torch::Tensor down_priority_,
    torch::Tensor down_order_
)
{
    // const dim3 blocks((x.size(0) + threads - 1) / threads);

    const size_t size_0 = rounding_number_.size(0);
    const size_t size_1 = rounding_number_.size(1);
    const size_t size_2 = rounding_number_.size(2);

    const size_t threads = 64;
    const dim3 grid((size_1 + threads - 1) / threads, size_0);

    AT_DISPATCH_FLOATING_TYPES(
        flip_number.type(), 
        "rounding_loop_forward_cuda",
        (
            [&] {
                rounding_loop_forward_cuda_kernel<scalar_t><<<grid, threads>>>(
                    flip_number.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
                    flip_up_number.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
                    flip_down_number.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),

                    rounding_error_sum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    rounding_number_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    rounding_error_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),

                    up_number_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    up_error_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    up_priority_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    up_order_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),

                    down_number_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    down_error_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    down_priority_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    down_order_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    
                    size_1,
                    size_2
                );
            }
        )
    );
    return;    
}