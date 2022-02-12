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
#include <iostream>
using namespace std;

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor> quant_forward_cuda(
    torch::Tensor x,
    torch::Tensor y);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> quant_forward_cpu(torch::Tensor x,
                            torch::Tensor y) {
                              
    // CHECK_INPUT(x);
    // CHECK_INPUT(y);

    auto z = quant_forward_cuda(x, y);

    return z;
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
    torch::Tensor down_order_);

void rounding_loop_forward_cpu(
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
    ) {
    
    rounding_loop_forward_cuda(
        flip_number,
        flip_up_number,
        flip_down_number,

        rounding_error_sum,
        rounding_number_, 
        rounding_error_, 

        up_number_, 
        up_error_, 
        up_priority_,
        up_order_, 

        down_number_, 
        down_error_, 
        down_priority_,
        down_order_
    );

    return;
}

// pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant", &quant_forward_cpu, "Quantization function");
  m.def("rounding_loop", &rounding_loop_forward_cpu, "Rounding loop function");
}