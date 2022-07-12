# *
# @file Different utility functions
# Copyright (c) Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao, 
# Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, Minyi Guo
# All rights reserved.
# This file is part of SQuant repository.
#
# SQuant is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SQuant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SQuant repository.  If not, see <http://www.gnu.org/licenses/>.
# *
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from quant_affine import *
import warnings
try:
    from quant_cuda import rounding_loop as SQuant_func
except ImportError:
    warnings.warn("CUDA-based SQuant is not installed! PyTorch-based SQuant will lead to a prolonged quantization process.")
    from squant_function import SQuant_func


logger = logging.getLogger(__name__)
class Quantizer(nn.Module):
    def __init__(self, mode="base", bit=8, is_signed=True, is_enable=False, is_input=False, args=None, operator=None):
        super(Quantizer, self).__init__()
        self.mode = mode
        self.register_buffer('bit', torch.tensor(1))
        self.bit = torch.tensor(bit)
        self.is_signed = is_signed
        self.is_enable = is_enable
        self.is_enable_input = is_enable
        self.is_input = is_input
        self.args = args
        self.operator = operator

        self.percent = self.args.percent / 100
        self.is_sigma = False
        if args.sigma > 0:
            self.percent = args.sigma
            self.is_sigma = True
        
        self.cr = args.constraint_radius
        self.pe = args.packed_element

        self.name = None
        self.has_zero = True
        self.quant_weight_tensor = None
        self.register_buffer('x_max', torch.tensor(1.0))
        self.register_buffer('x_min', torch.tensor(1.0))
        self.has_inited_quant_para = False

        self.squant_k = True
        self.squant_c = True

        self.is_perchannel = True
        if is_input:
            # Input shouldn't be per-channel quantizaton！
            self.is_perchannel = False

        self.tensor_sum = None
        self.tensor_sum_cov = None
        

    def disable_input_quantization(self):
        self.is_enable_input = False
        
    def enable_quantization(self, name):
        self.name = name
        self.is_enable = True

    def disable_quantization(self, name):
        self.name = name
        self.is_enable = False

    def _sigma(self, tensor):
        if not self.is_signed:
            return tensor[tensor > 0].std()
        return tensor.std()

    def updata_packed_element(self, tensor):
        s = tensor.view(tensor.shape[0], tensor.shape[1], -1).shape
        if s[-1] % self.pe != 0:
            self.pe = int(self.pe / 2)

    def updata_signed(self, tensor):
        if tensor.min() < 0:
            self.is_signed = True
            if self.is_input:
                logger.info("Warning!: Signed input!")

    def convert_tensor(self, values):
        values = torch.Tensor(list(set(values)))
        values, _ = torch.sort(values)
        return values

    def adaptive_round(self, x, t_min = None, t_max = None):
        # Get the rounding integer and fraction.
        rounding_number = x.round()
        rounding_error  = rounding_number - x
            
        up_number = rounding_number.clone()
        up_error  = rounding_error.clone()
        up_error[x >= t_max]  = 0.0
        up_error[up_error > 0]  = 0.0
        up_priority = up_error.clone().abs()

        up_error[up_error != 0]  += 1
        up_number[up_error != 0] += 1

        down_number = rounding_number.clone()
        down_error  = rounding_error.clone()
        down_error[x <= t_min]  = 0.0
        down_error[down_error < 0]  = 0.0
        down_priority = down_error.clone().abs()

        down_error[down_error != 0]  -= 1
        down_number[down_error != 0] -= 1

        flip_number = torch.tensor([0.0], device=x.device)
        flip_up_number = torch.tensor([0.0], device=x.device)
        flip_down_number = torch.tensor([0.0], device=x.device)

        conver_shape = x.view(x.shape[0], x.shape[1], -1).shape
        if conver_shape[2] == 1:
            self.squant_k = False

        if self.squant_k:
            rounding_error_sum = rounding_error.view(conver_shape).sum(-1)
            _, up_order = torch.sort(up_priority.view(conver_shape), descending=True)
            _, down_order = torch.sort(down_priority.view(conver_shape), descending=True)
            up_priority *= 0.0
            down_priority *= 0.0

            SQuant_func(
                flip_number,
                flip_up_number,
                flip_down_number,
                
                rounding_error_sum,
                rounding_number.view(conver_shape), 
                rounding_error.view(conver_shape), 

                up_number.view(conver_shape), 
                up_error.view(conver_shape), 
                up_priority.view(conver_shape), 
                up_order, 

                down_number.view(conver_shape), 
                down_error.view(conver_shape), 
                down_priority.view(conver_shape),
                down_order,
            )
        
        if self.squant_c:
            conver_shape = (1, x.shape[0], -1)
            rounding_error_sum = rounding_error.view(conver_shape).sum(-1)
            _, up_order = torch.sort(up_priority.view(conver_shape), descending=True)
            _, down_order = torch.sort(down_priority.view(conver_shape), descending=True)

            SQuant_func(
                flip_number,
                flip_up_number,
                flip_down_number,
                
                rounding_error_sum,
                rounding_number.view(conver_shape), 
                rounding_error.view(conver_shape), 

                up_number.view(conver_shape), 
                up_error.view(conver_shape), 
                up_priority.view(conver_shape), 
                up_order, 

                down_number.view(conver_shape), 
                down_error.view(conver_shape), 
                down_priority.view(conver_shape),
                down_order
            )


        assert (rounding_number.unique().numel() <= 2 ** self.bit.item())
        return rounding_number

    @torch.no_grad()
    def _init_quant_para(self, data):
        if self.has_inited_quant_para == False:
            logger.info("QUANT %d bit: %s " %(self.bit.item(), self.name))
            self.updata_packed_element(data)
            self.updata_signed(data)

            x_max = data.max()
            x_min = data.min()
            alpha = self.percent * data.abs().max()
            if self.is_sigma:
                sigma = self._sigma(data)
                alpha = self.percent * sigma
                if self.is_signed:
                    # We also consider the signed activation. Other framworks will skip this tensor.
                    alpha = self.percent * sigma / 1.25

                # For a higher bit-width, using a wider range still will not cause accuracy loss.
                if self.bit < 6:
                    # For small bit, need clip.
                    alpha = min(alpha, x_max)
                
            if self.mode == "squant-e":
                self.squant_k = False
                self.squant_c = False
                self.mode = "squant"
            elif self.mode == "squant-k":
                self.squant_c = False
                self.mode = "squant"
            elif self.mode == "squant-c":
                self.squant_k = False
                self.mode = "squant"
            
            if self.mode == "squant":
                def _quant(tensor):
                    if self.is_perchannel:
                        x_max = tensor.view(tensor.shape[0], -1).max(1).values
                        x_max = x_max.unsqueeze(1)
                        x_min = tensor.view(tensor.shape[0], -1).min(1).values
                        x_min = x_min.unsqueeze(1)
                    else:
                        x_max = tensor.max()
                        x_min = tensor.min()
                        
                    scale, zero_point = asymmetric_linear_quantization_params(self.bit, x_min, x_max)
                    quant_tensor = linear_quantize(tensor, scale, zero_point, inplace=False)

                    n = 2 ** (self.bit - 1)

                    if self.mode == "squant":
                        quant_tensor = self.adaptive_round(quant_tensor, -n, n - 1)
                    else:
                        quant_tensor = quant_tensor.round()

                    quant_tensor = torch.clamp(quant_tensor, -n, n - 1)
                    quant_tensor = linear_dequantize(quant_tensor, scale, zero_point, inplace=False)
                    return quant_tensor

                if not self.is_input:
                    #Weight quantization
                    start = time.perf_counter()
                    self.quant_weight_tensor = _quant(data)
                    elapsed = (time.perf_counter() - start)
                    logger.info("Quantzation time: %f ms" %(elapsed * 1000))
                else:
                    #Activation quantization
                    # min
                    if self.is_signed:
                        self.x_min = -alpha
                    else:
                        self.x_min.data = torch.zeros_like(alpha)
                    # max
                    self.x_max.data = alpha
            else:
                raise RuntimeError("Unsupported mode: " + self.mode)
                
        self.has_inited_quant_para = True
        
    def _forward(self, data):
        tensor = AsymmetricQuantFunction.apply(data, self.bit, self.x_min, self.x_max)
        return tensor
    
    def tensor_forward(self, tensor, image_size):
        self.image_size = image_size
        if self.mode == "base":
            return tensor
        if not self.is_enable:
            return tensor
        if self.is_input:
            if not self.is_enable_input:
                return tensor

        with torch.no_grad():

            self._init_quant_para(tensor)        
            if self.is_input:                               
                return self._forward(tensor)
            else:
                return self.quant_weight_tensor

class TensorQuantizer(Quantizer):
    def __init__(self, **kwargs):
        super(TensorQuantizer, self).__init__(**kwargs)

    def forward(self, tensor, image_size = 0):
        return self.tensor_forward(tensor, image_size)

class ActivationQuantizer(nn.Module):
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(ActivationQuantizer, self).__init__()        
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_output  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, is_input=True)

    def forward(self, output):
        return self.quant_output(output)

class LinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(LinearQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, is_input=True)
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args)

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input): 
        input = self.quant_input(input)
        weight = self.quant_weight(self.weight)
        # logger.info(input.unique().numel(), self.quant_input.name)
        return F.linear(input, weight, self.bias)


class Conv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(Conv2dQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, is_input=True)
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args)


    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        input = self.quant_input(input)       
        weight = self.quant_weight(self.weight) 
        # logger.info(input.unique().numel(), self.quant_input.name, "input")
        return self._conv_forward(input, weight)