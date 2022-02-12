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
import os
import torch
import logging
import uuid
from quant_modules import Quantizer

quant_args = {}
def set_quantizer(args):
    global quant_args
    quant_args.update({'mode' : args.mode, 'wbit': args.wbit, 'abit': args.abit, 'args' : args})

logger = logging.getLogger(__name__)

def set_util_logging(filename):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler()
        ]
    )

def tag_info(args):
    if args.tag == "":
        return ""
    else:
        return "_" + args.tag

def get_ckpt_path(args):
    path='squant_log'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, args.model+"_"+args.dataset)
    if not os.path.isdir(path):
        os.mkdir(path)
    pathname = args.mode + '_W' + str(args.wbit) + 'A' + str(args.wbit)
    num = int(uuid.uuid4().hex[0:4], 16)
    pathname += '_' + str(num)
    path = os.path.join(path, pathname)
    if not os.path.isdir(path):
        os.mkdir(path)    
    return path
    
def get_ckpt_filename(path, epoch):
    return os.path.join(path, 'ckpt_' + str(epoch) + '.pth')

def get_log_filename(args):
    dire = ['checkpoint', args.dataset, args.model]
    path=''
    for d in dire:
        path = os.path.join(path, d)
        if not os.path.isdir(path):
            os.mkdir(path)
    return os.path.join(path, 'ckpt_' + args.mode + '_' + '_'.join(map(lambda x: str(x), args.bit)) + tag_info(args) + '.txt')


def disable_input_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            module.disable_input_quantization()

def enable_quantization(model):
    for name, module in model.named_modules():
        # print("Enabling module:", name)
        if isinstance(module, Quantizer):
            # print("Enabling module:", name)
            module.enable_quantization(name)

def disable_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            # print("Disabling module:", name)
            module.disable_quantization(name)