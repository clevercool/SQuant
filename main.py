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
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models

from pytorchcv.model_provider import get_model
from dataloader import *
from quant_utils import *
from quant_model import *


parser = argparse.ArgumentParser(description='SQuant')
parser.add_argument('--mode', default='int', type=str,
                    help='quantizer mode')
parser.add_argument('--dataset', default='imagenet', type=str, 
                    help='dataset name')
parser.add_argument('--dataset_path', default='/state/partition/imagenet-raw-data', type=str, 
                    help='dataset path')
parser.add_argument('--model', default='resnet18', type=str, 
                    help='model name')
parser.add_argument('--wbit', '-wb', default='8', type=int, 
                    help='weight bit width')
parser.add_argument('--abit', '-ab', default='8', type=int, 
                    help='activation bit width')
parser.add_argument('--batch_size', default=256, type=int, 
                    help='batch_size num')
parser.add_argument('--disable_quant', "-dq", default=False, action='store_true', 
                    help='disable quant')
parser.add_argument('--disable_activation_quant', "-daq", default=False, action='store_true', 
                    help='quant_activation')
parser.add_argument('--percent', '-p', default='100', type=int, 
                    help='percent')
parser.add_argument('--constraint_radius', '-cr', default='1.0', type=float, 
                    help='Constraint radius')
parser.add_argument('--packed_element', '-pe', default='1', type=int, 
                    help='Packed Element radius')
parser.add_argument('--sigma', '-s', default='0', type=float, 
                    help='Init activation range with Batchnorm Sigma')
args = parser.parse_args()

### logging setting
output_path = get_ckpt_path(args)
set_util_logging(output_path + "/training.log")
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(output_path + "/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(output_path)
logger.info(args)

### Model
logger.info('==> Building model..')
model = get_model(args.model, pretrained=True)

### Random
if args.model.startswith('inception'):
    rand_input = torch.rand([args.batch_size, 3, 299, 299], dtype=torch.float, requires_grad=False).cuda()
else:
    rand_input = torch.rand([args.batch_size, 3, 224, 224], dtype=torch.float, requires_grad=False).cuda()

### Set Quantizer
logger.info('==> Setting quantizer..')
set_quantizer(args)
quantized_model = quantize_model(model)

if args.disable_quant:
    disable_quantization(quantized_model)
else:
    enable_quantization(quantized_model)

if args.disable_activation_quant:
    disable_input_quantization(quantized_model)
     
set_first_last_layer(quantized_model)
quantized_model.cuda()

logger.info("SQuant Start!")
quantized_model.eval()
quantized_model(rand_input)
logger.info("SQuant has Done!")

@torch.no_grad()
def test(quantized_model_):
    quantized_model_.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = quantized_model_(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        _, predicted_5 = outputs.topk(5, 1, True, True)
        predicted_5 = predicted_5.t()
        correct_ = predicted_5.eq(targets.view(1, -1).expand_as(predicted_5))
        correct_5 += correct_[:5].reshape(-1).float().sum(0, keepdim=True).item()
        if batch_idx % 10 == 0 or batch_idx == len(testloader) - 1:
            logger.info('test: [batch: %d/%d ] | Loss: %.3f | Acc: %.3f%% (%d/%d)/ %.3f%% (%d/%d)'
                        % (batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct_5/total, correct_5, total))

        ave_loss = test_loss/total

    acc = 100.*correct/total

    logger.info("Final accuracy: %.3f" % acc)


### Load validation data
logger.info('==> Preparing data..')
testloader = getTestData(args.dataset,
                        batch_size=args.batch_size,
                        path=args.dataset_path,
                        for_inception=args.model.startswith('inception'))
### Validation
test(quantized_model)