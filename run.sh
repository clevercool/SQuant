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

# Resnet18
python -u main.py --model resnet18 --mode squant -wb 4 -ab 4 -s 12
python -u main.py --model resnet18 --mode squant -wb 6 -ab 6 -s 14
python -u main.py --model resnet18 --mode squant -wb 8 -ab 8 -s 25

#Resnet50
python -u main.py --model resnet50 --mode squant -wb 4 -ab 4 -s 12
python -u main.py --model resnet50 --mode squant -wb 6 -ab 6 -s 14
python -u main.py --model resnet50 --mode squant -wb 8 -ab 8 -s 25

#InceptionV3
python -u main.py --model inceptionv3 --mode squant -wb 4 -ab 4 -s 12
python -u main.py --model inceptionv3 --mode squant -wb 6 -ab 6 -s 14
python -u main.py --model inceptionv3 --mode squant -wb 8 -ab 8 -s 25

#SqueezeNext
python -u main.py --model sqnxt23_w2 --mode squant -wb 4 -ab 4 -s 12
python -u main.py --model sqnxt23_w2 --mode squant -wb 6 -ab 6 -s 14
python -u main.py --model sqnxt23_w2 --mode squant -wb 8 -ab 8 -s 25

#ShuffleNet
python -u main.py --model shufflenet_g1_w1 --mode squant -wb 6 -ab 6 -s 14
python -u main.py --model shufflenet_g1_w1 --mode squant -wb 8 -ab 8 -s 25
