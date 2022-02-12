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

# SQuant granularity
python -u main.py --model resnet18 --mode squant-e -wb 3 -daq
python -u main.py --model resnet18 --mode squant-c -wb 3 -daq
python -u main.py --model resnet18 --mode squant-k -wb 3 -daq
python -u main.py --model resnet18 --mode squant -wb 3 -daq
python -u main.py --model resnet18 --mode squant-e -wb 4 -daq
python -u main.py --model resnet18 --mode squant-c -wb 4 -daq
python -u main.py --model resnet18 --mode squant-k -wb 4 -daq
python -u main.py --model resnet18 --mode squant -wb 4 -daq

# Ablation 2
python -u main.py --model resnet18 --mode squant -wb 3 -daq
python -u main.py --model resnet18 --mode squant -wb 4 -daq
python -u main.py --model resnet18 --mode squant -wb 5 -daq