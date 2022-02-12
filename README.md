# SQuant ICLR2022
```
@inproceedings{
guo2022squant,
title={{SQ}uant: On-the-Fly Data-Free Quantization via Diagonal Hessian Approximation},
author={Cong Guo and Yuxian Qiu and Jingwen Leng and Xiaotian Gao and Chen Zhang and Yunxin Liu and Fan Yang and Yuhao Zhu and Minyi Guo},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=JXhROKNZzOc}
}
```

## Environment

```bash
conda create -n squant python==3.8
conda activate squant
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install pytorchcv==0.0.51
pip install ./squant-kernel
```

## ImageNet Path

Please set your dataset path using "--dataset_path /your/imagenet_path".

## Run SQuant

```bash
# Quantize resnet-18 in 4-bit with SQuant
python -u main.py --model resnet18 --mode squant -wb 4 -ab 4  -s 12

# Test all models in Table 1, 2, and 3 of the ICLR2022 paper.
./run.sh

# Ablation study results in Table 4 and 5 of the ICLR2022 paper.
./run_ablation.sh
```

## Quantization Rules
1. The weight and activation should be quantized correctly when inference.
    - Weight is per-(output) channel quantization.
      - Affine quantzation, same as ZeroQ and GDFQ.
      - For all results of SQuant, we only set the min/max as the quantization range.
    - Activation (input) must be per-tensor quantization: all elements must have ONE quantization range.
      - The clip method is a BN-based method from DFQ with a wider clip range.
      - All models have the same range setting (alpha) of the BN-based method under the same bit-width (similar with [ACIQ (alpha)](https://github.com/submission2019/AnalyticalScaleForIntegerQuantization/blob/3246ee8cbfb747d7ef821c8cecc50283a73eaf92/pytorch_quantizer/quantization/qtypes/int_quantizer.py#L10)).
    - The first layer input does not need quantization. 
      - Same as other frameworks, such as ZeroQ and GDFQ.
    - The last layer (FC) input uses 8bit quantization. 
      - Lower bit-width than other frameworks, such as ZeroQ and GDFQ.
      - Note that other frameworks didn't quantize the FC input activation after average pooling.


2. The quantization procedure MUST not involve any real dataset information. (data-free)
    - Weight quantization should finish before the inference without training/validation dataset.
    - Activation can only quantize in inference runtime. However, its quantization range should be set before inference without a training/validation dataset.

3. Test all models
    - All models use the same hyperparameters under the same bit-width.
    - The results should fit the results (with $\pm 0.2$ error) presented in the manuscript.

3. Ablation study
    - Reproduce the same results as in the manuscript.
