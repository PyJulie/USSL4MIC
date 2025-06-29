# Universal Semi-Supervised Learning for Medical Image Classification

This repository implements the Universal Semi-Supervised Learning (USSL) framework for medical image classification as described in the paper "Universal Semi-Supervised Learning for Medical Image Classification" (arXiv:2304.04059).

## Overview

The USSL framework addresses the challenge of semi-supervised learning in open-set scenarios where unlabeled data may contain samples from:
- Unknown classes (UKC) that are not present in the labeled training data
- Unknown domains (UKD) with different imaging conditions or datasets

## Key Components

1. **Dual-path Outlier Estimation**: Identifies samples from unknown classes using both feature and classifier levels
2. **VAE-based Domain Detection**: Uses Variational AutoEncoder to detect samples from unknown domains
3. **Domain Adaptation**: Applies adversarial training to adapt features from unknown domains
4. **Unified Training**: Combines SSL techniques with domain adaptation for optimal performance

## Project Structure

```
USSL/
├── configs/                 # Configuration files
├── data/                    # Data loading and preprocessing
├── models/                  # Model architectures
├── utils/                   # Utility functions
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
└── requirements.txt         # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config configs/dermatology.yaml
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

## Datasets

The framework supports various medical image datasets:
- ISIC 2019 (Dermatology)
- Dermnet (Dermatology)
- Ophthalmology datasets

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ju2024universal,
  title={Universal Semi-supervised Learning for Medical Image Classification},
  author={Ju, Lie and Wu, Yicheng and Feng, Wei and Yu, Zhen and Wang, Lin and Zhu, Zhuoting and Ge, Zongyuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={355--365},
  year={2024},
  organization={Springer}
}
``` 
