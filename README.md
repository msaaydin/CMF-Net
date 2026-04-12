# CMF-Net: Cross-Modal Fusion Network for Automated Identification of Transthyretin Cardiac Amyloidosis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"Automated Identification of Transthyretin Cardiac Amyloidosis Using Cross-Modal Neural Networks on 99mTc Pyrophosphate Imaging and Clinical Data"**.

## Overview

We propose two multimodal deep learning frameworks — **Late Fusion (LF)** and **Cross-Modal Fusion Network (CMF-Net)** — that combine **99mTc-PYP scintigraphy images** with **clinical metadata** (age, gender, BMI, EF, etc.) to automate the detection of transthyretin cardiac amyloidosis (ATTR-CA).

<p align="center">
  <img src="https://github.com/msaaydin/CMF-Net/blob/main/data/data/cmfnet_architecture.png" alt="CMF-Net Architecture" width="700"/>
</p>

### Key Results

| Model | Backbone | Accuracy | F1-Score |
|-------|----------|----------|----------|
| Standard CNN | EfficientNet | 0.8182 | 0.8333 |
| Late Fusion | EfficientNet | 0.8636 | 0.8750 |
| **CMF-Net** | **EfficientNet** | **0.9091** | **0.9143** |

> CMF-Net with EfficientNet backbone achieved **90.9% accuracy** and **91.4% F1-score**, outperforming both image-only CNNs and Late Fusion models.

## Repository Structure

```
CMF-Net/
├── cmnet.py                    # Main CMF-Net training & evaluation (original split)
├── cmnet_revision_loo.py       # Leave-One-Out cross-validation
├── cmnet_revision_mccv.py      # Monte Carlo Cross-Validation (K=100)
├── late_fusion.py              # Late Fusion model training & evaluation
├── standart_CNN.py             # Standard CNN (image-only) baseline
├── helpers.py                  # Utility functions (data loading, metrics, etc.)
├── data.xlsx                   # Clinical metadata and split information
├── best_model_9091.pth         # Best pre-trained CMF-Net model (EfficientNet)
├── data/
│   └── data/
│       ├── positive/           # 99mTc-PYP images (ATTR-CA positive)
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── negative/           # 99mTc-PYP images (ATTR-CA negative)
│           ├── train/
│           ├── val/
│           └── test/
├── grad_cam_results/           # Grad-CAM visualizations for model interpretability
└── README.md
```

## Dataset

The dataset consists of **109 patients** (62 positive, 47 negative) who underwent 99mTc-PYP scintigraphy. Each sample includes:

- **Image data**: 99mTc-PYP scintigraphy scan (anterior view)
- **Clinical metadata**: Age, Gender, BMI, Ejection Fraction (EF), and other clinical features stored in `data.xlsx`

The dataset is split into **70% training**, **10% validation**, and **20% test** sets with stratified sampling.

## Installation

### Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn opencv-python tqdm openpyxl
```

### Tested Environment

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (GPU recommended)

## Usage

### 1. Standard Training & Evaluation (Original Split)

```bash
python cmnet.py
```

This trains CMF-Net with all backbone models (DenseNet121, ResNet18/34/50, MobileNetV2/V3, EfficientNet) using the predefined train/val/test split. Results are saved to the `cmnet/` directory.

### 2. Leave-One-Out Cross-Validation

```bash
python cmnet_revision_loo.py
```

Performs LOO evaluation where each sample is used as the test set exactly once. Results are saved to the `rev1/` directory.

### 3. Monte Carlo Cross-Validation (MCCV)

```bash
python cmnet_revision_mccv.py
```

Performs MCCV with **K=100 iterations**, randomly splitting the data into 70/10/20 (train/val/test) with stratified sampling in each iteration. Early stopping (patience=10) is applied based on validation F1-score. Results are saved to the `rev_mccv/` directory.

### 4. Late Fusion & Standard CNN Baselines

```bash
python late_fusion.py
python standart_CNN.py
```

## Supported Backbone Models

| Backbone | Parameters | ImageNet Pretrained |
|----------|-----------|-------------------|
| DenseNet121 | 7.0M | ✅ |
| ResNet18 | 11.2M | ✅ |
| ResNet34 | 21.3M | ✅ |
| ResNet50 | 23.5M | ✅ |
| MobileNetV2 | 2.2M | ✅ |
| MobileNetV3-Small | 1.5M | ✅ |
| EfficientNet-B0 | 4.0M | ✅ |

## Model Architecture

CMF-Net consists of three branches:

1. **CNN Branch**: Extracts visual features from 99mTc-PYP images using a pretrained backbone
2. **Text Branch**: Processes clinical metadata through fully connected layers
3. **Cross-Modal Fusion**: Combines both modalities via concatenation and joint classification layers

```
Image Input ──► CNN Backbone ──► Visual Features ──┐
                                                    ├──► Fusion ──► FC Layers ──► Prediction
Clinical Data ──► FC Layers ──► Text Features ──────┘
```

## Evaluation Methods

### Monte Carlo Cross-Validation (MCCV)

To provide robust and unbiased performance estimates, we employed MCCV with K=100 iterations:

- **Stratified random split** (70% train / 10% val / 20% test) in each iteration
- **Early stopping** (patience=10) based on validation F1-score
- **Statistical significance** tested via pairwise Wilcoxon signed-rank tests

### MCCV Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| DenseNet121 | 0.789 ± 0.099 | 0.809 ± 0.112 | 0.885 ± 0.163 | 0.828 ± 0.096 |
| ResNet18 | 0.725 ± 0.125 | 0.749 ± 0.136 | 0.891 ± 0.193 | 0.788 ± 0.113 |
| ResNet34 | 0.757 ± 0.127 | 0.788 ± 0.137 | 0.881 ± 0.194 | 0.805 ± 0.121 |
| ResNet50 | 0.757 ± 0.112 | 0.757 ± 0.120 | 0.922 ± 0.106 | 0.821 ± 0.075 |
| MobileNetV2 | 0.833 ± 0.094 | 0.856 ± 0.108 | 0.892 ± 0.127 | 0.862 ± 0.077 |
| MobileNetV3 | 0.824 ± 0.111 | 0.875 ± 0.119 | 0.855 ± 0.173 | 0.845 ± 0.122 |
| **EfficientNet** | **0.981 ± 0.030** | **0.971 ± 0.045** | **1.000 ± 0.000** | **0.985 ± 0.024** |

> EfficientNet significantly outperforms all other backbones (Wilcoxon signed-rank test, p < 0.001).

## Grad-CAM Visualizations

Model interpretability is provided through Grad-CAM heatmaps, highlighting the image regions most influential for the classification decision. Pre-generated visualizations are available in the `grad_cam_results/` directory.

## Pre-trained Model

The best-performing model (`best_model_9091.pth`) is included in the repository. To load and use it:

```python
import torch
from cmnet import MultimodalClassifier

model = MultimodalClassifier(model_name="efficientnet", num_text_features=5, num_classes=2)
model.load_state_dict(torch.load("best_model_9091.pth", map_location="cpu"))
model.eval()
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{aydin2025cmfnet,
  title={Automated Identification of Transthyretin Cardiac Amyloidosis Using Cross-Modal Neural Networks on 99mTc Pyrophosphate Imaging and Clinical Data},
  author={Aydin, M. S. and others},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an [issue](https://github.com/msaaydin/CMF-Net/issues) on GitHub.
