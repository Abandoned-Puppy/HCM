# Hyperspherical Confidence Mapping (HCM)

This repository provides the official implementation of Hyperspherical Confidence Mapping (HCM)
including both:

- Full CIFAR-10 + multi-OOD evaluation pipeline (ResNet-18)

- 1D regression toy example showing how HCM applies to scalar targets

- Two Moons (Updated!)

This codebase is released to support reproducibility for anonymous peer review.
All scripts are self-contained and lightweight.

---

## Overview

HCM decomposes a model's output into:

- a **unit-norm direction vector** `d`
- a **scalar magnitude** `R`

The prediction is reconstructed as:

$\hat{y} = \hat{R} \hat{d}.$

Uncertainty is quantified by measuring the deviation of `d` from the hypersphere constraint:

$u(x) = \hat{R}\big|\|\hat{d}\|_2-1\big|,$

(in classification case, $\big|\|\hat{d}\|_2-1\big|$)

This repository implements HCM on **ResNet-18 (CIFAR-10)**, but the components are modular and can be applied to any backbone.

---

## Dependencies

Only the following packages are required:
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.23.0
pillow>=10.0.0
scikit-learn>=1.3.0
thop>=0.1.1.post2209072238 # optional (only for FLOPs/MACs)


---

## File Structure

├── hcm_cifar10.py # Full training + OOD evaluation + runtime instrumentation
├── Toy_example.py          # 1D regression example (scalar target case)
├── README.md
├── requirements.txt
└── data/ # Dataset directory (created manually)
    ├── cifar-10-batches-py/ # Automatically downloaded by torchvision
    ├── cifar-100-python/ # Automatically downloaded
    ├── svhn/ # Automatically downloaded
    ├── dtd/ # Automatically downloaded
    ├── tiny-imagenet-200/ # Manual download required (see below)
    └── val/ # Places365 val folder (manual)

---


## CIFAR-10 Training
Train HCM on CIFAR-10:
>python hcm_cifar10.py --epochs 100 --batch 64

Key arguments:
| Argument   | Description                    |
| ---------- | ------------------------------ |
| `--epochs` | Training epochs (default: 100) |
| `--batch`  | Batch size (default: 64)       |
| `--seed`   | Random seed                    |
| `--lr`     | Learning rate (default: 0.1)   |
| `--oods`   | List of OOD datasets           |


---

## OOD Evaluation

The script automatically evaluates HCM on:

- CIFAR-100
- SVHN
- DTD
- TinyImageNet
- MNIST
- Places365

Metrics computed:

- AUROC
- AUPR
- FPR@95TPR

Examples will be printed during execution.

---



##  Dataset Setup

Most datasets are handled automatically, but some OOD datasets require manual download.

### ** Automatically downloaded (no action needed)**

These datasets are fetched by `torchvision` on first run:

- CIFAR-10  
- CIFAR-100  
- SVHN  
- MNIST  
- DTD  

They will appear under:
data/
cifar-10-batches-py/
cifar-100-python/
svhn/
mnist/
dtd/

---

### ** Manually downloaded datasets (required for full OOD evaluation)**

Some large datasets must be downloaded manually due to license restrictions.

---

### **1. TinyImageNet (manual)**  
Download from the official source:

https://www.kaggle.com/c/tiny-imagenet

Extract it so that the directory looks like:
data/tiny-imagenet-200/
├── train/
├── val/
└── test/
└── images/

---

### **2. Places365 (manual)**

Download validation images from:

http://places2.csail.mit.edu/

Extract under:
data/val/
├── airfield
├── airplane_cabin
...

---
## 1D Regression Toy Example

This script demonstrates how HCM applies to 1-dimensional regression,
addressing the reviewer’s question on how hyperspherical decomposition works when the target is a scalar.

>python Toy_example.py --mode mixture --epochs 500

Available modes:

| Mode              | Description                                |
| ----------------- | ------------------------------------------ |
| `gaussian_hetero` | Heteroscedastic Gaussian noise             |
| `laplace_hetero`  | Heteroscedastic Laplace noise              |
| `mixture`         | Bimodal Gaussian mixture noise             |
| `multi_y_x2`      | Multi-valued regression: ($y = \pm\sqrt{x}$) |


Output:
>hcm_1d_result.png

>Example band visualization (mean ± 1σ, 2σ, 3σ) is automatically saved.
