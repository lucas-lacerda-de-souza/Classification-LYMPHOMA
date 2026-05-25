# 🧠 When Artificial Intelligence Falls Short: Improving Head and Neck Lymphoma Classification through Multimodal Analysis in a Multicenter Cohort.

**Author:** Lucas Lacerda de Souza
**Year:** 2026

---

# 📘 Project Overview

This repository presents a computationally explainable multimodal artificial intelligence framework developed for the histopathological classification of head and neck lymphoid lesions using:

* Histopathological H&E image analysis
* Nuclear morphometric descriptors
* Clinicopathological variables
* Vision transformer–based segmentation
* Attention-based multiple instance learning (MIL)
* Explainable artificial intelligence approaches

The framework was designed to classify four diagnostic categories:

| Class | Diagnostic Category            |
| ----- | ------------------------------ |
| 0     | Aggressive B-cell lymphoma     |
| 1     | Indolent/small B-cell lymphoma |
| 2     | NK/T-cell lymphoma             |
| 3     | Reactive lymphoid lesion       |

The pipeline integrates:

* UNI foundation model embeddings
* CellViT++ nuclear segmentation
* Morphometric feature extraction
* Multimodal deep learning
* SHAP explainability analysis
* Attention-based MIL aggregation
* AI-guided diagnostic support workflows

The system was developed as a research-oriented decision-support framework for computational hematopathology and is not intended for autonomous clinical diagnosis.

# 🔬 Computational Pipeline

<img width="642" height="824" alt="Figure 1" src="https://github.com/user-attachments/assets/f0e1686b-78ad-4133-9c66-4812e3055cb0" />

The computational framework integrates:

1. Whole-slide image preprocessing
2. Patch extraction from H&E slides
3. UNI foundation model feature extraction
4. CellViT++ nuclear segmentation
5. Morphometric feature extraction
6. Multimodal fusion learning
7. Attention-based MIL aggregation
8. SHAP explainability analysis
9. External multicentre validation

The workflow combines image-derived representations, nuclear morphology, and structured clinicopathological variables into a unified multimodal classification framework.

# 🖥️ Environment and Hardware

All experiments were performed using the following configuration:

| Component        | Specification                    |
| ---------------- | -------------------------------- |
| Operating System | Ubuntu 20.04.1 LTS               |
| Python           | 3.12.11                          |
| PyTorch          | 2.8.0 (CUDA 12.8)                |
| CPU              | Intel Xeon W-2295                |
| RAM              | 125 GB                           |
| GPU              | 3 × NVIDIA RTX 3090 (24 GB each) |

The environment supports:

* Multi-GPU training
* Vision transformer inference
* Whole-slide image processing
* Large-scale histopathological workflows
* Mixed precision optimisation

# 📦 Environment and Dependencies

## Conda Channels

```bash
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels defaults
```

## Core Dependencies

```bash
python=3.12.11
pytorch=2.8.0
torchvision=0.19.0
torchaudio=2.8.0
cudatoolkit=12.8
numpy=1.26.4
pandas=2.2.3
scikit-learn=1.5.2
matplotlib=3.9.2
seaborn=0.13.2
pillow=10.4.0
tqdm=4.66.5
openpyxl=3.1.5
```

These libraries were used for:

* Deep learning
* Whole-slide image analysis
* Nuclear morphometry
* Segmentation
* Statistical analysis
* Explainability
* Visualisation

# 🧠 Model Architectures

The repository includes the following computational frameworks:

| Model                   | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| UNI Foundation Model    | Histopathological feature extraction                      |
| CellViT++               | Nuclear instance segmentation                             |
| Attention-based MIL     | Slide-level aggregation                                   |
| Multimodal Fusion Model | Integration of image, morphometric, and clinical features |
| XGBoost + SHAP          | Traditional machine learning and explainability           |

CellViT++ and UNI were used using their original implementations without architectural modifications. Only downstream integration, inference, and analysis pipelines are included in this repository.

# 🧬 Features Used

The multimodal framework integrates:

## Histopathological Features

* H&E image patches
* Whole-slide image representations
* UNI-derived embeddings
* Attention-based spatial representations

## Segmentation Features

* Nuclear masks
* Cell-level embeddings
* CellViT++ segmentation outputs

## Morphometric Features

* Nuclear area
* Nuclear perimeter
* Circularity
* Eccentricity

## Clinicopathological Features

* Age
* Sex
* Anatomical location

# 📊 Evaluation Metrics

The computational framework supports evaluation using:

## Classification Metrics

* Accuracy
* Precision
* Recall
* Weighted F1-score
* Sensitivity
* Specificity
* ROC AUC
* Cohen’s kappa
* Balanced accuracy

## Segmentation Metrics

* Dice coefficient
* Intersection over Union (IoU)
* Precision
* Recall

## Calibration Metrics

* Expected calibration error (ECE)
* Brier score

## Explainability

* SHAP feature importance
* Attention-based interpretability
* Feature attribution analysis

# 📂 Repository Structure

```text
DATA/                    → Synthetic example data and directory structures
MODELS/                  → Model architectures and inference pipelines
RESULTS/                 → Study results and supplementary outputs

INFERENCE.py             → Inference script
MODEL_CARD.md            → Model documentation
README.md                → Repository documentation
REQUIREMENTS.txt         → Dependency list
LICENSE.txt              → Repository license
```

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/lucas-lacerda-de-souza/Classification-LYMPHOMA.git

cd Classification-LYMPHOMA
```

## Create Environment

```bash
conda env create -f environment.yml

conda activate lymphoma-ai
```

# ⚡ Quick Start

## Run Inference

```bash
python INFERENCE.py \
    --input_dir ./data/example_slides \
    --output_dir ./results/
```

# 🧠 Compliance with TRIPOD-AI and CLAIM Guidelines

This repository was structured according to:

* TRIPOD-AI
* CLAIM 2024

to improve:

* Transparency
* Reproducibility
* Explainability
* Clinical interpretability

The repository includes:

* Dataset organisation
* Model architecture documentation
* Training configuration
* External validation
* Explainability methods
* Ethical considerations
* Intended use statements

# ⚖️ Ethics

This study was approved by:

* Piracicaba Dental School, University of Campinas, Brazil
  Protocol: `67064422.9.1001.5418`

* West of Scotland Research Ethics Service
  Protocol: `20/WS/0017`

The study followed the principles of the Declaration of Helsinki.
All collected data were fully anonymised.

# 🔒 Data Availability

Due to ethical restrictions and patient confidentiality regulations:

* Whole-slide images are not publicly distributed
* Raw clinical metadata are not publicly shared
* Patient-identifiable data are not included

To support reproducibility, this repository provides:

* Synthetic organisational examples
* Representative patch structures
* Example inference pipelines
* Documentation and reproducibility guidelines

# 💻 Code Availability

The complete computational framework is publicly available on GitHub:

[https://github.com/lucas-lacerda-de-souza/Classification-LYMPHOMA](https://github.com/lucas-lacerda-de-souza/Classification-LYMPHOMA)

The repository includes:

* Inference scripts
* Model architectures
* Evaluation pipelines
* Explainability workflows
* Documentation

# 🧠 Model Weights

Pretrained weights and checkpoints are available through Zenodo:

[https://doi.org/10.5281/zenodo.17661989](https://doi.org/10.5281/zenodo.17661989)

Available resources include:

* UNI checkpoints
* MIL models
* CellViT++ segmentation weights
* Multimodal classifiers

---

# 📚 Citation

```bibtex
@article{delasouza2025lymphoma,
  title={When Artificial Intelligence Falls Short: Improving Head and Neck Lymphoma Classification through Multimodal Analysis in a Multicenter Cohort},
  author={Souza, Lucas Lacerda de and collaborators},
  journal={2026},
  year={2026}
}
```
