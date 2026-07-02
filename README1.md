# GazeRefine 👁️

## Expert Gaze as a Test-Time Prompt for Training-Free Medical Image Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/<your-org>/gazerefine)

GazeRefine is a **training-free** and **zero-shot** medical image segmentation framework that uses **expert gaze** as an inference-time prompt to guide frozen DINOv3 representations.

Instead of segmentation masks, clicks, boxes, adapters, prompt encoders, or fine-tuning, GazeRefine transforms eye-tracking scanpaths into gaze-guided semantic prototypes and iteratively refines them in the frozen DINOv3 feature space.

> No training. No fine-tuning. No segmentation foundation model. Just expert gaze and frozen DINOv3 features.

---

# Overview

Medical image segmentation often requires dense annotations and task-specific training. GazeRefine explores a different paradigm: using the natural visual attention of clinicians to steer a frozen vision foundation model directly at test time.

### Key Features

* ✅ Training-free
* ✅ Zero-shot
* ✅ Human-in-the-loop segmentation
* ✅ Frozen DINOv3 backbone
* ✅ No segmentation masks for training
* ✅ No fine-tuning or adapters
* ✅ No SAM / MedSAM dependency

---

# Pipeline

<p align="center">
  <img src="/img/zero-shot_archi.jpg" width="100%">
</p>

<p align="center">
<b>Figure 1.</b> Overview of GazeRefine. Expert gaze scanpaths are converted into gaze priors that initialize foreground and background prototypes in the frozen DINOv3 feature space. Prototypes are refined through foreground-background cleaning and kNN affinity propagation before producing the final segmentation mask.
</p>

---

# Qualitative Results

<p align="center">
  <img src="/img/seg_results.png" width="80%">
</p>

<p align="center">
<b>Figure 2.</b> Qualitative comparison on Kvasir-SEG and NCI-ISBI datasets. Columns show: Input Image, gaze, Ground Truth, GazeRefine (Ours), GazeMedSAMv2, and GazeSAM.
</p>

Our method produces more complete object delineation while remaining fully training-free and operating directly on frozen DINOv3 features.

---

# ?

| Method                | Supervision          | Kvasir-SEG Dice (%) ↑ | NCI-ISBI Dice (%) ↑ |
| --------------------- | -------------------- | --------------------- | ------------------- |
| **GazeRefine (Ours)** | **Zero-Shot + Gaze** | **89.49 ± 0.12**      | **76.53 ± 0.17**    |


---

# Installation

```bash
git clone https://github.com/MohammedOussamaBEN/GazeRefine.git
cd GazeRefine

pip install -r requirements.txt
```

Requirements:

* Python ≥ 3.10
* PyTorch ≥ 2.1
* timm ≥ 1.0

---

# Quick Start

## Single Image Prediction

Input:

* Medical image (`.jpg`, `.jpeg`, `.png`)
* Eye-tracking fixation CSV

Example:

```bash
python scripts/predict_single.py \
    --image examples/image.png \
    --fixations examples/fixations.csv \
    --output output_mask.png
```

Expected fixation CSV format:

```csv
x,y,duration
340,221,180
356,228,145
368,244,205
...
```

Python API:

```python
from scripts.predict_single import predict

mask = predict(
    image_path="image.png",
    fixation_csv="fixations.csv"
)
```

---

# Datasets

## Kvasir-SEG

https://datasets.simula.no/kvasir-seg/

## NCI-ISBI 2013 Prostate MRI

https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/

Please follow the original dataset licenses and usage agreements.

---

# Repository Structure

```text
GazeRefine/
├── gazerefine/
├── configs/
├── scripts/
├── notebooks/
├── figures/
│   ├── pipeline.png
│   └── results.png
├── examples/
└── requirements.txt
```

---

# Citation

```bibtex
@inproceedings{gazerefine2026,
  title={GazeRefine: Expert Gaze as a Test-Time Prompt for Training-Free Medical Image Segmentation},
  author={Anonymous Authors},
  year={2026}
}
```

---
