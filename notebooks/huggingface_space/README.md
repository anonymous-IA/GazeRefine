---
title: GazeRefine
emoji: 👁️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Zero-shot, training-free gaze-guided medical segmentation
---

# GazeRefine — Expert Gaze as a Test-Time Prompt

Interactive demo for **GazeRefine**, a training-free, zero-shot framework
that turns expert eye-gaze into an inference-time prompt for medical image
segmentation. Frozen DINOv3 patch features + gaze-anchored
foreground/background prototypes + recurrent contrastive cleaning + kNN
affinity propagation — no masks, no clicks-as-boxes, no fine-tuning, no
adapters, no prompt encoder.

## How to use
1. Upload a colonoscopy or grayscale-MRI-style image.
2. Click on the image 1–5 times where a clinician's gaze would land on the
   structure of interest (a polyp, the prostate, ...). Each click adds a
   numbered fixation marker; the slider controls that fixation's relative
   duration/weight before your next click.
3. Pick a hyperparameter preset (tuned per-modality, see the paper).
4. Press **Run GazeRefine** to get the gaze-prior overlay and the predicted
   segmentation mask.

## Notes
- Inference uses a frozen `vit_large_patch16_dinov3.lvd1689m` backbone from
  `timm`. First run will download the checkpoint.
- CPU inference works but is slow; a GPU Space is recommended for a smooth
  demo.
- This Space is for research/demonstration only — it is **not** a clinical
  diagnostic tool.

Full code, configs, and the unified evaluation pipeline:
[GitHub repository](https://github.com/<your-org>/gazerefine).
