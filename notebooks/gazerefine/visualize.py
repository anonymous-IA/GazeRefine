"""visualize.py — Plotting / overlay helpers.

``save_prediction`` reproduces the 4-panel (image / gaze / GT / prediction)
grid used during batch evaluation. ``overlay_heatmap`` and ``overlay_mask``
are lighter-weight, matplotlib-free PIL helpers meant for the Gradio demo
and the example notebook, where you typically want a single composited
image rather than a saved subplot file.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .constants import IMG_MEAN, IMG_STD


def _denormalize(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) uint8-range float array in [0, 1]."""
    mean = torch.tensor(IMG_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMG_STD, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def save_prediction(
    img: torch.Tensor,
    gaze_heatmap: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    name: str,
    out_dir: str = "predictions",
) -> None:
    """Save a 4-panel [image | gaze heatmap | GT mask | predicted mask] figure."""
    os.makedirs(out_dir, exist_ok=True)

    img_np = _denormalize(img)
    gaze_np = gaze_heatmap.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img_np); ax[0].set_title("Image")
    ax[1].imshow(gaze_np, cmap="jet"); ax[1].set_title("Gaze Heatmap")
    ax[2].imshow(gt_np, cmap="gray"); ax[2].set_title("GT mask")
    ax[3].imshow(pred_np, cmap="gray"); ax[3].set_title("Predicted mask")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=100, bbox_inches="tight")
    plt.close()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Composite a [0, 1] heatmap (any H'xW', will be resized) onto a PIL image
    using a jet colormap. Used to show the gaze prior over the original image."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heat_img = Image.fromarray((cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8))
    heat_img = heat_img.resize(image.size, resample=Image.BILINEAR)

    base = image.convert("RGB")
    return Image.blend(base, heat_img, alpha=alpha)


def overlay_mask(
    image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (255, 60, 60), alpha: float = 0.45
) -> Image.Image:
    """Composite a binary/soft [0, 1] mask onto a PIL image as a solid color wash."""
    mask = np.asarray(mask, dtype=np.float32)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)

    base = image.convert("RGB")
    color_layer = Image.new("RGB", base.size, color)
    composited = Image.composite(color_layer, base, mask_img.point(lambda p: int(p * alpha)))
    return composited
