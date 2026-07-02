"""metrics.py — Dice / IoU against a binarized prediction."""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_metrics(preds: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> dict:
    """
    preds, mask : (B, 1, H, W). ``preds`` is the continuous [0, 1] GazeRefine
    output, ``mask`` is the ground-truth binary mask.
    """
    pred_bin = (preds > thr).float()
    tp = (pred_bin * mask).sum((1, 2, 3))
    fp = (pred_bin * (1 - mask)).sum((1, 2, 3))
    fn = ((1 - pred_bin) * mask).sum((1, 2, 3))

    dice_per = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou_per = tp / (tp + fp + fn + 1e-8)

    return {
        "dice": dice_per.mean().item(),
        "iou": iou_per.mean().item(),
        "dice_per": dice_per,
        "iou_per": iou_per,
    }
