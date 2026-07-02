"""
GazeRefine
==========
Expert gaze as a test-time prompt for training-free medical image segmentation.

This package exposes the full zero-shot pipeline described in the GazeRefine
paper: a frozen DINOv3 backbone, gaze -> heatmap conversion, gaze-anchored
foreground/background prototype construction, and recurrent refinement
(contrastive background cleaning + kNN affinity propagation).

Nothing in this package is trained. There are no learned weights other than
the frozen, pretrained DINOv3 backbone loaded from `timm`.
"""

from .backbone import FrozenDINOv3
from .gaze import get_scanpath, generate_gaze_heatmap
from .model import GazeRefine, knn_affinity_refinement
from .metrics import compute_metrics
from .visualize import save_prediction, overlay_heatmap, overlay_mask

__all__ = [
    "FrozenDINOv3",
    "get_scanpath",
    "generate_gaze_heatmap",
    "GazeRefine",
    "knn_affinity_refinement",
    "compute_metrics",
    "save_prediction",
    "overlay_heatmap",
    "overlay_mask",
]

__version__ = "0.1.0"
