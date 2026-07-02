"""
GazeRefine
==========
Expert gaze as a test-time prompt for training-free medical image segmentation.

Quick start
-----------
    from scripts.predict_single import predict
    mask = predict(image_path="image.png", fixation_csv="fixations.csv")
    mask.save("output_mask.png")
"""

from .backbone  import FrozenDINOv3
from .gaze      import get_scanpath, load_fixation_csv, generate_gaze_heatmap
from .model     import GazeRefine, knn_affinity_refinement
from .metrics   import compute_metrics
from .visualize import save_prediction, overlay_heatmap, overlay_mask

__all__ = [
    "FrozenDINOv3",
    "get_scanpath",
    "load_fixation_csv",
    "generate_gaze_heatmap",
    "GazeRefine",
    "knn_affinity_refinement",
    "compute_metrics",
    "save_prediction",
    "overlay_heatmap",
    "overlay_mask",
]

__version__ = "0.1.0"
