"""
gaze.py — Turn an eye-tracking scanpath into a spatial prior.

Implements the gaze heatmap from the paper:

    H(u, v) = Σ_m  d̃_m · exp( -((u-xm)² + (v-ym)²) / 2σ² )

i.e. a duration-weighted sum of 2D Gaussians, one per fixation, on the
DINOv3 patch grid.  The result is min-max normalized and used as the soft
foreground prior W_fg; (1-H) gives the background complement W_bg.

───────────────────────────────────────────────────────────────────────────
Simple CSV format (what predict_single.py / the README document):

    x,y,duration
    340,221,180
    356,228,145
    ...

    x, y     — fixation position in *raw pixel* coordinates of the original image.
               load_fixation_csv() divides these by the image width/height to
               produce the normalized [0,1] coords the model needs.
    duration — fixation duration in any consistent unit (ms typical).

───────────────────────────────────────────────────────────────────────────
EyeLink/Tobii batch format (used by the full-dataset loaders in datasets.py):

    IMAGE                  image filename the fixation belongs to
    CURRENT_FIX_INDEX      fixation order within the trial
    CURRENT_FIX_X          fixation x, already normalized to [0, 1]
    CURRENT_FIX_Y          fixation y, already normalized to [0, 1]
    CURRENT_FIX_DURATION   fixation duration (ms or any consistent unit)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── EyeLink-style column names (used by datasets.py / batch evaluation) ──
TIME_COL    = "CURRENT_FIX_DURATION"
FIX_IDX_COL = "CURRENT_FIX_INDEX"
FIX_X_COL   = "CURRENT_FIX_X"
FIX_Y_COL   = "CURRENT_FIX_Y"
IMG_COL     = "IMAGE"


# ═══════════════════════════════════════════════════════════════════════════
#  Simple CSV loader  (README-documented format)
# ═══════════════════════════════════════════════════════════════════════════

def load_fixation_csv(
    csv_path: str,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Load a simple  x,y,duration  fixation CSV and return a
    ``(1, L, 3)`` scanpath tensor ready to pass into GazeRefine.

    Parameters
    ----------
    csv_path     : path to a CSV with columns ``x``, ``y``, ``duration``.
                   ``x`` and ``y`` are raw pixel coordinates of the
                   *original* (not resized) image.
    image_width  : original image width  — used to normalize x → [0, 1].
    image_height : original image height — used to normalize y → [0, 1].

    Returns
    -------
    (1, L, 3) float32 tensor  [x_norm, y_norm, t_norm].
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # accept both `x`/`y` and `fix_x`/`fix_y` column naming variants
    x_col = "x" if "x" in df.columns else "fix_x"
    y_col = "y" if "y" in df.columns else "fix_y"
    d_col = "duration"

    x = df[x_col].to_numpy(np.float32) / image_width
    y = df[y_col].to_numpy(np.float32) / image_height
    t = df[d_col].to_numpy(np.float32)

    # normalize duration so the longest fixation = 1.0
    t_min, t_max = t.min(), t.max()
    t = (t - t_min) / (t_max - t_min + 1e-8)

    seq = np.stack([x, y, t], axis=1).astype(np.float32)  # (L, 3)
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, L, 3)


# ═══════════════════════════════════════════════════════════════════════════
#  Batch-evaluation helpers  (used by datasets.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_scanpath(df: pd.DataFrame, max_len: int, time_col: str = TIME_COL) -> torch.Tensor:
    """Build a zero-padded ``(max_len, 3)`` scanpath tensor from a
    fixation-report DataFrame already filtered to a single image.

    Columns: ``[CURRENT_FIX_X, CURRENT_FIX_Y, CURRENT_FIX_DURATION]``
    already normalized to [0,1] as produced by the EyeLink/Tobii export
    pipeline (datasets.py normalizes these before calling this function).
    """
    if len(df) == 0:
        return torch.zeros(max_len, 3)

    df = df.sort_values(FIX_IDX_COL)
    x = df[FIX_X_COL].to_numpy(np.float32)
    y = df[FIX_Y_COL].to_numpy(np.float32)
    t = df[time_col].to_numpy(np.float32)

    t_min, t_max = t.min(), t.max()
    t = (t - t_min) / (t_max - t_min + 1e-8)

    seq = np.stack([x, y, t], axis=1)
    if len(seq) > max_len:
        seq = seq[:max_len]
    if len(seq) < max_len:
        seq = np.concatenate([seq, np.zeros((max_len - len(seq), 3), np.float32)], axis=0)

    return torch.tensor(seq, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Heatmap generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_gaze_heatmap(
    fixation: torch.Tensor, h_patch: int, sigma: float = 2.0
) -> torch.Tensor:
    """Convert a batch of scanpaths into duration-weighted Gaussian heatmaps.

    Parameters
    ----------
    fixation : (B, L, 3) tensor  [x_norm, y_norm, t_norm], zero-padded.
    h_patch  : patch-grid side length (74 for IMG_SIZE=1184 with patch_size=16).
    sigma    : Gaussian spread in patch units. Larger → softer, more diffuse
               foreground prior; smaller → tighter, closer to exactly the
               fixated patches.

    Returns
    -------
    (B, h_patch, h_patch) float32 tensor, each sample min-max normalized
    to [0, 1] independently.
    """
    B, L, _ = fixation.shape
    device = fixation.device

    y_coords = torch.arange(h_patch, device=device, dtype=torch.float32)
    x_coords = torch.arange(h_patch, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    heatmaps = []
    for b in range(B):
        valid = fixation[b].abs().sum(dim=-1) > 1e-8   # filter zero-padding rows
        valid_fix = fixation[b][valid]

        h_b = torch.zeros(h_patch, h_patch, device=device)
        if len(valid_fix) > 0:
            for k in range(valid_fix.shape[0]):
                x_k, y_k, t_k = valid_fix[k, 0], valid_fix[k, 1], valid_fix[k, 2]
                gx = x_k * (h_patch - 1)
                gy = y_k * (h_patch - 1)
                dist_sq = (grid_x - gx) ** 2 + (grid_y - gy) ** 2
                h_b += t_k * torch.exp(-dist_sq / (2 * sigma ** 2))

            h_max = h_b.max()
            if h_max > 1e-8:
                h_b = h_b / h_max
        heatmaps.append(h_b)

    return torch.stack(heatmaps, dim=0)
