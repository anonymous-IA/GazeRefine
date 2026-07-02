"""
gaze.py — Turn an eye-tracking scanpath into a spatial prior.

Implements Eq. (gaze heatmap) from the paper:

    H(u, v) = sum_m  d_m * exp( -((u - x_m)^2 + (v - y_m)^2) / (2 * sigma^2) )

i.e. a duration-weighted sum of 2D Gaussians, one per fixation, evaluated on
the DINOv3 patch grid. The result is min-max normalized and used directly as
the soft foreground prior; ``1 - H`` is the background complement.

Expected CSV columns (standard EyeLink-style exports, e.g. SR Research /
Tobii fixation reports):

    IMAGE                   image filename the fixation belongs to
    CURRENT_FIX_INDEX       fixation order within the trial
    CURRENT_FIX_X            fixation x in image-normalized [0, 1] coords
    CURRENT_FIX_Y            fixation y in image-normalized [0, 1] coords
    CURRENT_FIX_DURATION     fixation duration (ms or any consistent unit)

If your fixation coordinates are in raw pixels rather than [0, 1], divide by
the screen/image width and height before building the CSV — GazeRefine
expects normalized coordinates so it stays resolution-independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

TIME_COL = "CURRENT_FIX_DURATION"
FIX_IDX_COL = "CURRENT_FIX_INDEX"
FIX_X_COL = "CURRENT_FIX_X"
FIX_Y_COL = "CURRENT_FIX_Y"
IMG_COL = "IMAGE"


def get_scanpath(df: pd.DataFrame, max_len: int, time_col: str = TIME_COL) -> torch.Tensor:
    """Build a zero-padded ``(max_len, 3)`` scanpath tensor from a fixation-report
    DataFrame already filtered down to a single image.

    Columns of the output: ``[x_norm, y_norm, t_norm]``, all in ``[0, 1]``.
    ``t_norm`` is the fixation duration min-max normalized within that image's
    own scanpath (longest fixation -> 1.0). Rows are sorted by
    ``CURRENT_FIX_INDEX`` so temporal order is preserved (only the spatial +
    duration channels are currently used by the model, but order is kept for
    future scanpath-aware extensions).
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


def generate_gaze_heatmap(
    fixation: torch.Tensor, h_patch: int, sigma: float = 2.0
) -> torch.Tensor:
    """Convert a batch of scanpaths into duration-weighted Gaussian heatmaps on
    the DINOv3 patch grid.

    Parameters
    ----------
    fixation : (B, L, 3) tensor of ``[x_norm, y_norm, t_norm]``, zero-padded.
    h_patch : side length of the square patch grid (e.g. 37 for a 518/14 ViT,
        37*2=74 for the 518/16 setup used in our experiments — see
        ``gazerefine.model.H_PATCH``).
    sigma : standard deviation (in patch units) of each fixation's Gaussian.
        Larger sigma -> a softer, more diffuse foreground prior; smaller
        sigma -> a tighter prior centered exactly on the fixated patches.

    Returns
    -------
    (B, h_patch, h_patch) tensor, each sample independently min-max
    normalized to ``[0, 1]`` (an all-zero map stays all-zero if a sample has
    no valid fixations).
    """
    B, L, _ = fixation.shape
    device = fixation.device

    y_coords = torch.arange(h_patch, device=device, dtype=torch.float32)
    x_coords = torch.arange(h_patch, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    heatmaps = []
    for b in range(B):
        # zero-padding rows are exactly all-zero -> filter them out
        valid = fixation[b].abs().sum(dim=-1) > 1e-8
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
