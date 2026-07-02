"""
predict_single.py — Run GazeRefine on a single image + fixation CSV.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Command-line usage (matches the README exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python scripts/predict_single.py \\
        --image   examples/image.png \\
        --fixations examples/fixations.csv \\
        --output  output_mask.png

Optional flags:
    --preset        colonoscopy | mri           (default: colonoscopy)
    --threshold     0.5                          binarization threshold
    --save_overlay                               also save a colour overlay PNG
    --device        cuda | cpu                   (auto-detected by default)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Python API (matches the README exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from scripts.predict_single import predict

    mask = predict(
        image_path="image.png",
        fixation_csv="fixations.csv",
    )
    # `mask` is a PIL Image of the binary segmentation mask.
    # Save it:
    mask.save("output_mask.png")

    # Extended API — also get overlays and raw arrays:
    result = predict(
        image_path="image.png",
        fixation_csv="fixations.csv",
        preset="mri",           # "colonoscopy" (default) or "mri"
        threshold=0.5,
        return_all=True,
    )
    result["mask"].save("mask.png")
    result["gaze_overlay"].save("gaze.png")
    result["mask_overlay"].save("overlay.png")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Fixation CSV format
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    x,y,duration
    340,221,180
    356,228,145
    368,244,205
    ...

    x, y     — fixation position in *raw pixel* coordinates of the input image.
               (These are automatically normalized by the image size internally.)
    duration — fixation duration in any consistent unit (milliseconds typical).
               The model only uses *relative* durations, so the unit does not matter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

# allow `python scripts/predict_single.py` from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazerefine import GazeRefine, overlay_heatmap, overlay_mask
from gazerefine.constants import IMG_MEAN, IMG_STD, IMG_SIZE
from gazerefine.gaze import load_fixation_csv
import torchvision.transforms as T


# ── per-modality hyperparameter presets ────────────────────────────────────
#    These match the exact settings used to produce the paper's Table 1 numbers.
PRESETS: dict[str, dict] = {
    "colonoscopy": dict(
        sigma=2.0,
        contrast_method="difference",
        max_iters=5,
        gaze_anchor_weight=0.5,
        knn_refine=True,
        knn_k=20,
        knn_temp=0.1,
    ),
    "mri": dict(
        sigma=1.5,
        contrast_method="difference",
        max_iters=1,
        gaze_anchor_weight=0.8,
        knn_refine=True,
        knn_k=3,
        knn_temp=0.1,
    ),
}

# one shared backbone name for both presets
DINO_NAME = "vit_large_patch16_dinov3.lvd1689m"

# module-level model cache: avoids reloading the backbone across repeated calls
# (useful when this module is imported by the Gradio Space or a notebook loop)
_MODEL_CACHE: dict[str, GazeRefine] = {}


def _get_model(preset: str, device: torch.device) -> GazeRefine:
    """Load (or return a cached) GazeRefine model for the given preset."""
    if preset not in _MODEL_CACHE:
        cfg = PRESETS[preset]
        _MODEL_CACHE[preset] = GazeRefine(dino_name=DINO_NAME, **cfg).to(device).eval()
    return _MODEL_CACHE[preset]


# ── main public function ────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    image_path: "str | Path | Image.Image",
    fixation_csv: "str | Path",
    preset: str = "colonoscopy",
    threshold: float = 0.5,
    device: str | None = None,
    return_all: bool = False,
) -> "Image.Image | dict":
    """Run GazeRefine on one image and return the predicted segmentation mask.

    Parameters
    ----------
    image_path   : path to the input image (.jpg / .jpeg / .png) **or** an
                   already-loaded ``PIL.Image`` (used by the Gradio Space).
    fixation_csv : path to the fixation CSV (``x,y,duration`` columns,
                   pixel coordinates — see module docstring for the format).
    preset       : ``"colonoscopy"`` (default, Kvasir-SEG settings) or
                   ``"mri"`` (NCI-ISBI prostate-MRI settings).
    threshold    : binarization cutoff applied to the [0, 1] soft mask.
    device       : ``"cuda"`` / ``"cpu"`` — auto-detected when ``None``.
    return_all   : when ``True``, return a dict with the binary mask PIL Image
                   **plus** ``gaze_overlay``, ``mask_overlay``, and the raw
                   numpy arrays ``preds`` and ``gaze_heatmap``.
                   When ``False`` (default), return only the mask PIL Image.

    Returns
    -------
    ``PIL.Image`` of the binary mask, **or** a dict (see ``return_all``).
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {list(PRESETS)}, got {preset!r}")

    _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── load the image ──
    if isinstance(image_path, Image.Image):
        pil_image = image_path.convert("RGB")
    else:
        pil_image = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_image.size

    # ── load fixations and normalize pixel → [0, 1] ──
    fix_t = load_fixation_csv(str(fixation_csv), image_width=img_w, image_height=img_h)
    fix_t = fix_t.to(_device)

    # ── preprocess the image for DINOv3 ──
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])
    img_t = tf(pil_image).unsqueeze(0).to(_device)  # (1, 3, IMG_SIZE, IMG_SIZE)

    # ── run the model ──
    model = _get_model(preset, _device)
    out   = model(img_t, fix_t)

    # ── decode outputs ──
    soft_mask = out["preds"][0, 0].cpu().numpy()    # (H, W) float in [0, 1]
    gaze      = out["gaze_heatmap"][0].cpu().numpy()  # (h, w) float in [0, 1]
    bin_mask  = (soft_mask > threshold).astype(np.uint8) * 255

    mask_pil = Image.fromarray(bin_mask, mode="L")

    if not return_all:
        return mask_pil

    return dict(
        mask          = mask_pil,
        gaze_overlay  = overlay_heatmap(pil_image, gaze),
        mask_overlay  = overlay_mask(pil_image, (bin_mask / 255).astype(np.float32)),
        preds         = soft_mask,
        gaze_heatmap  = gaze,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="predict_single.py",
        description="GazeRefine — zero-shot gaze-guided segmentation on a single image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # colonoscopy polyp (default preset):
  python scripts/predict_single.py \\
      --image examples/images/kvasir_sample.jpg \\
      --fixations examples/fixations/kvasir_sample.csv \\
      --output output_mask.png

  # prostate MRI:
  python scripts/predict_single.py \\
      --image examples/images/prostate_sample.png \\
      --fixations examples/fixations/prostate_sample.csv \\
      --output output_mask.png \\
      --preset mri

  # save overlays too:
  python scripts/predict_single.py \\
      --image examples/images/kvasir_sample.jpg \\
      --fixations examples/fixations/kvasir_sample.csv \\
      --output output_mask.png \\
      --save_overlay
        """,
    )
    ap.add_argument("--image",       required=True,
                    help="Path to the input image (.jpg / .jpeg / .png).")
    ap.add_argument("--fixations",   required=True,
                    help="Path to the fixation CSV (x,y,duration — pixel coordinates).")
    ap.add_argument("--output",      required=True,
                    help="Where to save the predicted binary mask (.png).")
    ap.add_argument("--preset",      default="colonoscopy",
                    choices=list(PRESETS),
                    help="Hyperparameter preset: 'colonoscopy' (default) or 'mri'.")
    ap.add_argument("--threshold",   type=float, default=0.5,
                    help="Binarization threshold applied to the soft mask (default: 0.5).")
    ap.add_argument("--save_overlay", action="store_true",
                    help="Also save a colour overlay PNG next to --output.")
    ap.add_argument("--device",      default=None,
                    help="'cuda' or 'cpu' — auto-detected when not given.")
    return ap


def main():
    args = _build_parser().parse_args()

    print(f"[GazeRefine] image    : {args.image}")
    print(f"[GazeRefine] fixations: {args.fixations}")
    print(f"[GazeRefine] preset   : {args.preset}")
    print(f"[GazeRefine] threshold: {args.threshold}")

    result = predict(
        image_path   = args.image,
        fixation_csv = args.fixations,
        preset       = args.preset,
        threshold    = args.threshold,
        device       = args.device,
        return_all   = args.save_overlay,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(result, dict):
        result["mask"].save(output_path)
        print(f"[GazeRefine] mask saved  → {output_path}")
        if args.save_overlay:
            overlay_path = output_path.with_stem(output_path.stem + "_overlay")
            result["mask_overlay"].save(overlay_path)
            gaze_path = output_path.with_stem(output_path.stem + "_gaze")
            result["gaze_overlay"].save(gaze_path)
            print(f"[GazeRefine] overlay     → {overlay_path}")
            print(f"[GazeRefine] gaze prior  → {gaze_path}")
    else:
        result.save(output_path)
        print(f"[GazeRefine] mask saved  → {output_path}")


if __name__ == "__main__":
    main()
