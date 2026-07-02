"""
app.py — GazeRefine interactive demo (Hugging Face Space).

Upload a medical image, click on it to drop a few fixation points the way a
clinician's gaze would land on the structure of interest, and GazeRefine
turns that into a segmentation mask — fully zero-shot, no training, no
prompt-specific architecture.

Run locally with:
    pip install -r huggingface_space/requirements.txt
    python huggingface_space/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from scripts.predict_single import predict

PRESETS = {
    "Colonoscopy / polyp (Kvasir-SEG settings)": dict(
        sigma=2.0, contrast_method="difference", max_iters=5,
        gaze_anchor_weight=0.5, knn_k=20, knn_temp=0.1,
    ),
    "Grayscale MRI / CT (prostate-MRI settings)": dict(
        sigma=1.5, contrast_method="difference", max_iters=1,
        gaze_anchor_weight=0.8, knn_k=3, knn_temp=0.1,
    ),
}
DINO_NAME = "vit_large_patch16_dinov3.lvd1689m"

POINT_COLORS = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#5ac8fa", "#007aff", "#af52de"]


def draw_points(image: Image.Image, points: list[tuple[float, float, float]]) -> Image.Image:
    """Render numbered fixation markers over the image for visual feedback."""
    if image is None:
        return None
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    w, h = vis.size
    r = max(6, min(w, h) // 80)
    for i, (x, y, dur) in enumerate(points):
        cx, cy = x * w, y * h
        color = POINT_COLORS[i % len(POINT_COLORS)]
        rad = r * (0.6 + 0.8 * dur)  # bigger marker = longer fixation
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=color, width=3)
        draw.text((cx + rad + 2, cy - rad), str(i + 1), fill=color)
    return vis


def on_select(image: Image.Image, points: list, duration: float, evt: gr.SelectData):
    if image is None:
        gr.Warning("Upload an image first.")
        return points, None
    w, h = image.size
    x_px, y_px = evt.index
    points = points + [(x_px / w, y_px / h, duration)]
    return points, draw_points(image, points)


def on_clear(image: Image.Image):
    return [], image


def on_image_change(image: Image.Image):
    # new image -> reset fixations
    return [], image


def run(image: Image.Image, points: list, preset_name: str, threshold: float):
    if image is None:
        gr.Warning("Upload an image first.")
        return None, None, None
    if len(points) == 0:
        gr.Warning("Click on the image at least once to place a fixation.")
        return None, None, None

    cfg = PRESETS[preset_name]
    out = predict(
        image=image,
        fixations=points,
        dino_name=DINO_NAME,
        threshold=threshold,
        **cfg,
    )
    mask_only = Image.fromarray((np.clip(out["mask_bin"], 0, 1) * 255).astype(np.uint8))
    return out["gaze_overlay"], out["mask_overlay"], mask_only


with gr.Blocks(title="GazeRefine — gaze-guided zero-shot segmentation") as demo:
    gr.Markdown(
        """
        # 👁️ GazeRefine — Expert Gaze as a Test-Time Prompt
        Training-free, zero-shot medical image segmentation. Upload an image, **click on it
        1–5 times** where a clinician would look, pick a preset, and run.
        No masks, clicks-as-bounding-boxes, fine-tuning, or adapters — just a frozen DINOv3
        backbone steered by your gaze. See the paper / code on GitHub (linked below).
        """
    )

    points_state = gr.State([])

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="1. Upload image, then click to place fixations", height=420)
            with gr.Row():
                duration_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Next fixation's relative duration")
                clear_btn = gr.Button("Clear fixations")
            preset = gr.Radio(list(PRESETS.keys()), value=list(PRESETS.keys())[0], label="2. Hyperparameter preset")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="3. Mask threshold")
            run_btn = gr.Button("Run GazeRefine", variant="primary")

        with gr.Column():
            gaze_out = gr.Image(label="Gaze prior over image", height=260)
            with gr.Row():
                mask_overlay_out = gr.Image(label="Predicted mask (overlay)", height=260)
                mask_only_out = gr.Image(label="Predicted mask (binary)", height=260)

    image_in.upload(on_image_change, inputs=[image_in], outputs=[points_state, image_in])
    image_in.select(on_select, inputs=[image_in, points_state, duration_slider], outputs=[points_state, image_in])
    clear_btn.click(on_clear, inputs=[image_in], outputs=[points_state, image_in])
    run_btn.click(run, inputs=[image_in, points_state, preset, threshold], outputs=[gaze_out, mask_overlay_out, mask_only_out])

    gr.Markdown(
        "Code: [GitHub](https://github.com/<your-org>/gazerefine) · "
        "Method: GazeRefine — frozen DINOv3 + gaze-anchored prototypes + recurrent "
        "foreground/background refinement, entirely training-free."
    )

if __name__ == "__main__":
    demo.launch()
