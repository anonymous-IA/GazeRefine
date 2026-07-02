# Examples

This folder holds the sample assets for the demo notebook and the
quick-start commands in the README.

## Structure

```
examples/
├── images/
│   ├── kvasir_sample.jpg          ← add a real Kvasir-SEG image here
│   └── prostate_sample.png        ← add a real NCI-ISBI axial slice here
└── fixations/
    ├── kvasir_sample.csv          ← example fixation CSV (colonoscopy)
    └── prostate_sample.csv        ← example fixation CSV (prostate MRI)
```

## Fixation CSV format

```csv
x,y,duration
340,221,180
356,228,145
368,244,205
...
```

| Column     | Meaning                                                             |
|------------|---------------------------------------------------------------------|
| `x`        | Fixation x in **raw pixel** coordinates of the original image      |
| `y`        | Fixation y in **raw pixel** coordinates of the original image      |
| `duration` | Fixation duration in any consistent unit (milliseconds is typical) |

`x` and `y` are automatically normalized inside `predict_single.py` by the
image's width and height — you do **not** need to pre-normalize them.

Only **relative** durations matter: the model internally min-max normalizes
them so that the longest fixation gets weight 1.0. The unit (ms, frames, …)
makes no difference as long as it's consistent within one file.

## Quick test with the provided examples

Once you have placed a real image in `examples/images/`, run:

```bash
# colonoscopy polyp
python scripts/predict_single.py \
    --image examples/images/kvasir_sample.jpg \
    --fixations examples/fixations/kvasir_sample.csv \
    --output output_mask.png \
    --save_overlay

# prostate MRI
python scripts/predict_single.py \
    --image examples/images/prostate_sample.png \
    --fixations examples/fixations/prostate_sample.csv \
    --output output_mask.png \
    --preset mri \
    --save_overlay
```

This produces:
- `output_mask.png` — binary segmentation mask
- `output_mask_overlay.png` — mask painted over the input image in red
- `output_mask_gaze.png` — gaze heatmap painted over the input image

## Generating your own fixation CSV

If you have an eye-tracker (EyeLink, Tobii, Pupil Labs, …), export the
fixation report for the image of interest, then:

1. Keep only the rows belonging to that image.
2. Rename the x/y coordinate columns to `x` and `y`, and the duration
   column to `duration`.
3. Make sure `x`/`y` are in **pixel** coordinates of the image (not screen
   coordinates — crop to the image region first if needed).
4. Save as a plain CSV.

If you don't have an eye-tracker and are using this for experimentation,
you can manually click on the structure of interest a few times and record
those pixel coordinates with durations of `1` (uniform weighting).
