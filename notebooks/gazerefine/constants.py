"""constants.py — shared sizing and normalization constants.

IMG_SIZE is fixed to an exact multiple of the ViT patch size so the patch
grid divides evenly with no rounding/cropping artifacts.
"""

PATCH_SIZE = 16
H_PATCH = 37 * 2                      # 74 patches per side
IMG_SIZE = H_PATCH * PATCH_SIZE       # 1184 px  (use 518 = 37*14 if you switch to a /14 ViT)
N_PATCHES = H_PATCH ** 2

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
