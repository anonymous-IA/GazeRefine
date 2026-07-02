"""
datasets.py — Image/mask/fixation loaders for the two reported benchmarks,
plus a small base class so adding a third dataset is just a few lines.

Expected folder layout (same for both datasets):

    <root>/
        images/    one image per case (.jpg/.png for Kvasir, .dcm for prostate MRI)
        masks/     matching binary mask, same basename, .png
    <fixation_csv>
        one row per fixation, see gazerefine.gaze for the expected columns.
        The IMAGE column must match an images/ filename (Kvasir) or the
        DICOM basename + ".jpg" (prostate — fixation collection was run on
        JPEG-rendered slices while the model reads the original DICOM).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from .constants import IMG_SIZE, IMG_MEAN, IMG_STD
from .gaze import get_scanpath, IMG_COL


class _BaseGazeDataset(Dataset):
    """Shared image/mask transform + fixation-grouping logic.

    Subclasses only need to implement ``_load_image(name)`` and provide the
    set of valid image ids that have a matching mask and fixation entries.
    """

    def __init__(self, root: str, fixation_csv: str, img_size: int = IMG_SIZE):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.img_size = img_size

        self.df = pd.read_csv(fixation_csv)
        self.df.columns = self.df.columns.str.strip()
        self.fix_df = self.df.groupby(IMG_COL)
        self.max_len = int(self.df.groupby(IMG_COL).size().max())
        print(f"[Dataset] max scanpath length = {self.max_len}")

        self.image_ids: list[str] = []  # set by subclass __init__

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMG_MEAN, IMG_STD),
        ])
        self.mask_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, name: str) -> Image.Image:
        raise NotImplementedError

    def _mask_path(self, name: str) -> str:
        raise NotImplementedError

    def _fixation_key(self, name: str) -> str:
        """CSV IMAGE-column key for this sample. Override if it differs from
        the on-disk basename (e.g. prostate MRI uses .jpg keys for .dcm files)."""
        return name

    def __getitem__(self, idx: int):
        name = self.image_ids[idx]

        image = self.img_tf(self._load_image(name))

        mask = self.mask_tf(Image.open(self._mask_path(name)).convert("L"))
        mask = (mask > 0.5).float()

        fix_rows = self.fix_df.get_group(self._fixation_key(name))
        fixation = get_scanpath(fix_rows, self.max_len)

        return {"image": image, "fixation": fixation, "mask": mask, "name": name}


class KvasirSEGDataset(_BaseGazeDataset):
    """Kvasir-SEG colonoscopy polyp segmentation. images/masks share filenames
    (e.g. ``cju0qkwl35piu0993l0dewei2.jpg`` in both folders)."""

    def __init__(self, root: str, fixation_csv: str, img_size: int = IMG_SIZE):
        super().__init__(root, fixation_csv, img_size)
        img_files = set(os.listdir(self.img_dir))
        mask_files = set(os.listdir(self.mask_dir))
        csv_imgs = set(self.df[IMG_COL].unique())
        self.image_ids = sorted(img_files & mask_files & csv_imgs)
        if not self.image_ids:
            raise RuntimeError("No overlap between images/, masks/ and the fixation CSV.")
        print(f"[Dataset] Kvasir-SEG valid samples = {len(self.image_ids)}")

    def _load_image(self, name: str) -> Image.Image:
        return Image.open(os.path.join(self.img_dir, name)).convert("RGB")

    def _mask_path(self, name: str) -> str:
        return os.path.join(self.mask_dir, name)


class ProstateMRIDataset(_BaseGazeDataset):
    """NCI-ISBI prostate MRI. images/ holds DICOM (.dcm), masks/ holds PNG,
    and the fixation CSV references each case as ``<basename>.jpg`` (the
    format the eye-tracking session was actually rendered/displayed in)."""

    def __init__(self, root: str, fixation_csv: str, img_size: int = IMG_SIZE):
        super().__init__(root, fixation_csv, img_size)

        dcm_basenames = {os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith(".dcm")}
        png_basenames = {os.path.splitext(f)[0] for f in os.listdir(self.mask_dir) if f.endswith(".png")}
        csv_basenames = {os.path.splitext(f)[0] for f in self.df[IMG_COL].unique()}

        self.image_ids = sorted(dcm_basenames & png_basenames & csv_basenames)
        if not self.image_ids:
            raise RuntimeError("No overlap between images/ (.dcm), masks/ (.png) and the fixation CSV.")
        print(f"[Dataset] Prostate MRI valid samples = {len(self.image_ids)}")

    def _load_image(self, name: str) -> Image.Image:
        import pydicom  # local import: optional dependency, only needed for DICOM datasets

        dcm_path = os.path.join(self.img_dir, f"{name}.dcm")
        dicom = pydicom.dcmread(dcm_path)
        arr = dicom.pixel_array.astype(np.float32)

        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr = np.zeros_like(arr)

        rgb = np.stack([arr, arr, arr], axis=-1)  # grayscale -> 3-channel, DINOv3 expects RGB
        return Image.fromarray((rgb * 255).astype(np.uint8))

    def _mask_path(self, name: str) -> str:
        return os.path.join(self.mask_dir, f"{name}.png")

    def _fixation_key(self, name: str) -> str:
        return f"{name}.jpg"


DATASET_REGISTRY = {
    "kvasir": KvasirSEGDataset,
    "prostate_mri": ProstateMRIDataset,
}
