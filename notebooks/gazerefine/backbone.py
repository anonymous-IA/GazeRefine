"""
backbone.py — Frozen DINOv3 feature extractor.

GazeRefine never updates the backbone. We load a pretrained DINOv3 ViT via
`timm`, freeze every parameter, and pull out raw patch tokens from one or
more transformer blocks using forward hooks. No adapters, no projections,
no fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenDINOv3(nn.Module):
    """Completely frozen DINOv3 ViT backbone that exposes raw patch tokens.

    Parameters
    ----------
    model_name : str
        Any DINOv3 variant available in `timm` (e.g.
        ``"vit_large_patch16_dinov3.lvd1689m"``,
        ``"vit_base_patch16_dinov3.lvd1689m"``). Larger backbones generally
        give cleaner semantic separation but cost more memory/compute.
    extract_mode : {"last", "all"}
        - ``"last"``: use only the final block's patch tokens (fast, the
          default used in our reported results).
        - ``"all"``: pool patch tokens from 4 evenly-spaced blocks
          (0, n/4, 3n/4, n-1) and average the resulting similarity maps in
          ``GazeRefine``. This sometimes helps on harder modalities at the
          cost of ~4x compute.
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch16_dinov3.lvd1689m",
        extract_mode: str = "last",
    ):
        super().__init__()
        import timm  # local import: keeps `timm` optional for users who only read the code

        print(f"[GazeRefine] Loading frozen backbone: {model_name} (extract_mode={extract_mode})")
        bb = timm.create_model(model_name, pretrained=True, num_classes=0)
        for p in bb.parameters():
            p.requires_grad_(False)
        bb.eval()

        self.backbone = bb
        self.embed_dim = bb.embed_dim
        self.num_blocks = len(bb.blocks)

        if extract_mode == "last":
            self.levels = [-1]
        elif extract_mode == "all":
            self.levels = [0, self.num_blocks // 4, (self.num_blocks * 3) // 4, self.num_blocks - 1]
        else:
            raise ValueError(f"extract_mode must be 'last' or 'all', got {extract_mode!r}")

        print(f"[GazeRefine] Hooking transformer blocks at levels: {self.levels}")
        self._feats: dict[int, torch.Tensor] = {}
        for lvl in self.levels:
            bb.blocks[lvl].register_forward_hook(self._make_hook(lvl))

    def _make_hook(self, lvl: int):
        def _hook_fn(module, inp, out):
            # DINOv3 token layout: [CLS, register_1..register_k, patch_1..patch_N]
            n_prefix = 1 + getattr(self.backbone, "num_register_tokens", 4)
            self._feats[lvl] = out[:, n_prefix:, :]  # (B, N, D) — patch tokens only
        return _hook_fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the frozen backbone and return a list of (B, N, D) patch-token tensors,
        one per hooked level, in the order given by ``self.levels``."""
        self._feats.clear()
        self.backbone(x)
        return [self._feats[lvl] for lvl in self.levels]
