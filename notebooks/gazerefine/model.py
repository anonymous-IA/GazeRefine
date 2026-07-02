"""
model.py — The GazeRefine zero-shot segmentation model.

This is a 1:1 refactor of the two task-specific scripts (Kvasir-SEG polyp /
NCI-ISBI prostate MRI) into a single, dataset-agnostic module. The numerics
are unchanged from the original experiments — only structure, naming and
comments were cleaned up. Every per-dataset difference (sigma, kNN k,
gaze-anchor weight, max_iters, ...) is now a constructor / config argument
instead of a hardcoded default, see ``configs/*.yaml``.

Maps onto the paper (Section 2) as follows:

    gaze.generate_gaze_heatmap   -> H(u, v)                          (Eq. gaze prior)
    W_fg, W_bg                   -> W_fg^(0), W_bg^(0)                (gaze-derived weights)
    F_proto_init, B_proto_init   -> F^(0), B^(0)                      (initial prototypes)
    contrast_method="difference" -> S_i^(t) = max(0, s_fg - alpha*s_bg)  (contrastive cleaning)
    knn_affinity_refinement      -> S̄_i^(t)                           (kNN affinity propagation)
    gaze_anchor_weight           -> lambda                            (anchor blending strength)
    convergence check            -> ||F^(t+1) - F^(t)|| < eps
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FrozenDINOv3
from .gaze import generate_gaze_heatmap
from .constants import IMG_SIZE, PATCH_SIZE

EPS = 1e-12


def knn_affinity_refinement(
    Pv: torch.Tensor, S: torch.Tensor, k: int = 5, temperature: float = 0.05
) -> torch.Tensor:
    """Propagate a per-patch score map ``S`` across its k nearest neighbors in
    frozen DINOv3 feature space (patch-to-patch affinity), encouraging
    coherent, object-level responses instead of isolated high-confidence
    patches. Corresponds to S̄ in the paper.

    Pv : (B, N, D) raw (unnormalized) patch embeddings.
    S  : (B, N)    current per-patch score to be smoothed.
    """
    B, N, D = Pv.shape
    Pv_norm = F.normalize(Pv, p=2, dim=-1)

    sim_matrix = torch.bmm(Pv_norm, Pv_norm.transpose(1, 2))            # (B, N, N)
    topk_vals, topk_indices = torch.topk(sim_matrix, k=k, dim=-1)        # (B, N, k)
    weights = F.softmax(topk_vals / temperature, dim=-1)                 # (B, N, k)

    S_expanded = S.unsqueeze(1).expand(B, N, N)
    S_neighbors = torch.gather(S_expanded, dim=2, index=topk_indices)    # (B, N, k)

    return torch.sum(weights * S_neighbors, dim=-1)                     # (B, N)


class GazeRefine(nn.Module):
    """Zero-shot, training-free, gaze-guided segmentation model.

    The model has exactly one set of *learned* weights: the frozen,
    pretrained DINOv3 backbone. Everything else — prototype construction,
    contrastive cleaning, kNN propagation, gaze anchoring — is a closed-form
    operation re-run from scratch on every image at inference time.

    Parameters
    ----------
    dino_name : timm DINOv3 checkpoint name.
    img_size, patch_size : input resolution / ViT patch size. Must satisfy
        ``img_size % patch_size == 0``.
    sigma : Gaussian spread (in patch units) for the gaze heatmap.
    extract_mode : ``"last"`` (1 block) or ``"all"`` (4 blocks, averaged).
    contrast_method : ``"difference"`` (paper default, contrastive cleaning),
        ``"softmax"`` (foreground/background softmax ratio), or
        ``"original"`` (plain foreground cosine similarity, no background
        suppression — kept for the ablation in Table 2).
    temperature : softmax temperature, only used when contrast_method="softmax".
    max_iters : maximum recurrent refinement iterations (T in the paper).
    knn_refine : whether to apply kNN affinity propagation each iteration.
    knn_k, knn_temp : kNN neighborhood size / softmax temperature.
    gaze_anchor_weight : lambda — how strongly each iteration's prototypes
        are pulled back toward the original gaze-only prototypes. Higher =
        trust the raw fixations more; lower = let the model drift further
        from the initial gaze region.
    """

    def __init__(
        self,
        dino_name: str = "vit_large_patch16_dinov3.lvd1689m",
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
        sigma: float = 2.0,
        extract_mode: str = "last",
        contrast_method: str = "difference",
        temperature: float = 0.05,
        max_iters: int = 10,
        knn_refine: bool = True,
        knn_k: int = 5,
        knn_temp: float = 0.1,
        gaze_anchor_weight: float = 0.6,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be a multiple of patch_size"
        self.img_size = img_size
        self.h_patch = img_size // patch_size
        self.sigma = sigma
        self.contrast_method = contrast_method
        self.temperature = temperature
        self.max_iters = max_iters
        self.knn_refine = knn_refine
        self.knn_k = knn_k
        self.knn_temp = knn_temp
        self.gaze_anchor_weight = gaze_anchor_weight

        self.visual_enc = FrozenDINOv3(dino_name, extract_mode=extract_mode)

    # ------------------------------------------------------------------ #
    def _gaze_weights(self, fixation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scanpath -> heatmap -> normalized foreground/background spatial weights."""
        gaze_heatmap = generate_gaze_heatmap(fixation, h_patch=self.h_patch, sigma=self.sigma)
        H_flat = gaze_heatmap.view(gaze_heatmap.size(0), -1)

        W_fg = H_flat / (H_flat.sum(dim=-1, keepdim=True) + EPS)
        bg_w = 1.0 - H_flat
        W_bg = bg_w / (bg_w.sum(dim=-1, keepdim=True) + EPS)
        return gaze_heatmap, W_fg, W_bg

    def _score(self, Pv_norm: torch.Tensor, F_proto_norm: torch.Tensor, B_proto_norm: torch.Tensor) -> torch.Tensor:
        """Foreground/background contrastive scoring for one level, one iteration."""
        if self.contrast_method == "original":
            S = torch.bmm(Pv_norm, F_proto_norm.unsqueeze(-1)).squeeze(-1)
            return torch.clamp(S, min=0.0)

        sim_fg = torch.bmm(Pv_norm, F_proto_norm.unsqueeze(-1)).squeeze(-1)
        sim_bg = torch.bmm(Pv_norm, B_proto_norm.unsqueeze(-1)).squeeze(-1)

        if self.contrast_method == "difference":
            return torch.clamp(sim_fg - sim_bg, min=0.0)
        elif self.contrast_method == "softmax":
            stacked = torch.stack([sim_fg, sim_bg], dim=-1) / self.temperature
            probs = F.softmax(stacked, dim=-1)
            return probs[:, :, 0]
        raise ValueError(f"Unknown contrast_method: {self.contrast_method!r}")

    def _refine_level(self, Pv: torch.Tensor, W_fg: torch.Tensor, W_bg: torch.Tensor) -> torch.Tensor:
        """Run the full recurrent gaze-anchored refinement loop for one DINOv3 level.
        Returns the final (B, N) per-patch foreground score map for that level."""
        B = Pv.size(0)
        device = Pv.device
        Pv_norm = F.normalize(Pv, p=2, dim=-1)

        # Initial gaze-only prototypes — F^(0), B^(0)
        F_proto_init = torch.sum(W_fg.unsqueeze(-1) * Pv, dim=1)
        B_proto_init = torch.sum(W_bg.unsqueeze(-1) * Pv, dim=1)

        W_fg_curr, W_bg_curr = W_fg.clone(), W_bg.clone()
        best_S = W_fg.clone()
        S_prev = W_fg.clone()
        active = torch.ones(B, dtype=torch.bool, device=device)

        for _ in range(self.max_iters):
            W_fg_norm = W_fg_curr / (W_fg_curr.sum(dim=-1, keepdim=True) + EPS)
            W_bg_norm = W_bg_curr / (W_bg_curr.sum(dim=-1, keepdim=True) + EPS)

            F_proto_curr = torch.sum(W_fg_norm.unsqueeze(-1) * Pv, dim=1)
            B_proto_curr = torch.sum(W_bg_norm.unsqueeze(-1) * Pv, dim=1)

            # Anchor toward the initial gaze-only prototypes — lambda blending
            lam = self.gaze_anchor_weight
            F_proto = (1 - lam) * F_proto_curr + lam * F_proto_init
            B_proto = (1 - lam) * B_proto_curr + lam * B_proto_init
            F_proto_norm = F.normalize(F_proto, p=2, dim=-1)
            B_proto_norm = F.normalize(B_proto, p=2, dim=-1)

            S_iter = self._score(Pv_norm, F_proto_norm, B_proto_norm)
            if self.knn_refine:
                S_iter = knn_affinity_refinement(Pv, S_iter, k=self.knn_k, temperature=self.knn_temp)

            S_new = S_prev.clone()
            S_new[active] = S_iter[active]

            # Collapse prevention: if an active sample's map flattened to ~0,
            # roll it back and freeze it instead of letting it degenerate.
            collapsed = S_new.max(dim=-1)[0] < 1e-4
            freeze = active & collapsed
            S_new[freeze] = S_prev[freeze]
            active = active & ~collapsed

            best_S = S_new.clone()
            if not active.any():
                break

            # Turn the refined score into the next iteration's spatial weights
            S_sig = torch.sigmoid(S_new)
            W_fg_next = S_sig / (S_sig.sum(dim=-1, keepdim=True) + EPS)
            bg_w = 1.0 - S_sig
            bg_w = torch.clamp(bg_w - bg_w.min(dim=-1, keepdim=True).values, min=0.0)
            W_bg_next = bg_w / (bg_w.sum(dim=-1, keepdim=True) + EPS)

            F_proto_next = torch.sum(W_fg_next.unsqueeze(-1) * Pv, dim=1)
            F_proto_next_norm = F.normalize(F_proto_next, p=2, dim=-1)

            # Convergence: foreground prototype stopped moving
            diff = torch.abs(F_proto_next_norm - F_proto_norm).mean(dim=-1)
            if torch.all(diff < 1e-6):
                break

            S_prev = S_new.clone()
            W_fg_curr[active] = W_fg_next[active]
            W_bg_curr[active] = W_bg_next[active]

        return best_S

    # ------------------------------------------------------------------ #
    def forward(self, image: torch.Tensor, fixation: torch.Tensor) -> dict:
        """
        image : (B, 3, H, W) normalized RGB tensor (ImageNet mean/std).
        fixation : (B, L, 3) zero-padded scanpath, see ``gazerefine.gaze.get_scanpath``.

        Returns a dict with:
            preds        (B, 1, H, W) final mask in [0, 1] — threshold at 0.5 for a binary mask
            gaze_heatmap (B, h, w) the raw gaze prior, useful for visualization
            final_map    (B, N) the un-upsampled patch-level score map
        """
        B = image.size(0)
        Pv_list = self.visual_enc(image)
        gaze_heatmap, W_fg, W_bg = self._gaze_weights(fixation)

        all_maps = [self._refine_level(Pv, W_fg, W_bg) for Pv in Pv_list]
        final_map = torch.stack(all_maps, dim=1).mean(dim=1)  # average across levels

        grid = final_map.view(B, 1, self.h_patch, self.h_patch)
        upsampled = F.interpolate(grid, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # per-sample min-max normalization -> ready-to-threshold mask
        flat = upsampled.view(B, -1)
        m_min = flat.min(dim=-1, keepdim=True).values.view(B, 1, 1, 1)
        m_max = flat.max(dim=-1, keepdim=True).values.view(B, 1, 1, 1)
        preds = (upsampled - m_min) / (m_max - m_min + 1e-8)

        return dict(preds=preds, gaze_heatmap=gaze_heatmap, final_map=final_map)
