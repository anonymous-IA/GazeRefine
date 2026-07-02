"""
run_eval.py — Unified zero-shot evaluation entrypoint.

Usage
-----
    python scripts/run_eval.py --config configs/kvasir.yaml
    python scripts/run_eval.py --config configs/prostate_mri.yaml

    # override anything from the YAML on the command line:
    python scripts/run_eval.py --config configs/kvasir.yaml --root /data/Kvasir-SEG --max_iters 8
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `import gazerefine`

from gazerefine import GazeRefine, compute_metrics, save_prediction
from gazerefine.datasets import DATASET_REGISTRY


def load_config(path: str, overrides: argparse.Namespace) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(overrides).items():
        if k != "config" and v is not None:
            cfg[k] = v
    return cfg


@torch.no_grad()
def eval_epoch(model, loader, device, threshold: float, pred_dir: str, save_num: int) -> dict:
    model.eval()
    all_dice, all_iou = [], []
    n = len(loader)
    saved = 0

    for i, batch in enumerate(loader):
        img = batch["image"].to(device)
        fix = batch["fixation"].to(device)
        msk = batch["mask"].to(device)

        out = model(img, fix)
        preds = out["preds"]
        m = compute_metrics(preds, msk, thr=threshold)
        all_dice.append(m["dice_per"].cpu())
        all_iou.append(m["iou_per"].cpu())

        if saved < save_num:
            for ridx in range(img.size(0)):
                if saved >= save_num:
                    break
                pred_bin = (preds[ridx] > threshold).float()
                save_prediction(
                    img[ridx], out["gaze_heatmap"][ridx], msk[ridx], pred_bin,
                    name=f"zeroshot_sample_{batch['name'][ridx]}.png", out_dir=pred_dir,
                )
                saved += 1

        if (i + 1) % max(1, n // 5) == 0:
            print(f"  [{i + 1}/{n}] dice={m['dice']:.4f} iou={m['iou']:.4f}")

    all_dice = torch.cat(all_dice)
    all_iou = torch.cat(all_iou)
    return {
        "dice": all_dice.mean().item(), "dice_std": all_dice.std().item(),
        "iou": all_iou.mean().item(), "iou_std": all_iou.std().item(),
    }


def main():
    ap = argparse.ArgumentParser("GazeRefine — zero-shot gaze-guided segmentation")
    ap.add_argument("--config", required=True, help="path to a YAML config, e.g. configs/kvasir.yaml")
    # optional CLI overrides for anything in the YAML
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--fixation_csv", type=str, default=None)
    ap.add_argument("--pred_dir", type=str, default=None)
    ap.add_argument("--dino_name", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--extract_mode", type=str, default=None, choices=["all", "last"])
    ap.add_argument("--contrast_method", type=str, default=None, choices=["difference", "softmax", "original"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--gaze_anchor_weight", type=float, default=None)
    ap.add_argument("--knn_k", type=int, default=None)
    ap.add_argument("--save_num", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GazeRefine] device={device}, dataset={cfg['dataset']}")

    DatasetCls = DATASET_REGISTRY[cfg["dataset"]]
    dataset = DatasetCls(cfg["root"], cfg["fixation_csv"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=False, pin_memory=True)
    print(f"[GazeRefine] total samples to evaluate: {len(dataset)}")

    model = GazeRefine(
        dino_name=cfg["dino_name"],
        sigma=cfg["sigma"],
        extract_mode=cfg["extract_mode"],
        contrast_method=cfg["contrast_method"],
        temperature=cfg["temperature"],
        max_iters=cfg["max_iters"],
        knn_refine=cfg["knn_refine"],
        knn_k=cfg["knn_k"],
        knn_temp=cfg["knn_temp"],
        gaze_anchor_weight=cfg["gaze_anchor_weight"],
    ).to(device)

    print(f"[GazeRefine] running zero-shot evaluation on {cfg['dataset']}...")
    metrics = eval_epoch(model, loader, device, threshold=cfg["threshold"], pred_dir=cfg["pred_dir"], save_num=cfg["save_num"])

    print(f"\n{'═' * 60}")
    print(f"[Results] Dice: {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"[Results] IoU:  {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
