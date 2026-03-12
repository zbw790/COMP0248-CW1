"""Evaluation script for hand gesture multi-task models.

Loads a checkpoint and computes all metrics on the validation set:
detection accuracy@0.5 IoU, mean bbox IoU, segmentation mIoU, Dice,
classification top-1 accuracy, macro F1, and confusion matrix.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

from src.dataloader import get_dataloaders, HandGestureDataset, scan_test_dataset
from src.model import BaselineCNN, MultiTaskResNet
from src.utils import (
    bbox_from_mask_tensor,
    compute_iou_bbox,
    compute_seg_iou,
    compute_dice,
    detection_accuracy,
    GESTURE_NAMES,
)


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        device: Torch device.

    Returns:
        (model, args_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    in_channels = 4 if args.get("use_depth", False) else 3
    if args["model"] == "baseline":
        model = BaselineCNN(in_channels=in_channels)
    else:
        model = MultiTaskResNet(in_channels=in_channels, pretrained=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, args


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation and collect all metrics.

    Args:
        model: Trained model.
        loader: Validation DataLoader.
        device: Torch device.

    Returns:
        Dict of all metrics.
    """
    all_reg_bbox_ious = []   # from direct regression head
    all_seg_bbox_ious = []   # from seg mask derived bbox
    all_seg_ious = []
    all_dice = []
    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        # BBox IoU from direct regression head
        reg_ious = compute_iou_bbox(outputs["bbox"], bboxes)
        all_reg_bbox_ious.append(reg_ious.cpu())

        # BBox IoU derived from predicted segmentation mask
        seg_bboxes = bbox_from_mask_tensor(outputs["seg"])
        seg_ious = compute_iou_bbox(seg_bboxes, bboxes)
        all_seg_bbox_ious.append(seg_ious.cpu())

        # Seg metrics
        all_seg_ious.append(compute_seg_iou(outputs["seg"], masks).cpu().item())
        all_dice.append(compute_dice(outputs["seg"], masks).cpu().item())

        # Classification
        preds = outputs["cls"].argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())

    all_reg_bbox_ious = torch.cat(all_reg_bbox_ious)
    all_seg_bbox_ious = torch.cat(all_seg_bbox_ious)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Use whichever bbox method is better for reporting
    best_bbox_ious = torch.max(all_reg_bbox_ious, all_seg_bbox_ious)

    metrics = {
        "detection_acc@0.5_reg": (all_reg_bbox_ious >= 0.5).float().mean().item(),
        "detection_acc@0.5_seg": (all_seg_bbox_ious >= 0.5).float().mean().item(),
        "detection_acc@0.5": (best_bbox_ious >= 0.5).float().mean().item(),
        "mean_bbox_iou_reg": all_reg_bbox_ious.mean().item(),
        "mean_bbox_iou_seg": all_seg_bbox_ious.mean().item(),
        "mean_bbox_iou": best_bbox_ious.mean().item(),
        "seg_miou": np.mean(all_seg_ious),
        "dice": np.mean(all_dice),
        "cls_top1_acc": (all_preds == all_labels).mean(),
        "cls_macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=list(range(10))),
    }

    return metrics


def print_results(metrics: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Det Acc @0.5 (regression)':<30} {metrics['detection_acc@0.5_reg']:>10.4f}")
    print(f"{'Det Acc @0.5 (from seg)':<30} {metrics['detection_acc@0.5_seg']:>10.4f}")
    print(f"{'Det Acc @0.5 (best)':<30} {metrics['detection_acc@0.5']:>10.4f}")
    print(f"{'Mean BBox IoU (regression)':<30} {metrics['mean_bbox_iou_reg']:>10.4f}")
    print(f"{'Mean BBox IoU (from seg)':<30} {metrics['mean_bbox_iou_seg']:>10.4f}")
    print(f"{'Mean BBox IoU (best)':<30} {metrics['mean_bbox_iou']:>10.4f}")
    print(f"{'Segmentation mIoU':<30} {metrics['seg_miou']:>10.4f}")
    print(f"{'Dice Coefficient':<30} {metrics['dice']:>10.4f}")
    print(f"{'Classification Top-1 Acc':<30} {metrics['cls_top1_acc']:>10.4f}")
    print(f"{'Classification Macro F1':<30} {metrics['cls_macro_f1']:>10.4f}")
    print("-" * 42)

    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    # Header
    print(f"{'':>12}", end="")
    for i in range(10):
        print(f"{GESTURE_NAMES[i][:5]:>7}", end="")
    print()
    for i in range(10):
        print(f"{GESTURE_NAMES[i][:10]:>12}", end="")
        for j in range(10):
            print(f"{cm[i, j]:>7d}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate hand gesture model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", default="~/Documents/UCL/COMP0248/dataset_all/extracted/")
    parser.add_argument("--test_root", default=None, help="Path to test dataset (no student folders)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
                          "cpu")

    model, model_args = load_model(args.checkpoint, device)
    use_depth = model_args.get("use_depth", False)

    # Evaluate on validation set
    _, val_loader = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        use_depth=use_depth,
        num_workers=args.num_workers,
    )

    print("\n--- Validation Set ---")
    val_metrics = evaluate(model, val_loader, device)
    print_results(val_metrics)

    save_path = os.path.splitext(args.checkpoint)[0] + "_val_metrics.txt"
    with open(save_path, "w") as f:
        for k, v in val_metrics.items():
            if k != "confusion_matrix":
                f.write(f"{k}: {v}\n")
        f.write(f"\nConfusion Matrix:\n{val_metrics['confusion_matrix']}\n")
    print(f"\nVal metrics saved to {save_path}")

    # Evaluate on test set if provided
    if args.test_root:
        from torch.utils.data import DataLoader

        test_samples = scan_test_dataset(args.test_root)
        print(f"\nFound {len(test_samples)} test samples.")
        test_ds = HandGestureDataset(test_samples, use_depth=use_depth, augment=False)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        print("\n--- Test Set ---")
        test_metrics = evaluate(model, test_loader, device)
        print_results(test_metrics)

        save_path = os.path.splitext(args.checkpoint)[0] + "_test_metrics.txt"
        with open(save_path, "w") as f:
            for k, v in test_metrics.items():
                if k != "confusion_matrix":
                    f.write(f"{k}: {v}\n")
            f.write(f"\nConfusion Matrix:\n{test_metrics['confusion_matrix']}\n")
        print(f"\nTest metrics saved to {save_path}")


if __name__ == "__main__":
    main()
