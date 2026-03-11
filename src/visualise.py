"""Visualisation utilities for qualitative results and training analysis.

Generates:
    - Overlay images with predicted mask and bounding box
    - Confusion matrix heatmap
    - Training curves (loss, metrics over epochs)
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.dataloader import get_dataloaders, HandGestureDataset, scan_dataset, split_by_student
from src.model import BaselineCNN, MultiTaskResNet
from src.evaluate import load_model
from src.utils import GESTURE_NAMES, bbox_from_mask_tensor


def denormalise(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised image tensor back to displayable numpy array.

    Args:
        img_tensor: (C, H, W) tensor, ImageNet-normalised.

    Returns:
        (H, W, 3) uint8 numpy array.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor[:3].cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def visualise_predictions(model, loader, device, save_dir: str, num_samples: int = 16):
    """Generate overlay visualisations of model predictions.

    Args:
        model: Trained model.
        loader: DataLoader.
        device: Torch device.
        save_dir: Directory to save images.
        num_samples: Number of samples to visualise.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks_gt = batch["mask"]
    bboxes_gt = batch["bbox"]
    labels_gt = batch["label"]

    with torch.no_grad():
        outputs = model(images)

    pred_masks = (torch.sigmoid(outputs["seg"]) > 0.5).cpu().float()
    pred_bboxes = bbox_from_mask_tensor(outputs["seg"]).cpu()
    pred_labels = outputs["cls"].argmax(1).cpu()

    n = min(num_samples, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img = denormalise(images[i].cpu())
        h, w = img.shape[:2]

        # Original + GT
        ax = axes[i, 0]
        ax.imshow(img)
        ax.set_title(f"GT: {GESTURE_NAMES[labels_gt[i]]}", fontsize=10)
        # GT bbox
        bx = bboxes_gt[i]
        rect = patches.Rectangle(
            (bx[0] * w, bx[1] * h), (bx[2] - bx[0]) * w, (bx[3] - bx[1]) * h,
            linewidth=2, edgecolor="green", facecolor="none"
        )
        ax.add_patch(rect)
        # GT mask overlay
        mask_gt = masks_gt[i, 0].numpy()
        overlay = np.zeros((*mask_gt.shape, 4))
        overlay[mask_gt > 0.5] = [0, 1, 0, 0.3]
        ax.imshow(overlay)
        ax.axis("off")

        # Prediction overlay
        ax = axes[i, 1]
        ax.imshow(img)
        ax.set_title(f"Pred: {GESTURE_NAMES[pred_labels[i]]}", fontsize=10)
        bx = pred_bboxes[i]
        rect = patches.Rectangle(
            (bx[0] * w, bx[1] * h), (bx[2] - bx[0]) * w, (bx[3] - bx[1]) * h,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        mask_pred = pred_masks[i, 0].numpy()
        overlay = np.zeros((*mask_pred.shape, 4))
        overlay[mask_pred > 0.5] = [1, 0, 0, 0.3]
        ax.imshow(overlay)
        ax.axis("off")

        # Mask comparison
        ax = axes[i, 2]
        combined = np.zeros((*mask_gt.shape, 3))
        combined[mask_gt > 0.5, 1] = 1       # GT green
        combined[mask_pred > 0.5, 0] = 1     # Pred red
        # Overlap -> yellow
        ax.imshow(combined)
        ax.set_title("Green=GT, Red=Pred, Yellow=Overlap", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction visualisations to {save_dir}/predictions.png")


def plot_confusion_matrix(model, loader, device, save_dir: str):
    """Generate and save confusion matrix plot.

    Args:
        model: Trained model.
        loader: DataLoader.
        device: Torch device.
        save_dir: Directory to save plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            outputs = model(images)
            preds = outputs["cls"].argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    short_names = [g.split("_")[1] for g in GESTURE_NAMES]

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=short_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Gesture Classification Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_dir}/confusion_matrix.png")


def plot_training_curves(history_path: str, save_dir: str):
    """Plot training curves from saved history JSON.

    Args:
        history_path: Path to *_history.json file.
        save_dir: Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_total = [h["train_loss"]["total"] for h in history]
    val_total = [h["val_metrics"]["loss"]["total"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].plot(epochs, train_total, label="Train")
    axes[0, 0].plot(epochs, val_total, label="Val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Individual losses
    for key, color in [("bbox", "blue"), ("seg", "orange"), ("cls", "green")]:
        train_vals = [h["train_loss"][key] for h in history]
        val_vals = [h["val_metrics"]["loss"][key] for h in history]
        axes[0, 1].plot(epochs, train_vals, f"--", color=color, alpha=0.6, label=f"Train {key}")
        axes[0, 1].plot(epochs, val_vals, color=color, label=f"Val {key}")
    axes[0, 1].set_title("Individual Losses")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Seg metrics
    seg_iou = [h["val_metrics"]["seg_iou"] for h in history]
    dice = [h["val_metrics"]["dice"] for h in history]
    axes[1, 0].plot(epochs, seg_iou, label="Seg mIoU")
    axes[1, 0].plot(epochs, dice, label="Dice")
    axes[1, 0].set_title("Segmentation Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Classification + BBox
    cls_acc = [h["val_metrics"]["cls_acc"] for h in history]
    bbox_iou = [h["val_metrics"]["bbox_iou"] for h in history]
    axes[1, 1].plot(epochs, cls_acc, label="Cls Accuracy")
    axes[1, 1].plot(epochs, bbox_iou, label="BBox IoU")
    axes[1, 1].set_title("Classification & Detection")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_dir}/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Visualise model results")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--history", help="Path to training history JSON")
    parser.add_argument("--data_root", default="~/Documents/UCL/COMP0248/dataset_all/extracted/")
    parser.add_argument("--save_dir", default="~/Documents/UCL/COMP0248/project/results/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
                          "cpu")

    if args.checkpoint:
        model, model_args = load_model(args.checkpoint, device)
        use_depth = model_args.get("use_depth", False)

        _, val_loader = get_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            use_depth=use_depth,
            num_workers=args.num_workers,
        )

        visualise_predictions(model, val_loader, device, save_dir)
        plot_confusion_matrix(model, val_loader, device, save_dir)

    if args.history:
        plot_training_curves(args.history, save_dir)

    if not args.checkpoint and not args.history:
        print("Provide --checkpoint and/or --history. See --help.")


if __name__ == "__main__":
    main()
