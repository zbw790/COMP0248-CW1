"""Training script for multi-task hand gesture models.

Supports training both BaselineCNN and MultiTaskResNet with configurable
hyperparameters via command-line arguments.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataloader import get_dataloaders
from src.model import BaselineCNN, MultiTaskResNet
from src.utils import compute_iou_bbox, compute_seg_iou, compute_dice


def get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_loss(
    outputs: dict,
    batch: dict,
    w_bbox: float = 1.0,
    w_seg: float = 1.0,
    w_cls: float = 1.0,
) -> tuple:
    """Compute weighted multi-task loss.

    Args:
        outputs: Model outputs dict with 'seg', 'bbox', 'cls'.
        batch: Batch dict with 'mask', 'bbox', 'label'.
        w_bbox: Weight for bbox loss.
        w_seg: Weight for segmentation loss.
        w_cls: Weight for classification loss.

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual losses.
    """
    loss_bbox_l1 = nn.SmoothL1Loss()(outputs["bbox"], batch["bbox"])
    # GIoU-style loss for better bbox learning
    pred_b, gt_b = outputs["bbox"], batch["bbox"]
    x1 = torch.max(pred_b[:, 0], gt_b[:, 0])
    y1 = torch.max(pred_b[:, 1], gt_b[:, 1])
    x2 = torch.min(pred_b[:, 2], gt_b[:, 2])
    y2 = torch.min(pred_b[:, 3], gt_b[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_pred = (pred_b[:, 2] - pred_b[:, 0]).clamp(min=0) * (pred_b[:, 3] - pred_b[:, 1]).clamp(min=0)
    area_gt = (gt_b[:, 2] - gt_b[:, 0]).clamp(min=0) * (gt_b[:, 3] - gt_b[:, 1]).clamp(min=0)
    union = area_pred + area_gt - inter + 1e-7
    iou = inter / union
    loss_bbox_iou = (1 - iou).mean()
    loss_bbox = loss_bbox_l1 + loss_bbox_iou

    # Combined BCE + Dice loss for segmentation
    loss_seg_bce = nn.BCEWithLogitsLoss()(outputs["seg"], batch["mask"])
    pred_sig = torch.sigmoid(outputs["seg"])
    inter_seg = (pred_sig * batch["mask"]).sum(dim=(1, 2, 3))
    denom_seg = pred_sig.sum(dim=(1, 2, 3)) + batch["mask"].sum(dim=(1, 2, 3)) + 1e-7
    loss_seg_dice = (1 - 2.0 * inter_seg / denom_seg).mean()
    loss_seg = loss_seg_bce + loss_seg_dice
    loss_cls = nn.CrossEntropyLoss()(outputs["cls"], batch["label"])

    total = w_bbox * loss_bbox + w_seg * loss_seg + w_cls * loss_cls

    return total, {
        "bbox": loss_bbox.item(),
        "seg": loss_seg.item(),
        "cls": loss_cls.item(),
        "total": total.item(),
    }


def train_one_epoch(model, loader, optimizer, device, loss_weights):
    """Train for one epoch."""
    model.train()
    running = {"bbox": 0, "seg": 0, "cls": 0, "total": 0}
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)
        labels = batch["label"].to(device)
        b = {"mask": masks, "bbox": bboxes, "label": labels}

        outputs = model(images)
        loss, ld = compute_loss(outputs, b, **loss_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = images.size(0)
        for k in running:
            running[k] += ld[k] * bs
        n += bs

    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device, loss_weights):
    """Validate and compute losses + metrics."""
    model.eval()
    running = {"bbox": 0, "seg": 0, "cls": 0, "total": 0}
    all_ious, all_seg_ious, all_dice, correct, n = [], [], [], 0, 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)
        labels = batch["label"].to(device)
        b = {"mask": masks, "bbox": bboxes, "label": labels}

        outputs = model(images)
        _, ld = compute_loss(outputs, b, **loss_weights)

        bs = images.size(0)
        for k in running:
            running[k] += ld[k] * bs
        n += bs

        # Metrics
        all_ious.append(compute_iou_bbox(outputs["bbox"], bboxes).mean().item())
        all_seg_ious.append(compute_seg_iou(outputs["seg"], masks).item())
        all_dice.append(compute_dice(outputs["seg"], masks).item())
        correct += (outputs["cls"].argmax(1) == labels).sum().item()

    metrics = {
        "loss": {k: v / n for k, v in running.items()},
        "bbox_iou": sum(all_ious) / len(all_ious),
        "seg_iou": sum(all_seg_ious) / len(all_seg_ious),
        "dice": sum(all_dice) / len(all_dice),
        "cls_acc": correct / n,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train hand gesture multi-task model")
    parser.add_argument("--model", choices=["baseline", "resnet"], default="resnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--w_bbox", type=float, default=1.0)
    parser.add_argument("--w_seg", type=float, default=1.0)
    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--data_root", default="~/Documents/UCL/COMP0248/dataset_all/extracted/")
    parser.add_argument("--save_dir", default="../checkpoints/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    save_dir = Path(os.path.expanduser(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    in_channels = 4 if args.use_depth else 3
    loss_weights = {"w_bbox": args.w_bbox, "w_seg": args.w_seg, "w_cls": args.w_cls}

    # Build model
    if args.model == "baseline":
        model = BaselineCNN(in_channels=in_channels)
    else:
        model = MultiTaskResNet(in_channels=in_channels, pretrained=True)
    model = model.to(device)
    print(f"Model: {args.model}, params: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_loader, val_loader = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        use_depth=args.use_depth,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )

    # Optimiser & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training history
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(model, train_loader, optimizer, device, loss_weights)
        val_metrics = validate(model, val_loader, device, loss_weights)
        scheduler.step()

        elapsed = time.time() - t0
        vl = val_metrics["loss"]["total"]

        record = {
            "epoch": epoch,
            "train_loss": train_losses,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train: {train_losses['total']:.4f} | "
            f"Val: {vl:.4f} | "
            f"BBox IoU: {val_metrics['bbox_iou']:.4f} | "
            f"Seg IoU: {val_metrics['seg_iou']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"Cls Acc: {val_metrics['cls_acc']:.4f}"
        )

        # Save best
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args)},
                save_dir / f"{args.model}_best.pth",
            )
            print(f"  -> Saved best model (val_loss={vl:.4f})")

    # Save final + history
    torch.save(
        {"epoch": args.epochs, "model_state_dict": model.state_dict(), "args": vars(args)},
        save_dir / f"{args.model}_final.pth",
    )
    with open(save_dir / f"{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
