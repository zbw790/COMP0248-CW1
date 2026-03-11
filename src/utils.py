"""Utility functions for IoU, Dice, bbox operations, and metric helpers."""

import torch
import numpy as np
from typing import Tuple


def bbox_from_mask(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Extract bounding box [x_min, y_min, x_max, y_max] from a binary mask.

    Args:
        mask: Binary mask of shape (H, W) with nonzero values indicating the object.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) normalised to [0, 1].
        Returns (0, 0, 0, 0) if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    h, w = mask.shape
    x_min, x_max = xs.min() / w, xs.max() / w
    y_min, y_max = ys.min() / h, ys.max() / h
    return (x_min, y_min, x_max, y_max)


def compute_iou_bbox(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute IoU between predicted and target bounding boxes.

    Args:
        pred: Predicted boxes of shape (N, 4) as [x1, y1, x2, y2].
        target: Target boxes of shape (N, 4) as [x1, y1, x2, y2].

    Returns:
        IoU values of shape (N,).
    """
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_target - inter + 1e-7

    return inter / union


def compute_seg_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean IoU for binary segmentation masks.

    Args:
        pred: Predicted mask logits of shape (N, 1, H, W).
        target: Target binary masks of shape (N, 1, H, W).

    Returns:
        Mean IoU scalar tensor.
    """
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection + 1e-7
    return (intersection / union).mean()


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Dice coefficient for binary segmentation masks.

    Args:
        pred: Predicted mask logits of shape (N, 1, H, W).
        target: Target binary masks of shape (N, 1, H, W).

    Returns:
        Mean Dice coefficient scalar tensor.
    """
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    denom = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-7
    return (2.0 * intersection / denom).mean()


def detection_accuracy(pred_bbox: torch.Tensor, target_bbox: torch.Tensor,
                       iou_threshold: float = 0.5) -> float:
    """Compute detection accuracy at a given IoU threshold.

    Args:
        pred_bbox: Predicted boxes (N, 4).
        target_bbox: Target boxes (N, 4).
        iou_threshold: IoU threshold for a correct detection.

    Returns:
        Fraction of predictions with IoU >= threshold.
    """
    ious = compute_iou_bbox(pred_bbox, target_bbox)
    return (ious >= iou_threshold).float().mean().item()


def bbox_from_mask_tensor(seg_logits: torch.Tensor) -> torch.Tensor:
    """Derive bounding boxes from predicted segmentation mask logits.

    Args:
        seg_logits: Predicted mask logits of shape (N, 1, H, W).

    Returns:
        Bounding boxes of shape (N, 4) as [x1, y1, x2, y2] normalised to [0, 1].
    """
    pred_bin = (torch.sigmoid(seg_logits) > 0.5).squeeze(1)  # (N, H, W)
    N, H, W = pred_bin.shape
    bboxes = torch.zeros(N, 4, device=seg_logits.device)

    for i in range(N):
        mask = pred_bin[i]
        ys, xs = torch.where(mask > 0)
        if len(xs) == 0:
            continue
        bboxes[i, 0] = xs.min().float() / W
        bboxes[i, 1] = ys.min().float() / H
        bboxes[i, 2] = xs.max().float() / W
        bboxes[i, 3] = ys.max().float() / H

    return bboxes


GESTURE_NAMES = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",
]
