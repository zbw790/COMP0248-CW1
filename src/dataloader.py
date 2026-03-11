"""Dataset and DataLoader for hand gesture RGB-D data.

Scans the dataset directory for annotated frames, builds samples with
RGB (+ optional depth), segmentation mask, bounding box, and gesture class.
"""

import os
import re
import glob
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from src.utils import bbox_from_mask

# Gesture folder prefix -> class index
GESTURE_MAP: Dict[str, int] = {
    "G01": 0, "G02": 1, "G03": 2, "G04": 3, "G05": 4,
    "G06": 5, "G07": 6, "G08": 7, "G09": 8, "G10": 9,
}

IMG_SIZE = 256


class HandGestureDataset(Dataset):
    """Dataset for hand gesture detection, segmentation, and classification.

    Each sample contains:
        - image: (C, 256, 256) tensor (C=3 for RGB, C=4 for RGBD)
        - mask: (1, 256, 256) binary tensor
        - bbox: (4,) tensor [x1, y1, x2, y2] normalised to [0, 1]
        - label: int gesture class (0-9)
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, Optional[str], int]],
        use_depth: bool = False,
        augment: bool = False,
    ):
        """Initialise the dataset.

        Args:
            samples: List of (rgb_path, annotation_path, depth_raw_path, gesture_class).
            use_depth: If True, load depth and create 4-channel input.
            augment: If True, apply data augmentation (train mode).
        """
        self.samples = samples
        self.use_depth = use_depth
        self.augment = augment

        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, ann_path, depth_path, label = self.samples[idx]

        # Load RGB
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # Load annotation mask
        mask = Image.open(ann_path).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        # Augmentation: horizontal flip
        do_flip = False
        if self.augment and torch.rand(1).item() > 0.5:
            do_flip = True
            rgb = TF.hflip(rgb)
            mask = TF.hflip(mask)

        # Augmentation: color jitter (RGB only)
        if self.augment:
            rgb = self.color_jitter(rgb)

        # Convert to tensors
        rgb_tensor = TF.to_tensor(rgb)  # (3, H, W), [0, 1]
        mask_np = np.array(mask)
        mask_binary = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # (1, H, W)

        # Load depth if requested
        if self.use_depth and depth_path is not None and os.path.exists(depth_path):
            depth = np.load(depth_path).astype(np.float32)
            depth = np.array(Image.fromarray(depth).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            if do_flip:
                depth = np.fliplr(depth).copy()
            # Normalise depth to [0, 1]
            d_max = depth.max()
            if d_max > 0:
                depth = depth / d_max
            depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
            img_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)
        else:
            img_tensor = rgb_tensor

        # Normalise RGB channels with ImageNet stats
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_tensor[:3] = TF.normalize(img_tensor[:3], mean, std)

        # Extract bbox from mask
        bbox = bbox_from_mask(mask_binary)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }


def scan_dataset(root: str) -> List[Tuple[str, str, Optional[str], int]]:
    """Scan the dataset directory and collect all annotated samples.

    Args:
        root: Path to the extracted dataset (containing student folders).

    Returns:
        List of (rgb_path, annotation_path, depth_raw_path_or_None, gesture_class).
    """
    samples = []
    root = os.path.expanduser(root)

    # Find all student folders
    student_dirs = sorted(glob.glob(os.path.join(root, "*_*")))

    for student_dir in student_dirs:
        if not os.path.isdir(student_dir):
            continue
        # Find gesture folders
        for gesture_dir in sorted(glob.glob(os.path.join(student_dir, "G*_*"))):
            gesture_name = os.path.basename(gesture_dir)
            gesture_prefix = gesture_name[:3]  # e.g. "G01"
            if gesture_prefix not in GESTURE_MAP:
                continue
            gesture_class = GESTURE_MAP[gesture_prefix]

            # Find clip folders
            for clip_dir in sorted(glob.glob(os.path.join(gesture_dir, "clip*"))):
                ann_dir = os.path.join(clip_dir, "annotation")
                rgb_dir = os.path.join(clip_dir, "rgb")
                depth_raw_dir = os.path.join(clip_dir, "depth_raw")

                if not os.path.isdir(ann_dir) or not os.path.isdir(rgb_dir):
                    continue

                # Each annotated frame
                for ann_file in sorted(os.listdir(ann_dir)):
                    if not ann_file.lower().endswith(".png"):
                        continue
                    ann_path = os.path.join(ann_dir, ann_file)
                    rgb_path = os.path.join(rgb_dir, ann_file)

                    if not os.path.exists(rgb_path):
                        # Try matching by frame number
                        continue

                    # Depth raw: same filename but .npy
                    depth_stem = os.path.splitext(ann_file)[0]
                    depth_path = os.path.join(depth_raw_dir, depth_stem + ".npy")
                    if not os.path.exists(depth_path):
                        depth_path = None

                    samples.append((rgb_path, ann_path, depth_path, gesture_class))

    return samples


def scan_test_dataset(root: str) -> List[Tuple[str, str, Optional[str], int]]:
    """Scan the test dataset directory (no student layer).

    Test data structure: root/G01_call/clip01/{rgb,depth,depth_raw,annotation}/

    Args:
        root: Path to the test dataset directory.

    Returns:
        List of (rgb_path, annotation_path, depth_raw_path_or_None, gesture_class).
    """
    samples = []
    root = os.path.expanduser(root)

    for gesture_dir in sorted(glob.glob(os.path.join(root, "G*_*"))):
        gesture_name = os.path.basename(gesture_dir)
        gesture_prefix = gesture_name[:3]
        if gesture_prefix not in GESTURE_MAP:
            continue
        gesture_class = GESTURE_MAP[gesture_prefix]

        for clip_dir in sorted(glob.glob(os.path.join(gesture_dir, "clip*"))):
            ann_dir = os.path.join(clip_dir, "annotation")
            rgb_dir = os.path.join(clip_dir, "rgb")
            depth_raw_dir = os.path.join(clip_dir, "depth_raw")

            if not os.path.isdir(ann_dir) or not os.path.isdir(rgb_dir):
                continue

            for ann_file in sorted(os.listdir(ann_dir)):
                if not ann_file.lower().endswith(".png"):
                    continue
                ann_path = os.path.join(ann_dir, ann_file)
                rgb_path = os.path.join(rgb_dir, ann_file)

                if not os.path.exists(rgb_path):
                    continue

                depth_stem = os.path.splitext(ann_file)[0]
                depth_path = os.path.join(depth_raw_dir, depth_stem + ".npy")
                if not os.path.exists(depth_path):
                    depth_path = None

                samples.append((rgb_path, ann_path, depth_path, gesture_class))

    return samples


def split_by_student(
    samples: List[Tuple[str, str, Optional[str], int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List, List]:
    """Split samples into train/val by student identity (no data leakage).

    Args:
        samples: All collected samples.
        val_ratio: Fraction of students for validation.
        seed: Random seed.

    Returns:
        (train_samples, val_samples)
    """
    # Group by student
    student_samples: Dict[str, List] = {}
    for s in samples:
        # Extract student folder name from path
        parts = Path(s[0]).parts
        # Find the part that matches student pattern (digits_name)
        student = None
        for p in parts:
            if re.match(r"^\d+_\w+", p):
                student = p
                break
        if student is None:
            student = "unknown"
        student_samples.setdefault(student, []).append(s)

    students = sorted(student_samples.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(students)

    n_val = max(1, int(len(students) * val_ratio))
    val_students = set(students[:n_val])

    train, val = [], []
    for st in students:
        if st in val_students:
            val.extend(student_samples[st])
        else:
            train.extend(student_samples[st])

    return train, val


def get_dataloaders(
    root: str = "~/Documents/UCL/COMP0248/dataset_all/extracted/",
    batch_size: int = 16,
    use_depth: bool = False,
    num_workers: int = 4,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Args:
        root: Dataset root directory.
        batch_size: Batch size.
        use_depth: Whether to include depth channel.
        num_workers: Number of data loading workers.
        val_ratio: Validation split ratio.

    Returns:
        (train_loader, val_loader)
    """
    all_samples = scan_dataset(root)
    print(f"Found {len(all_samples)} annotated samples total.")

    train_samples, val_samples = split_by_student(all_samples, val_ratio=val_ratio)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = HandGestureDataset(train_samples, use_depth=use_depth, augment=True)
    val_ds = HandGestureDataset(val_samples, use_depth=use_depth, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print("Image shape:", batch["image"].shape)
    print("Mask shape:", batch["mask"].shape)
    print("Bbox shape:", batch["bbox"].shape)
    print("Labels:", batch["label"])
