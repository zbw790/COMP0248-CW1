# Hand Gesture Detection, Segmentation & Classification

Multi-task PyTorch model for hand gesture recognition from RGB-D images (COMP0248 coursework).

## Project Structure

```
project/
├── src/
│   ├── dataloader.py   # Dataset scanning, loading, augmentation
│   ├── model.py        # BaselineCNN and MultiTaskResNet models
│   ├── train.py        # Training loop with multi-task loss
│   ├── evaluate.py     # Full metric evaluation
│   ├── visualise.py    # Qualitative results and plots
│   └── utils.py        # IoU, Dice, bbox utilities
├── checkpoints/        # Saved model weights
├── results/            # Visualisation outputs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Models

### BaselineCNN
Simple 5-layer convolutional encoder with three heads (bbox, segmentation, classification). No pretrained weights.

### MultiTaskResNet (Improved)
ResNet-18 backbone (ImageNet pretrained) with:
- FPN-style decoder for segmentation
- Global pooled features for bbox regression and classification
- Optional 4-channel input for RGB-D

## Usage

### Train

```bash
cd src/

# Train baseline (RGB only)
python train.py --model baseline --epochs 50 --batch_size 16 --lr 1e-3

# Train improved model (RGB only)
python train.py --model resnet --epochs 50 --batch_size 16 --lr 1e-3

# Train with depth (RGBD, 4-channel)
python train.py --model resnet --epochs 50 --use_depth

# Custom loss weights
python train.py --model resnet --w_bbox 2.0 --w_seg 1.0 --w_cls 1.0
```

### Evaluate

```bash
python evaluate.py --checkpoint ../checkpoints/resnet_best.pth
```

Outputs: detection accuracy @0.5 IoU, mean bbox IoU, segmentation mIoU, Dice coefficient, classification top-1 accuracy, macro F1, and confusion matrix.

### Visualise

```bash
# Prediction overlays + confusion matrix
python visualise.py --checkpoint ../checkpoints/resnet_best.pth

# Training curves
python visualise.py --history ../checkpoints/resnet_history.json

# Both
python visualise.py --checkpoint ../checkpoints/resnet_best.pth --history ../checkpoints/resnet_history.json
```

## Data

Expected structure at `~/Documents/UCL/COMP0248/dataset_all/extracted/`:
```
<studentno>_<name>/
  G01_call/clip01/{rgb,depth,depth_raw,annotation}/
  G02_dislike/clip01/...
  ...
```

- 10 gesture classes (G01-G10)
- ~3299 annotated frames across 31 students
- Train/val split by student (80/20) to prevent data leakage

## Metrics

| Metric | Description |
|--------|-------------|
| Detection Acc @0.5 IoU | Fraction of predictions with bbox IoU ≥ 0.5 |
| Mean BBox IoU | Average IoU between predicted and GT bboxes |
| Seg mIoU | Mean intersection-over-union for binary segmentation |
| Dice | Dice coefficient for segmentation |
| Cls Top-1 Acc | Classification accuracy |
| Cls Macro F1 | Macro-averaged F1 across all 10 classes |
