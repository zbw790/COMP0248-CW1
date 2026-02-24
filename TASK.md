# COMP0248 Coursework 1: Hand Gesture Detection, Segmentation & Classification

## Task
Build a PyTorch model that takes an RGB-D image of a hand gesture and:
1. Detects the hand (bounding box)
2. Segments the hand (pixel-wise mask)
3. Classifies the gesture (10 classes)

## Data Location
- Training data: ~/Documents/UCL/COMP0248/dataset_all/extracted/
- 31 student folders, each with structure: <studentno>_<name>/G01_call/clip01/{rgb,depth,depth_raw,annotation}/
- RGB: 640x480 uint8 PNG
- Annotation masks: 640x480 uint8 PNG (0=bg, 255=hand) - only 2 keyframes per clip have annotations
- Depth raw: 640x480 uint16 NPY
- Depth vis: 640x480 uint8 PNG (colormap visualization)
- 10 gestures: G01_call, G02_dislike, G03_like, G04_ok, G05_one, G06_palm, G07_peace, G08_rock, G09_stop, G10_three
- ~22800 RGB frames total, ~3299 annotation masks
- Test data not yet available (releases Feb 27)

## Key Constraints
- Must use PyTorch with custom torch.nn.Module
- NO high-level frameworks (YOLO, Detectron2, MMDetection, segmentation_models_pytorch, pre-built Mask R-CNN)
- CAN use torchvision pretrained backbones (ResNet etc) as feature extractors
- Must write own training loop, loss computation, evaluation scripts
- Need baseline model AND improved model for comparison

## Models to Build

### Baseline: SimpleMultiTaskNet
- Simple CNN encoder (few conv layers)
- Three heads: bbox regression (4 values), segmentation (pixel mask), classification (10 classes)
- Basic losses: L1 for bbox, BCE for segmentation, CE for classification

### Main Model: MultiTaskResNet  
- Pretrained ResNet-18/34 backbone (frozen or fine-tuned)
- FPN-like multi-scale features
- Detection head: predict bbox from features
- Segmentation head: upsampling decoder to predict mask
- Classification head: global average pooling + FC layers
- Combined multi-task loss with learned/tuned weights

## Data Loading Strategy
- Only use frames that HAVE annotation masks (since we need masks for training segmentation)
- For frames with masks: derive bounding box from mask (min/max of nonzero pixels)
- Gesture class from folder name (G01=0, G02=1, ..., G10=9)
- Use RGB + depth (4-channel input) OR RGB-only, compare both
- Train/val split: 80/20 by student (so val uses different hands)
- Resize to 256x256 for training

## Evaluation Metrics Required
- Detection: accuracy@0.5 IoU, mean bbox IoU
- Segmentation: mean IoU, Dice coefficient
- Classification: top-1 accuracy, macro F1, confusion matrix
- Qualitative overlays (mask + bbox on images)

## Deliverables Structure
project/
├── src/
│   ├── dataloader.py      # data loading + preprocessing
│   ├── model.py           # both baseline and main model
│   ├── train.py           # training loop
│   ├── evaluate.py        # all metrics
│   ├── visualise.py       # plotting masks, boxes, confusion matrices
│   └── utils.py           # helpers
├── weights/               # saved checkpoints
├── results/               # predictions, logs, plots
├── requirements.txt
└── README.md

## Important Notes
- Code must be modular, documented, reproducible
- Apple Silicon Mac (MPS backend available)
- Python 3.11
