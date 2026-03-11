"""Multi-task models for hand gesture detection, segmentation, and classification.

Provides two models:
    - BaselineCNN: Simple 5-layer convolutional encoder with three task heads.
    - MultiTaskResNet: ResNet-18 backbone with FPN-style decoder and task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaselineCNN(nn.Module):
    """Simple baseline CNN with 5 conv layers and 3 task heads.

    Architecture:
        - 5 convolutional blocks (conv -> BN -> ReLU -> MaxPool)
        - Segmentation head: transposed convolutions to upsample to input size
        - BBox head: FC layers outputting 4 values
        - Classification head: FC layers outputting 10 classes
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        """Initialise BaselineCNN.

        Args:
            in_channels: Number of input channels (3 for RGB, 4 for RGBD).
            num_classes: Number of gesture classes.
        """
        super().__init__()
        # Encoder: 5 conv blocks, 256x256 -> 8x8
        self.enc1 = self._block(in_channels, 32)   # -> 128
        self.enc2 = self._block(32, 64)              # -> 64
        self.enc3 = self._block(64, 128)             # -> 32
        self.enc4 = self._block(128, 256)            # -> 16
        self.enc5 = self._block(256, 512)            # -> 8

        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 16
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 128
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 256
        )

        # BBox head
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C, 256, 256).

        Returns:
            Dict with keys 'seg' (N,1,256,256), 'bbox' (N,4), 'cls' (N,10).
        """
        f = self.enc1(x)
        f = self.enc2(f)
        f = self.enc3(f)
        f = self.enc4(f)
        f = self.enc5(f)

        seg = self.seg_decoder(f)
        bbox = self.bbox_head(f)
        cls = self.cls_head(f)

        return {"seg": seg, "bbox": bbox, "cls": cls}


class MultiTaskResNet(nn.Module):
    """Improved model using ResNet-18 backbone with FPN-style decoder.

    Architecture:
        - ResNet-18 backbone (pretrained, first conv modified for optional 4-ch)
        - FPN-style lateral connections + top-down pathway for segmentation
        - Global average pooled features for bbox and classification heads
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True):
        """Initialise MultiTaskResNet.

        Args:
            in_channels: Number of input channels (3 for RGB, 4 for RGBD).
            num_classes: Number of gesture classes.
            pretrained: Whether to use pretrained ResNet-18 weights.
        """
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Modify first conv if needed for 4-channel input
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    # Initialise extra channels as mean of RGB weights
                    new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(
                        1, in_channels - 3, 1, 1
                    )
            resnet.conv1 = new_conv

        # Backbone stages
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 64, stride 4
        self.layer2 = resnet.layer2  # 128, stride 8
        self.layer3 = resnet.layer3  # 256, stride 16
        self.layer4 = resnet.layer4  # 512, stride 32

        # FPN lateral connections
        self.lat4 = nn.Conv2d(512, 256, 1)
        self.lat3 = nn.Conv2d(256, 256, 1)
        self.lat2 = nn.Conv2d(128, 256, 1)
        self.lat1 = nn.Conv2d(64, 256, 1)

        # FPN smooth convolutions
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, 3, padding=1)

        # Segmentation head from FPN features
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

        # BBox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C, 256, 256).

        Returns:
            Dict with keys 'seg' (N,1,256,256), 'bbox' (N,4), 'cls' (N,10).
        """
        input_size = x.shape[2:]

        # Backbone
        c0 = self.stem(x)       # stride 4
        c1 = self.layer1(c0)    # stride 4, 64ch
        c2 = self.layer2(c1)    # stride 8, 128ch
        c3 = self.layer3(c2)    # stride 16, 256ch
        c4 = self.layer4(c3)    # stride 32, 512ch

        # FPN top-down
        p4 = self.smooth4(self.lat4(c4))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest"))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest"))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="nearest"))

        # Segmentation from finest FPN level
        seg = self.seg_head(p1)
        seg = F.interpolate(seg, size=input_size, mode="bilinear", align_corners=False)

        # Global features for bbox + classification
        feat = self.pool(c4).flatten(1)  # (N, 512)
        bbox = self.bbox_head(feat)
        cls = self.cls_head(feat)

        return {"seg": seg, "bbox": bbox, "cls": cls}


if __name__ == "__main__":
    # Quick test
    for Model, name in [(BaselineCNN, "Baseline"), (MultiTaskResNet, "ResNet")]:
        for ch in [3, 4]:
            m = Model(in_channels=ch)
            x = torch.randn(2, ch, 256, 256)
            out = m(x)
            print(f"{name} (ch={ch}): seg={out['seg'].shape}, bbox={out['bbox'].shape}, cls={out['cls'].shape}")
