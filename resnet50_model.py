from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models


CLASS_NAMES = ["dustin", "eleven", "hopper", "lucas", "steve"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class FaceResNet50(nn.Module):
    def __init__(self, num_classes: int = 5, dropout_p: float = 0.4):
        super().__init__()

        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.backbone = models.resnet50(weights=weights)
        except Exception:
            self.backbone = models.resnet50(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.backbone.fc(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_from_layer3(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in (self.backbone.layer3, self.backbone.layer4, self.backbone.fc):
            for param in module.parameters():
                param.requires_grad = True

    def get_optimizer(self, phase: int):
        if phase == 1:
            params = self.backbone.fc.parameters()
            return Adam(params, lr=1e-3)

        if phase == 2:
            params = list(self.backbone.layer3.parameters())
            params += list(self.backbone.layer4.parameters())
            params += list(self.backbone.fc.parameters())
            return Adam(params, lr=1e-4)

        raise ValueError("phase must be 1 or 2")


def build_model(num_classes: int = 5, dropout_p: float = 0.4) -> FaceResNet50:
    return FaceResNet50(num_classes=num_classes, dropout_p=dropout_p)
