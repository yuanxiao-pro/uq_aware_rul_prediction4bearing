#!/usr/bin/env python3
"""
Table 6: Multi-Scale Convolution Module 复现。
双分支多尺度（kernel 5 / kernel 3）→ 融合 → 后续卷积与全连接，输出 64 维特征（可接 RUL 头）。
输入 (batch, 13, seq_len)，1D 卷积适用于振动/时序。
"""

import torch
import torch.nn as nn
from typing import Optional


class MultiScaleConvModule(nn.Module):
    """
    Table 6 多尺度卷积模块。
    分支1: Conv1(13→256,k5) → AvgPool1(k5) → BN(256)
    分支2: Conv2(13→256,k3) → AvgPool2(k5) → BN(256)
    融合后: Conv3(256→128,k5) → AvgPool3(k3) → Conv4(128→64,k3) → MaxPool1(k3) → FC(64→64) + ReLU。
    """

    def __init__(self, in_channels: int = 13):
        super().__init__()
        # --- 分支1: kernel 5 ---
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(256)

        # --- 分支2: kernel 3 ---
        self.conv2 = nn.Conv1d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        # --- 融合后序列（两分支相加得 256 通道）---
        self.conv3 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64, 64)
        self.relu_fc = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, seq_len)
        return: (batch, 64)
        """
        # 分支1
        h1 = self.conv1(x)
        h1 = self.relu1(h1)
        h1 = self.avgpool1(h1)
        h1 = self.bn1(h1)

        # 分支2
        h2 = self.conv2(x)
        h2 = self.relu2(h2)
        h2 = self.avgpool2(h2)
        h2 = self.bn2(h2)

        # 融合（相加，保持 256 通道）
        h = h1 + h2

        h = self.conv3(h)
        h = self.relu3(h)
        h = self.avgpool3(h)

        h = self.conv4(h)
        h = self.relu4(h)
        h = self.maxpool1(h)

        h = h.mean(dim=2)
        h = self.fc1(h)
        h = self.relu_fc(h)
        return h


class MultiScaleCNN(nn.Module):
    """
    多尺度卷积模块 + 可选 RUL 输出头。
    输入 (batch, 13, seq_len)，输出 (batch,) 或 (batch, 1) 作为 RUL 点估计。
    """

    def __init__(self, in_channels: int = 13, num_classes: int = 1):
        super().__init__()
        self.backbone = MultiScaleConvModule(in_channels=in_channels)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)


def build_multi_scale_cnn(
    in_channels: int = 13,
    num_classes: int = 1,
    device: Optional[torch.device] = None,
) -> MultiScaleCNN:
    model = MultiScaleCNN(in_channels=in_channels, num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_multi_scale_cnn(in_channels=13, device=dev)
    x = torch.randn(4, 13, 256, device=dev)
    y = model(x)
    print("MultiScaleCNN output shape:", y.shape)
