#!/usr/bin/env python3
"""
Res_GRU：3 层 GRU + 残差式跳跃连接（第 1 层与第 3 层输出拼接）+ 全连接，输出 RUL 点估计。
不使用 BDL 框架，纯 PyTorch。
"""

import torch
import torch.nn as nn
from typing import Optional


class ResGRU(nn.Module):
    """
    3 层 GRU，通道数 13 / 64 / 64；将第 1 层与第 3 层的输出拼接后经全连接（64 维）+ ReLU，输出 RUL 点估计。
    输入 x: (batch, seq_len, input_dim)，输出 (batch,) 或 (batch, 1)。
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: tuple = (13, 64, 64),
        fc_dim: int = 64,
    ):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.hidden_dims = hidden_dims

        self.gru1 = nn.GRU(input_dim, h1, batch_first=True)
        self.gru2 = nn.GRU(h1, h2, batch_first=True)
        self.gru3 = nn.GRU(h2, h3, batch_first=True)

        concat_dim = h1 + h3  # 13 + 64 = 77
        self.fc = nn.Linear(concat_dim, fc_dim)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        return: (batch,) 或 (batch, 1)，RUL 点估计
        """
        out1, _ = self.gru1(x)           # (B, L, h1)
        out2, _ = self.gru2(out1)        # (B, L, h2)
        out3, _ = self.gru3(out2)        # (B, L, h3)

        h1_last = out1[:, -1, :]         # (B, h1)
        h3_last = out3[:, -1, :]         # (B, h3)
        cat = torch.cat([h1_last, h3_last], dim=1)  # (B, h1+h3)

        h = self.fc(cat)
        h = self.relu(h)
        rul = self.out(h)                # (B, 1)
        return rul.squeeze(-1)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        """返回最后一层全连接 ReLU 后的 64 维特征，(batch, seq_len, input_dim) -> (batch, 64)。"""
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)
        out3, _ = self.gru3(out2)
        h1_last = out1[:, -1, :]
        h3_last = out3[:, -1, :]
        cat = torch.cat([h1_last, h3_last], dim=1)
        h = self.relu(self.fc(cat))
        return h


def build_res_gru(
    input_dim: int = 13,
    hidden_dims: tuple = (13, 64, 64),
    fc_dim: int = 64,
    device: Optional[torch.device] = None,
) -> ResGRU:
    model = ResGRU(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        fc_dim=fc_dim,
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_res_gru(input_dim=13, device=dev)
    x = torch.randn(4, 64, 13, device=dev)
    y = model(x)
    print("Res_GRU output shape:", y.shape)  # (4,)
