#!/usr/bin/env python3
"""
MSCRGAT：Multi-Scale CNN 与 Res_GRU 特征拼接 + 3 层全连接预测器，输出经 Sigmoid。
heteroscedastic=True 时预测头输出 2 维 (mu_logit, log_var)，得到 (μ, σ) 用于偶然不确定性。
build 函数在 adversarial_mscrgat.py 中：build_mscrgat(config, device)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from multi_scale_cnn import MultiScaleConvModule
from res_gru import ResGRU


class MSCRGAT(nn.Module):
    """
    多尺度 CNN 特征 (64 维) + Res_GRU 特征 (64 维) 拼接 -> 3 层全连接 -> Sigmoid。
    heteroscedastic=True 时最后一层输出 2 维 (mu_logit, log_var)，μ=sigmoid(mu_logit)，σ=softplus(log_var)+eps。
    """

    def __init__(
        self,
        input_dim: int = 13,
        seq_len: Optional[int] = None,
        predictor_hidden: tuple = (64, 64, 32),
        dropout: float = 0.5,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.mscnn = MultiScaleConvModule(in_channels=input_dim)
        self.res_gru = ResGRU(
            input_dim=input_dim,
            hidden_dims=(13, 64, 64),
            fc_dim=64,
        )
        concat_dim = 64 + 64
        hidden = list(predictor_hidden)
        layers = []
        in_f = concat_dim
        for h in hidden:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_f = h
        out_dim = 2 if heteroscedastic else 1
        layers.append(nn.Linear(in_f, out_dim))
        self.predictor = nn.Sequential(*layers)
        self.sigmoid_out = nn.Sigmoid()
        self.feature_dim = concat_dim  # 128，供域判别器 / MMD 使用
        self._eps = 1e-6

    def forward(self, x: torch.Tensor, feature: bool = False):
        """
        x: (batch, seq_len, input_dim)，例如 (B, L, 13)
        heteroscedastic=False:
          feature=False: return (batch,) RUL 预测 [0,1]
          feature=True:  return (rul, feat)
        heteroscedastic=True:
          feature=False: return (mu, sigma)，mu 点预测，sigma 偶然不确定性
          feature=True:  return (mu, sigma, feat)
        """
        x_cnn = x.permute(0, 2, 1)
        feat_cnn = self.mscnn(x_cnn)
        feat_gru = self.res_gru.forward_feature(x)
        cat = torch.cat([feat_cnn, feat_gru], dim=1)
        out = self.predictor(cat)

        if self.heteroscedastic:
            mu_logit = out[:, 0]
            log_var = out[:, 1]
            mu = self.sigmoid_out(mu_logit)
            sigma = F.softplus(log_var) + self._eps
            if feature:
                return mu, sigma, cat
            return mu, sigma
        else:
            rul = self.sigmoid_out(out).squeeze(-1)
            if feature:
                return rul, cat
            return rul


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from adversarial_mscrgat import build_mscrgat
    except ImportError:
        from adversarial_fbtcn import build_mscrgat  # type: ignore
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"input_dim": 13, "predictor_hidden": (64, 64, 32)}
    model = build_mscrgat(cfg, device=dev)
    x = torch.randn(4, 64, 13, device=dev)
    y = model(x)
    print("MSCRGAT output shape:", y.shape)
    print("MSCRGAT output range (ReLU):", y.min().item(), y.max().item())
