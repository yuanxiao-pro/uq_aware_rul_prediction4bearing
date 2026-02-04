#  定义 TCN 卷积+残差 模块
from torch.nn.utils import parametrizations
import torch
import torch.nn as nn
from bayesian_torch.layers import Conv1dReparameterization, LinearReparameterization

import sys
sys.path.append('..')
from loss_function import compute_au_nll, compute_au_nll_with_pos, compute_au_nll_with_crps_and_pos, compute_au_nll_with_crps, compute_au_nll_with_crps_wide_intervals_v2
from function_kl import get_bayesian_model_parameters, get_bayesian_model_mu_rho, calculate_function_kl
from metrics import picp, nmpiw,cwc, ece, aleatoric_uncertainty, epistemic_uncertainty, ood_detection, sharpness,mae,rmse
from stable_fbtcn_training import model_train_stable, StabilizedAUNLL, get_stable_optimizer
import torch.nn.functional as F

# 定义裁剪模块
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class BayesianTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, conv_posterior_rho_init, dropout=0.2):
        super().__init__()
        self.conv1 = Conv1dReparameterization(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=conv_posterior_rho_init
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = Conv1dReparameterization(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=conv_posterior_rho_init
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        # self.downsample = Conv1dReparameterization(
        #     in_channels, out_channels, 1,
        #     prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3
        # ) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        kl = 0.0
        out, kl1 = self.conv1(x)
        out = self.chomp1(out)
        kl = kl + kl1
        out = self.relu1(out)
        out = self.dropout1(out)
        out, kl2 = self.conv2(out)
        out = self.chomp2(out)
        kl = kl + kl2
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res), kl
    


class SelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        # 将输入维度从input_dim（通道数）映射到attention_dim
        self.query = nn.Linear(input_dim, embed_dim)
        self.key = nn.Linear(input_dim, embed_dim) 
        self.value = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) 其中input_dim是TCN输出的通道数
        # 对最后一维（通道维度）进行线性变换
        q = self.query(x)  # (batch_size, seq_len, embed_dim)
        k = self.key(x)    # (batch_size, seq_len, embed_dim)
        v = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # 计算注意力权重
        attention_weights = F.softmax(torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)), dim=-1)
        # 应用注意力权重
        output = torch.matmul(attention_weights, v)  # (batch_size, seq_len, embed_dim)
        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    支持任意输入特征维度，自动映射到embed_dim
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, input_dim=None):
        """
        Args:
            embed_dim: 嵌入维度（输出维度）
            num_heads: 注意力头的数量
            dropout: dropout比率
            input_dim: 输入特征维度，必须提供
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除"
        assert input_dim is not None, "input_dim必须提供"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.input_dim = input_dim
        
        # 创建Q, K, V线性层：将输入特征维度映射到embed_dim
        self.query = nn.Linear(input_dim, embed_dim)
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        
        # 输出投影层
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape为 (batch_size, seq_len, input_dim)
        Returns:
            output: 输出张量，shape为 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, input_features = x.shape
        assert input_features == self.input_dim, f"输入特征维度 {input_features} 与预期 {self.input_dim} 不匹配"
        
        # 1. 通过线性层得到Q, K, V
        # x: (batch_size, seq_len, input_features) -> (batch_size, seq_len, embed_dim)
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # 2. 重塑为多头形式
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在 Q, K, V 的shape都是 (batch_size, num_heads, seq_len, head_dim)
        
        # 3. 计算注意力分数
        # Q @ K^T: (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))
        
        # 4. 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 5. 应用注意力权重到V
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attention_weights, V)
        
        # 6. 拼接所有头
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 7. 通过输出投影层
        output = self.output_proj(attn_output)  # (batch_size, seq_len, embed_dim)
        
        return output

class BayesianTCN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_channels,
        attention_dim,
        conv_posterior_rho_init,
        output_posterior_rho_init,
        kernel_size=2,
        dropout=0.2,
        output_dim=1,
        attention_mode: str = "self",
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        self.kl_modules = []
        self.attention_mode = attention_mode
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            block = BayesianTemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1)*dilation_size,
                dropout=dropout,
                conv_posterior_rho_init=conv_posterior_rho_init
            )
            layers.append(block)
        self.network = nn.ModuleList(layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 获取TCN最后一层的输出通道数
        tcn_output_channels = num_channels[-1] if len(num_channels) > 0 else input_dim
        
        # 当 attention_mode 为 none 时跳过自注意力
        # 修改：传递TCN输出通道数作为SelfAttention的输入维度
        self.attention = None if attention_mode == "none" else SelfAttention(tcn_output_channels, attention_dim)
        
        # 定义mu和sigma层，输入维度是注意力层的输出维度embed_dim
        # posterior_rho_init=-5时，后验的对数方差为-10 因为对数方差 log(var) = 2 * rho，rho=-5，所以 log(var)=2*(-5)=-10
        self.mu = LinearReparameterization(
            in_features=attention_dim, out_features=output_dim,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=output_posterior_rho_init  # 增大初始后验方差，使均值预测更具波动性
        )
        self.sigma = LinearReparameterization(
            in_features=attention_dim, out_features=output_dim,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=output_posterior_rho_init
        )

        # 自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def generate_init_params(self, sample_input):
        """
        根据输入数据的形状初始化模型参数
        Args:
            sample_input: 输入数据的一个样本（用于推断参数形状）
        Returns:
            params_dict: 包含所有参数的字典
        """
        # 确保不计算梯度
        with torch.no_grad():
            # 前向传播一次（不保存计算图）
            _ = self.forward(sample_input)
            params_dict = {k: v.clone() for k, v in self.state_dict().items()}
            return params_dict
        
    def forward(self, x, feature=False):
        # x: [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        x = x.permute(0, 2, 1)
        kl = 0.0
        
        # ========== TCN Backbone (特征提取层) ==========
        for block in self.network:
            x, kl_block = block(x)
            kl = kl + kl_block
        # 此时 x shape: [batch, channels, seq_len] 例如 [batch, 32, seq_len]
        
        # ========== 自注意力机制 (保留通道信息) ==========
        # 转置为序列格式: [batch, channels, seq_len] -> [batch, seq_len, channels]
        # 不再对channels维度做平均，保留所有通道信息
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len] -> [batch, seq_len, channels]
        
        if self.attention is not None:
            # SelfAttention期望输入: [batch, seq_len, channels]
            # 输出: [batch, seq_len, attention_dim]
            x = self.attention(x)  # [batch, seq_len, channels] -> [batch, seq_len, attention_dim]
        
        # 转回Conv1d格式用于后续池化: [batch, seq_len, attention_dim] -> [batch, attention_dim, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, attention_dim] -> [batch, attention_dim, seq_len]
        # ========== 池化和输出层 ==========
        x = self.adaptive_pool(x)  # [batch, embed_dim, seq_len] -> [batch, embed_dim, 1]
        x = x.view(x.size(0), -1)  # [batch, embed_dim, 1] -> [batch, embed_dim]

        mu, kl_mu = self.mu(x)
        sigma, kl_sigma = self.sigma(x)
        sigma = F.softplus(sigma) # sigma是AU
        kl = kl + (kl_mu+kl_sigma)
        if feature:
            return mu, sigma, kl, x
        else:
            return mu, sigma, kl

    def kl_loss(self):
        kl = 0.0
        for m in self.children():  # 只遍历直接子模块
            if hasattr(m, "kl_loss"):
                kl = kl + m.kl_loss()
        return kl