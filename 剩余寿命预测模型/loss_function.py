import torch
import torch.distributions as dist
import math
import torch.nn as nn


def compute_smooth_l1_task_loss(
    targets: torch.Tensor,
    mean: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    r"""
    任务损失：Smooth L1 Loss（Huber 回归损失的一种）。

    数学公式（r 为预测值与标签的差值，r_i = μ_i - y_i ）：

        f(r) = {  0.5 * r^2           if |r| < β
               {  β * (|r| - 0.5*β)   if |r| ≥ β

    批量的损失为均值：

        L_smooth_L1 = (1/N) * Σ_i f(μ_i - y_i)

    其中 N 为样本数，μ_i 为预测值，y_i 为标签（真实 RUL）。β 控制二次/线性分界，默认 β=1。
    """
    criterion = nn.SmoothL1Loss(reduction="mean", beta=beta)
    return criterion(mean, targets)


def _rbf_kernel(F_i: torch.Tensor, F_j: torch.Tensor, gamma: float) -> torch.Tensor:
    r"""
    式 (19) 高斯 RBF 核：φ(F_i, F_j) = exp(-||F_i - F_j||^2 / (2γ^2))
    """
    # ||F_i - F_j||^2 = ||F_i||^2 + ||F_j||^2 - 2 <F_i, F_j>
    sq_dists = (
        (F_i * F_i).sum(dim=1, keepdim=True)
        + (F_j * F_j).sum(dim=1)
        - 2.0 * (F_i @ F_j.T)
    )
    return torch.exp(-sq_dists / (2.0 * gamma * gamma))


def compute_mmd_loss(
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    r"""
    MMD 损失，式 (18) 的平方展开形式 + 式 (19) 高斯核。

    L_MMD = || (1/n) Σ_i φ(F_i^s) - (1/m) Σ_j φ(F_j^t) ||
    平方后展开为核形式（φ 为式 (19) 高斯核）：
    L_MMD^2 = (1/n^2) Σ_i Σ_j φ(F_i^s, F_j^s) + (1/m^2) Σ_i Σ_j φ(F_i^t, F_j^t)
              - (2/(n·m)) Σ_i Σ_j φ(F_i^s, F_j^t)

    式 (19)：φ(F_i, F_j) = exp(-||F_i - F_j||^2 / (2γ^2))
    feat_s: 源域特征 [n, d]，feat_t: 目标域特征 [m, d]。返回标量。
    """
    n, m = feat_s.size(0), feat_t.size(0)
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=feat_s.device, dtype=feat_s.dtype)
    k_ss = _rbf_kernel(feat_s, feat_s, gamma)
    k_tt = _rbf_kernel(feat_t, feat_t, gamma)
    k_st = _rbf_kernel(feat_s, feat_t, gamma)
    # 对角为 1，求均值时去掉对角可避免 bias，此处按标准 MMD 含对角
    term_ss = k_ss.sum() / (n * n)
    term_tt = k_tt.sum() / (m * m)
    term_st = k_st.sum() / (n * m)
    mmd_sq = term_ss + term_tt - 2.0 * term_st
    return mmd_sq.clamp(min=0.0)


def compute_au_nll(targets: torch.Tensor, 
                  mean: torch.Tensor, 
                  sigma: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """计算Aleatoric Uncertainty NLL损失
    torch.square(targets - mean) * torch.exp(-sigma)
    sigma是对数方差，所以需要先exp
    """
    s = torch.square(targets - mean)
    loss1 = torch.exp(-sigma) * s
    loss2 = sigma
    loss = 0.5 * (loss1 + loss2)
    return loss.mean()
    # criterion = nn.GaussianNLLLoss(full=False, eps=eps, reduction='mean')
    # nll_loss = criterion(mean, targets, sigma)
    # return nll_loss


def compute_crps(targets: torch.Tensor, 
                mean: torch.Tensor, 
                sigma: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """
    计算CRPS (Continuous Ranked Probability Score) 损失
    
    CRPS衡量预测分布与真实值之间的差异，对于高斯分布有解析解。
    
    对于预测分布 N(μ, σ²) 和真实值 y：
    CRPS = σ * [z * (2 * Φ(z) - 1) + 2 * φ(z) - 1/√π]
    
    其中 z = (y - μ) / σ，Φ是标准正态分布的CDF，φ是标准正态分布的PDF。
    
    Args:
        targets: 真实值，shape (batch_size,) 或 (batch_size, 1)
        mean: 预测均值，shape (batch_size,) 或 (batch_size, 1)
        sigma: 预测的对数方差 (log_var)，shape (batch_size,) 或 (batch_size, 1)
        eps: 数值稳定性参数，防止除零
    
    Returns:
        CRPS损失值（标量）
    """
    # 确保sigma是方差（从对数方差转换）
    # sigma是对数方差，需要转换为标准差
    # std = torch.sqrt(torch.exp(sigma) + eps)  # exp(sigma) = variance, sqrt = std
    std = torch.sqrt(sigma + eps)  # exp(sigma) = variance, sqrt = std
    
    # 计算标准化残差 z = (y - μ) / σ
    z = (targets - mean) / (std + eps)
    
    # 创建标准正态分布
    normal = dist.Normal(torch.zeros_like(z), torch.ones_like(z))
    
    # 计算CDF和PDF
    cdf_z = normal.cdf(z)  # Φ(z)
    pdf_z = torch.exp(normal.log_prob(z))  # φ(z) = exp(log_prob(z))
    
    # 计算CRPS的解析解
    # CRPS = σ * [z * (2 * Φ(z) - 1) + 2 * φ(z) - 1/√π]
    sqrt_pi = math.sqrt(math.pi)
    crps = std * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / sqrt_pi)
    
    return crps.mean()

def compute_au_nll_with_pos(
    targets: torch.Tensor, 
    mean: torch.Tensor, 
    var: torch.Tensor,
    lambda_pos: float = 0.1,  # 恒正损失的权重系数
    eps: float = 1e-8
) -> torch.Tensor:
    """计算Aleatoric Uncertainty NLL损失，并加入恒正约束
    
    Args:
        targets: 真实值 (y_true)
        mean: 预测均值 (y_pred)
        var: 预测的方差var
        lambda_pos: 恒正损失的权重（默认0.1）
    
    Returns:
        总损失 = NLL损失 + 恒正惩罚项
    """
    # 1. 原始NLL损失（对数似然损失）
    criterion = nn.GaussianNLLLoss(full=False, eps=eps, reduction='mean')
    nll_loss = criterion(mean, targets, var)

    # 2. 恒正损失（当 mean <= 0 时施加惩罚）
    positive_loss = torch.mean(torch.relu(-mean))  # ReLU(-mean) 对负值进行惩罚

    # 3. 总损失 = NLL损失 + λ * 恒正损失
    total_loss = nll_loss + lambda_pos * positive_loss

    return total_loss


def compute_au_nll_with_crps(
    targets: torch.Tensor, 
    mean: torch.Tensor, 
    var: torch.Tensor,
    lambda_crps: float = 0.2,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算结合NLL和CRPS的损失函数
    
    这个损失函数结合了：
    1. NLL损失：衡量预测分布的对数似然
    2. CRPS损失：衡量预测分布与真实值的概率距离
    
    总损失 = (1 - λ_crps) * NLL + λ_crps * CRPS
    
    Args:
        targets: 真实值 (y_true)，shape (batch_size,) 或 (batch_size, 1)
        mean: 预测均值 (y_pred)，shape (batch_size,) 或 (batch_size, 1)
        sigma: 预测的对数方差 (log_var)，shape (batch_size,) 或 (batch_size, 1)
        lambda_crps: CRPS损失的权重系数（0-1之间），默认0.5
            - lambda_crps=0: 仅使用NLL损失
            - lambda_crps=1: 仅使用CRPS损失
            - lambda_crps=0.5: 平衡两种损失
        eps: 数值稳定性参数
    
    Returns:
        总损失 = (1 - λ_crps) * NLL + λ_crps * CRPS
    """
    # 1. 计算NLL损失
    s = torch.square(targets - mean)
    loss1 = torch.exp(-var) * s  # exp(-sigma) * (y_true - y_pred)^2
    loss2 = var                  # log_var项
    nll_loss = 0.5 * (loss1 + loss2).mean()
    
    # 2. 计算CRPS损失
    crps_loss = compute_crps(targets, mean, var, eps)
    
    # 3. 组合损失
    total_loss = (1 - lambda_crps) * nll_loss + lambda_crps * crps_loss
    
    return total_loss


def compute_au_nll_with_crps_and_pos(
    targets: torch.Tensor, 
    mean: torch.Tensor, 
    var: torch.Tensor,
    lambda_crps: float = 0.5,
    lambda_pos: float = 0.1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算结合NLL、CRPS和恒正约束的损失函数
    
    这个损失函数结合了：
    1. NLL损失：衡量预测分布的对数似然
    2. CRPS损失：衡量预测分布与真实值的概率距离
    3. 恒正损失：确保预测值为正（适用于剩余寿命等非负值预测）
    
    总损失 = (1 - λ_crps) * NLL + λ_crps * CRPS + λ_pos * Positive_Loss
    
    Args:
        targets: 真实值 (y_true)，shape (batch_size,) 或 (batch_size, 1)
        mean: 预测均值 (y_pred)，shape (batch_size,) 或 (batch_size, 1)
        var: 预测的方差 (var)，shape (batch_size,) 或 (batch_size, 1)
        lambda_crps: CRPS损失的权重系数（0-1之间），默认0.5
        lambda_pos: 恒正损失的权重系数，默认0.1
        eps: 数值稳定性参数
    
    Returns:
        总损失 = (1 - λ_crps) * NLL + λ_crps * CRPS + λ_pos * Positive_Loss
    """
    # 1. 计算NLL损失
    # s = torch.square(targets - mean)
    # loss1 = torch.exp(-sigma) * s  # exp(-sigma) * (y_true - y_pred)^2
    # loss2 = sigma                  # log_var项
    # nll_loss = 0.5 * (loss1 + loss2).mean()
    criterion = nn.GaussianNLLLoss(full=False, eps=eps, reduction='mean')
    nll_loss = criterion(mean, targets, var)
    
    # 2. 计算CRPS损失
    crps_loss = compute_crps(targets, mean, var, eps)
    
    # 3. 恒正损失（当 mean <= 0 时施加惩罚）
    positive_loss = torch.mean(torch.relu(-mean))  # ReLU(-mean) 对负值进行惩罚
    
    # 4. 组合损失
    total_loss = (1 - lambda_crps) * nll_loss + lambda_crps * crps_loss + lambda_pos * positive_loss
    
    return total_loss


def compute_au_nll_with_crps_wide_intervals(
    targets: torch.Tensor, 
    mean: torch.Tensor, 
    sigma: torch.Tensor,
    lambda_crps: float = 0.2,
    lambda_uncertainty: float = 0.1,
    target_log_var: float = 0.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算结合NLL、CRPS和不确定性正则化的损失函数（鼓励更大的区间宽度）
    
    这个损失函数结合了：
    1. NLL损失：衡量预测分布的对数似然
    2. CRPS损失：衡量预测分布与真实值的概率距离
    3. 不确定性正则化项：鼓励模型输出更大的不确定性（更宽的区间）
    
    总损失 = (1 - λ_crps) * NLL + λ_crps * CRPS - λ_uncertainty * Uncertainty_Regularization
    
    注意：不确定性正则化项是负的，因为我们要"鼓励"更大的不确定性。
    但为了与损失函数最小化一致，我们使用负号，这样优化过程会自然地增加不确定性。
    
    Args:
        targets: 真实值 (y_true)，shape (batch_size,) 或 (batch_size, 1)
        mean: 预测均值 (y_pred)，shape (batch_size,) 或 (batch_size, 1)
        sigma: 预测的对数方差 (log_var)，shape (batch_size,) 或 (batch_size, 1)
        lambda_crps: CRPS损失的权重系数（0-1之间），默认0.2
        lambda_uncertainty: 不确定性正则化项的权重，默认0.1
            - 正值：鼓励更大的不确定性（更宽的区间）
            - 值越大，区间宽度增加越多
        target_log_var: 目标对数方差值，默认0.0（对应标准差=1.0）
            - 如果sigma < target_log_var，正则化项会鼓励增加sigma
        eps: 数值稳定性参数
    
    Returns:
        总损失 = (1 - λ_crps) * NLL + λ_crps * CRPS - λ_uncertainty * Uncertainty_Reg
    """
    # 1. 计算NLL损失
    # s = torch.square(targets - mean)
    # loss1 = torch.exp(-sigma) * s  # exp(-sigma) * (y_true - y_pred)^2
    # loss2 = sigma                  # log_var项
    # nll_loss = 0.5 * (loss1 + loss2).mean()
    criterion = nn.GaussianNLLLoss(full=False, eps=eps, reduction='mean')
    nll_loss = criterion(mean, targets, sigma)
    
    # 2. 计算CRPS损失
    crps_loss = compute_crps(targets, mean, sigma, eps)
    
    # 3. 不确定性正则化项：鼓励更大的不确定性
    # 方法1: 鼓励sigma接近或大于target_log_var
    # 使用负号是因为我们要"奖励"更大的不确定性，但损失函数要最小化
    uncertainty_reg = -torch.mean(sigma - target_log_var)
    # 或者使用更温和的方式：只对小于target_log_var的sigma进行惩罚
    # uncertainty_reg = torch.mean(torch.relu(target_log_var - sigma))
    
    # 4. 组合损失
    # 注意：uncertainty_reg前面是负号，因为我们要鼓励更大的不确定性
    total_loss = (1 - lambda_crps) * nll_loss + lambda_crps * crps_loss - lambda_uncertainty * uncertainty_reg
    
    return total_loss


def compute_au_nll_with_crps_wide_intervals_v2(
    targets: torch.Tensor, 
    mean: torch.Tensor, 
    sigma: torch.Tensor,
    lambda_crps: float = 0.5,
    lambda_uncertainty: float = 0.05,
    min_std: float = 0.1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算结合NLL、CRPS和最小标准差约束的损失函数（版本2：更稳定的方法）
    
    这个版本使用最小标准差约束，确保预测区间不会太窄。
    
    总损失 = (1 - λ_crps) * NLL + λ_crps * CRPS + λ_uncertainty * Min_Std_Penalty
    
    Args:
        targets: 真实值 (y_true)
        mean: 预测均值 (y_pred)
        sigma: 预测的对数方差 (log_var)
        lambda_crps: CRPS损失的权重系数，默认0.2
        lambda_uncertainty: 最小标准差惩罚的权重，默认0.05
        min_std: 最小标准差阈值，默认0.1
            - 如果预测的标准差小于min_std，会施加惩罚
        eps: 数值稳定性参数
    
    Returns:
        总损失
    """
    # 1. 计算NLL损失
    # s = torch.square(targets - mean)
    # loss1 = torch.exp(-sigma) * s
    # loss2 = sigma

    # loss1 = s / sigma
    # loss2 = torch.pi * torch.log(sigma)
    # nll_loss = (0.5 * loss1 + loss2).mean()

    criterion = nn.GaussianNLLLoss(full=False, eps=eps, reduction='mean')
    nll_loss = criterion(mean, targets, sigma)
    # return loss
    
    # 2. 计算CRPS损失
    crps_loss = compute_crps(targets, mean, sigma, eps)
    
    # 3. 最小标准差惩罚：如果标准差太小，施加惩罚
    std = torch.sqrt(torch.exp(sigma) + eps)
    min_std_penalty = torch.mean(torch.relu(min_std - std))  # 如果std < min_std，施加惩罚
    
    # 4. 组合损失
    total_loss = (1 - lambda_crps) * nll_loss + lambda_crps * crps_loss + lambda_uncertainty * min_std_penalty
    
    return total_loss


# ==================== 使用示例 ====================
"""
使用示例：

1. 仅使用CRPS损失：
   loss = compute_crps(y_true, y_pred_mean, y_pred_log_var)

2. 结合NLL和CRPS（推荐）：
   loss = compute_au_nll_with_crps(y_true, y_pred_mean, y_pred_log_var, lambda_crps=0.5)
   # lambda_crps=0.5 表示平衡NLL和CRPS
   # lambda_crps=0.0 表示仅使用NLL（等同于compute_au_nll）
   # lambda_crps=1.0 表示仅使用CRPS

3. 结合NLL、CRPS和恒正约束（适用于剩余寿命预测）：
   loss = compute_au_nll_with_crps_and_pos(
       y_true, y_pred_mean, y_pred_log_var,
       lambda_crps=0.5,  # CRPS权重
       lambda_pos=0.1    # 恒正约束权重
   )

4. 如果需要增加区间宽度，使用带不确定性正则化的版本：
   loss = compute_au_nll_with_crps_wide_intervals(
       y_true, y_pred_mean, y_pred_log_var,
       lambda_crps=0.2,        # CRPS权重
       lambda_uncertainty=0.1, # 不确定性正则化权重（越大，区间越宽）
       target_log_var=0.0      # 目标对数方差（对应标准差=1.0）
   )
   
   或者使用版本2（更稳定）：
   loss = compute_au_nll_with_crps_wide_intervals_v2(
       y_true, y_pred_mean, y_pred_log_var,
       lambda_crps=0.2,        # CRPS权重
       lambda_uncertainty=0.05, # 最小标准差惩罚权重
       min_std=0.1             # 最小标准差阈值
   )

在训练循环中使用：
   from loss_function import compute_au_nll_with_crps_wide_intervals
   
   # 在训练循环中
   mu, sigma, kl = model(data)
   loss = compute_au_nll_with_crps_wide_intervals(
       labels, mu, sigma, 
       lambda_crps=0.2, 
       lambda_uncertainty=0.1
   )
   total_loss = loss + kl_weight * kl
   total_loss.backward()

CRPS的优势：
- CRPS直接优化预测分布的质量，而不仅仅是点预测
- CRPS对预测区间的校准更敏感，有助于改善不确定性量化
- CRPS在概率预测任务中通常比NLL更稳定
- 结合NLL和CRPS可以同时优化似然性和概率距离

为什么CRPS不会自动增加区间宽度？
====================================
CRPS衡量的是预测分布与真实值之间的概率距离。它会在以下两者之间找到平衡：
1. 过于自信（区间太窄）：如果真实值落在区间外，CRPS会很大
2. 过于保守（区间太宽）：虽然覆盖率高，但CRPS也会增加（因为分布更分散）

因此，CRPS倾向于找到"最优"的区间宽度，而不是"最大"的区间宽度。

如果需要增加区间宽度，可以：
1. 使用 compute_au_nll_with_crps_wide_intervals（添加不确定性正则化）
2. 增加 lambda_uncertainty 参数值
3. 调整 target_log_var 或 min_std 参数
4. 在模型配置中调整先验分布的方差参数
"""