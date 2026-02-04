import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import MultivariateNormal, kl_divergence
import copy
def calculate_moments(model,params_mean, params_logvar, inputs):
    """
    根据输入的均值和方差，将输入分布进行局部线性化
    """
    # 把参数均值和参数对数方差都拆分为特征层和输出层
    params_feature_mean, params_final_layer_mean = split_params(params_mean)
    params_feature_logvar, params_final_layer_logvar = split_params(params_logvar)
    # 从特征参数的均值和对数方差中采样一组参数
    # print("params_feature_mean", params_feature_mean)
    # print("params_feature_logvar", params_feature_logvar)
    # print("params_final_layer_mean", params_final_layer_mean)
    # print("params_final_layer_logvar", params_final_layer_logvar)
    params_feature_sample = sample_parameters(params_feature_mean, params_feature_logvar)
    # 将从特征参数采样的参数与最终层参数的均值合并，以获得完整的模型参数
    params_partial_sample = merge_params(params_feature_sample, params_final_layer_mean)
    # 获得模型输出和特征样本
    # 保存当前参数
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    # 加载采样参数
    model.load_state_dict(params_partial_sample, strict=False)
    # 预测
    with torch.no_grad():
        # output = model(inputs)
        preds_f_sample, _, _, feature_sample = model(inputs, feature=True)
    # 恢复原参数
    model.load_state_dict(original_state)
    # print("feature_sample.shape", feature_sample.shape[1])
    n_samples = preds_f_sample.shape[1]
    feature_dim = feature_sample.shape[1]
    
    # final_layer_var_weights,final_layer_var_bias分别是最终层权重和偏置项的对数方差，通过取指数得到真实方差
    # 那么sigma.rho_weight要不要考虑进去呢
    final_layer_var_weights = torch.exp(params_final_layer_logvar["mu.rho_weight"])
    final_layer_var_bias = torch.exp(params_final_layer_logvar["mu.rho_bias"])

    # num_classes = 1
    # feature_times_var = (final_layer_var_weights.repeat(n_samples, 1).
    #                     reshape(n_samples, feature_dim, num_classes) * feature_sample[:, :,None]).permute(2, 0, 1)
    # preds_f_cov = torch.matmul(feature_times_var, feature_sample.T).permute(1, 2, 0)
    # preds_f_cov += preds_f_cov + final_layer_var_bias[None, None, :]
    
    # Step 1: 重复 final_layer_var_weights n_samples 次，形状变为 (n_samples, feature_dim)
    repeated_weights = final_layer_var_weights.repeat(n_samples, 1)  # 形状: (n_samples, feature_dim)
    
    # Step 2: 重塑为 (n_samples, feature_dim, 1) （因为 self.num_classes=1）
    reshaped_weights = repeated_weights.unsqueeze(-1)  # 形状: (n_samples, feature_dim, 1)
    
    # Step 3: 扩展 feature_sample 增加一个维度
    feature_sample_expanded = feature_sample.unsqueeze(-1)  # 形状: (n_samples, feature_dim, 1)
    
    # Step 4: 逐元素相乘
    feature_times_var = reshaped_weights * feature_sample_expanded  # 形状: (n_samples, feature_dim, 1)
    
    # Step 5: 转置维度为 (1, n_samples, feature_dim)
    # 使用 permute 来重新排列维度
    feature_times_var_transposed = feature_times_var.permute(2, 0, 1)  # 形状: (1, n_samples, feature_dim)
    
    # Step 6: 矩阵乘法 feature_times_var_transposed (1, n_samples, feature_dim) 与 feature_sample.T (feature_dim, n_samples)
    # 结果形状: (1, n_samples, n_samples)
    matmul_result = torch.matmul(feature_times_var_transposed, feature_sample.T)  # 形状: (1, n_samples, n_samples)
    
    # Step 7: 转置结果为 (n_samples, n_samples, 1)
    # 使用 permute 来重新排列维度
    preds_f_cov = matmul_result.permute(1, 2, 0)  # 形状: (n_samples, n_samples, 1)
    
    # Step 8: 添加 final_layer_var_bias
    # 确保 final_layer_var_bias 被扩展为 (1, 1, 1)
    if final_layer_var_bias.dim() == 0:
        # 如果 final_layer_var_bias 是标量，扩展为 (1, 1, 1)
        final_layer_var_bias_expanded = final_layer_var_bias.unsqueeze(-1).unsqueeze(-1)  # 形状: (1, 1, 1)
    else:
        # 如果 final_layer_var_bias 已经是 (self.num_classes,)，假设 self.num_classes=1
        final_layer_var_bias_expanded = final_layer_var_bias.unsqueeze(-1).unsqueeze(-1)  # 形状: (1, 1, 1)
    # 广播加法
    preds_f_cov = preds_f_cov + final_layer_var_bias_expanded  # 形状: (n_samples, n_samples, 1)

    # print("preds_f_cov", preds_f_cov)
    return preds_f_sample, preds_f_cov


def calculate_function_kl(
    params_variational_mean, 
    params_variational_logvar, 
    inputs, 
    model,         # PyTorch 模型
    # prior_feature_logvar=-50.0,
    # prior_var=1.0,
    # seed=42
):
    """
    PyTorch 实现的函数空间 KL 散度计算
    1.参数初始化​​：通过模型初始化或直接加载预定义参数，设置先验分布的均值
    2.先验方差设置​​：特征层使用极小对数方差（强先验约束），最终层使用较大对数方差（弱约束）
    3.分布计算​​：分别计算先验分布和变分分布在函数空间的均值和协方差矩阵
    4.KL散度计算​​：通过蒙特卡洛采样得到的分布样本，计算两者之间的KL散度
    """
    model_copy = copy.deepcopy(model)

    params_dict = model_copy.generate_init_params(inputs)
    params_prior_mean, params_prior_logvar = get_bayesian_model_mu_rho_from_dict(params_dict)
    # print("params_prior_mean", params_prior_mean)
    # print("params_prior_logvar", params_prior_logvar)


    '''这段注释别删'''
    # feature_prior_logvar = prior_feature_logvar
    # final_layer_prior_logvar = torch.log(torch.tensor(prior_var))

    # # 初始化先验logvar参数为零
    # params_prior_logvar_init = zeros_like_params(params_prior_mean)
    # # 按"dense"层类型拆分为特征层和最终层初始化参数
    # params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = split_params(params_prior_logvar_init)
    # # 分别设置特征层和最终层的先验对数方差（特征层用固定小值，最终层用固定大值）
    # params_feature_prior_logvar = zeros_like_params(params_feature_prior_logvar_init, feature_prior_logvar)
    # params_final_layer_prior_logvar = zeros_like_params(params_final_layer_prior_logvar_init, final_layer_prior_logvar)
    # # 合并分层后的先验logvar参数，得到完整的先验参数分布
    # params_prior_logvar = merge_params(params_feature_prior_logvar, params_final_layer_prior_logvar)
    # print("params_prior_logvar", params_prior_logvar)
    '''这段注释别删'''


    # 对先验分布进行局部线性化
    preds_f_prior_mean, preds_f_prior_cov = calculate_moments(model_copy, params_prior_mean, params_prior_logvar, inputs)
    # 对变分分布进行局部线性化
    preds_f_variational_mean, preds_f_variational_cov = calculate_moments(model_copy, params_variational_mean, params_variational_logvar, inputs)
    
    # 计算KL散度
    fkl = 0
    n_samples = preds_f_variational_mean.shape[0]
    cov_jitter = 1e-4
    num_classes = 1
    device = preds_f_prior_cov.device  # 获取当前张量的device
    for j in range(num_classes):
        # 保证 mean 是一维，cov 是二维
        _preds_f_prior_mean = preds_f_prior_mean[:, j].reshape(-1)
        _preds_f_prior_cov = preds_f_prior_cov[:, :, j] + torch.eye(n_samples, device=device) * cov_jitter

        _preds_f_variational_mean = preds_f_variational_mean[:, j].reshape(-1)
        _preds_f_variational_cov = preds_f_variational_cov[:, :, j] + torch.eye(n_samples, device=device) * cov_jitter
        # 对每个类别：提取均值向量并转置，添加抖动项到协方差矩阵（确保数值稳定性）

        q = MultivariateNormal(loc=_preds_f_variational_mean, covariance_matrix=_preds_f_variational_cov)
        p = MultivariateNormal(loc=_preds_f_prior_mean, covariance_matrix=_preds_f_prior_cov)
        kl = kl_divergence(q, p)
        fkl = fkl + kl

    return fkl

def get_bayesian_model_parameters(model):
    """
    获取 bayesian-torch 编写的模型的所有参数（包括均值、方差等），并冻结参数的梯度
    返回一个字典，键为参数名，值为参数的 tensor。
    """
    params = {}
    for name, param in model.named_parameters():
        param.requires_grad = False
        params[name] = param.data.clone()
    return params

def get_bayesian_model_mu_rho(model):
    """
    获取BayesianTCN模型所有贝叶斯层的参数均值(mu)和对数方差(rho)字典
    返回两个字典：mu_dict, rho_dict
    """
    mu_dict = {}
    rho_dict = {}
    for name, param in model.named_parameters():
        if 'mu_' in name:
            mu_dict[name] = param.data.clone()
        elif 'rho_' in name:
            rho_dict[name] = param.data.clone()
    return mu_dict, rho_dict

def get_bayesian_model_mu_rho_from_dict(params_dict):
    """
    从参数字典中提取所有贝叶斯层的参数均值(mu)和对数方差(rho)，返回两个字典：mu_dict, rho_dict
    """
    mu_dict = {}
    rho_dict = {}
    for name, param in params_dict.items():
        if 'mu_' in name:
            mu_dict[name] = param.clone()
        elif 'rho_' in name:
            rho_dict[name] = param.clone()
    return mu_dict, rho_dict

def split_params(params_dict):
    """手动拆分参数为特征层和最终层
    输出层必须是双头输出的，标记为mu和sigma，其他层为特征层
    """
    feature_params = {k: v for k, v in params_dict.items() if not (k.startswith('mu.') or k.startswith('sigma.'))}
    output_params = {k: v for k, v in params_dict.items() if k.startswith('mu.') or k.startswith('sigma.')}
    # print("output_params", output_params)
    return feature_params, output_params

def merge_params(params_1, params_2):
    """
    合并两个参数字典，params_2中的键会覆盖params_1中的同名键。
    """
    merged = params_1.copy()
    merged.update(params_2)
    return merged

def zeros_like_params(params_dict, delta=0.0):
    """
    根据一个参数字典，生成一个结构和形状完全一样、但数值全为0的新字典。
    """
    return {k: torch.zeros_like(v) + delta for k, v in params_dict.items()}

def sample_parameters(params_mu, params_logvar):
    """
    根据均值和对数方差参数字典，采样一组BNN参数
    """
    sampled_params = {}
    for k in params_mu:
        # 将 mu 的 key 替换成 rho 的 key
        # print("k", k)
        rho_key = k.replace('mu_', 'rho_')
        if rho_key not in params_logvar:
            raise KeyError(f"{rho_key} not found in params_rho")
        mu = params_mu[k]
        rho = params_logvar[rho_key]
        std = torch.exp(rho)  # 或 softplus(rho)，视你的实现
        eps = torch.randn_like(std)
        sampled_params[k] = mu + std * eps
    return sampled_params