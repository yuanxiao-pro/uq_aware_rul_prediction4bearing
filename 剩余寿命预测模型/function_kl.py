import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import MultivariateNormal, kl_divergence
import copy

# 添加可视化功能的导入
try:
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib未安装，可视化功能不可用")

def calculate_moments(model,params_mean, params_logvar, inputs, debug_nan=False):
    """
    根据输入的均值和方差，将输入分布进行局部线性化
    """
    if debug_nan:
        print("🔍 calculate_moments开始...")

    # 把参数均值和参数对数方差都拆分为特征层和输出层
    params_feature_mean, params_final_layer_mean = split_params(params_mean)
    params_feature_logvar, params_final_layer_logvar = split_params(params_logvar)
    
    if debug_nan:
        print(f"  特征层参数数量: {len(params_feature_mean)}")
        print(f"  输出层参数数量: {len(params_final_layer_mean)}")
        
        # 检查输出层参数
        for name, param in params_final_layer_logvar.items():
            if torch.isnan(param).any():
                print(f"❌ 输出层参数包含NaN: {name}")
            if torch.isinf(param).any():
                print(f"⚠️ 输出层参数包含Inf: {name}, 值范围: [{param.min():.2f}, {param.max():.2f}]")
    
    # 从特征参数的均值和对数方差中采样一组参数
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
    
    if debug_nan:
        print(f"  模型输出形状: {preds_f_sample.shape}")
        print(f"  特征样本形状: {feature_sample.shape}")
        if torch.isnan(preds_f_sample).any():
            print("❌ 模型输出包含NaN")
        if torch.isnan(feature_sample).any():
            print("❌ 特征样本包含NaN")
    
    n_samples = preds_f_sample.shape[1]
    feature_dim = feature_sample.shape[1]
    
    # final_layer_var_weights,final_layer_var_bias分别是最终层权重和偏置项的对数方差，通过取指数得到真实方差
    # 那么sigma.rho_weight要不要考虑进去呢
    final_layer_var_weights = torch.exp(params_final_layer_logvar["mu.rho_weight"])
    final_layer_var_bias = torch.exp(params_final_layer_logvar["mu.rho_bias"])
    
    if debug_nan:
        print(f"  最终层权重方差: min={final_layer_var_weights.min():.8f}, max={final_layer_var_weights.max():.8f}")
        print(f"  最终层偏置方差: min={final_layer_var_bias.min():.8f}, max={final_layer_var_bias.max():.8f}")
        
        if torch.isnan(final_layer_var_weights).any():
            print("❌ 最终层权重方差包含NaN")
        if torch.isnan(final_layer_var_bias).any():
            print("❌ 最终层偏置方差包含NaN")
        if torch.isinf(final_layer_var_weights).any():
            print("⚠️ 最终层权重方差包含Inf")
        if torch.isinf(final_layer_var_bias).any():
            print("⚠️ 最终层偏置方差包含Inf")

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
    
    if debug_nan:
        print(f"  协方差矩阵计算前: 对角线min={torch.diag(preds_f_cov[:,:,0]).min():.8f}, max={torch.diag(preds_f_cov[:,:,0]).max():.8f}")
        if torch.isnan(preds_f_cov).any():
            print("❌ 协方差矩阵计算过程中出现NaN")
    
    # 广播加法
    preds_f_cov = preds_f_cov + final_layer_var_bias_expanded  # 形状: (n_samples, n_samples, 1)

    if debug_nan:
        print(f"  最终协方差矩阵: 对角线min={torch.diag(preds_f_cov[:,:,0]).min():.8f}, max={torch.diag(preds_f_cov[:,:,0]).max():.8f}")
        if torch.isnan(preds_f_cov).any():
            print("❌ 最终协方差矩阵包含NaN")
        print("✅ calculate_moments完成")

    return preds_f_sample, preds_f_cov


def calculate_function_kl(
    inputs, 
    model,         # PyTorch 模型
    init_model,
    enable_diagnosis=False,    # 是否启用可视化诊断
    diagnosis_save_path="fkl.png",  # 诊断图保存路径
    diagnosis_threshold=1000,  # 触发诊断的KL阈值
    debug_nan=True,  # 启用NaN调试
):
    """
    PyTorch 实现的函数空间 KL 散度计算
    1.参数初始化​​：通过模型初始化或直接加载预定义参数，设置先验分布的均值
    2.先验方差设置​​：特征层使用极小对数方差（强先验约束），最终层使用较大对数方差（弱约束）
    3.分布计算​​：分别计算先验分布和变分分布在函数空间的均值和协方差矩阵
    4.KL散度计算​​：通过蒙特卡洛采样得到的分布样本，计算两者之间的KL散度
    
    Args:
        inputs: 输入数据
        model: 当前训练的模型
        init_model: 初始模型（先验）
        enable_diagnosis: 是否启用可视化诊断
        diagnosis_save_path: 诊断图保存路径
        diagnosis_threshold: 触发诊断的KL阈值
        debug_nan: 启用NaN调试
    """
    if debug_nan:
        print("🔍 Function KL计算开始，启用NaN调试...")
    
    try:
        model_copy = copy.deepcopy(model) # 用来局部线性化
        
        '''初始化先验分布'''
        params_prior_mean, params_prior_logvar = get_bayesian_model_mu_rho(init_model)
        
        # 检查先验参数是否包含NaN
        if debug_nan:
            for name, param in params_prior_mean.items():
                if torch.isnan(param).any():
                    print(f"❌ 检测到先验均值NaN: {name}")
                    return torch.tensor(float('nan'))
            for name, param in params_prior_logvar.items():
                if torch.isnan(param).any():
                    print(f"❌ 检测到先验对数方差NaN: {name}")
                    return torch.tensor(float('nan'))
        
        # 调整先验方差设置，避免过于极端的值
        feature_prior_logvar = -10  # 从-20调整到-10，exp(-10) ≈ 4.5e-5
        final_layer_prior_logvar = 1   # 从-10调整到1，exp(1) ≈ 2.718

        params_prior_logvar_init = {key: torch.zeros_like(value) for key,value in params_prior_logvar.items()}
        params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = split_params(params_prior_logvar_init)
        params_feature_prior_logvar = {key: torch.zeros_like(value) + feature_prior_logvar for key,value in params_feature_prior_logvar_init.items()} 
        params_final_layer_prior_logvar = {key: torch.zeros_like(value) + final_layer_prior_logvar for key,value in params_final_layer_prior_logvar_init.items()}
        params_prior_logvar = merge_params(params_feature_prior_logvar, params_final_layer_prior_logvar)
        
        '''线性化先验分布'''
        if debug_nan:
            print("📊 计算先验分布moments...")
        preds_f_prior_mean, preds_f_prior_cov = calculate_moments(model_copy, params_prior_mean, params_prior_logvar, inputs, debug_nan)
        
        # 检查先验分布计算结果
        if debug_nan:
            if torch.isnan(preds_f_prior_mean).any():
                print("❌ 先验均值包含NaN")
                print(f"先验均值形状: {preds_f_prior_mean.shape}")
                print(f"先验均值统计: min={preds_f_prior_mean.min():.6f}, max={preds_f_prior_mean.max():.6f}")
                return torch.tensor(float('nan'))
            if torch.isnan(preds_f_prior_cov).any():
                print("❌ 先验协方差包含NaN")
                print(f"先验协方差形状: {preds_f_prior_cov.shape}")
                print(f"先验协方差对角线统计: min={torch.diag(preds_f_prior_cov[:,:,0]).min():.6f}, max={torch.diag(preds_f_prior_cov[:,:,0]).max():.6f}")
                return torch.tensor(float('nan'))
            print(f"✅ 先验分布计算正常: 均值={preds_f_prior_mean.mean():.6f}, 协方差对角线={torch.diag(preds_f_prior_cov[:,:,0]).mean():.6f}")

        '''线性化变分分布'''
        params_variational_mean, params_variational_logvar = get_bayesian_model_mu_rho(model)
        
        # 检查变分参数是否包含NaN
        if debug_nan:
            for name, param in params_variational_mean.items():
                if torch.isnan(param).any():
                    print(f"❌ 检测到变分均值NaN: {name}")
                    return torch.tensor(float('nan'))
            for name, param in params_variational_logvar.items():
                if torch.isnan(param).any():
                    print(f"❌ 检测到变分对数方差NaN: {name}")
                    return torch.tensor(float('nan'))
                if torch.isinf(param).any():
                    print(f"⚠️ 检测到变分对数方差Inf: {name}, 值范围: [{param.min():.2f}, {param.max():.2f}]")
        
        if debug_nan:
            print("📊 计算变分分布moments...")
        preds_f_variational_mean, preds_f_variational_cov = calculate_moments(model_copy, params_variational_mean, params_variational_logvar, inputs, debug_nan)
        
        # 检查变分分布计算结果
        if debug_nan:
            if torch.isnan(preds_f_variational_mean).any():
                print("❌ 变分均值包含NaN")
                print(f"变分均值形状: {preds_f_variational_mean.shape}")
                print(f"变分均值统计: min={preds_f_variational_mean.min():.6f}, max={preds_f_variational_mean.max():.6f}")
                return torch.tensor(float('nan'))
            if torch.isnan(preds_f_variational_cov).any():
                print("❌ 变分协方差包含NaN")
                print(f"变分协方差形状: {preds_f_variational_cov.shape}")
                print(f"变分协方差对角线统计: min={torch.diag(preds_f_variational_cov[:,:,0]).min():.6f}, max={torch.diag(preds_f_variational_cov[:,:,0]).max():.6f}")
                return torch.tensor(float('nan'))
            print(f"✅ 变分分布计算正常: 均值={preds_f_variational_mean.mean():.6f}, 协方差对角线={torch.diag(preds_f_variational_cov[:,:,0]).mean():.6f}")
        
        # 计算KL散度
        fkl = 0
        n_samples = preds_f_variational_mean.shape[0]
        cov_jitter = 1e-4  # 增加抖动项
        num_classes = 1
        device = preds_f_prior_cov.device
        
        if debug_nan:
            print(f"📊 开始计算KL散度: n_samples={n_samples}, num_classes={num_classes}")
        
        for j in range(num_classes):
            # 保证 mean 是一维，cov 是二维
            _preds_f_prior_mean = preds_f_prior_mean[:, j].reshape(-1)
            _preds_f_prior_cov = preds_f_prior_cov[:, :, j]

            _preds_f_variational_mean = preds_f_variational_mean[:, j].reshape(-1)
            _preds_f_variational_cov = preds_f_variational_cov[:, :, j]
            
            # 🔧 强化协方差矩阵正定性修正
            _preds_f_prior_cov = ensure_positive_definite(_preds_f_prior_cov, min_eigenvalue=cov_jitter, debug=debug_nan, name="先验")
            _preds_f_variational_cov = ensure_positive_definite(_preds_f_variational_cov, min_eigenvalue=cov_jitter, debug=debug_nan, name="变分")
            
            # 详细的协方差矩阵检查
            if debug_nan:
                print(f"📊 类别 {j} 协方差矩阵检查:")
                
                # 检查协方差矩阵的数学性质
                prior_eigs = torch.linalg.eigvals(_preds_f_prior_cov).real
                var_eigs = torch.linalg.eigvals(_preds_f_variational_cov).real
                
                print(f"  先验协方差: 最小特征值={prior_eigs.min():.8f}, 最大特征值={prior_eigs.max():.8f}")
                print(f"  变分协方差: 最小特征值={var_eigs.min():.8f}, 最大特征值={var_eigs.max():.8f}")
                
                # 检查条件数
                prior_cond = torch.linalg.cond(_preds_f_prior_cov)
                var_cond = torch.linalg.cond(_preds_f_variational_cov)
                print(f"  协方差矩阵条件数: 先验={prior_cond:.2e}, 变分={var_cond:.2e}")
                
                if prior_cond > 1e12 or var_cond > 1e12:
                    print("⚠️ 协方差矩阵条件数过大，可能导致数值不稳定")

            try:
                # 尝试创建多元正态分布
                q = MultivariateNormal(loc=_preds_f_variational_mean, covariance_matrix=_preds_f_variational_cov)
                p = MultivariateNormal(loc=_preds_f_prior_mean, covariance_matrix=_preds_f_prior_cov)
                
                if debug_nan:
                    print(f"✅ 成功创建多元正态分布")
                
                # 计算KL散度
                kl = kl_divergence(q, p)
                
                if debug_nan:
                    print(f"📊 KL散度计算结果: {kl.item():.6f}")
                
                # 检查KL散度结果
                if torch.isnan(kl):
                    print(f"❌ KL散度计算得到NaN!")
                    print(f"  先验均值: {_preds_f_prior_mean[:5]}")
                    print(f"  变分均值: {_preds_f_variational_mean[:5]}")
                    print(f"  先验协方差对角线: {torch.diag(_preds_f_prior_cov)[:5]}")
                    print(f"  变分协方差对角线: {torch.diag(_preds_f_variational_cov)[:5]}")
                    return torch.tensor(float('nan'))
                
                if torch.isinf(kl):
                    print(f"❌ KL散度计算得到Inf: {kl.item()}")
                    return torch.tensor(float('inf'))
                
                fkl = fkl + kl
                
            except Exception as e:
                pass
                # print(f"❌ 多元正态分布或KL散度计算失败")
                # 作为最后的备选方案，使用参数空间KL近似
                # print("🔄 尝试使用参数空间KL近似...")
                # try:
                #     param_kl_approx = approximate_function_kl_with_parameter_kl(model, init_model)
                #     print(f"📊 参数空间KL近似: {param_kl_approx:.6f}")
                #     return param_kl_approx
                # except:
                #     return torch.tensor(float('nan'))

        # 最终检查
        if debug_nan:
            if torch.isnan(fkl):
                # print(f"❌ 最终Function KL为NaN!")
                return fkl
            elif torch.isinf(fkl):
                # print(f"⚠️ 最终Function KL为Inf: {fkl.item()}")
                return fkl
            else:
                print(f"✅ Function KL计算完成: {fkl.item():.6f}")

        # 可视化诊断功能
        if enable_diagnosis and (fkl.item() > diagnosis_threshold):
            print(f"⚠️  检测到异常大的Function KL: {fkl.item():.2f}, 启动诊断...")
            diagnosis_result = visualize_kl_diagnosis(
                model=model, 
                init_model=init_model, 
                inputs=inputs,
                function_kl_value=fkl.item(),
                save_path=diagnosis_save_path
            )
            if diagnosis_result:
                print("📊 诊断完成，请查看可视化结果")
        elif enable_diagnosis:
            print(f"✅ Function KL正常: {fkl.item():.2f}")

        return fkl
        
    except Exception as e:
        # print(f"❌ Function KL计算过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return torch.tensor(float('nan'))

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
    用来构造变分分布
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
    用来构造先验分布
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

def ensure_positive_definite(matrix, min_eigenvalue=1e-6, debug=False, name=""):
    """
    确保协方差矩阵正定
    
    Args:
        matrix: 输入协方差矩阵
        min_eigenvalue: 最小特征值阈值
        debug: 是否输出调试信息
        name: 矩阵名称（用于调试）
    
    Returns:
        修正后的正定矩阵
    """
    try:
        # 首先确保矩阵对称
        matrix = (matrix + matrix.t()) / 2
        
        # 计算特征值分解
        eigenvals, eigenvecs = torch.linalg.eigh(matrix)
        
        if debug and name:
            print(f"  🔧 {name}协方差矩阵修正: 原始最小特征值={eigenvals.min():.8f}")
        
        # 修正负特征值和过小的特征值
        eigenvals_corrected = torch.clamp(eigenvals, min=min_eigenvalue)
        
        # 重构矩阵
        matrix_corrected = eigenvecs @ torch.diag(eigenvals_corrected) @ eigenvecs.t()
        
        if debug and name:
            print(f"  ✅ {name}协方差矩阵修正完成: 新最小特征值={eigenvals_corrected.min():.8f}")
        
        return matrix_corrected
        
    except Exception as e:
        if debug:
            print(f"  ❌ {name}协方差矩阵修正失败: {e}")
        # 备选方案：直接添加对角抖动项
        device = matrix.device
        n = matrix.size(0)
        return matrix + torch.eye(n, device=device) * min_eigenvalue

def approximate_function_kl_with_parameter_kl(model, init_model):
    """
    当Function KL计算失败时，使用参数空间KL近似
    
    Args:
        model: 当前模型
        init_model: 初始模型
    
    Returns:
        参数空间KL散度近似值
    """
    var_mu, var_rho = get_bayesian_model_mu_rho(model)
    prior_mu, prior_rho = get_bayesian_model_mu_rho(init_model)
    
    total_kl = 0.0
    
    for name in var_mu.keys():
        if name in prior_mu:
            mu_var = var_mu[name].flatten()
            mu_prior = prior_mu[name].flatten()
            
            rho_name = name.replace('mu_', 'rho_')
            if rho_name in var_rho and rho_name in prior_rho:
                var_var = torch.exp(var_rho[rho_name]).flatten()
                var_prior = torch.exp(prior_rho[rho_name]).flatten()
                
                # 单变量高斯KL散度：KL(q||p) = 0.5 * (log(σ_p²/σ_q²) + σ_q²/σ_p² + (μ_q-μ_p)²/σ_p² - 1)
                kl_layer = 0.5 * (
                    torch.log(var_prior / (var_var + 1e-8)) + 
                    var_var / (var_prior + 1e-8) + 
                    (mu_var - mu_prior).pow(2) / (var_prior + 1e-8) - 1
                ).sum()
                
                total_kl += kl_layer
    
    # 缩放因子，使其与Function KL量级相近
    scale_factor = 0.1
    return total_kl * scale_factor

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
        # print("sample_parameters rho", rho)
        std = torch.exp(rho)  # 或 softplus(rho)，视你的实现
        eps = torch.randn_like(std)
        sampled_params[k] = mu + std * eps
    return sampled_params

def visualize_kl_diagnosis(model, init_model, inputs, function_kl_value, save_path=None):
    """
    在function_kl计算过程中进行可视化诊断
    
    Args:
        model: 当前模型
        init_model: 初始模型
        inputs: 输入数据
        function_kl_value: 计算得到的Function KL值
        save_path: 保存路径
    """
    if not VISUALIZATION_AVAILABLE:
        print("可视化功能不可用，跳过诊断")
        return None
    
    try:
        # 获取参数分布
        var_mu, var_rho = get_bayesian_model_mu_rho(model)
        prior_mu, prior_rho = get_bayesian_model_mu_rho(init_model)
        
        # 计算一些关键统计量
        var_mu_values = torch.cat([v.flatten() for v in var_mu.values()]).cpu().numpy()
        prior_mu_values = torch.cat([v.flatten() for v in prior_mu.values()]).cpu().numpy()
        var_variance = torch.cat([torch.exp(v).flatten() for v in var_rho.values()]).cpu().numpy()
        prior_variance = torch.cat([torch.exp(v).flatten() for v in prior_rho.values()]).cpu().numpy()
        
        # 创建简化的可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Function KL Diagnosis (KL={function_kl_value:.2f})', fontsize=14, fontweight='bold')
        
        # 1. 参数均值分布
        ax = axes[0, 0]
        ax.hist(prior_mu_values, bins=30, alpha=0.7, label='Prior Mean', color='blue', density=True)
        ax.hist(var_mu_values, bins=30, alpha=0.7, label='Variational Mean', color='red', density=True)
        ax.set_title('Parameter Mean Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 参数方差分布（对数尺度）
        ax = axes[0, 1]
        ax.hist(np.log10(prior_variance + 1e-8), bins=30, alpha=0.7, label='Prior Variance(log)', color='blue', density=True)
        ax.hist(np.log10(var_variance + 1e-8), bins=30, alpha=0.7, label='Variational Variance(log)', color='red', density=True)
        ax.set_title('Parameter Variance Distribution(log10)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 参数偏移分析
        ax = axes[1, 0]
        param_diff = var_mu_values - prior_mu_values[:len(var_mu_values)]
        ax.hist(param_diff, bins=30, alpha=0.7, color='green', density=True)
        ax.axvline(np.mean(param_diff), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean Shift={np.mean(param_diff):.4f}')
        ax.set_title('Parameter Shift Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 关键统计信息
        ax = axes[1, 1]
        ax.axis('off')
        
        # 计算关键统计
        var_ratio = np.mean(var_variance) / np.mean(prior_variance)
        param_shift = np.std(param_diff)
        
        stats_text = f"""
KL Divergence Diagnosis Report:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Function KL: {function_kl_value:.2f}
Parameter Mean Shift: {np.mean(param_diff):.6f}
Parameter Shift Std: {param_shift:.6f}
Variance Ratio: {var_ratio:.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Diagnosis Results:
"""
        
        if function_kl_value > 1000:
            stats_text += "⚠️ KL divergence too large!\n"
        if var_ratio > 5 or var_ratio < 0.2:
            stats_text += f"⚠️ Abnormal variance ratio: {var_ratio:.2f}\n"
        if param_shift > 0.1:
            stats_text += f"⚠️ Parameter shift too large: {param_shift:.4f}\n"
            
        if function_kl_value < 100 and 0.2 <= var_ratio <= 5 and param_shift <= 0.1:
            stats_text += "✅ Distribution status normal"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"KL diagnosis figure saved: {save_path}")
        
        plt.show()
        
        # 返回诊断信息
        return {
            'function_kl': function_kl_value,
            'param_shift_mean': np.mean(param_diff),
            'param_shift_std': param_shift,
            'variance_ratio': var_ratio,
            'diagnosis': 'normal' if function_kl_value < 100 else 'abnormal'
        }
        
    except Exception as e:
        print(f"Visualization diagnosis failed: {e}")
        return None

def diagnose_kl_issues(model, init_model, context_inputs, save_prefix="kl_diagnosis"):
    """
    便捷的KL散度问题诊断函数
    
    Args:
        model: 当前训练的模型
        init_model: 初始模型（先验）
        context_inputs: 上下文输入数据
        save_prefix: 保存文件的前缀
    
    Returns:
        dict: 包含诊断结果的字典
    """
    print("🔍 Starting KL divergence issue diagnosis...")
    
    # 1. 计算Function KL并自动诊断
    try:
        function_kl = calculate_function_kl(
            inputs=context_inputs,
            model=model,
            init_model=init_model,
            enable_diagnosis=True,
            diagnosis_save_path=f"{save_prefix}_function_kl.png",
            diagnosis_threshold=100  # 降低阈值，更容易触发诊断
        )
        
        print(f"📊 Function KL calculation completed: {function_kl.item():.2f}")
        
    except Exception as e:
        print(f"❌ Function KL calculation failed: {e}")
        function_kl = torch.tensor(float('inf'))
    
    # 2. 进行参数空间分析
    try:
        var_mu, var_rho = get_bayesian_model_mu_rho(model)
        prior_mu, prior_rho = get_bayesian_model_mu_rho(init_model)
        
        # 计算参数统计
        var_mu_values = torch.cat([v.flatten() for v in var_mu.values()]).cpu().numpy()
        prior_mu_values = torch.cat([v.flatten() for v in prior_mu.values()]).cpu().numpy()
        var_variance = torch.cat([torch.exp(v).flatten() for v in var_rho.values()]).cpu().numpy()
        prior_variance = torch.cat([torch.exp(v).flatten() for v in prior_rho.values()]).cpu().numpy()
        
        param_diff = var_mu_values - prior_mu_values[:len(var_mu_values)]
        
        # 参数空间KL近似
        param_kl = 0
        for name in var_mu.keys():
            if name in prior_mu:
                mu_var = var_mu[name].flatten()
                mu_prior = prior_mu[name].flatten()
                
                rho_name = name.replace('mu_', 'rho_')
                if rho_name in var_rho and rho_name in prior_rho:
                    var_var = torch.exp(var_rho[rho_name]).flatten()
                    var_prior = torch.exp(prior_rho[rho_name]).flatten()
                    
                    # 单变量高斯KL散度
                    kl_layer = 0.5 * (
                        torch.log(var_prior / var_var) + 
                        var_var / var_prior + 
                        (mu_var - mu_prior).pow(2) / var_prior - 1
                    ).sum().item()
                    param_kl += kl_layer
        
        diagnosis_result = {
            'function_kl': function_kl.item() if torch.isfinite(function_kl) else float('inf'),
            'param_kl_approx': param_kl,
            'param_shift_mean': np.mean(param_diff),
            'param_shift_std': np.std(param_diff),
            'variance_ratio': np.mean(var_variance) / np.mean(prior_variance),
            'prior_var_mean': np.mean(prior_variance),
            'current_var_mean': np.mean(var_variance),
            'total_params': len(var_mu_values),
        }
        
        # 3. 打印诊断报告
        print("\n" + "="*50)
        print("📋 KL Divergence Diagnosis Report")
        print("="*50)
        print(f"Function KL divergence:     {diagnosis_result['function_kl']:.2f}")
        print(f"Parameter space KL approx:  {diagnosis_result['param_kl_approx']:.2f}")
        print(f"Parameter mean shift:       {diagnosis_result['param_shift_mean']:.6f}")
        print(f"Parameter shift std:        {diagnosis_result['param_shift_std']:.6f}")
        print(f"Variance ratio:             {diagnosis_result['variance_ratio']:.3f}")
        print(f"Prior average variance:     {diagnosis_result['prior_var_mean']:.6f}")
        print(f"Current average variance:   {diagnosis_result['current_var_mean']:.6f}")
        print(f"Total parameter count:      {diagnosis_result['total_params']}")
        
        # 4. 问题诊断和建议
        print("\n" + "-"*50)
        print("🔧 Problem Diagnosis and Suggestions:")
        print("-"*50)
        
        if diagnosis_result['function_kl'] > 1000:
            print("❗ Function KL散度过大 (>1000)")
            print("   建议: 使用参数空间KL替代，或添加KL裁剪")
        
        if diagnosis_result['param_kl_approx'] > 500:
            print("❗ 参数空间KL散度过大 (>500)")
            print("   建议: 减小学习率，增加KL权重warm-up")
            
        if diagnosis_result['variance_ratio'] > 5:
            print("❗ 方差增长过快 (>5倍)")
            print("   建议: 检查rho初始化，降低学习率")
        elif diagnosis_result['variance_ratio'] < 0.2:
            print("❗ 方差衰减过快 (<0.2倍)")
            print("   建议: 增加学习率，检查KL权重是否过大")
            
        if abs(diagnosis_result['param_shift_mean']) > 0.1:
            print(f"❗ 参数均值偏移过大 ({diagnosis_result['param_shift_mean']:.4f})")
            print("   建议: 检查梯度裁剪，降低学习率")
            
        if diagnosis_result['param_shift_std'] > 0.5:
            print(f"❗ 参数偏移不一致 (std={diagnosis_result['param_shift_std']:.4f})")
            print("   建议: 检查不同层的学习率设置")
        
        # 5. 如果一切正常
        if (diagnosis_result['function_kl'] < 100 and 
            diagnosis_result['param_kl_approx'] < 500 and
            0.2 <= diagnosis_result['variance_ratio'] <= 5 and
            abs(diagnosis_result['param_shift_mean']) <= 0.1):
            print("✅ 所有指标正常，模型训练状态良好")
        
        print("="*50)
        
        return diagnosis_result
        
    except Exception as e:
        print(f"❌ 参数分析失败: {e}")
        return {'error': str(e), 'function_kl': function_kl.item() if torch.isfinite(function_kl) else float('inf')}

# 添加快速诊断入口
def quick_kl_check(model, init_model, context_inputs):
    """
    快速KL检查，只打印关键信息
    """
    try:
        fkl = calculate_function_kl(context_inputs, model, init_model)
        var_mu, var_rho = get_bayesian_model_mu_rho(model)
        prior_mu, prior_rho = get_bayesian_model_mu_rho(init_model)
        
        var_values = torch.cat([torch.exp(v).flatten() for v in var_rho.values()]).cpu().numpy()
        prior_values = torch.cat([torch.exp(v).flatten() for v in prior_rho.values()]).cpu().numpy()
        var_ratio = np.mean(var_values) / np.mean(prior_values)
        
        status = "🟢 正常" if fkl.item() < 100 else "🔴 异常" if fkl.item() > 1000 else "🟡 警告"
        print(f"KL快检: {status} | Function KL: {fkl.item():.1f} | 方差比例: {var_ratio:.2f}")
        
        return fkl.item()
    except Exception as e:
        print(f"KL快检失败: {e}")
        return float('inf')

def calculate_function_kl_robust(
    inputs, 
    model, 
    init_model,
    feature_prior_logvar=-10,      # 可配置的特征层先验对数方差
    final_layer_prior_logvar=-5,   # 可配置的输出层先验对数方差
    cov_jitter=1e-4,               # 可配置的协方差抖动项
    use_parameter_kl_fallback=True, # 是否使用参数空间KL作为后备
    enable_diagnosis=False,
    debug_nan=False
):
    """
    稳健的Function KL计算函数，支持参数配置
    
    Args:
        inputs: 输入数据
        model: 当前模型
        init_model: 初始模型
        feature_prior_logvar: 特征层先验对数方差
        final_layer_prior_logvar: 输出层先验对数方差
        cov_jitter: 协方差矩阵抖动项
        use_parameter_kl_fallback: 是否使用参数KL作为后备
        enable_diagnosis: 是否启用诊断
        debug_nan: 是否启用NaN调试
    
    Returns:
        Function KL散度值
    """
    if debug_nan:
        print("🔍 稳健Function KL计算开始...")
    
    try:
        model_copy = copy.deepcopy(model)
        
        '''初始化先验分布'''
        params_prior_mean, params_prior_logvar = get_bayesian_model_mu_rho(init_model)
        
        # 使用可配置的先验方差
        params_prior_logvar_init = {key: torch.zeros_like(value) for key,value in params_prior_logvar.items()}
        params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = split_params(params_prior_logvar_init)
        params_feature_prior_logvar = {key: torch.zeros_like(value) + feature_prior_logvar for key,value in params_feature_prior_logvar_init.items()} 
        params_final_layer_prior_logvar = {key: torch.zeros_like(value) + final_layer_prior_logvar for key,value in params_final_layer_prior_logvar_init.items()}
        params_prior_logvar = merge_params(params_feature_prior_logvar, params_final_layer_prior_logvar)
        
        '''计算分布'''
        preds_f_prior_mean, preds_f_prior_cov = calculate_moments(model_copy, params_prior_mean, params_prior_logvar, inputs, debug_nan)
        
        params_variational_mean, params_variational_logvar = get_bayesian_model_mu_rho(model)
        preds_f_variational_mean, preds_f_variational_cov = calculate_moments(model_copy, params_variational_mean, params_variational_logvar, inputs, debug_nan)
        
        # 计算KL散度
        fkl = 0
        n_samples = preds_f_variational_mean.shape[0]
        num_classes = 1
        device = preds_f_prior_cov.device
        
        for j in range(num_classes):
            _preds_f_prior_mean = preds_f_prior_mean[:, j].reshape(-1)
            _preds_f_prior_cov = preds_f_prior_cov[:, :, j]

            _preds_f_variational_mean = preds_f_variational_mean[:, j].reshape(-1)
            _preds_f_variational_cov = preds_f_variational_cov[:, :, j]
            
            # 确保协方差矩阵正定
            _preds_f_prior_cov = ensure_positive_definite(_preds_f_prior_cov, min_eigenvalue=cov_jitter, debug=debug_nan, name="先验")
            _preds_f_variational_cov = ensure_positive_definite(_preds_f_variational_cov, min_eigenvalue=cov_jitter, debug=debug_nan, name="变分")
            
            try:
                q = MultivariateNormal(loc=_preds_f_variational_mean, covariance_matrix=_preds_f_variational_cov)
                p = MultivariateNormal(loc=_preds_f_prior_mean, covariance_matrix=_preds_f_prior_cov)
                kl = kl_divergence(q, p)
                
                if torch.isnan(kl) or torch.isinf(kl):
                    raise ValueError(f"KL divergence计算异常: {kl}")
                
                fkl = fkl + kl
                
            except Exception as e:
                if debug_nan:
                    # print(f"❌ 多元正态分布KL计算失败: {e}")
                    pass
                if use_parameter_kl_fallback:
                    if debug_nan:
                        print("🔄 使用参数空间KL近似...")
                    return approximate_function_kl_with_parameter_kl(model, init_model)
                else:
                    return torch.tensor(float('nan'))
        
        if debug_nan:
            print(f"✅ 稳健Function KL计算完成: {fkl.item():.6f}")
        
        return fkl
        
    except Exception as e:
        if debug_nan:
            print(f"❌ 稳健Function KL计算失败: {e}")
        
        if use_parameter_kl_fallback:
            if debug_nan:
                print("🔄 最终使用参数空间KL近似...")
            return approximate_function_kl_with_parameter_kl(model, init_model)
        else:
            return torch.tensor(float('nan'))

# 便捷的配置预设
def get_function_kl_config(stability_level="medium"):
    """
    获取不同稳定性级别的Function KL配置
    
    Args:
        stability_level: "low", "medium", "high", "ultra"
    
    Returns:
        配置字典
    """
    configs = {
        "low": {
            "feature_prior_logvar": -15,
            "final_layer_prior_logvar": -8,
            "cov_jitter": 1e-6,
            "use_parameter_kl_fallback": False
        },
        "medium": {
            "feature_prior_logvar": -10,
            "final_layer_prior_logvar": -5,
            "cov_jitter": 1e-4,
            "use_parameter_kl_fallback": True
        },
        "high": {
            "feature_prior_logvar": -8,
            "final_layer_prior_logvar": -3,
            "cov_jitter": 1e-3,
            "use_parameter_kl_fallback": True
        },
        "ultra": {
            "feature_prior_logvar": -5,
            "final_layer_prior_logvar": -2,
            "cov_jitter": 1e-2,
            "use_parameter_kl_fallback": True
        }
    }
    return configs.get(stability_level, configs["medium"])