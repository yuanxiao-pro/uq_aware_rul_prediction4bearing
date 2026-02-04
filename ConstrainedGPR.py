import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from scipy.optimize import minimize
from skopt import BayesSearchCV

class ConstrainedGP(GaussianProcessRegressor):
    """
    带约束的高斯过程回归器，强制满足：
    1. 输出值在[0,1]区间
    2. 整体负斜率趋势
    3. 正截距
    """
    def __init__(self, kernel=None, alpha=1e-5, **kwargs):
        if kernel is None:
            # 默认正定核函数
            kernel = (ConstantKernel(1.0, (0.1, 10.0)) * 
                     RBF(length_scale=2.0, length_scale_bounds=(0.1, 10.0)) + 
                     WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e-1)))
            
        super().__init__(kernel=kernel, alpha=alpha, **kwargs)
    
    def _constrained_loss(self, theta):
        self.kernel_.theta = theta
        K = self.kernel_(self.X_train_)
        K += np.eye(K.shape[0]) * self.alpha
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train_))
            log_likelihood = -0.5 * np.dot(self.y_train_.T, alpha)
            log_likelihood -= np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            return np.inf
        
        # 约束惩罚项
        penalty = 0
        
        # 约束惩罚项
        try:
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train_))
            
            # # 正截距约束 (初始值 f(0) > 0.5)
            # # 假设第一个训练点接近t=0
            # if len(self.X_train_) > 0:
            #     initial_pred = np.dot(K[0], alpha)  # 第一个点的预测值
            #     penalty += 100 * max(0, 0.5 - initial_pred)
            
            # 值域约束 (预测值应在合理范围内)
            y_pred = K @ alpha
            penalty += 10 * (np.sum(np.maximum(0, y_pred - 2)) + np.sum(np.maximum(0, -3 - y_pred)))
        except:
            penalty += 1000  # 如果计算失败，添加大惩罚
        
        return -(log_likelihood - penalty)

    def fit(self, X, y):
        super().fit(X, y)
        res = minimize(self._constrained_loss, 
                      self.kernel_.theta, 
                      method='L-BFGS-B',
                      bounds=self.kernel_.bounds)
        self.kernel_.theta = res.x
        return self

def logistic_transform(y, epsilon=1e-6):
    """将数据压缩到(0,1)区间"""
    y = np.clip(y, epsilon, 1-epsilon)
    return np.log(y / (1 - y))

def inverse_logistic(y_trans):
    """logistic逆变换"""
    return 1 / (1 + np.exp(-y_trans))

def build_temporal_features(t, period=24):
    """构造时序特征"""
    return np.column_stack([
        t,
        np.sin(2 * np.pi * t / period),
        np.cos(2 * np.pi * t / period)
    ])

class OnlineUpdater:
    """在线更新模块"""
    def __init__(self, model, window_size=50):
        self.model = model
        self.window = []
        self.window_size = window_size
    
    def update(self, new_t, new_y):
        self.window.append((new_t, new_y))
        if len(self.window) > self.window_size:
            self.window.pop(0)
        
        X_new = np.array([t for t,_ in self.window]).reshape(-1, 1)
        X_new = build_temporal_features(X_new)
        y_new = logistic_transform(np.array([y for _,y in self.window]))
        self.model.fit(X_new, y_new)

def plot_results(t_train, y_train, t_test, y_pred, y_std=None):
    """可视化结果"""
    plt.figure(figsize=(12, 6))
    
    # 训练数据
    plt.scatter(t_train, y_train, c='k', label='Observations')
    
    # 预测曲线
    plt.plot(t_test, y_pred, 'b-', label='Prediction')
    
    # 不确定性区间
    if y_std is not None:
        upper = inverse_logistic(inverse_logistic(y_pred) + 1.96*y_std)
        lower = inverse_logistic(inverse_logistic(y_pred) - 1.96*y_std)
        plt.fill_between(t_test[:,0], lower, upper, alpha=0.2, color='blue')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(-0.1, 1.1)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=1, color='k', linestyle='--')
    plt.legend()
    plt.title('Constrained GP Regression')
    plt.savefig('constrained_gpr.png')

def main():
    # 1. 生成模拟数据
    np.random.seed(42)
    t_train = np.linspace(0, 10, 50).reshape(-1, 1)
    y_true = 0.8 - 0.07 * t_train[:,0] + 0.05 * np.random.normal(size=len(t_train))

    y_train = np.clip(y_true, 0, 1)
    
    # 2. 特征工程
    X_train = build_temporal_features(t_train)
    
    # 3. 模型初始化 - 使用正定核函数，增加边界范围
    kernel = (ConstantKernel(1.0, (0.1, 100.0)) * 
              RBF(length_scale=2.0, length_scale_bounds=(0.1, 50.0)) + 
              WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e-1)))
    gp = ConstrainedGP(kernel=kernel, alpha=1e-2)  # 增加alpha提高数值稳定性
    
    # 4. 超参数优化 (可选)
    USE_HYPEROPT = True  # 设置为True启用超参数优化
    
    if USE_HYPEROPT:
        # 首先检查实际的参数结构
        print("Available kernel parameters:", gp.kernel.get_params().keys())
        
        # 根据实际核函数结构定义参数空间
        param_space = {
            'alpha': (1e-5, 1e-2, 'log-uniform')
        }
        
        # 添加核函数参数（需要根据实际结构调整）
        kernel_params = gp.kernel.get_params()
        print("Kernel parameter values:", kernel_params)
        
        # 检查核函数边界
        try:
            kernel_bounds = gp.kernel.bounds
            print("Kernel bounds:", kernel_bounds)
        except:
            print("Cannot access kernel bounds")
        
        # 只添加数值类型的参数，增加边界范围以避免收敛警告
        for param_name, param_value in kernel_params.items():
            if isinstance(param_value, (int, float)) and param_value > 0:
                if 'constant_value' in param_name:
                    param_space[f'kernel__{param_name}'] = (0.1, 100.0)  # 增加上界
                elif 'length_scale' in param_name:
                    param_space[f'kernel__{param_name}'] = (0.1, 50.0)   # 增加上界
                elif 'noise_level' in param_name:
                    param_space[f'kernel__{param_name}'] = (1e-6, 1e-1, 'log-uniform')
        
        print("Parameter space:", param_space)
        opt = BayesSearchCV(gp, param_space, n_iter=10, cv=3)
        opt.fit(X_train, logistic_transform(y_train))
        gp = opt.best_estimator_
        print("Best parameters:", opt.best_params_)
    else:
        print("Skipping hyperparameter optimization")
    
    # 5. 训练模型
    gp.fit(X_train, logistic_transform(y_train))
    
    # 6. 预测
    t_test = np.linspace(0, 12, 100).reshape(-1, 1)
    X_test = build_temporal_features(t_test)
    y_pred, y_std = gp.predict(X_test, return_std=True)
    y_pred = inverse_logistic(y_pred)
    
    # 7. 可视化
    plot_results(t_train, y_train, t_test, y_pred, y_std)
    
    # 8. 在线更新演示
    updater = OnlineUpdater(gp, window_size=30)
    new_data = np.linspace(10, 15, 10)
    for t in new_data:
        new_y = 0.7 - 0.05*t + 0.05*np.random.randn()
        updater.update(t, new_y)
    
    # 预测新范围
    t_update = np.linspace(0, 15, 150).reshape(-1, 1)
    X_update = build_temporal_features(t_update)
    y_update = inverse_logistic(gp.predict(X_update))
    plot_results(np.concatenate([t_train, new_data.reshape(-1,1)]),
                np.concatenate([y_train, [max(0, min(1, 0.7-0.05*x+0.05*np.random.randn())) for x in new_data]]),
                t_update, y_update)

if __name__ == "__main__":
    main()