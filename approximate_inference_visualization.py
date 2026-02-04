"""
近似推断中分布逼近过程的可视化
Visualization of Distribution Approximation in Approximate Inference

展示变分推断中，变分分布（Variational Distribution）如何逐步逼近真实后验分布（True Posterior）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
import matplotlib.patches as mpatches

def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    计算两个高斯分布之间的KL散度: KL(N(μ1,σ1²) || N(μ2,σ2²))
    
    KL(p||q) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
    """
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

def approximate_inference_animation():
    """
    创建近似推断过程的动画
    展示变分分布逐步逼近真实后验分布
    """
    # 设置真实后验分布（目标分布）
    true_posterior_mu = 2.0
    true_posterior_sigma = 0.8
    
    # 初始变分分布（远离真实分布）
    initial_variational_mu = -1.5
    initial_variational_sigma = 1.5
    
    # 优化步数
    n_steps = 100
    
    # 生成优化轨迹（模拟梯度下降过程）
    # 使用指数衰减来模拟优化过程
    mu_trajectory = []
    sigma_trajectory = []
    kl_trajectory = []
    
    current_mu = initial_variational_mu
    current_sigma = initial_variational_sigma
    
    for step in range(n_steps):
        mu_trajectory.append(current_mu)
        sigma_trajectory.append(current_sigma)
        
        # 计算当前KL散度
        kl = kl_divergence_gaussian(
            current_mu, current_sigma,
            true_posterior_mu, true_posterior_sigma
        )
        kl_trajectory.append(kl)
        
        # 模拟梯度下降更新（向真实分布方向移动）
        # 使用指数衰减和学习率
        learning_rate_mu = 0.05
        learning_rate_sigma = 0.03
        
        # 梯度方向：向真实分布移动
        grad_mu = -(current_mu - true_posterior_mu) / (current_sigma**2 + 1e-6)
        grad_sigma = -(current_sigma - true_posterior_sigma) / (current_sigma + 1e-6)
        
        # 更新参数（添加一些随机性使过程更真实）
        current_mu = current_mu - learning_rate_mu * grad_mu
        current_sigma = current_sigma - learning_rate_sigma * grad_sigma
        
        # 确保sigma为正
        current_sigma = np.maximum(current_sigma, 0.1)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 6))
    
    # 子图1: 分布对比
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_xlim(-4, 5)
    ax1.set_ylim(0, 0.6)
    ax1.set_xlabel('x', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution Approximation Process', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 参数轨迹
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_xlabel('Optimization Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Trajectory', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: KL散度变化
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_xlabel('Optimization Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax3.set_title('KL Divergence Minimization', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 准备x轴数据
    x = np.linspace(-4, 5, 1000)
    
    # 绘制真实后验分布（固定）
    true_pdf = norm.pdf(x, true_posterior_mu, true_posterior_sigma)
    line_true, = ax1.plot(x, true_pdf, 'b-', linewidth=3, label='True Posterior', alpha=0.7)
    
    # 变分分布（会更新）
    variational_pdf = norm.pdf(x, current_mu, current_sigma)
    line_variational, = ax1.plot(x, variational_pdf, 'r--', linewidth=2.5, 
                                 label='Variational Distribution', alpha=0.7)
    
    # 填充区域显示差异
    fill = ax1.fill_between(x, np.minimum(true_pdf, variational_pdf), 
                            np.maximum(true_pdf, variational_pdf),
                            alpha=0.2, color='gray', label='Difference')
    
    # 参数轨迹线
    line_mu, = ax2.plot([], [], 'r-', linewidth=2, label='Variational μ')
    line_sigma, = ax2.plot([], [], 'g-', linewidth=2, label='Variational σ')
    line_true_mu = ax2.axhline(y=true_posterior_mu, color='b', linestyle='--', 
                              linewidth=2, label='True Posterior μ', alpha=0.7)
    line_true_sigma = ax2.axhline(y=true_posterior_sigma, color='b', linestyle=':', 
                                 linewidth=2, label='True Posterior σ', alpha=0.7)
    
    # KL散度轨迹
    line_kl, = ax3.plot([], [], 'purple', linewidth=2, label='KL Divergence')
    
    # 添加图例
    ax1.legend(loc='upper right', fontsize=10)
    ax2.legend(loc='best', fontsize=10)
    ax3.legend(loc='upper right', fontsize=10)
    
    # 添加文本显示当前KL值
    text_kl = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        # 更新当前参数
        current_mu = mu_trajectory[frame]
        current_sigma = sigma_trajectory[frame]
        current_kl = kl_trajectory[frame]
        
        # 更新变分分布
        variational_pdf = norm.pdf(x, current_mu, current_sigma)
        line_variational.set_ydata(variational_pdf)
        
        # 更新填充区域
        ax1.collections.clear()
        ax1.fill_between(x, np.minimum(true_pdf, variational_pdf),
                        np.maximum(true_pdf, variational_pdf),
                        alpha=0.2, color='gray')
        
        # 更新参数轨迹
        steps = np.arange(frame + 1)
        line_mu.set_data(steps, mu_trajectory[:frame+1])
        line_sigma.set_data(steps, sigma_trajectory[:frame+1])
        ax2.set_xlim(0, n_steps)
        ax2.set_ylim(-2, 3)
        
        # 更新KL散度轨迹
        line_kl.set_data(steps, kl_trajectory[:frame+1])
        ax3.set_xlim(0, n_steps)
        if len(kl_trajectory[:frame+1]) > 0:
            ax3.set_ylim(max(kl_trajectory) * 1.1, min(kl_trajectory) * 0.1)
        
        # 更新文本
        text_kl.set_text(f'Step: {frame}/{n_steps-1}\nKL Divergence: {current_kl:.4f}')
        
        return line_variational, line_mu, line_sigma, line_kl, text_kl
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=n_steps, interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    return fig, anim

def static_comparison_plot():
    """
    创建静态对比图，展示不同优化阶段的分布对比
    """
    # 真实后验分布
    true_mu = 2.0
    true_sigma = 0.8
    
    # 不同优化阶段的变分分布
    stages = [
        {'mu': -1.5, 'sigma': 1.5, 'label': 'Initial (Step 0)', 'color': 'red', 'alpha': 0.3},
        {'mu': 0.0, 'sigma': 1.2, 'label': 'Middle (Step 30)', 'color': 'orange', 'alpha': 0.5},
        {'mu': 1.5, 'sigma': 0.9, 'label': 'Near Convergence (Step 70)', 'color': 'green', 'alpha': 0.6},
        {'mu': 1.95, 'sigma': 0.82, 'label': 'Converged (Step 100)', 'color': 'purple', 'alpha': 0.7},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution Approximation at Different Stages', fontsize=16, fontweight='bold')
    
    x = np.linspace(-4, 5, 1000)
    true_pdf = norm.pdf(x, true_mu, true_sigma)
    
    for idx, (ax, stage) in enumerate(zip(axes.flatten(), stages)):
        variational_pdf = norm.pdf(x, stage['mu'], stage['sigma'])
        kl = kl_divergence_gaussian(stage['mu'], stage['sigma'], true_mu, true_sigma)
        
        # 绘制真实后验分布
        ax.plot(x, true_pdf, 'b-', linewidth=3, label='True Posterior', alpha=0.8)
        
        # 绘制变分分布
        ax.plot(x, variational_pdf, '--', linewidth=2.5, 
               color=stage['color'], label=stage['label'], alpha=stage['alpha'])
        
        # 填充差异区域
        ax.fill_between(x, np.minimum(true_pdf, variational_pdf),
                       np.maximum(true_pdf, variational_pdf),
                       alpha=0.15, color='gray')
        
        ax.set_xlim(-4, 5)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{stage["label"]}\nKL Divergence: {kl:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig

def kl_landscape_visualization():
    """
    可视化KL散度在参数空间中的景观
    """
    true_mu = 2.0
    true_sigma = 0.8
    
    # 创建参数网格
    mu_range = np.linspace(-1, 4, 50)
    sigma_range = np.linspace(0.3, 1.5, 50)
    Mu, Sigma = np.meshgrid(mu_range, sigma_range)
    
    # 计算KL散度
    KL = np.zeros_like(Mu)
    for i in range(len(sigma_range)):
        for j in range(len(mu_range)):
            KL[i, j] = kl_divergence_gaussian(
                Mu[i, j], Sigma[i, j],
                true_mu, true_sigma
            )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 3D表面图
    ax1 = axes[0]
    contour = ax1.contourf(Mu, Sigma, KL, levels=30, cmap='viridis')
    ax1.contour(Mu, Sigma, KL, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax1.plot(true_mu, true_sigma, 'r*', markersize=20, label='True Posterior', zorder=5)
    ax1.set_xlabel('μ (Mean)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('σ (Standard Deviation)', fontsize=12, fontweight='bold')
    ax1.set_title('KL Divergence Landscape', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    plt.colorbar(contour, ax=ax1, label='KL Divergence')
    ax1.grid(True, alpha=0.3)
    
    # 优化轨迹叠加
    ax2 = axes[1]
    contour2 = ax2.contourf(Mu, Sigma, KL, levels=30, cmap='viridis', alpha=0.6)
    ax2.contour(Mu, Sigma, KL, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # 生成优化轨迹
    initial_mu, initial_sigma = -1.5, 1.5
    current_mu, current_sigma = initial_mu, initial_sigma
    trajectory_mu = [current_mu]
    trajectory_sigma = [current_sigma]
    
    for _ in range(50):
        learning_rate_mu = 0.05
        learning_rate_sigma = 0.03
        grad_mu = -(current_mu - true_mu) / (current_sigma**2 + 1e-6)
        grad_sigma = -(current_sigma - true_sigma) / (current_sigma + 1e-6)
        current_mu = current_mu - learning_rate_mu * grad_mu
        current_sigma = current_sigma - learning_rate_sigma * grad_sigma
        current_sigma = np.maximum(current_sigma, 0.1)
        trajectory_mu.append(current_mu)
        trajectory_sigma.append(current_sigma)
    
    ax2.plot(trajectory_mu, trajectory_sigma, 'r-', linewidth=2.5, 
            label='Optimization Trajectory', marker='o', markersize=4, alpha=0.8)
    ax2.plot(initial_mu, initial_sigma, 'go', markersize=12, 
            label='Initial Point', zorder=5)
    ax2.plot(true_mu, true_sigma, 'r*', markersize=20, 
            label='True Posterior (Target)', zorder=5)
    ax2.set_xlabel('μ (Mean)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('σ (Standard Deviation)', fontsize=12, fontweight='bold')
    ax2.set_title('Optimization Trajectory in KL Landscape', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def prior_variational_comparison():
    """
    按照kl_landscape_visualization的风格，在一张图片里画出先验分布和变分分布
    用2个同心椭圆表示，一个表示先验分布，另一个表示变分分布
    """
    # 设置先验分布参数
    prior_mu = 0.0
    prior_sigma = 1.0
    
    # 设置变分分布参数（优化后的）
    variational_mu = 1.95
    variational_sigma = 0.82
    
    # 创建参数网格（用于显示KL散度景观作为背景）
    mu_range = np.linspace(-2, 4, 50)
    sigma_range = np.linspace(0.3, 1.5, 50)
    Mu, Sigma = np.meshgrid(mu_range, sigma_range)
    
    # 计算KL散度（以变分分布为目标分布，作为背景）
    KL = np.zeros_like(Mu)
    for i in range(len(sigma_range)):
        for j in range(len(mu_range)):
            KL[i, j] = kl_divergence_gaussian(
                Mu[i, j], Sigma[i, j],
                variational_mu, variational_sigma
            )
    
    # 创建单个图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 绘制KL散度景观作为背景
    contour = ax.contourf(Mu, Sigma, KL, levels=30, cmap='viridis', alpha=0.8)
    ax.contour(Mu, Sigma, KL, levels=20, colors='black', alpha=0.2, linewidths=0.5)
    
    # 绘制先验分布的椭圆（置信区域）
    # 椭圆参数：中心(μ, σ)，长轴和短轴与分布参数相关
    from matplotlib.patches import Ellipse
    
    # 先验分布椭圆：以(prior_mu, prior_sigma)为中心
    # 椭圆大小与分布的不确定性相关
    prior_ellipse_width = 2 * prior_sigma * 0.5  # 长轴（μ方向）
    prior_ellipse_height = 2 * prior_sigma * 0.3  # 短轴（σ方向）
    prior_ellipse = Ellipse(
        (prior_mu, prior_sigma),
        prior_ellipse_width,
        prior_ellipse_height,
        edgecolor='blue',
        facecolor='blue',
        alpha=0.3,
        linewidth=2.5,
        label='Prior Distribution'
    )
    ax.add_patch(prior_ellipse)
    
    # 变分分布椭圆：以(variational_mu, variational_sigma)为中心
    variational_ellipse_width = 2 * variational_sigma * 0.5  # 长轴（μ方向）
    variational_ellipse_height = 2 * variational_sigma * 0.3  # 短轴（σ方向）
    variational_ellipse = Ellipse(
        (variational_mu, variational_sigma),
        variational_ellipse_width,
        variational_ellipse_height,
        edgecolor='red',
        facecolor='red',
        alpha=0.3,
        linewidth=2.5,
        linestyle='--',
        label='Variational Distribution'
    )
    ax.add_patch(variational_ellipse)
    
    # 标记椭圆中心点
    ax.plot(prior_mu, prior_sigma, 'bo', markersize=10, 
            markeredgecolor='white', markeredgewidth=2, zorder=5)
    ax.plot(variational_mu, variational_sigma, 'r*', markersize=15, 
            markeredgecolor='white', markeredgewidth=2, zorder=5)
    
    ax.set_xlabel('μ (Mean)', fontsize=12, fontweight='bold')
    ax.set_ylabel('σ (Standard Deviation)', fontsize=12, fontweight='bold')
    ax.set_title('Prior vs Variational Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    plt.colorbar(contour, ax=ax, label='KL Divergence (w.r.t. Variational)')
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(mu_range.min(), mu_range.max())
    ax.set_ylim(sigma_range.min(), sigma_range.max())
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Generating approximate inference visualizations...")
    
    # 1. 创建静态对比图
    print("1. Creating static comparison plot...")
    fig1 = static_comparison_plot()
    fig1.savefig('approximate_inference_stages.png', dpi=300, bbox_inches='tight')
    print("   Saved: approximate_inference_stages.png")
    
    # 2. 创建KL散度景观图
    print("2. Creating KL divergence landscape...")
    fig2 = kl_landscape_visualization()
    fig2.savefig('kl_divergence_landscape.png', dpi=300, bbox_inches='tight')
    print("   Saved: kl_divergence_landscape.png")
    
    # 2.5. 创建先验分布和变分分布对比图
    print("2.5. Creating prior vs variational comparison...")
    fig2_5 = prior_variational_comparison()
    fig2_5.savefig('prior_variational_comparison.svg', dpi=1000, bbox_inches='tight', format='svg')
    print("   Saved: prior_variational_comparison.svg")
    
    # 3. 创建动画（可选，需要保存为gif）
    print("3. Creating animation (interactive)...")
    fig3, anim = approximate_inference_animation()
    
    # 保存动画为gif（需要pillow）
    try:
        anim.save('approximate_inference_animation.gif', writer='pillow', fps=20)
        print("   Saved: approximate_inference_animation.gif")
    except Exception as e:
        print(f"   Warning: Could not save animation as GIF: {e}")
        print("   Animation will be displayed interactively.")
    
    # 显示所有图形
    plt.show()
    
    print("\nVisualization complete!")

