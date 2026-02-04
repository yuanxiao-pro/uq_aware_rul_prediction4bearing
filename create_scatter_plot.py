"""
生成散点图：48个点，4种颜色，无横纵轴
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成48个点的坐标
n_points = 48
n_colors = 4
points_per_color = n_points // n_colors  # 每种颜色12个点

# 定义4种颜色的中心点位置（更靠近中心）
centers = [
    (-0.02, 0.02),   # 红色中心
    (0.02, 0.02),    # 青色中心
    (-0.02, -0.02),  # 蓝色中心
    (0.02, -0.02)    # 橙色中心
]

# 生成更密集的随机坐标（围绕中心点）
x = []
y = []
for i in range(n_colors):
    center_x, center_y = centers[i]
    # 使用更小的标准差使点更密集（0.12表示点非常集中）
    x.extend(np.random.normal(center_x, 0.01, points_per_color))
    y.extend(np.random.normal(center_y, 0.01, points_per_color))

x = np.array(x)
y = np.array(y)

# 定义4种颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # 红、青、蓝、橙
# 或者使用其他颜色组合：
# colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # 红、蓝、绿、橙
# colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F5']  # 红、绿、蓝、紫

# 为每个点分配颜色
color_list = []
for i in range(n_colors):
    color_list.extend([colors[i]] * points_per_color)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制散点图
scatter = ax.scatter(x, y, c=color_list, s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

# 移除所有轴线和刻度
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 设置相等的坐标轴比例，使图形更美观
ax.set_aspect('equal', adjustable='box')

# 移除边距
plt.tight_layout(pad=0)

# 保存图片
plt.savefig('scatter_plot_48_points.png', dpi=300, bbox_inches='tight', pad_inches=0)
print("散点图已保存为: scatter_plot_48_points.png")

# 显示图片
plt.show()

