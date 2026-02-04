import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt

def DE(data, m=4, epsilon=30):
    """
    高效计算数据集的多样性熵
    """
    data = np.asarray(data).flatten()
    N = len(data)
    M = N - (m - 1)
    if M <= 1:
        return 0.0

    # 构造滑动窗口矩阵
    X = np.lib.stride_tricks.sliding_window_view(data, m).T  # shape (m, M)

    # 计算相邻窗口的余弦相似度
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    numerator = np.sum(X1 * X2, axis=0)
    denominator = np.sqrt(np.sum(X1 * X1, axis=0)) * np.sqrt(np.sum(X2 * X2, axis=0))
    dist = numerator / denominator
    dist = np.round(dist, 4)

    # 统计直方图
    counts, _ = np.histogram(dist, bins=epsilon, range=(-1, 1))
    P = counts / np.sum(counts)
    ep = np.log(epsilon)
    valid = P > 0
    de = -np.sum(P[valid] * np.log(P[valid]) / ep)
    return de

