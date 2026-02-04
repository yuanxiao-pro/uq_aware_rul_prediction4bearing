import numpy as np
from typing import Union, Tuple

def safe_divide_matrix(a, b, default=0):
    """
    安全除法，处理除零情况
    
    Args:
        a: 被除数
        b: 除数
        default: 除零时的默认值
        
    Returns:
        除法结果
    """
    # 创建结果数组
    result = np.full_like(a, default, dtype=float)
    # 使用 where 条件进行除法
    mask = b != 0
    result[mask] = a[mask] / b[mask]
    return result

def safe_multiply(a, b, default=0):
    """
    安全乘法，处理无效值
    
    Args:
        a: 第一个乘数
        b: 第二个乘数
        default: 无效值时的默认值
        
    Returns:
        乘法结果
    """
    # 创建结果数组
    result = np.full_like(a, default, dtype=float)
    # 使用 where 条件进行乘法
    mask = ~(np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b))
    result[mask] = a[mask] * b[mask]
    return result

def histcount(data: np.ndarray, bins: Union[int, np.ndarray] = 10, 
              range: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算直方图的计数和边界
    
    Args:
        data: 输入数据数组
        bins: 直方图的箱数或边界数组
        range: 直方图的范围，格式为 (min, max)
        
    Returns:
        counts: 每个箱的计数
        edges: 箱的边界
    """
    # 如果没有指定范围，使用数据的最大最小值
    if range is None:
        range = (np.min(data), np.max(data))
    
    # 计算直方图
    counts, edges = np.histogram(data, bins=bins, range=range)
    
    return counts, edges

def example_usage():
    """
    示例使用
    """
    # 创建示例数据
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)  # 1000个正态分布随机数
    
    # 使用默认参数
    counts, edges = histcount(data)
    print("默认箱数(10)的结果:")
    print("计数:", counts)
    print("边界:", edges)
    
    # 使用自定义箱数
    counts, edges = histcount(data, bins=20)
    print("\n自定义箱数(20)的结果:")
    print("计数:", counts)
    print("边界:", edges)
    
    # 使用自定义范围
    counts, edges = histcount(data, bins=10, range=(-2, 2))
    print("\n自定义范围(-2到2)的结果:")
    print("计数:", counts)
    print("边界:", edges)
    
    # 使用自定义边界
    custom_edges = np.array([-3, -2, -1, 0, 1, 2, 3])
    counts, edges = histcount(data, bins=custom_edges)
    print("\n使用自定义边界的结果:")
    print("计数:", counts)
    print("边界:", edges)
    
    # 测试安全除法
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 0, 3], [4, 5, 0]])
    result = safe_divide_matrix(a, b)
    print("\n安全除法结果:")
    print(result)
    
    # 测试安全乘法
    c = np.array([[1, np.nan, 3], [4, 5, np.inf]])
    d = np.array([[1, 2, 3], [4, 5, 6]])
    result = safe_multiply(c, d)
    print("\n安全乘法结果:")
    print(result)

if __name__ == "__main__":
    example_usage() 