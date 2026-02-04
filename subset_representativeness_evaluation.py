"""
振动信号子集代表性评估模块

用于评估一个子集的振动信号在相应工况中是否最具代表性。

评估特征说明：
==================

1. 统计特征相似度 (Statistical Feature Similarity)
   - 均值相似度：子集与全集的均值差异
   - 方差相似度：子集与全集的方差差异
   - 偏度相似度：子集与全集的偏度差异
   - 峰度相似度：子集与全集的峰度差异
   - 变异系数相似度：衡量数据离散程度的一致性

2. 分布相似度 (Distribution Similarity)
   - Kolmogorov-Smirnov检验：评估累积分布函数的差异
   - KL散度：衡量概率分布的差异
   - 直方图重叠度：评估分布形状的相似性
   - 分位数相似度：评估不同分位点的相似性

3. 特征空间覆盖度 (Feature Space Coverage)
   - PCA主成分覆盖度：在主要特征方向上的覆盖范围
   - 特征空间密度：子集在特征空间中的分布密度
   - 边界样本覆盖：是否包含极端值样本
   - 聚类中心距离：与工况聚类中心的距离

4. 多样性熵相似度 (Diversity Entropy Similarity)
   - DE值相似度：子集与全集的多样性熵差异
   - 特征DE值相关性：各特征DE值的相关性

5. 频域特征相似度 (Frequency Domain Similarity)
   - 能量比相似度：频域能量分布的相似性
   - 频谱平坦度相似度：频谱形状的相似性
   - FFT特征相似度：频域特征的相似性

6. 时间序列特征相似度 (Time Series Similarity)
   - 峰值因子相似度：峰值特征的相似性
   - 峰值振动相似度：振动幅值的相似性
   - 分形维数相似度：信号复杂度的相似性

7. 综合代表性评分 (Comprehensive Representativeness Score)
   - 加权综合评分：综合以上所有特征
   - 代表性等级：优秀/良好/一般/较差
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import load
from DE import DE
import warnings
warnings.filterwarnings('ignore')


class SubsetRepresentativenessEvaluator:
    """
    子集代表性评估器
    
    用于评估一个子集（如某个轴承的数据）在相应工况（如c1, c2, c3）中是否最具代表性。
    """
    
    def __init__(self, data_dir='datasetresult/xjtu', m=4, epsilon=30):
        """
        初始化评估器
        
        Args:
            data_dir: 数据目录
            m: DE参数m
            epsilon: DE参数epsilon
        """
        self.data_dir = data_dir
        self.m = m
        self.epsilon = epsilon
        self.feature_names = [
            'Kurtosis', 'Fractal Dimension', 'Peak factor',
            'Energy ratio', 'Spectral flatness', 'Mean', 'Variance', 
            'Skewness', 'Peak vibration', 'DE', 'FFT_mean'
        ]
        # 自动检测数据集类型
        self.dataset_type = self._detect_dataset_type()
    
    def _detect_dataset_type(self):
        """
        自动检测数据集类型（xjtu有c前缀，femto没有）
        
        Returns:
            'xjtu' 或 'femto' 或 'other'
        """
        # 检查目录是否存在，如果不存在尝试使用绝对路径
        if not os.path.exists(self.data_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            abs_data_dir = os.path.join(script_dir, self.data_dir)
            if os.path.exists(abs_data_dir):
                self.data_dir = abs_data_dir
                print(f"Auto-detected absolute path: {self.data_dir}")
            else:
                # 目录不存在，返回'other'，但会在使用时报错
                return 'other'
        
        try:
            all_files = os.listdir(self.data_dir)
        except FileNotFoundError:
            return 'other'
        features_files = [f for f in all_files if f.endswith('_features_df')]
        
        if len(features_files) == 0:
            return 'other'
        
        # 检查是否有c前缀的文件
        has_prefix = any(f.startswith('c') and f[1].isdigit() for f in features_files)
        
        if has_prefix:
            return 'xjtu'
        else:
            # 检查是否是femto数据集（通常包含Bearing但无c前缀）
            has_bearing = any('Bearing' in f for f in features_files)
            if has_bearing:
                return 'femto'
            else:
                return 'other'
    
    def load_condition_data(self, condition_prefix=None):
        """
        加载某个工况下的所有轴承数据
        
        Args:
            condition_prefix: 工况前缀
                - 对于xjtu数据集: 'c1', 'c2', 'c3' 等
                - 对于femto数据集: None 或 '' (加载所有文件，因为femto没有工况前缀)
        
        Returns:
            all_features: 所有轴承的特征数据列表
            bearing_names: 轴承名称列表
        """
        # 检查目录是否存在
        if not os.path.exists(self.data_dir):
            # 尝试使用绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            abs_data_dir = os.path.join(script_dir, self.data_dir)
            if os.path.exists(abs_data_dir):
                self.data_dir = abs_data_dir
                print(f"Using absolute path: {self.data_dir}")
            else:
                raise FileNotFoundError(
                    f"Data directory not found: {self.data_dir}\n"
                    f"Also tried: {abs_data_dir}\n"
                    f"Current working directory: {os.getcwd()}"
                )
        
        all_files = os.listdir(self.data_dir)
        
        # 根据数据集类型选择文件过滤方式
        if self.dataset_type == 'xjtu':
            # xjtu数据集：需要匹配前缀
            if condition_prefix is None or condition_prefix == '':
                print("Warning: xjtu dataset requires condition_prefix (e.g., 'c1', 'c2', 'c3')")
                return [], []
            features_files = [f for f in all_files 
                            if f.startswith(condition_prefix) and f.endswith('_features_df')]
        elif self.dataset_type == 'femto':
            # femto数据集：没有前缀，加载所有特征文件
            if condition_prefix is not None and condition_prefix != '':
                print(f"Warning: femto dataset doesn't use condition_prefix. Ignoring '{condition_prefix}' and loading all files.")
            features_files = [f for f in all_files if f.endswith('_features_df')]
        else:
            # 其他数据集：尝试匹配前缀（如果提供），否则加载所有
            if condition_prefix is None or condition_prefix == '':
                features_files = [f for f in all_files if f.endswith('_features_df')]
            else:
                features_files = [f for f in all_files 
                                if f.startswith(condition_prefix) and f.endswith('_features_df')]
        
        all_features = []
        bearing_names = []
        
        for f in features_files:
            try:
                features_path = os.path.join(self.data_dir, f)
                features_df = load(features_path)
                
                # 提取轴承名称
                bearing_name = f.replace('_features_df', '').replace('.csv', '')
                bearing_names.append(bearing_name)
                all_features.append(features_df)
                
                print(f"Loaded {bearing_name}: shape {features_df.shape}")
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
        
        return all_features, bearing_names
    
    def calculate_statistical_similarity(self, subset_features, full_features):
        """
        计算统计特征相似度
        
        Args:
            subset_features: 子集特征DataFrame
            full_features: 全集特征DataFrame
        
        Returns:
            similarity_scores: 各统计特征的相似度字典
        """
        scores = {}
        
        for feature in self.feature_names:
            if feature not in subset_features.columns or feature not in full_features.columns:
                continue
            
            subset_data = subset_features[feature].values
            full_data = full_features[feature].values
            
            # 去除NaN值
            subset_data = subset_data[~np.isnan(subset_data)]
            full_data = full_data[~np.isnan(full_data)]
            
            if len(subset_data) == 0 or len(full_data) == 0:
                continue
            
            # 1. 均值相似度（使用相对误差）
            mean_subset = np.mean(subset_data)
            mean_full = np.mean(full_data)
            if mean_full != 0:
                mean_sim = 1 - abs(mean_subset - mean_full) / (abs(mean_full) + 1e-8)
            else:
                mean_sim = 1 - abs(mean_subset - mean_full)
            scores[f'{feature}_mean_sim'] = max(0, mean_sim)
            
            # 2. 方差相似度
            var_subset = np.var(subset_data)
            var_full = np.var(full_data)
            if var_full != 0:
                var_sim = 1 - abs(var_subset - var_full) / (var_full + 1e-8)
            else:
                var_sim = 1 - abs(var_subset - var_full)
            scores[f'{feature}_var_sim'] = max(0, var_sim)
            
            # 3. 偏度相似度
            skew_subset = stats.skew(subset_data)
            skew_full = stats.skew(full_data)
            if abs(skew_full) > 1e-8:
                skew_sim = 1 - abs(skew_subset - skew_full) / (abs(skew_full) + 1e-8)
            else:
                skew_sim = 1 - abs(skew_subset - skew_full)
            scores[f'{feature}_skew_sim'] = max(0, skew_sim)
            
            # 4. 峰度相似度
            kurt_subset = stats.kurtosis(subset_data)
            kurt_full = stats.kurtosis(full_data)
            if abs(kurt_full) > 1e-8:
                kurt_sim = 1 - abs(kurt_subset - kurt_full) / (abs(kurt_full) + 1e-8)
            else:
                kurt_sim = 1 - abs(kurt_subset - kurt_full)
            scores[f'{feature}_kurt_sim'] = max(0, kurt_sim)
            
            # 5. 变异系数相似度
            cv_subset = np.std(subset_data) / (np.mean(subset_data) + 1e-8)
            cv_full = np.std(full_data) / (np.mean(full_data) + 1e-8)
            if cv_full != 0:
                cv_sim = 1 - abs(cv_subset - cv_full) / (abs(cv_full) + 1e-8)
            else:
                cv_sim = 1 - abs(cv_subset - cv_full)
            scores[f'{feature}_cv_sim'] = max(0, cv_sim)
        
        return scores
    
    def calculate_distribution_similarity(self, subset_features, full_features):
        """
        计算分布相似度
        
        Args:
            subset_features: 子集特征DataFrame
            full_features: 全集特征DataFrame
        
        Returns:
            similarity_scores: 各分布相似度字典
        """
        scores = {}
        
        for feature in self.feature_names:
            if feature not in subset_features.columns or feature not in full_features.columns:
                continue
            
            subset_data = subset_features[feature].values
            full_data = full_features[feature].values
            
            # 去除NaN值
            subset_data = subset_data[~np.isnan(subset_data)]
            full_data = full_data[~np.isnan(full_data)]
            
            if len(subset_data) == 0 or len(full_data) == 0:
                continue
            
            # 1. Kolmogorov-Smirnov检验
            try:
                ks_statistic, ks_pvalue = stats.ks_2samp(subset_data, full_data)
                # KS统计量越小越好，转换为相似度（0-1）
                ks_sim = 1 - min(ks_statistic, 1.0)
                scores[f'{feature}_ks_sim'] = ks_sim
                scores[f'{feature}_ks_pvalue'] = ks_pvalue
            except:
                scores[f'{feature}_ks_sim'] = 0.0
            
            # 2. KL散度（需要将数据转换为概率分布）
            try:
                # 使用直方图估计概率分布
                min_val = min(np.min(subset_data), np.min(full_data))
                max_val = max(np.max(subset_data), np.max(full_data))
                bins = np.linspace(min_val, max_val, 50)
                
                hist_subset, _ = np.histogram(subset_data, bins=bins, density=True)
                hist_full, _ = np.histogram(full_data, bins=bins, density=True)
                
                # 归一化
                hist_subset = hist_subset / (np.sum(hist_subset) + 1e-10)
                hist_full = hist_full / (np.sum(hist_full) + 1e-10)
                
                # 计算KL散度
                kl_div = stats.entropy(hist_subset + 1e-10, hist_full + 1e-10)
                # KL散度转换为相似度（0-1），使用指数衰减
                kl_sim = np.exp(-kl_div)
                scores[f'{feature}_kl_sim'] = kl_sim
            except:
                scores[f'{feature}_kl_sim'] = 0.0
            
            # 3. Jensen-Shannon散度
            try:
                js_div = jensenshannon(hist_subset, hist_full)
                js_sim = 1 - js_div  # JS散度在[0,1]范围内
                scores[f'{feature}_js_sim'] = js_sim
            except:
                scores[f'{feature}_js_sim'] = 0.0
            
            # 4. 分位数相似度
            try:
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                quantile_sims = []
                for q in quantiles:
                    q_subset = np.quantile(subset_data, q)
                    q_full = np.quantile(full_data, q)
                    if abs(q_full) > 1e-8:
                        q_sim = 1 - abs(q_subset - q_full) / (abs(q_full) + 1e-8)
                    else:
                        q_sim = 1 - abs(q_subset - q_full)
                    quantile_sims.append(max(0, q_sim))
                scores[f'{feature}_quantile_sim'] = np.mean(quantile_sims)
            except:
                scores[f'{feature}_quantile_sim'] = 0.0
        
        return scores
    
    def calculate_feature_space_coverage(self, subset_features, full_features):
        """
        计算特征空间覆盖度
        
        Args:
            subset_features: 子集特征DataFrame
            full_features: 全集特征DataFrame
        
        Returns:
            coverage_scores: 覆盖度评分字典
        """
        scores = {}
        
        # 准备数据
        valid_features = [f for f in self.feature_names if f in subset_features.columns and f in full_features.columns]
        if len(valid_features) == 0:
            return scores
        
        subset_data = subset_features[valid_features].values
        full_data = full_features[valid_features].values
        
        # 去除NaN
        subset_data = subset_data[~np.isnan(subset_data).any(axis=1)]
        full_data = full_data[~np.isnan(full_data).any(axis=1)]
        
        if len(subset_data) == 0 or len(full_data) == 0:
            return scores
        
        # 标准化
        scaler = StandardScaler()
        full_data_scaled = scaler.fit_transform(full_data)
        subset_data_scaled = scaler.transform(subset_data)
        
        # 1. PCA主成分覆盖度
        try:
            pca = PCA(n_components=min(3, len(valid_features)))
            pca.fit(full_data_scaled)
            
            # 计算在主成分空间中的覆盖范围
            full_pca = pca.transform(full_data_scaled)
            subset_pca = pca.transform(subset_data_scaled)
            
            # 计算每个主成分方向的覆盖度
            for i in range(pca.n_components_):
                full_range = np.max(full_pca[:, i]) - np.min(full_pca[:, i])
                subset_range = np.max(subset_pca[:, i]) - np.min(subset_pca[:, i])
                if full_range > 1e-8:
                    coverage = subset_range / full_range
                    scores[f'pca_component_{i+1}_coverage'] = min(1.0, coverage)
                else:
                    scores[f'pca_component_{i+1}_coverage'] = 1.0
            
            # 平均PCA覆盖度
            pca_coverages = [scores.get(f'pca_component_{i+1}_coverage', 0) 
                           for i in range(pca.n_components_)]
            scores['pca_avg_coverage'] = np.mean(pca_coverages)
        except Exception as e:
            print(f"PCA calculation error: {e}")
            scores['pca_avg_coverage'] = 0.0
        
        # 2. 特征空间密度（使用KMeans聚类）
        try:
            n_clusters = min(5, len(subset_data) // 10 + 1)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(full_data_scaled)
                
                # 计算子集样本到聚类中心的平均距离
                subset_distances = []
                for point in subset_data_scaled:
                    distances = [np.linalg.norm(point - center) for center in kmeans.cluster_centers_]
                    subset_distances.append(min(distances))
                
                # 计算全集样本到聚类中心的平均距离
                full_distances = []
                for point in full_data_scaled:
                    distances = [np.linalg.norm(point - center) for center in kmeans.cluster_centers_]
                    full_distances.append(min(distances))
                
                # 相似度：距离越接近越好
                avg_subset_dist = np.mean(subset_distances)
                avg_full_dist = np.mean(full_distances)
                if avg_full_dist > 1e-8:
                    density_sim = 1 - abs(avg_subset_dist - avg_full_dist) / (avg_full_dist + 1e-8)
                else:
                    density_sim = 1.0
                scores['cluster_density_sim'] = max(0, density_sim)
            else:
                scores['cluster_density_sim'] = 1.0
        except Exception as e:
            print(f"Clustering calculation error: {e}")
            scores['cluster_density_sim'] = 0.0
        
        # 3. 边界样本覆盖（检查是否包含极端值）
        try:
            # 计算每个特征的边界值覆盖度
            boundary_coverages = []
            for i, feature in enumerate(valid_features):
                full_feature = full_data[:, i]
                subset_feature = subset_data[:, i]
                
                # 检查是否包含最大值和最小值
                has_min = np.min(subset_feature) <= np.min(full_feature) + 1e-6
                has_max = np.max(subset_feature) >= np.max(full_feature) - 1e-6
                
                boundary_coverages.append((has_min + has_max) / 2)
            
            scores['boundary_coverage'] = np.mean(boundary_coverages)
        except:
            scores['boundary_coverage'] = 0.0
        
        return scores
    
    def calculate_de_similarity(self, subset_features, full_features):
        """
        计算多样性熵相似度
        
        Args:
            subset_features: 子集特征DataFrame
            full_features: 全集特征DataFrame
        
        Returns:
            de_scores: DE相似度评分字典
        """
        scores = {}
        
        # 计算子集和全集的DE值
        subset_de = {}
        full_de = {}
        
        for feature in self.feature_names:
            if feature not in subset_features.columns or feature not in full_features.columns:
                continue
            
            subset_data = subset_features[feature].values
            full_data = full_features[feature].values
            
            subset_data = subset_data[~np.isnan(subset_data)]
            full_data = full_data[~np.isnan(full_data)]
            
            if len(subset_data) > 10:
                try:
                    subset_de[feature] = DE(subset_data, m=self.m, epsilon=self.epsilon)
                except:
                    subset_de[feature] = 0.0
            
            if len(full_data) > 10:
                try:
                    full_de[feature] = DE(full_data, m=self.m, epsilon=self.epsilon)
                except:
                    full_de[feature] = 0.0
        
        # 计算DE值相似度
        de_similarities = []
        for feature in subset_de.keys():
            if feature in full_de:
                de_subset = subset_de[feature]
                de_full = full_de[feature]
                if abs(de_full) > 1e-8:
                    de_sim = 1 - abs(de_subset - de_full) / (abs(de_full) + 1e-8)
                else:
                    de_sim = 1 - abs(de_subset - de_full)
                de_similarities.append(max(0, de_sim))
                scores[f'{feature}_de_sim'] = max(0, de_sim)
        
        if len(de_similarities) > 0:
            scores['avg_de_sim'] = np.mean(de_similarities)
        else:
            scores['avg_de_sim'] = 0.0
        
        return scores
    
    def calculate_comprehensive_score(self, all_scores):
        """
        计算综合代表性评分
        
        Args:
            all_scores: 所有相似度评分的字典
        
        Returns:
            comprehensive_score: 综合评分（0-1）
            score_breakdown: 评分分解
        """
        # 定义各维度的权重
        # weights = {
        #     'statistical': 0.20,      # 统计特征相似度
        #     'distribution': 0.30,     # 分布相似度
        #     'coverage': 0.20,         # 特征空间覆盖度
        #     # 'de': 0.15,               # 多样性熵相似度
        #     'de': 0.30,               # 多样性熵相似度
        #     # 'other': 0.10             # 其他特征
        # }
        weights = {
            'statistical': 0.0,      # 统计特征相似度
            'distribution': 0.0,     # 分布相似度
            'coverage': 0.0,         # 特征空间覆盖度
            # 'de': 0.15,               # 多样性熵相似度
            'de': 1,               # 多样性熵相似度
            # 'other': 0.10             # 其他特征
        }
        
        # 提取各维度评分
        statistical_scores = [v for k, v in all_scores.items() 
                            if any(x in k for x in ['_mean_sim', '_var_sim', '_skew_sim', '_kurt_sim', '_cv_sim'])]
        distribution_scores = [v for k, v in all_scores.items() 
                             if any(x in k for x in ['_ks_sim', '_kl_sim', '_js_sim', '_quantile_sim'])]
        coverage_scores = [v for k, v in all_scores.items() 
                         if any(x in k for x in ['_coverage', 'density_sim'])]
        de_scores = [v for k, v in all_scores.items() if '_de_sim' in k]
        
        # 计算各维度平均分
        stat_avg = np.mean(statistical_scores) if statistical_scores else 0.0
        dist_avg = np.mean(distribution_scores) if distribution_scores else 0.0
        cov_avg = np.mean(coverage_scores) if coverage_scores else 0.0
        de_avg = np.mean(de_scores) if de_scores else 0.0
        
        # 综合评分
        comprehensive_score = (
            weights['statistical'] * stat_avg +
            weights['distribution'] * dist_avg +
            weights['coverage'] * cov_avg +
            weights['de'] * de_avg
        )
        
        score_breakdown = {
            'statistical_similarity': stat_avg,
            'distribution_similarity': dist_avg,
            'feature_space_coverage': cov_avg,
            'de_similarity': de_avg,
            'comprehensive_score': comprehensive_score
        }
        
        return comprehensive_score, score_breakdown
    
    def evaluate_subset(self, subset_bearing_name, condition_prefix=None):
        """
        评估子集代表性
        
        Args:
            subset_bearing_name: 子集轴承名称
                - 对于xjtu数据集: 如 'c1_Bearing1_1'
                - 对于femto数据集: 如 'Bearing1_1'
            condition_prefix: 工况前缀
                - 对于xjtu数据集: 'c1', 'c2', 'c3' 等
                - 对于femto数据集: None 或 '' (因为femto没有工况前缀)
        
        Returns:
            evaluation_result: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"Evaluating representativeness of {subset_bearing_name}")
        if condition_prefix:
            print(f"Condition: {condition_prefix}")
        else:
            print(f"Dataset Type: {self.dataset_type} (no condition prefix)")
        print(f"{'='*60}\n")
        
        # 1. 加载工况下的所有数据
        all_features_list, bearing_names = self.load_condition_data(condition_prefix)
        
        if len(all_features_list) == 0:
            print("No data found for this condition!")
            return None
        
        # 2. 找到子集数据
        subset_idx = None
        for i, name in enumerate(bearing_names):
            # 支持多种匹配方式
            if (subset_bearing_name in name or 
                name in subset_bearing_name or 
                subset_bearing_name == name or
                name.endswith(subset_bearing_name) or
                subset_bearing_name.endswith(name)):
                subset_idx = i
                break
        
        if subset_idx is None:
            condition_str = condition_prefix if condition_prefix else "all files"
            print(f"Subset {subset_bearing_name} not found in condition {condition_str}!")
            print(f"Available bearings: {bearing_names}")
            return None
        
        subset_features = all_features_list[subset_idx]
        
        # 3. 合并其他所有轴承数据作为全集
        full_features_list = [all_features_list[i] for i in range(len(all_features_list)) if i != subset_idx]
        if len(full_features_list) > 0:
            full_features = pd.concat(full_features_list, ignore_index=True)
        else:
            full_features = subset_features.copy()
        
        print(f"Subset shape: {subset_features.shape}")
        print(f"Full set shape: {full_features.shape}\n")
        
        # 4. 计算各项相似度
        print("Calculating statistical similarity...")
        stat_scores = self.calculate_statistical_similarity(subset_features, full_features)
        
        print("Calculating distribution similarity...")
        dist_scores = self.calculate_distribution_similarity(subset_features, full_features)
        
        print("Calculating feature space coverage...")
        coverage_scores = self.calculate_feature_space_coverage(subset_features, full_features)
        
        print("Calculating DE similarity...")
        de_scores = self.calculate_de_similarity(subset_features, full_features)
        
        # 5. 合并所有评分
        all_scores = {**stat_scores, **dist_scores, **coverage_scores, **de_scores}
        
        # 6. 计算综合评分
        comprehensive_score, score_breakdown = self.calculate_comprehensive_score(all_scores)
        
        # 7. 确定代表性等级
        if comprehensive_score >= 0.85:
            level = "优秀 (Excellent)"
        elif comprehensive_score >= 0.70:
            level = "良好 (Good)"
        elif comprehensive_score >= 0.55:
            level = "一般 (Fair)"
        else:
            level = "较差 (Poor)"
        
        # 8. 构建结果
        result = {
            'subset_name': subset_bearing_name,
            'condition': condition_prefix if condition_prefix else f'{self.dataset_type}_all',
            'dataset_type': self.dataset_type,
            'subset_size': len(subset_features),
            'full_set_size': len(full_features),
            'comprehensive_score': comprehensive_score,
            'representativeness_level': level,
            'score_breakdown': score_breakdown,
            'detailed_scores': all_scores
        }
        
        return result
    
    def compare_all_subsets(self, condition_prefix=None):
        """
        比较工况下所有子集的代表性
        
        Args:
            condition_prefix: 工况前缀
                - 对于xjtu数据集: 'c1', 'c2', 'c3' 等
                - 对于femto数据集: None 或 '' (比较所有文件)
        
        Returns:
            comparison_results: 比较结果DataFrame
        """
        print(f"\n{'='*60}")
        if condition_prefix:
            print(f"Comparing all subsets in condition {condition_prefix}")
        else:
            print(f"Comparing all subsets in dataset (type: {self.dataset_type})")
        print(f"{'='*60}\n")
        
        # 加载所有数据
        all_features_list, bearing_names = self.load_condition_data(condition_prefix)
        
        if len(all_features_list) == 0:
            print("No data found!")
            return None
        
        # 排除DE列，不参与计算
        for i in range(len(all_features_list)):
            if 'DE' in all_features_list[i].columns:
                all_features_list[i] = all_features_list[i].drop(columns=['DE'])
                print(f"Excluded 'DE' column from {bearing_names[i]}")
        
        results = []
        
        # 评估每个子集
        for i, bearing_name in enumerate(bearing_names):
            subset_features = all_features_list[i]
            
            # 合并其他数据作为全集
            full_features_list = [all_features_list[j] for j in range(len(all_features_list)) if j != i]
            if len(full_features_list) > 0:
                full_features = pd.concat(full_features_list, ignore_index=True)
            else:
                full_features = subset_features.copy()
            
            print("full_features.shape: ", full_features.shape)
            # 计算各项评分
            stat_scores = self.calculate_statistical_similarity(subset_features, full_features)
            dist_scores = self.calculate_distribution_similarity(subset_features, full_features)
            coverage_scores = self.calculate_feature_space_coverage(subset_features, full_features)
            de_scores = self.calculate_de_similarity(subset_features, full_features)
            
            all_scores = {**stat_scores, **dist_scores, **coverage_scores, **de_scores}
            comprehensive_score, score_breakdown = self.calculate_comprehensive_score(all_scores)
            print({
                'bearing_name': bearing_name,
                'subset_size': len(subset_features),
                'comprehensive_score': comprehensive_score,
                **score_breakdown
            })
            results.append({
                'bearing_name': bearing_name,
                'subset_size': len(subset_features),
                'comprehensive_score': comprehensive_score,
                **score_breakdown
            })
        
        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('comprehensive_score', ascending=False)
        
        return results_df
    
    def find_most_representative_subset(self, condition_prefix=None, top_k=1, 
                                       filter_func=None, save_comparison=True):
        """
        在同工况的子集中筛选最具代表性的子集
        
        Args:
            condition_prefix: 工况前缀
                - 对于xjtu数据集: 'c1', 'c2', 'c3' 等
                - 对于femto数据集: None 或 '' (在所有文件中查找)
            top_k: 返回前k个最具代表性的子集，默认1
            filter_func: 可选的过滤函数，用于进一步筛选子集
                - 例如: lambda name: 'train' in name  # 只考虑train类型
                - 例如: lambda name: 'Bearing1' in name  # 只考虑Bearing1系列
            save_comparison: 是否保存比较结果到CSV文件
        
        Returns:
            most_representative: 最具代表性的子集结果列表（按代表性评分排序）
            comparison_df: 所有子集的比较结果DataFrame
        """
        print(f"\n{'='*60}")
        if condition_prefix:
            print(f"Finding most representative subset(s) in condition {condition_prefix}")
        else:
            print(f"Finding most representative subset(s) in dataset (type: {self.dataset_type})")
        if filter_func:
            print("Note: Filter function is applied")
        print(f"{'='*60}\n")
        
        # 1. 比较所有子集
        comparison_df = self.compare_all_subsets(condition_prefix)
        
        if comparison_df is None or len(comparison_df) == 0:
            print("No data found for comparison!")
            return [], None
        
        # 2. 应用过滤函数（如果提供）
        if filter_func:
            original_count = len(comparison_df)
            comparison_df = comparison_df[comparison_df['bearing_name'].apply(filter_func)]
            filtered_count = len(comparison_df)
            print(f"Filtered from {original_count} to {filtered_count} subsets")
            
            if len(comparison_df) == 0:
                print("No subsets match the filter criteria!")
                return [], None
        
        # 3. 按综合评分排序，取前top_k个
        comparison_df_sorted = comparison_df.sort_values('comprehensive_score', ascending=False)
        top_subsets = comparison_df_sorted.head(top_k)
        
        # 4. 获取最具代表性的子集的详细信息
        most_representative = []
        for idx, row in top_subsets.iterrows():
            bearing_name = row['bearing_name']
            print(f"\n{'='*60}")
            print(f"Most Representative Subset #{len(most_representative) + 1}: {bearing_name}")
            print(f"{'='*60}")
            print(f"Comprehensive Score: {row['comprehensive_score']:.4f}")
            print(f"Subset Size: {row['subset_size']}")
            print(f"\nScore Breakdown:")
            print(f"  Statistical Similarity: {row['statistical_similarity']:.4f}")
            print(f"  Distribution Similarity: {row['distribution_similarity']:.4f}")
            print(f"  Feature Space Coverage: {row['feature_space_coverage']:.4f}")
            print(f"  DE Similarity: {row['de_similarity']:.4f}")
            
            # 评估该子集以获取完整结果
            result = self.evaluate_subset(bearing_name, condition_prefix)
            if result:
                most_representative.append(result)
        
        # 5. 保存比较结果
        if save_comparison:
            condition_str = condition_prefix if condition_prefix else f'{self.dataset_type}_all'
            filename = f'{condition_str}_subset_comparison.csv'
            comparison_df_sorted.to_csv(filename, index=False)
            print(f"\nComparison results saved to {filename}")
        
        return most_representative, comparison_df_sorted
    
    def find_most_representative_by_bearing_series(self, condition_prefix=None, 
                                                   bearing_series=None, save_comparison=True):
        """
        在相同工况下，按Bearing系列分组，找到每组内最具代表性的子集
        
        Args:
            condition_prefix: 工况前缀
                - 对于xjtu数据集: 'c1', 'c2', 'c3' 等
                - 对于femto数据集: None 或 '' (在所有文件中查找)
            bearing_series: Bearing系列，如 'Bearing1', 'Bearing2', 'Bearing3'
                - 如果为None，会自动检测所有Bearing系列并分别处理
            save_comparison: 是否保存比较结果到CSV文件
        
        Returns:
            results_dict: 字典，键为Bearing系列，值为该系列最具代表性的子集结果
            all_comparisons: 所有Bearing系列的比较结果字典
        """
        print(f"\n{'='*60}")
        if condition_prefix:
            print(f"Finding most representative subset by Bearing series in condition {condition_prefix}")
        else:
            print(f"Finding most representative subset by Bearing series in dataset (type: {self.dataset_type})")
        print(f"{'='*60}\n")
        
        # 1. 加载所有数据
        all_features_list, bearing_names = self.load_condition_data(condition_prefix)
        
        if len(all_features_list) == 0:
            print("No data found!")
            return {}, {}
        
        # 2. 检测所有Bearing系列（如果未指定）
        if bearing_series is None:
            import re
            bearing_series_set = set()
            for name in bearing_names:
                # 提取Bearing系列（如Bearing1, Bearing2等）
                match = re.search(r'(Bearing\d+)', name)
                if match:
                    bearing_series_set.add(match.group(1))
            bearing_series_list = sorted(list(bearing_series_set))
            print(f"Auto-detected Bearing series: {bearing_series_list}")
        else:
            bearing_series_list = [bearing_series] if isinstance(bearing_series, str) else bearing_series
        
        results_dict = {}
        all_comparisons = {}
        
        # 3. 对每个Bearing系列分别处理
        for series in bearing_series_list:
            print(f"\n{'='*60}")
            print(f"Processing {series} series")
            print(f"{'='*60}\n")
            
            # 筛选该系列的所有子集
            series_filter = lambda name: series in name
            most_representative, comparison_df = self.find_most_representative_subset(
                condition_prefix=condition_prefix,
                top_k=1,
                filter_func=series_filter,
                save_comparison=True  # 稍后统一保存
            )
            
            if most_representative and len(most_representative) > 0:
                results_dict[series] = most_representative[0]
                all_comparisons[series] = comparison_df
                
                print(f"\n{'='*60}")
                print(f"Most Representative Subset for {series}:")
                print(f"{'='*60}")
                result = most_representative[0]
                print(f"  Name: {result['subset_name']}")
                print(f"  Comprehensive Score: {result['comprehensive_score']:.4f}")
                print(f"  Level: {result['representativeness_level']}")
                print(f"  Subset Size: {result['subset_size']}")
            else:
                print(f"No representative subset found for {series}")
        
        # 4. 保存所有比较结果
        if save_comparison:
            condition_str = condition_prefix if condition_prefix else f'{self.dataset_type}_all'
            for series, comp_df in all_comparisons.items():
                if comp_df is not None and len(comp_df) > 0:
                    filename = f'{condition_str}_{series}_comparison.csv'
                    comp_df.to_csv(filename, index=False)
                    print(f"\nComparison results for {series} saved to {filename}")
        
        return results_dict, all_comparisons
    
    def visualize_bearing_series_comparison(self, results_dict, all_comparisons=None, save_path=None):
        """
        可视化各Bearing系列的代表性子集比较结果
        
        Args:
            results_dict: find_most_representative_by_bearing_series返回的结果字典
            all_comparisons: 所有Bearing系列的比较结果字典（可选）
            save_path: 保存路径
        """
        if not results_dict:
            print("No results to visualize!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Most Representative Subset by Bearing Series', fontsize=16, fontweight='bold')
        
        series_list = sorted(results_dict.keys())
        scores = [results_dict[s]['comprehensive_score'] for s in series_list]
        names = [results_dict[s]['subset_name'] for s in series_list]
        
        # 1. 各系列最具代表性子集的综合评分
        ax1 = axes[0, 0]
        colors = ['green' if s >= 0.85 else 'orange' if s >= 0.70 else 'yellow' if s >= 0.55 else 'red' 
                 for s in scores]
        bars = ax1.barh(range(len(series_list)), scores, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(series_list)))
        ax1.set_yticklabels(series_list, fontsize=12)
        ax1.set_xlabel('Comprehensive Score', fontweight='bold')
        ax1.set_title('Most Representative Subset Score by Series', fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(True, axis='x')
        
        # 添加数值标签和子集名称
        for i, (bar, score, name) in enumerate(zip(bars, scores, names)):
            ax1.text(score + 0.01, i, f'{score:.3f}\n({name})', va='center', fontsize=9, fontweight='bold')
        
        # 2. 各维度评分对比
        ax2 = axes[0, 1]
        dims = ['statistical_similarity', 'distribution_similarity', 
                'feature_space_coverage', 'de_similarity']
        dim_labels = ['Statistical', 'Distribution', 'Coverage', 'DE']
        
        x = np.arange(len(series_list))
        width = 0.2
        
        for i, (dim, label) in enumerate(zip(dims, dim_labels)):
            dim_scores = [results_dict[s]['score_breakdown'][dim] for s in series_list]
            offset = (i - 1.5) * width
            ax2.barh(x + offset, dim_scores, width, label=label, alpha=0.7)
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(series_list, fontsize=12)
        ax2.set_xlabel('Score', fontweight='bold')
        ax2.set_title('Dimension-wise Scores by Series', fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, axis='x')
        
        # 3. 子集规模对比
        ax3 = axes[1, 0]
        subset_sizes = [results_dict[s]['subset_size'] for s in series_list]
        full_sizes = [results_dict[s]['full_set_size'] for s in series_list]
        coverage = [s / f * 100 for s, f in zip(subset_sizes, full_sizes)]
        
        x_pos = np.arange(len(series_list))
        ax3_twin = ax3.twinx()
        
        bars1 = ax3.bar(x_pos - 0.2, subset_sizes, 0.4, label='Subset Size', alpha=0.7, color='skyblue')
        bars2 = ax3.bar(x_pos + 0.2, full_sizes, 0.4, label='Full Set Size', alpha=0.7, color='lightcoral')
        line = ax3_twin.plot(x_pos, coverage, 'o-', color='green', linewidth=2, markersize=8, label='Coverage %')
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(series_list, fontsize=12)
        ax3.set_ylabel('Size', fontweight='bold')
        ax3_twin.set_ylabel('Coverage (%)', fontweight='bold', color='green')
        ax3.set_title('Subset Size and Coverage by Series', fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 汇总信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_lines = ["Summary by Bearing Series:", "=" * 40]
        for series in series_list:
            result = results_dict[series]
            info_lines.append(f"\n{series}:")
            info_lines.append(f"  Subset: {result['subset_name']}")
            info_lines.append(f"  Score: {result['comprehensive_score']:.4f}")
            info_lines.append(f"  Level: {result['representativeness_level']}")
            info_lines.append(f"  Size: {result['subset_size']}")
        
        info_text = "\n".join(info_lines)
        ax4.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                family='monospace', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_comparison_results(self, comparison_df, top_k=5, save_path=None):
        """
        可视化比较结果
        
        Args:
            comparison_df: 比较结果DataFrame
            top_k: 显示前k个最具代表性的子集
            save_path: 保存路径
        """
        if comparison_df is None or len(comparison_df) == 0:
            print("No data to visualize!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Subset Representativeness Comparison', fontsize=16, fontweight='bold')
        
        # 取前top_k个
        top_df = comparison_df.head(top_k)
        
        # 1. 综合评分柱状图
        ax1 = axes[0, 0]
        colors = ['green' if s >= 0.85 else 'orange' if s >= 0.70 else 'yellow' if s >= 0.55 else 'red' 
                 for s in top_df['comprehensive_score']]
        bars = ax1.barh(range(len(top_df)), top_df['comprehensive_score'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_df)))
        ax1.set_yticklabels(top_df['bearing_name'], fontsize=10)
        ax1.set_xlabel('Comprehensive Score', fontweight='bold')
        ax1.set_title(f'Top {top_k} Most Representative Subsets', fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(True, axis='x')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, top_df['comprehensive_score'])):
            ax1.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # 2. 各维度评分对比（前top_k个）
        ax2 = axes[0, 1]
        x = np.arange(len(top_df))
        width = 0.2
        
        dims = ['statistical_similarity', 'distribution_similarity', 
                'feature_space_coverage', 'de_similarity']
        dim_labels = ['Statistical', 'Distribution', 'Coverage', 'DE']
        
        for i, (dim, label) in enumerate(zip(dims, dim_labels)):
            offset = (i - 1.5) * width
            ax2.barh(x + offset, top_df[dim], width, label=label, alpha=0.7)
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(top_df['bearing_name'], fontsize=9)
        ax2.set_xlabel('Score', fontweight='bold')
        ax2.set_title('Dimension-wise Scores Comparison', fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, axis='x')
        
        # 3. 综合评分分布直方图
        ax3 = axes[1, 0]
        ax3.hist(comparison_df['comprehensive_score'], bins=20, color='skyblue', 
                edgecolor='black', alpha=0.7)
        ax3.axvline(comparison_df['comprehensive_score'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {comparison_df["comprehensive_score"].mean():.3f}')
        ax3.axvline(comparison_df['comprehensive_score'].median(), color='green', 
                   linestyle='--', linewidth=2, label=f'Median: {comparison_df["comprehensive_score"].median():.3f}')
        ax3.set_xlabel('Comprehensive Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Score Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_text = f"""
        Comparison Statistics:
        ----------------------
        Total Subsets: {len(comparison_df)}
        Top {top_k} Subsets Shown
        
        Score Statistics:
        Mean: {comparison_df['comprehensive_score'].mean():.4f}
        Median: {comparison_df['comprehensive_score'].median():.4f}
        Std: {comparison_df['comprehensive_score'].std():.4f}
        Min: {comparison_df['comprehensive_score'].min():.4f}
        Max: {comparison_df['comprehensive_score'].max():.4f}
        
        Most Representative:
        {top_df.iloc[0]['bearing_name']}
        Score: {top_df.iloc[0]['comprehensive_score']:.4f}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_evaluation_result(self, result, save_path=None):
        """
        可视化评估结果
        
        Args:
            result: 评估结果字典
            save_path: 保存路径
        """
        if result is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Representativeness Evaluation: {result["subset_name"]}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 综合评分雷达图
        ax1 = axes[0, 0]
        # 获取所有键值对，排除comprehensive_score
        breakdown = result['score_breakdown']
        breakdown_items = [(k, v) for k, v in breakdown.items() if k != 'comprehensive_score']
        
        # 检查是否有足够的数据点
        if len(breakdown_items) < 3:
            # 如果数据点太少，跳过雷达图，显示提示信息
            ax1.text(0.5, 0.5, f'Insufficient data for radar chart\n(Need at least 3 dimensions, got {len(breakdown_items)})', 
                    ha='center', va='center', fontsize=12, transform=ax1.transAxes)
            ax1.set_title('Score Breakdown (Insufficient Data)', fontweight='bold')
        else:
            # 分离键和值
            original_keys = [k for k, v in breakdown_items]
            values = [v for k, v in breakdown_items]
            
            # 将键转换为显示格式（下划线替换为空格，首字母大写）
            categories = [k.replace('_', ' ').title() for k in original_keys]
            
            # 添加第一个值到末尾以闭合图形
            values_plot = values + [values[0]]
            categories_plot = categories + [categories[0]]
            
            angles = np.linspace(0, 2 * np.pi, len(categories_plot), endpoint=True)
            ax1.plot(angles, values_plot, 'o-', linewidth=2, label='Scores')
            ax1.fill(angles, values_plot, alpha=0.25)
            
            # 确保刻度和标签数量一致
            num_ticks = len(categories)  # 原始类别数量
            ax1.set_xticks(angles[:num_ticks])
            ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('Score Breakdown', fontweight='bold')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 综合评分柱状图
        ax2 = axes[0, 1]
        score = result['comprehensive_score']
        level = result['representativeness_level']
        color = 'green' if score >= 0.85 else 'orange' if score >= 0.70 else 'yellow' if score >= 0.55 else 'red'
        ax2.barh([0], [score], color=color, alpha=0.7)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Comprehensive Score', fontweight='bold')
        ax2.set_title(f'Overall Score: {score:.4f}\n{level}', fontweight='bold')
        ax2.grid(True, axis='x')
        
        # 3. 各维度评分对比
        ax3 = axes[1, 0]
        breakdown = result['score_breakdown']
        dims = [k.replace('_', ' ').title() for k in breakdown.keys() if k != 'comprehensive_score']
        dim_scores = [breakdown[k] for k in breakdown.keys() if k != 'comprehensive_score']
        colors_bar = ['green' if s >= 0.85 else 'orange' if s >= 0.70 else 'yellow' if s >= 0.55 else 'red' 
                     for s in dim_scores]
        ax3.barh(dims, dim_scores, color=colors_bar, alpha=0.7)
        ax3.set_xlim(0, 1)
        ax3.set_xlabel('Score', fontweight='bold')
        ax3.set_title('Dimension-wise Scores', fontweight='bold')
        ax3.grid(True, axis='x')
        
        # 4. 数据规模信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        condition_str = result.get('condition', 'N/A')
        dataset_type_str = result.get('dataset_type', 'N/A')
        info_text = f"""
        Subset Information:
        -------------------
        Name: {result['subset_name']}
        Dataset Type: {dataset_type_str}
        Condition: {condition_str}
        Subset Size: {result['subset_size']}
        Full Set Size: {result['full_set_size']}
        Coverage: {result['subset_size']/result['full_set_size']*100:.2f}%
        
        Comprehensive Score: {result['comprehensive_score']:.4f}
        Level: {result['representativeness_level']}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                family='monospace', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """主函数 - 查找各工况下最具代表性的子集"""
    
    print("\n" + "="*80)
    print("Finding Most Representative Subsets by Condition and Bearing Series")
    print("="*80)
    
    # ========== XJTU数据集 ==========
    # print("\n" + "="*80)
    # print("XJTU Dataset - Finding Most Representative Subsets by Condition")
    # print("="*80)
    
    # evaluator_xjtu = SubsetRepresentativenessEvaluator(
    #     data_dir='datasetresult/xjtu/',
    #     m=4,
    #     epsilon=30
    # )
    
    # # XJTU数据集有3个工况：c1, c2, c3
    # xjtu_conditions = ['c1', 'c2', 'c3']
    
    # for condition in xjtu_conditions:
    #     print(f"\n{'='*80}")
    #     print(f"Processing XJTU Condition: {condition}")
    #     print(f"{'='*80}")
        
    #     results_dict, all_comparisons = evaluator_xjtu.find_most_representative_by_bearing_series(
    #         condition_prefix=condition,
    #         bearing_series=None,  # 自动检测所有Bearing系列
    #         save_comparison=True
    #     )
        
    #     if results_dict:
    #         print(f"\n{'='*80}")
    #         print(f"Summary for {condition}:")
    #         print(f"{'='*80}")
    #         for series, result in sorted(results_dict.items()):
    #             print(f"{series:15s} -> {result['subset_name']:30s} "
    #                   f"(Score: {result['comprehensive_score']:.4f}, "
    #                   f"Level: {result['representativeness_level']})")
            
    #         # 可视化比较结果
    #         evaluator_xjtu.visualize_bearing_series_comparison(
    #             results_dict,
    #             all_comparisons,
    #             save_path=f'xjtu_{condition}_bearing_series_comparison.png'
    #         )
    
    # ========== FEMTO数据集 ==========
    print("\n" + "="*80)
    print("FEMTO Dataset - Finding Most Representative Subsets by Bearing Series")
    print("="*80)
    
    evaluator_femto = SubsetRepresentativenessEvaluator(
        data_dir='datasetresult/femto',
        m=4,
        epsilon=30
    )
    
    # FEMTO数据集没有工况前缀，直接按Bearing系列分组
    results_dict, all_comparisons = evaluator_femto.find_most_representative_by_bearing_series(
        condition_prefix=None,  # FEMTO没有工况前缀
        bearing_series=None,  # 自动检测所有Bearing系列
        save_comparison=True
    )
    
    if results_dict:
        print(f"\n{'='*80}")
        print("Summary for FEMTO Dataset:")
        print(f"{'='*80}")
        for series, result in sorted(results_dict.items()):
            print(f"{series:15s} -> {result['subset_name']:30s} "
                  f"(Score: {result['comprehensive_score']:.4f}, "
                  f"Level: {result['representativeness_level']})")
        
        # 可视化比较结果
        evaluator_femto.visualize_bearing_series_comparison(
            results_dict,
            all_comparisons,
            save_path='femto_bearing_series_comparison.png'
        )
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

