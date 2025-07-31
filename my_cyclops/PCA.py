import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample

class CelltypeAwareTransformer:
    """细胞类型感知的变换器类"""
    def __init__(self, selected_genes, final_pca, n_components):
        self.selected_genes = selected_genes
        self.final_pca = final_pca
        self.n_components = n_components
    
    def transform(self, X_scaled, times_test=None):
        # 选择相同的基因
        if X_scaled.shape[1] >= max(self.selected_genes) + 1:
            selected_test_expression = X_scaled[:, self.selected_genes]
            return self.final_pca.transform(selected_test_expression)
        else:
            # 如果基因数量不够，降级处理
            print("警告: 测试数据基因数量不足，使用前N个基因")
            available_genes = min(X_scaled.shape[1], self.n_components)
            fallback_expression = X_scaled[:, :available_genes]
            
            # 简单的0填充到期望维度
            if fallback_expression.shape[1] < self.n_components:
                padding = np.zeros((fallback_expression.shape[0], 
                                  self.n_components - fallback_expression.shape[1]))
                fallback_expression = np.hstack([fallback_expression, padding])
            
            return fallback_expression[:, :self.n_components]

class SimplePCATransformer:
    """简单PCA变换器类"""
    def __init__(self, pca):
        self.pca = pca
    
    def transform(self, X_scaled, times_test=None):
        return self.pca.transform(X_scaled)

def create_celltype_aware_eigengenes(expression_scaled, celltypes, times=None, n_components=50, period_hours=24.0):
    """
    创建细胞类型感知的eigengenes，确保对所有细胞类型都有效
    
    Args:
        expression_scaled: 标准化后的表达数据
        celltypes: 细胞类型标签
        times: 时间信息（如果有的话）
        n_components: 组件数量
        period_hours: 周期长度
    
    Returns:
        celltype_aware_components: 细胞类型感知的eigengenes
        transformer: 变换器对象
        explained_variance: 解释方差比
    """
    print("=== 创建细胞类型感知的Eigengenes ===")
    
    n_samples, n_genes = expression_scaled.shape
    
    if celltypes is not None:
        unique_celltypes = np.unique(celltypes)
        unique_celltypes = [ct for ct in unique_celltypes if ct != 'PADDING']
        print(f"发现细胞类型: {unique_celltypes}")
        print(f"每个细胞类型的样本数量:")
        for ct in unique_celltypes:
            ct_count = np.sum(celltypes == ct)
            print(f"  {ct}: {ct_count} 样本")
        
        # 方法1: 为每个细胞类型分别进行PCA，然后寻找共同模式
        print("为每个细胞类型分别进行PCA分析...")
        
        celltype_pcas = {}
        celltype_components = {}
        celltype_variances = {}
        
        # 为每个细胞类型计算PCA
        for ct in unique_celltypes:
            ct_mask = celltypes == ct
            ct_expression = expression_scaled[ct_mask]
            ct_times = times[ct_mask] if times is not None else None
            
            if ct_expression.shape[0] < 3:  # 样本太少，跳过
                print(f"  细胞类型 {ct} 样本太少，跳过")
                continue
            
            # 对该细胞类型进行PCA
            ct_pca = PCA(n_components=min(n_components, ct_expression.shape[0], ct_expression.shape[1]))
            ct_pca_result = ct_pca.fit_transform(ct_expression)
            
            celltype_pcas[ct] = ct_pca
            celltype_components[ct] = ct_pca_result
            celltype_variances[ct] = ct_pca.explained_variance_ratio_
            
            print(f"  {ct}: 提取了 {ct_pca_result.shape[1]} 个主成分")
        
        if len(celltype_pcas) == 0:
            print("没有足够的细胞类型数据，降级为普通PCA")
            return create_simple_pca_fallback(expression_scaled, n_components)
        
        # 方法2: 寻找跨细胞类型的共同特征
        print("寻找跨细胞类型的共同基因模式...")
        
        # 获取每个细胞类型的主要loading vectors
        common_gene_weights = []
        
        for gene_idx in range(min(n_genes, n_components * 2)):  # 考虑更多基因
            gene_importance_scores = []
            
            for ct in celltype_pcas.keys():
                # 计算该基因在该细胞类型前几个主成分中的重要性
                loadings = celltype_pcas[ct].components_  # (n_components, n_genes)
                gene_loadings = loadings[:min(5, loadings.shape[0]), gene_idx]  # 前5个主成分
                
                # 计算加权重要性（权重由解释方差决定）
                weights = celltype_variances[ct][:len(gene_loadings)]
                weighted_importance = np.sum(np.abs(gene_loadings) * weights)
                gene_importance_scores.append(weighted_importance)
            
            # 计算该基因在所有细胞类型中的平均重要性
            avg_importance = np.mean(gene_importance_scores)
            min_importance = np.min(gene_importance_scores)  # 最小重要性，确保对所有细胞类型都重要
            
            # 组合评分：平均重要性 + 最小重要性（确保公平性）
            combined_score = 0.7 * avg_importance + 0.3 * min_importance
            common_gene_weights.append((gene_idx, combined_score))
        
        # 按重要性排序，选择前n_components个基因
        common_gene_weights.sort(key=lambda x: x[1], reverse=True)
        selected_genes = [gene_idx for gene_idx, _ in common_gene_weights[:n_components]]
        
        print(f"选择了 {len(selected_genes)} 个跨细胞类型重要基因")
        
        # 方法3: 基于选定基因构建新的特征空间
        print("基于选定基因构建细胞类型感知的特征空间...")
        
        # 创建新的特征矩阵
        selected_expression = expression_scaled[:, selected_genes]
        
        # 为了确保不同细胞类型的模式都能被捕获，使用加权PCA
        # 权重确保每个细胞类型的贡献相等
        celltype_weights = np.ones(n_samples)
        for ct in unique_celltypes:
            ct_mask = celltypes == ct
            ct_count = np.sum(ct_mask)
            if ct_count > 0:
                # 平衡不同细胞类型的样本数量
                target_weight = n_samples / (len(unique_celltypes) * ct_count)
                celltype_weights[ct_mask] = target_weight
        
        # 创建平衡的数据集
        balanced_expression = []
        balanced_celltypes = []
        balanced_times = []
        
        # 计算目标样本数（每个细胞类型的样本数应该相等）
        min_samples = min([np.sum(celltypes == ct) for ct in unique_celltypes])
        target_samples_per_type = max(min_samples, 10)  # 至少10个样本
        
        print(f"每个细胞类型目标样本数: {target_samples_per_type}")
        
        for ct in unique_celltypes:
            ct_mask = celltypes == ct
            ct_expression = selected_expression[ct_mask]
            ct_celltypes = celltypes[ct_mask]
            ct_times_subset = times[ct_mask] if times is not None else None
            
            if len(ct_expression) >= target_samples_per_type:
                # 随机采样
                indices = np.random.choice(len(ct_expression), target_samples_per_type, replace=False)
                balanced_expression.append(ct_expression[indices])
                balanced_celltypes.extend(ct_celltypes[indices])
                if ct_times_subset is not None:
                    balanced_times.extend(ct_times_subset[indices])
            else:
                # 过采样
                indices = resample(range(len(ct_expression)), n_samples=target_samples_per_type, random_state=42)
                balanced_expression.append(ct_expression[indices])
                balanced_celltypes.extend(ct_celltypes[indices])
                if ct_times_subset is not None:
                    balanced_times.extend(ct_times_subset[indices])
        
        balanced_expression = np.vstack(balanced_expression)
        balanced_celltypes = np.array(balanced_celltypes)
        balanced_times = np.array(balanced_times) if balanced_times else None
        
        print(f"平衡后的数据维度: {balanced_expression.shape}")
        print(f"平衡后每个细胞类型的样本数:")
        for ct in unique_celltypes:
            ct_count = np.sum(balanced_celltypes == ct)
            print(f"  {ct}: {ct_count} 样本")
        
        # 在平衡的数据上进行最终PCA
        final_pca = PCA(n_components=n_components)
        balanced_components = final_pca.fit_transform(balanced_expression)
        
        # 将所有原始数据投影到这个空间
        original_components = final_pca.transform(selected_expression)
        
        explained_variance = final_pca.explained_variance_ratio_
        
        # 创建变换器（现在使用模块级别的类）
        transformer = CelltypeAwareTransformer(selected_genes, final_pca, n_components)
        
        print(f"成功创建细胞类型感知的 {n_components} 个eigengenes")
        print(f"前5个组件的解释方差比: {explained_variance[:5]}")
        
        return original_components, transformer, explained_variance
        
    else:
        # 没有细胞类型信息，降级为普通PCA
        print("没有细胞类型信息，使用普通PCA")
        return create_simple_pca_fallback(expression_scaled, n_components)

def create_simple_pca_fallback(expression_scaled, n_components):
    """PCA降级方案"""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(expression_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    # 使用模块级别的类
    transformer = SimplePCATransformer(pca)
    return components, transformer, explained_variance