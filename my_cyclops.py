import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

class ExpressionDataset(Dataset):
    """基因表达数据集"""
    def __init__(self, expressions, times=None, celltypes=None):
        self.expressions = torch.FloatTensor(expressions)
        self.times = torch.FloatTensor(times) if times is not None else None
        self.celltypes = celltypes
        
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        sample = {'expression': self.expressions[idx]}
        if self.times is not None:
            sample['time'] = self.times[idx]
        if self.celltypes is not None:
            sample['celltype'] = self.celltypes[idx]
        return sample

class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        phase_coords = self.encoder(x)
        
        reconstructed = self.decoder(phase_coords)
        
        return phase_coords, reconstructed
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, phase_coords):
        return self.decoder(phase_coords)

def coords_to_phase(coords):
    """将2D坐标转换为相位角度"""
    x, y = coords[:, 0], coords[:, 1]
    phase = torch.atan2(y, x)
    phase = torch.where(phase < 0, phase + 2*np.pi, phase)
    return phase

def phase_to_coords(phase):
    """将相位角度转换为2D坐标（单位圆上）"""
    x = torch.cos(phase)
    y = torch.sin(phase)
    return torch.stack([x, y], dim=1)

def time_to_phase(time_hours, period_hours=24.0):
    """将时间转换为相位"""
    return 2 * np.pi * time_hours / period_hours

def train_model(model, train_loader, val_loader, preprocessing_info, 
                num_epochs=100, lr=0.001, device='cuda',
                lambda_recon=1.0, lambda_time=0.5, lambda_sine=0.1,
                period_hours=24.0, save_dir='./model_checkpoints'):
    """训练模型（使用验证集而不是测试集）"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    recon_criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    # 收集训练集细胞类型信息
    if preprocessing_info['train_has_celltype']:
        all_celltypes = []
        for batch in train_loader:
            if 'celltype' in batch:
                all_celltypes.extend(batch['celltype'])
        unique_celltypes = list(set(all_celltypes))
        celltype_to_idx = {ct: idx for idx, ct in enumerate(unique_celltypes)}
    else:
        celltype_to_idx = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始训练...")
    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss_epoch = 0.0
            train_recon_loss_epoch = 0.0
            train_time_loss_epoch = 0.0
            train_sine_loss_epoch = 0.0

            for batch in train_loader:
                expressions = batch['expression'].to(device)
                times = batch.get('time', None)
                celltypes = batch.get('celltype', None)

                if times is not None:
                    times = times.to(device)

                optimizer.zero_grad()

                phase_coords, reconstructed = model(expressions)
                recon_loss = recon_criterion(reconstructed, expressions)
                
                # 只在训练集有相应信息时计算对应损失
                time_loss = torch.tensor(0.0, device=device)
                if preprocessing_info['train_has_time'] and times is not None:
                    time_loss = time_supervision_loss(phase_coords, times, 1.0, period_hours)
                
                sine_loss = torch.tensor(0.0, device=device)
                if preprocessing_info['train_has_celltype'] and celltypes is not None:
                    sine_loss = sine_prior_loss(phase_coords, celltypes, celltype_to_idx, 1.0)

                total_loss = lambda_recon * recon_loss + lambda_time * time_loss + lambda_sine * sine_loss
                total_loss.backward()
                optimizer.step()

                train_loss_epoch += total_loss.item()
                train_recon_loss_epoch += recon_loss.item()
                train_time_loss_epoch += time_loss.item()
                train_sine_loss_epoch += sine_loss.item()

            # 验证阶段
            model.eval()
            val_loss_epoch = 0.0
            val_recon_loss_epoch = 0.0
            val_time_loss_epoch = 0.0
            val_sine_loss_epoch = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    expressions = batch['expression'].to(device)
                    times = batch.get('time', None)
                    celltypes = batch.get('celltype', None)

                    if times is not None:
                        times = times.to(device)

                    phase_coords, reconstructed = model(expressions)
                    recon_loss = recon_criterion(reconstructed, expressions)
                    
                    # 只在有相应信息时计算对应损失
                    time_loss = torch.tensor(0.0, device=device)
                    if preprocessing_info['train_has_time'] and times is not None:
                        time_loss = time_supervision_loss(phase_coords, times, 1.0, period_hours)
                    
                    sine_loss = torch.tensor(0.0, device=device)
                    if preprocessing_info['train_has_celltype'] and celltypes is not None:
                        sine_loss = sine_prior_loss(phase_coords, celltypes, celltype_to_idx, 1.0)

                    total_loss = lambda_recon * recon_loss + lambda_time * time_loss + lambda_sine * sine_loss

                    val_loss_epoch += total_loss.item()
                    val_recon_loss_epoch += recon_loss.item()
                    val_time_loss_epoch += time_loss.item()
                    val_sine_loss_epoch += sine_loss.item()

            train_loss_avg = train_loss_epoch / len(train_loader)
            val_loss_avg = val_loss_epoch / len(val_loader)

            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)

            scheduler.step(val_loss_avg)

            if (epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'Train loss': f'{train_loss_avg:.4f}',
                    'Val loss': f'{val_loss_avg:.4f}'
                })

            if (epoch + 1) % 100 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg,
                    'preprocessing_info': preprocessing_info
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            pbar.update(1)

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'preprocessing_info': preprocessing_info,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    
    return train_losses, val_losses

def load_and_preprocess_train_data(train_file, n_components=50, random_state=42):
    """加载和预处理训练数据（不分割验证集）"""
    print("=== 加载训练数据 ===")
    df = pd.read_csv(train_file, low_memory=False)
    
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    
    print(f"训练集包含时间信息: {has_time}")
    print(f"训练集包含细胞类型信息: {has_celltype}")
    
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    n_samples = len(sample_columns)
    
    celltypes = None
    times = None
    
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
        print(f"训练集细胞类型: {np.unique(celltypes)}")
    
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"训练集时间范围: {times.min():.2f} - {times.max():.2f} 小时")
    
    # 提取基因表达数据
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    gene_names = gene_df['Gene_Symbol'].values
    expression_data = gene_df[sample_columns].values.T  # (n_samples, n_genes)
    
    print(f"训练集原始基因数量: {len(gene_names)}")
    print(f"训练集样本数量: {n_samples}")
    
    # 标准化训练数据
    print("进行训练数据标准化...")
    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_data)
    
    # 使用SVD进行基因选择（基于训练集）
    print(f"基于训练集进行奇异值分解，选择前 {n_components} 个最重要的基因...")
    U, s, Vt = np.linalg.svd(expression_scaled.T, full_matrices=False)
    
    # 计算基因重要性分数
    n_top_components = min(10, len(s))
    gene_importance = np.sum(np.abs(U[:, :n_top_components]) * s[:n_top_components], axis=1)
    
    # 选择最重要的基因
    top_gene_indices = np.argsort(gene_importance)[-n_components:][::-1]
    selected_genes = gene_names[top_gene_indices]
    selected_expression = expression_scaled[:, top_gene_indices]
    
    print(f"选择的基因数量: {len(selected_genes)}")
    print(f"选择的基因样例: {selected_genes[:10].tolist()}")
    print(f"训练样本数: {len(selected_expression)}")
    
    # 创建训练数据集（使用全部数据）
    train_dataset = ExpressionDataset(selected_expression, times, celltypes)
    
    preprocessing_info = {
        'scaler': scaler,
        'selected_gene_indices': top_gene_indices,
        'selected_genes': selected_genes,
        'gene_importance_scores': gene_importance[top_gene_indices],
        'all_gene_names': gene_names,
        'train_has_time': has_time,
        'train_has_celltype': has_celltype,
        'n_components': n_components,
        'svd_info': {
            'U': U[:, :n_top_components],
            's': s[:n_top_components],
            'Vt': Vt[:n_top_components, :]
        }
    }
    
    return train_dataset, preprocessing_info

def load_and_preprocess_test_data(test_file, preprocessing_info):
    """加载和预处理测试数据（使用训练时的预处理参数）"""
    print("\n=== 加载测试数据 ===")
    df = pd.read_csv(test_file, low_memory=False)
    
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    
    print(f"测试集包含时间信息: {has_time}")
    print(f"测试集包含细胞类型信息: {has_celltype}")
    
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    n_samples = len(sample_columns)
    
    celltypes = None
    times = None
    
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
        print(f"测试集细胞类型: {np.unique(celltypes)}")
    
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"测试集时间范围: {times.min():.2f} - {times.max():.2f} 小时")
    
    # 提取基因表达数据
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    test_gene_names = gene_df['Gene_Symbol'].values
    test_expression_data = gene_df[sample_columns].values.T  # (n_samples, n_genes)
    
    print(f"测试集原始基因数量: {len(test_gene_names)}")
    print(f"测试集样本数量: {n_samples}")
    
    # 使用训练时的scaler进行标准化
    scaler = preprocessing_info['scaler']
    selected_genes = preprocessing_info['selected_genes']
    n_components = preprocessing_info['n_components']
    
    print("使用训练集的标准化参数处理测试数据...")
    test_expression_scaled = scaler.transform(test_expression_data)
    
    # 构建对应的测试集表达矩阵
    test_selected_expression = np.zeros((n_samples, n_components))
    missing_genes = []
    found_genes = []
    
    for train_idx, gene in enumerate(selected_genes):
        if gene in test_gene_names:
            test_gene_idx = np.where(test_gene_names == gene)[0][0]
            test_selected_expression[:, train_idx] = test_expression_scaled[:, test_gene_idx]
            found_genes.append(gene)
        else:
            missing_genes.append(gene)
            # 对于缺失的基因，使用0填充
            test_selected_expression[:, train_idx] = 0
    
    print(f"测试集中找到的基因数量: {len(found_genes)}")
    if missing_genes:
        print(f"测试集中缺失的基因数量: {len(missing_genes)}")
        print(f"缺失基因样例: {missing_genes[:5]}")
    
    # 创建测试数据集
    test_dataset = ExpressionDataset(test_selected_expression, times, celltypes)
    
    # 更新preprocessing_info
    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info.update({
        'test_has_time': has_time,
        'test_has_celltype': has_celltype,
        'test_sample_columns': sample_columns,
        'found_genes': found_genes,
        'missing_genes': missing_genes
    })
    
    return test_dataset, test_preprocessing_info

def sine_prior_loss(phase_coords, celltypes, celltype_to_idx, lambda_sine=1.0):
    if celltypes is None:
        return torch.tensor(0.0, device=phase_coords.device)
    
    total_loss = 0.0
    count = 0
    
    unique_celltypes = np.unique(celltypes)
    
    for celltype in unique_celltypes:
        mask = np.array([ct == celltype for ct in celltypes])
        if mask.sum() < 2:
            continue
            
        celltype_coords = phase_coords[mask]
        
        phases = coords_to_phase(celltype_coords)
        
        mean_phase = torch.atan2(torch.mean(torch.sin(phases)), torch.mean(torch.cos(phases)))
        phase_diffs = torch.abs(phases - mean_phase)
        phase_diffs = torch.min(phase_diffs, 2*np.pi - phase_diffs)
        
        phase_variance = torch.mean(phase_diffs)
        total_loss += phase_variance
        count += 1
    
    return lambda_sine * (total_loss / count if count > 0 else 0.0)

def time_supervision_loss(phase_coords, true_times, lambda_time=1.0, period_hours=24.0):
    if true_times is None:
        return torch.tensor(0.0, device=phase_coords.device)
    
    true_phases = time_to_phase(true_times, period_hours)
    true_coords = phase_to_coords(true_phases)
    
    pred_phases = coords_to_phase(phase_coords)
    
    phase_diff = torch.abs(pred_phases - true_phases)
    phase_diff = torch.min(phase_diff, 2*np.pi - phase_diff)
    
    return lambda_time * torch.mean(phase_diff)

def train_model(model, train_loader, preprocessing_info, 
                num_epochs=100, lr=0.001, device='cuda',
                lambda_recon=1.0, lambda_time=0.5, lambda_sine=0.1,
                period_hours=24.0, save_dir='./model_checkpoints'):
    """训练模型（无验证集）"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 改为步长调度器
    
    recon_criterion = nn.MSELoss()
    
    train_losses = []
    
    # 收集训练集细胞类型信息
    if preprocessing_info['train_has_celltype']:
        all_celltypes = []
        for batch in train_loader:
            if 'celltype' in batch:
                all_celltypes.extend(batch['celltype'])
        unique_celltypes = list(set(all_celltypes))
        celltype_to_idx = {ct: idx for idx, ct in enumerate(unique_celltypes)}
    else:
        celltype_to_idx = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始训练...")
    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss_epoch = 0.0
            train_recon_loss_epoch = 0.0
            train_time_loss_epoch = 0.0
            train_sine_loss_epoch = 0.0

            for batch in train_loader:
                expressions = batch['expression'].to(device)
                times = batch.get('time', None)
                celltypes = batch.get('celltype', None)

                if times is not None:
                    times = times.to(device)

                optimizer.zero_grad()

                phase_coords, reconstructed = model(expressions)
                recon_loss = recon_criterion(reconstructed, expressions)
                
                # 只在训练集有相应信息时计算对应损失
                time_loss = torch.tensor(0.0, device=device)
                if preprocessing_info['train_has_time'] and times is not None:
                    time_loss = time_supervision_loss(phase_coords, times, 1.0, period_hours)
                
                sine_loss = torch.tensor(0.0, device=device)
                if preprocessing_info['train_has_celltype'] and celltypes is not None:
                    sine_loss = sine_prior_loss(phase_coords, celltypes, celltype_to_idx, 1.0)

                total_loss = lambda_recon * recon_loss + lambda_time * time_loss + lambda_sine * sine_loss
                total_loss.backward()
                optimizer.step()

                train_loss_epoch += total_loss.item()
                train_recon_loss_epoch += recon_loss.item()
                train_time_loss_epoch += time_loss.item()
                train_sine_loss_epoch += sine_loss.item()

            train_loss_avg = train_loss_epoch / len(train_loader)
            train_losses.append(train_loss_avg)

            scheduler.step()  # 步长调度器

            if (epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'Train loss': f'{train_loss_avg:.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })

            if (epoch + 1) % 100 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'preprocessing_info': preprocessing_info
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            pbar.update(1)

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'preprocessing_info': preprocessing_info,
        'train_losses': train_losses
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    
    return train_losses

def predict_and_save_phases(model, test_loader, preprocessing_info, device='cuda', save_dir='./results'):
    print("\n=== 预测测试集相位 ===")
    model.eval()
    
    all_phase_coords = []
    all_phases = []
    all_times = []
    all_celltypes = []
    sample_indices = []
    
    batch_start_idx = 0
    
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            times = batch.get('time', None)
            celltypes = batch.get('celltype', None)
            
            phase_coords, _ = model(expressions)
            
            phases = coords_to_phase(phase_coords)
            
            all_phase_coords.append(phase_coords.cpu().numpy())
            all_phases.append(phases.cpu().numpy())
            
            batch_size = expressions.shape[0]
            batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
            sample_indices.extend(batch_indices)
            batch_start_idx += batch_size
            
            if times is not None:
                all_times.append(times.cpu().numpy())
            if celltypes is not None:
                all_celltypes.extend(celltypes)
    
    phase_coords = np.vstack(all_phase_coords)
    phases = np.concatenate(all_phases)
    
    if all_times:
        times = np.concatenate(all_times)
    else:
        times = None
        
    if all_celltypes:
        celltypes = np.array(all_celltypes)
    else:
        celltypes = None
    
    os.makedirs(save_dir, exist_ok=True)
    
    results_data = {
        'Sample_Index': sample_indices,
        'Phase_X': phase_coords[:, 0],
        'Phase_Y': phase_coords[:, 1],
        'Predicted_Phase_Radians': phases,
        'Predicted_Phase_Degrees': phases * 180 / np.pi,
        'Predicted_Phase_Hours': phases * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
    }
    
    if times is not None:
        results_data['True_Time_Hours'] = times
        results_data['True_Phase_Radians'] = time_to_phase(times, preprocessing_info.get('period_hours', 24.0))
        results_data['Phase_Error_Radians'] = np.abs(phases - results_data['True_Phase_Radians'])
        results_data['Phase_Error_Radians'] = np.minimum(
            results_data['Phase_Error_Radians'], 
            2*np.pi - results_data['Phase_Error_Radians']
        )
        results_data['Phase_Error_Hours'] = results_data['Phase_Error_Radians'] * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
    
    if celltypes is not None:
        results_data['Cell_Type'] = celltypes
    
    results_df = pd.DataFrame(results_data)
    
    predictions_file = os.path.join(save_dir, 'phase_predictions.csv')
    results_df.to_csv(predictions_file, index=False)
    print(f"详细预测结果保存到: {predictions_file}")
    
    simple_results = results_df[['Sample_Index', 'Predicted_Phase_Hours']].copy()
    simple_results.columns = ['Sample_ID', 'Predicted_Phase_Hours']
    
    simple_file = os.path.join(save_dir, 'phase_predictions_simple.csv')
    simple_results.to_csv(simple_file, index=False)
    print(f"简化预测结果保存到: {simple_file}")
    
    print(f"\n=== 预测统计 ===")
    print(f"预测样本数量: {len(phases)}")
    print(f"预测相位范围: {phases.min():.3f} - {phases.max():.3f} 弧度")
    print(f"预测相位范围: {(phases * 180 / np.pi).min():.1f} - {(phases * 180 / np.pi).max():.1f} 度")
    print(f"预测时间范围: {results_data['Predicted_Phase_Hours'].min():.2f} - {results_data['Predicted_Phase_Hours'].max():.2f} 小时")
    
    if times is not None:
        mean_error_hours = np.mean(results_data['Phase_Error_Hours'])
        std_error_hours = np.std(results_data['Phase_Error_Hours'])
        print(f"平均预测误差: {mean_error_hours:.2f} ± {std_error_hours:.2f} 小时")
        
        for threshold in [1, 2, 3, 6]:
            accuracy = np.mean(results_data['Phase_Error_Hours'] <= threshold) * 100
            print(f"误差 ≤ {threshold}小时的样本比例: {accuracy:.1f}%")
    
    if celltypes is not None:
        print(f"\n按细胞类型统计:")
        celltype_stats = results_df.groupby('Cell_Type').agg({
            'Predicted_Phase_Hours': ['mean', 'std', 'count']
        }).round(2)
        print(celltype_stats)
    
    create_prediction_plots(results_df, save_dir)
    
    return results_df

def create_prediction_plots(results_df, save_dir):
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['Predicted_Phase_Hours'], bins=24, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Phase (Hours)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Phases')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['Phase_X'], results_df['Phase_Y'], alpha=0.6)
    plt.xlabel('Phase X')
    plt.ylabel('Phase Y')
    plt.title('Phase Distribution in Unit Circle')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    if 'True_Time_Hours' in results_df.columns:
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['True_Time_Hours'], results_df['Predicted_Phase_Hours'], alpha=0.6)
        plt.plot([0, 24], [0, 24], 'r--', label='Perfect Prediction')
        plt.xlabel('True Time (Hours)')
        plt.ylabel('Predicted Phase (Hours)')
        plt.title('True Time vs Predicted Phase')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.hist(results_df['Phase_Error_Hours'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (Hours)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True, alpha=0.3)
    else:
        if 'Cell_Type' in results_df.columns:
            plt.subplot(2, 2, 3)
            unique_celltypes = results_df['Cell_Type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_celltypes)))
            
            for i, celltype in enumerate(unique_celltypes):
                mask = results_df['Cell_Type'] == celltype
                plt.scatter(results_df.loc[mask, 'Phase_X'], 
                          results_df.loc[mask, 'Phase_Y'], 
                          c=[colors[i]], label=celltype, alpha=0.6)
            
            plt.xlabel('Phase X')
            plt.ylabel('Phase Y')
            plt.title('Phase Distribution by Cell Type')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测分析图表保存到: {os.path.join(save_dir, 'prediction_analysis.png')}")

def main():
    parser = argparse.ArgumentParser(description="训练相位自编码器模型")
    parser.add_argument("--train_file", required=True, help="训练数据文件路径")
    parser.add_argument("--test_file", default=None, help="测试数据文件路径（可选）")
    parser.add_argument("--n_components", type=int, default=50, help="选择的重要基因数量")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="重建损失权重")
    parser.add_argument("--lambda_time", type=float, default=0.5, help="时间监督损失权重")
    parser.add_argument("--lambda_sine", type=float, default=0.1, help="正弦先验损失权重")
    parser.add_argument("--period_hours", type=float, default=24.0, help="预期周期（小时）")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout比例")
    parser.add_argument("--device", default='cuda', help="设备 (cuda/cpu)")
    parser.add_argument("--save_dir", default='./phase_autoencoder_results', help="保存目录")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print("=== 相位自编码器训练 ===")
    print(f"训练数据文件: {args.train_file}")
    if args.test_file:
        print(f"测试数据文件: {args.test_file}")
    print(f"选择重要基因数: {args.n_components}")
    print(f"设备: {args.device}")
    
    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        args.train_file, args.n_components, args.random_seed
    )
    
    preprocessing_info['period_hours'] = args.period_hours
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = PhaseAutoEncoder(
        input_dim=args.n_components,
        dropout=args.dropout
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    train_losses = train_model(
        model=model,
        train_loader=train_loader,
        preprocessing_info=preprocessing_info,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        lambda_recon=args.lambda_recon,
        lambda_time=args.lambda_time,
        lambda_sine=args.lambda_sine,
        period_hours=args.period_hours,
        save_dir=args.save_dir
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== 训练完成 ===")
    print(f"训练结果保存到: {args.save_dir}")
    
    if args.test_file:
        print(f"\n=== 开始测试阶段 ===")
        
        test_dataset, test_preprocessing_info = load_and_preprocess_test_data(
            args.test_file, preprocessing_info
        )
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        prediction_results = predict_and_save_phases(
            model=model,
            test_loader=test_loader,
            preprocessing_info=test_preprocessing_info,
            device=args.device,
            save_dir=args.save_dir
        )
        
        print(f"\n=== 测试完成 ===")
        print(f"主要输出文件:")
        print(f"  - 模型权重: {args.save_dir}/final_model.pth")
        print(f"  - 详细预测: {args.save_dir}/phase_predictions.csv")
        print(f"  - 简化预测: {args.save_dir}/phase_predictions_simple.csv")
        print(f"  - 训练曲线: {args.save_dir}/training_curves.png")
        print(f"  - 预测分析: {args.save_dir}/prediction_analysis.png")
    else:
        print(f"\n未提供测试文件，只完成训练阶段")
        print(f"主要输出文件:")
        print(f"  - 模型权重: {args.save_dir}/final_model.pth")
        print(f"  - 训练曲线: {args.save_dir}/training_curves.png")

if __name__ == "__main__":
    main()