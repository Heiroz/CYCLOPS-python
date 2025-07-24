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

        self.encoder = nn.Linear(input_dim, 2)
        self.decoder = nn.Linear(2, input_dim)
    
    def forward(self, x):
        phase_coords = self.encoder(x)
        reconstructed = self.decoder(phase_coords)
        return phase_coords, reconstructed
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, phase_coords):
        return self.decoder(phase_coords)

def coords_to_phase(coords):
    x, y = coords[:, 0], coords[:, 1]
    phase = torch.atan2(y, x)
    phase = torch.where(phase < 0, phase + 2*np.pi, phase)
    return phase

def phase_to_coords(phase):
    x = torch.cos(phase)
    y = torch.sin(phase)
    return torch.stack([x, y], dim=1)

def time_to_phase(time_hours, period_hours=24.0):
    return 2 * np.pi * time_hours / period_hours

def load_and_preprocess_train_data(train_file, n_components=50, max_samples=100, random_state=42):
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
    
    print(f"原始样本数量: {n_samples}")
    print(f"最大样本数量限制: {max_samples}")
    
    celltypes = None
    times = None
    
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
        print(f"训练集细胞类型: {np.unique(celltypes)}")
    
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"训练集时间范围: {times.min():.2f} - {times.max():.2f} 小时")
    
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    gene_names = gene_df['Gene_Symbol'].values
    expression_data = gene_df[sample_columns].values.T
    
    print(f"训练集原始基因数量: {len(gene_names)}")
    
    print("进行训练数据标准化...")
    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_data)
    
    print(f"基于训练集进行奇异值分解，选择前 {n_components} 个最重要的基因...")
    U, s, Vt = np.linalg.svd(expression_scaled.T, full_matrices=False)
    
    n_top_components = min(n_components, len(s))
    gene_importance = np.sum(np.abs(U[:, :n_top_components]) * s[:n_top_components], axis=1)
    
    top_gene_indices = np.argsort(gene_importance)[-n_components:][::-1]
    selected_genes = gene_names[top_gene_indices]
    selected_expression = expression_scaled[:, top_gene_indices]
    
    if n_samples > max_samples:
        print(f"样本数量 ({n_samples}) 超过最大限制 ({max_samples})，进行截断...")
        np.random.seed(random_state)
        selected_indices = np.random.choice(n_samples, max_samples, replace=False)
        selected_indices = np.sort(selected_indices)
        
        selected_expression = selected_expression[selected_indices]
        if times is not None:
            times = times[selected_indices]
        if celltypes is not None:
            celltypes = celltypes[selected_indices]
        
        actual_samples = max_samples
        print(f"截断后样本数量: {actual_samples}")
        
    elif n_samples < max_samples:
        print(f"样本数量 ({n_samples}) 少于最大限制 ({max_samples})，进行0填充...")
        pad_size = max_samples - n_samples
        
        padding = np.zeros((pad_size, n_components))
        selected_expression = np.vstack([selected_expression, padding])
        
        if times is not None:
            times = np.concatenate([times, np.zeros(pad_size)])
        if celltypes is not None:
            celltypes = np.concatenate([celltypes, ['PADDING'] * pad_size])
        
        actual_samples = max_samples
        print(f"填充后样本数量: {actual_samples}")
        
    else:
        actual_samples = n_samples
        print(f"样本数量正好等于最大限制: {actual_samples}")
    
    print(f"最终使用的样本数量: {actual_samples}")
    print(f"选择的基因数量: {len(selected_genes)}")
    print(f"选择的基因样例: {selected_genes[:10].tolist()}")
    
    # 创建训练数据集（现在使用固定大小的数据）
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
        'max_samples': max_samples,
        'actual_samples': actual_samples,
        'original_samples': n_samples,
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
            test_selected_expression[:, train_idx] = 0
    
    print(f"测试集中找到的基因数量: {len(found_genes)}")
    if missing_genes:
        print(f"测试集中缺失的基因数量: {len(missing_genes)}")
        print(f"缺失基因样例: {missing_genes[:5]}")
    
    test_dataset = ExpressionDataset(test_selected_expression, times, celltypes)
    
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
    
    total_loss = torch.tensor(0.0, device=phase_coords.device)
    count = 0
    
    unique_celltypes = np.unique(celltypes)
    
    for celltype in unique_celltypes:
        mask = np.array([ct == celltype for ct in celltypes])
        if mask.sum() < 2:
            continue
            
        celltype_coords = phase_coords[mask]
        phases = coords_to_phase(celltype_coords)
        
        sin_values = torch.sin(phases)
        cos_values = torch.cos(phases)
        
        mean_sin = torch.mean(sin_values)
        mean_cos = torch.mean(cos_values)
        
        loss = mean_sin**2 + mean_cos**2
        total_loss += loss
        count += 1
    
    if count > 0:
        return lambda_sine * (total_loss / count)
    else:
        return torch.tensor(0.0, device=phase_coords.device)


def time_supervision_loss(phase_coords, true_times, lambda_time=1.0, period_hours=24.0):
    if true_times is None:
        return torch.tensor(0.0, device=phase_coords.device)
    
    true_phases = time_to_phase(true_times, period_hours)
    true_coords = phase_to_coords(true_phases)
    
    pred_phases = coords_to_phase(phase_coords)
    
    phase_diff = torch.abs(pred_phases - true_phases)
    phase_diff = torch.min(phase_diff, 2*np.pi - phase_diff)
    
    return lambda_time * torch.mean(phase_diff)

def train_model(model, train_dataset, preprocessing_info, 
                num_epochs=100, lr=0.001, device='cuda',
                lambda_recon=1.0, lambda_time=0.5, lambda_sine=0.1,
                period_hours=24.0, save_dir='./model_checkpoints'):
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    recon_criterion = nn.MSELoss()
    
    train_losses = []
    
    if preprocessing_info['train_has_celltype']:
        all_celltypes = []
        for i in range(len(train_dataset)):
            celltype = train_dataset[i].get('celltype', None)
            if celltype is not None and celltype != 'PADDING':
                all_celltypes.append(celltype)
        unique_celltypes = list(set(all_celltypes))
        celltype_to_idx = {ct: idx for idx, ct in enumerate(unique_celltypes)}
        print(f"训练集有效细胞类型: {unique_celltypes}")
    else:
        celltype_to_idx = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("准备训练数据...")
    all_expressions = []
    all_times = []
    all_celltypes = []
    valid_mask = []
    
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        all_expressions.append(sample['expression'])
        
        if 'celltype' in sample and sample['celltype'] == 'PADDING':
            is_valid = False
        else:
            is_valid = True
        valid_mask.append(is_valid)
        
        if 'time' in sample:
            all_times.append(sample['time'])
        if 'celltype' in sample:
            all_celltypes.append(sample['celltype'])
    
    expressions_tensor = torch.stack(all_expressions).to(device)
    valid_mask_tensor = torch.tensor(valid_mask, device=device)
    
    times_tensor = None
    if all_times:
        times_tensor = torch.stack(all_times).to(device)
    
    celltypes_array = None
    if all_celltypes:
        celltypes_array = np.array(all_celltypes)
    
    print(f"训练数据准备完成:")
    print(f"  - 总样本数: {len(expressions_tensor)}")
    print(f"  - 有效样本数: {valid_mask_tensor.sum().item()}")
    print(f"  - 填充样本数: {(~valid_mask_tensor).sum().item()}")
    
    print("开始训练...")
    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            model.train()
            
            optimizer.zero_grad()
            
            phase_coords, reconstructed = model(expressions_tensor)
            
            if valid_mask_tensor.sum() > 0:
                valid_expressions = expressions_tensor[valid_mask_tensor]
                valid_reconstructed = reconstructed[valid_mask_tensor]
                recon_loss = recon_criterion(valid_reconstructed, valid_expressions)
            else:
                recon_loss = torch.tensor(0.0, device=device)
            
            time_loss = torch.tensor(0.0, device=device)
            if preprocessing_info['train_has_time'] and times_tensor is not None:
                valid_phase_coords = phase_coords[valid_mask_tensor]
                valid_times = times_tensor[valid_mask_tensor]
                if len(valid_times) > 0:
                    time_loss = time_supervision_loss(valid_phase_coords, valid_times, 1.0, period_hours)
            
            sine_loss = torch.tensor(0.0, device=device)
            if preprocessing_info['train_has_celltype'] and celltypes_array is not None:
                valid_phase_coords = phase_coords[valid_mask_tensor]
                valid_celltypes = celltypes_array[valid_mask_tensor.cpu().numpy()]
                non_padding_mask = valid_celltypes != 'PADDING'
                if non_padding_mask.sum() > 0:
                    final_phase_coords = valid_phase_coords[non_padding_mask]
                    final_celltypes = valid_celltypes[non_padding_mask]
                    sine_loss = sine_prior_loss(final_phase_coords, final_celltypes, celltype_to_idx, 1.0)
            
            total_loss = lambda_recon * recon_loss + lambda_time * time_loss + lambda_sine * sine_loss
            print("recon_loss:", recon_loss.item())
            print("time_loss:", time_loss.item())
            print("sine_loss:", sine_loss.item())
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'Train loss': f'{total_loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'Time': f'{time_loss.item():.4f}',
                    'Sine': f'{sine_loss.item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            # if (epoch + 1) % 1000 == 0:
            #     checkpoint = {
            #         'epoch': epoch + 1,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'train_loss': total_loss.item(),
            #         'preprocessing_info': preprocessing_info
            #     }
            #     torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
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

def plot_gene_expression_by_phase(model, test_loader, preprocessing_info, device='cuda', save_dir='./results', n_genes_to_plot=10, custom_genes=None):
    """绘制基因表达量随预测相位变化的图表"""
    print("\n=== 绘制基因表达相位图 ===")
    model.eval()
    
    all_expressions = []
    all_phase_hours = []
    all_celltypes = []
    
    # 收集预测数据
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            celltypes = batch.get('celltype', None)
            
            # 获取预测相位
            phase_coords, _ = model(expressions)
            phases = coords_to_phase(phase_coords)
            phase_hours = phases.cpu().numpy() * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
            
            all_expressions.append(expressions.cpu().numpy())
            all_phase_hours.append(phase_hours)
            
            if celltypes is not None:
                all_celltypes.extend(celltypes)
    
    # 合并数据
    expressions_data = np.vstack(all_expressions)  # SVD处理后的数据
    phase_hours_data = np.concatenate(all_phase_hours)
    
    if all_celltypes:
        celltypes_data = np.array(all_celltypes)
        unique_celltypes = np.unique(celltypes_data)
        print(f"发现细胞类型: {unique_celltypes}")
    else:
        celltypes_data = None
        unique_celltypes = ['All_Samples']
    
    # 获取要绘制的基因信息和对应的表达数据
    if custom_genes is not None:
        print(f"使用用户指定的基因列表: {custom_genes}")
        
        # 需要重新加载原始测试数据来获取指定基因的表达量
        gene_expressions, gene_names_to_plot = get_custom_gene_expressions(
            preprocessing_info, custom_genes, phase_hours_data, celltypes_data
        )
        
        if gene_expressions is None:
            print("无法获取自定义基因表达数据，使用SVD选择的基因")
            selected_genes = preprocessing_info['selected_genes']
            n_genes_to_plot = min(n_genes_to_plot, len(selected_genes))
            gene_names_to_plot = selected_genes[:n_genes_to_plot]
            gene_expressions = expressions_data[:, :n_genes_to_plot]
        
    else:
        # 使用默认的SVD选择的基因
        selected_genes = preprocessing_info['selected_genes']
        n_genes_to_plot = min(n_genes_to_plot, len(selected_genes))
        gene_names_to_plot = selected_genes[:n_genes_to_plot]
        gene_expressions = expressions_data[:, :n_genes_to_plot]
        print(f"使用默认的前 {n_genes_to_plot} 个SVD选择的基因")
    
    print(f"将要绘制的基因: {gene_names_to_plot}")
    print(f"基因表达数据维度: {gene_expressions.shape}")
    
    # 创建相位分箱
    n_bins = 24  # 每小时一个分箱
    phase_bins = np.linspace(0, 24, n_bins + 1)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个细胞类型创建图表
    if celltypes_data is not None:
        # 按细胞类型分别绘制
        for celltype in unique_celltypes:
            if celltype == 'PADDING':
                continue
                
            celltype_mask = celltypes_data == celltype
            celltype_expressions = gene_expressions[celltype_mask]
            celltype_phases = phase_hours_data[celltype_mask]
            
            if len(celltype_expressions) < 10:  # 样本太少跳过
                continue
            
            plot_celltype_gene_expression(
                celltype_expressions, celltype_phases, gene_names_to_plot,
                phase_centers, phase_bins, celltype, save_dir
            )
    else:
        # 所有样本一起绘制
        plot_celltype_gene_expression(
            gene_expressions, phase_hours_data, gene_names_to_plot,
            phase_centers, phase_bins, 'All_Samples', save_dir
        )
    
    # 创建综合对比图
    if celltypes_data is not None and len(unique_celltypes) > 1:
        # 过滤掉PADDING类型
        valid_celltypes = [ct for ct in unique_celltypes if ct != 'PADDING']
        if len(valid_celltypes) > 1:
            plot_celltype_comparison(
                gene_expressions, phase_hours_data, celltypes_data, gene_names_to_plot,
                phase_centers, phase_bins, valid_celltypes, save_dir
            )

def get_custom_gene_expressions(preprocessing_info, custom_genes, phase_hours_data, celltypes_data):
    """获取自定义基因的表达数据"""
    try:
        # 从preprocessing_info中获取测试文件信息
        if 'test_sample_columns' not in preprocessing_info:
            print("警告: 无法获取测试文件信息")
            return None, None
            
        # 这里需要重新读取原始测试数据
        # 注意：这个函数假设我们可以访问原始测试文件
        # 在实际使用中，可能需要传入test_file路径
        print("注意: 需要重新读取原始测试数据来获取自定义基因表达量")
        print("当前实现将返回None，请在main函数中传入原始数据")
        return None, None
        
    except Exception as e:
        print(f"获取自定义基因表达数据时出错: {e}")
        return None, None

def get_original_gene_expressions(test_file, custom_genes, preprocessing_info, phase_hours_data, celltypes_data):
    """从原始测试文件中获取自定义基因的表达数据"""
    try:
        print(f"从原始文件重新读取自定义基因: {custom_genes}")
        
        # 重新读取原始测试数据
        df = pd.read_csv(test_file, low_memory=False)
        
        # 提取基因表达数据
        gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
        test_gene_names = gene_df['Gene_Symbol'].values
        
        sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
        test_expression_data = gene_df[sample_columns].values.T  # (n_samples, n_genes)
        
        # 使用训练时的scaler进行标准化
        scaler = preprocessing_info['scaler']
        test_expression_scaled = scaler.transform(test_expression_data)
        
        # 查找自定义基因
        found_genes = []
        gene_expressions_list = []
        
        for gene in custom_genes:
            if gene in test_gene_names:
                gene_idx = np.where(test_gene_names == gene)[0][0]
                gene_expression = test_expression_scaled[:, gene_idx]
                gene_expressions_list.append(gene_expression)
                found_genes.append(gene)
                print(f"  ✓ 找到基因: {gene}")
            else:
                print(f"  ✗ 基因 {gene} 不在测试数据中")
        
        if len(found_genes) == 0:
            print("错误: 没有找到任何指定的基因")
            return None, None
        
        # 转换为numpy数组
        gene_expressions = np.column_stack(gene_expressions_list)
        
        print(f"成功获取 {len(found_genes)} 个基因的表达数据")
        print(f"表达数据维度: {gene_expressions.shape}")
        
        return gene_expressions, np.array(found_genes)
        
    except Exception as e:
        print(f"从原始文件读取基因表达数据时出错: {e}")
        return None, None

def plot_celltype_gene_expression(expressions, phase_hours, gene_names, phase_centers, phase_bins, celltype, save_dir):
    """为单个细胞类型绘制基因表达图"""
    n_genes = len(gene_names)
    
    print(f"为细胞类型 {celltype} 绘制基因表达图")
    print(f"样本数量: {len(expressions)}")
    print(f"基因数量: {n_genes}")
    print(f"表达数据维度: {expressions.shape}")
    
    # 计算每个相位分箱的平均表达量和标准误差
    binned_expressions = np.zeros((len(phase_centers), n_genes))
    binned_std = np.zeros((len(phase_centers), n_genes))
    binned_counts = np.zeros(len(phase_centers))
    
    for i, (start, end) in enumerate(zip(phase_bins[:-1], phase_bins[1:])):
        mask = (phase_hours >= start) & (phase_hours < end)
        if mask.sum() > 0:
            binned_expressions[i] = np.mean(expressions[mask], axis=0)
            binned_std[i] = np.std(expressions[mask], axis=0) / np.sqrt(mask.sum())
            binned_counts[i] = mask.sum()
    
    # 创建图表 - 动态调整布局
    n_cols = min(5, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols  # 向上取整
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_genes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_genes > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, gene_name in enumerate(gene_names):
        ax = axes[i]
        
        # 绘制表达量曲线
        valid_mask = binned_counts > 0
        x_data = phase_centers[valid_mask]
        y_data = binned_expressions[valid_mask, i]
        y_err = binned_std[valid_mask, i]
        
        if len(x_data) == 0:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{gene_name}', fontsize=10)
            continue
        
        # 主要曲线
        ax.plot(x_data, y_data, 'o-', linewidth=2, markersize=4, alpha=0.8, label='Expression')
        ax.fill_between(x_data, y_data - y_err, y_data + y_err, alpha=0.3)
        
        # 尝试拟合正弦曲线
        if len(x_data) > 5:
            try:
                def sine_func(x, amplitude, phase_shift, offset):
                    return amplitude * np.sin(2 * np.pi * x / 24 + phase_shift) + offset
                
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(sine_func, x_data, y_data, maxfev=2000)
                
                # 绘制拟合曲线
                x_fit = np.linspace(0, 24, 100)
                y_fit = sine_func(x_fit, *popt)
                ax.plot(x_fit, y_fit, '--', color='red', alpha=0.7, label='Sine Fit')
                
                # 显示拟合参数
                amplitude, phase_shift, offset = popt
                peak_time = (-phase_shift * 24 / (2 * np.pi)) % 24
                ax.text(0.02, 0.98, f'Peak: {peak_time:.1f}h\nAmp: {amplitude:.3f}', 
                       transform=ax.transAxes, verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                print(f"拟合失败 {gene_name}: {e}")
        
        ax.set_title(f'{gene_name}', fontsize=10)
        ax.set_xlabel('Predicted Phase (Hours)')
        ax.set_ylabel('Expression Level')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        
        if i == 0:
            ax.legend(fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_genes, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Gene Expression vs Predicted Phase - {celltype}', fontsize=14)
    plt.tight_layout()
    
    filename = f'gene_expression_phase_{celltype}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"基因表达相位图已保存: {filepath}")

def plot_celltype_comparison(expressions, phase_hours, celltypes, gene_names, phase_centers, phase_bins, valid_celltypes, save_dir):
    """绘制不同细胞类型的基因表达对比图"""
    print("绘制细胞类型对比图...")
    print(f"有效细胞类型: {valid_celltypes}")
    print(f"选择的基因: {gene_names}")
    
    n_genes_to_compare = min(4, len(gene_names))
    top_genes = gene_names[:n_genes_to_compare]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_celltypes)))
    
    for i, gene_name in enumerate(top_genes):
        if i >= 4:
            break
            
        ax = axes[i]
        
        for celltype_idx, celltype in enumerate(valid_celltypes):
            celltype_mask = celltypes == celltype
            celltype_expressions = expressions[celltype_mask]
            celltype_phases = phase_hours[celltype_mask]
            
            if len(celltype_expressions) < 5:
                continue
            
            # 计算分箱平均值
            binned_expr = np.zeros(len(phase_centers))
            binned_counts = np.zeros(len(phase_centers))
            
            for bin_i, (start, end) in enumerate(zip(phase_bins[:-1], phase_bins[1:])):
                mask = (celltype_phases >= start) & (celltype_phases < end)
                if mask.sum() > 0:
                    binned_expr[bin_i] = np.mean(celltype_expressions[mask, i])
                    binned_counts[bin_i] = mask.sum()
            
            valid_mask = binned_counts > 0
            if valid_mask.sum() > 3:
                x_data = phase_centers[valid_mask]
                y_data = binned_expr[valid_mask]
                
                ax.plot(x_data, y_data, 'o-', color=colors[celltype_idx], 
                       label=celltype, linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_title(f'{gene_name}', fontsize=12)
        ax.set_xlabel('Predicted Phase (Hours)')
        ax.set_ylabel('Expression Level')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.legend(fontsize=8)
    
    for i in range(n_genes_to_compare, 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Gene Expression Comparison Across Cell Types', fontsize=16)
    plt.tight_layout()
    
    filename = 'gene_expression_celltype_comparison.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"细胞类型对比图已保存: {filepath}")

def plot_gene_expression_with_custom_data(model, test_loader, preprocessing_info, custom_gene_expressions, custom_gene_names, device='cuda', save_dir='./results'):
    """使用自定义基因数据绘制基因表达相位图"""
    print("\n=== 使用自定义基因数据绘制基因表达相位图 ===")
    model.eval()
    
    all_phase_hours = []
    all_celltypes = []
    
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            celltypes = batch.get('celltype', None)
            
            phase_coords, _ = model(expressions)
            phases = coords_to_phase(phase_coords)
            phase_hours = phases.cpu().numpy() * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
            
            all_phase_hours.append(phase_hours)
            
            if celltypes is not None:
                all_celltypes.extend(celltypes)
    
    phase_hours_data = np.concatenate(all_phase_hours)
    
    if all_celltypes:
        celltypes_data = np.array(all_celltypes)
        unique_celltypes = np.unique(celltypes_data)
        print(f"发现细胞类型: {unique_celltypes}")
    else:
        celltypes_data = None
        unique_celltypes = ['All_Samples']
    
    print(f"自定义基因: {custom_gene_names}")
    print(f"基因表达数据维度: {custom_gene_expressions.shape}")
    
    n_bins = 24
    phase_bins = np.linspace(0, 24, n_bins + 1)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    
    os.makedirs(save_dir, exist_ok=True)
    
    if celltypes_data is not None:
        for celltype in unique_celltypes:
            if celltype == 'PADDING':
                continue
                
            celltype_mask = celltypes_data == celltype
            celltype_expressions = custom_gene_expressions[celltype_mask]
            celltype_phases = phase_hours_data[celltype_mask]
            
            if len(celltype_expressions) < 10:
                continue
            
            plot_celltype_gene_expression(
                celltype_expressions, celltype_phases, custom_gene_names,
                phase_centers, phase_bins, celltype, save_dir
            )
    else:
        plot_celltype_gene_expression(
            custom_gene_expressions, phase_hours_data, custom_gene_names,
            phase_centers, phase_bins, 'All_Samples', save_dir
        )
    
    if celltypes_data is not None and len(unique_celltypes) > 1:
        valid_celltypes = [ct for ct in unique_celltypes if ct != 'PADDING']
        if len(valid_celltypes) > 1:
            plot_celltype_comparison(
                custom_gene_expressions, phase_hours_data, celltypes_data, custom_gene_names,
                phase_centers, phase_bins, valid_celltypes, save_dir
            )

def main():
    parser = argparse.ArgumentParser(description="训练相位自编码器模型")
    parser.add_argument("--train_file", required=True, help="训练数据文件路径")
    parser.add_argument("--test_file", default=None, help="测试数据文件路径（可选）")
    parser.add_argument("--n_components", type=int, default=50, help="选择的重要基因数量")
    parser.add_argument("--max_samples", type=int, default=100, help="最大样本数量（超过则截断，不足则填充）")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="重建损失权重")
    parser.add_argument("--lambda_time", type=float, default=0.5, help="时间监督损失权重")
    parser.add_argument("--lambda_sine", type=float, default=0.5, help="正弦先验损失权重")
    parser.add_argument("--period_hours", type=float, default=24.0, help="预期周期（小时）")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout比例")
    parser.add_argument("--device", default='cuda', help="设备 (cuda/cpu)")
    parser.add_argument("--save_dir", default='./phase_autoencoder_results', help="保存目录")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n_genes_plot", type=int, default=10, help="绘制的基因数量（当未指定custom_genes时使用）")
    parser.add_argument("--custom_genes", nargs='*', default=None, help="指定要绘制的基因列表，例如: --custom_genes GENE1 GENE2 GENE3")
    
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
    print(f"最大样本数量: {args.max_samples}")
    print(f"设备: {args.device}")
    
    if args.custom_genes:
        print(f"用户指定基因: {args.custom_genes}")
    else:
        print(f"将绘制前 {args.n_genes_plot} 个重要基因")
    
    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        args.train_file, args.n_components, args.max_samples, args.random_seed
    )
    
    preprocessing_info['period_hours'] = args.period_hours
    
    model = PhaseAutoEncoder(
        input_dim=args.n_components,
        dropout=args.dropout
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    train_losses = train_model(
        model=model,
        train_dataset=train_dataset,
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
        
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        _ = predict_and_save_phases(
            model=model,
            test_loader=test_loader,
            preprocessing_info=test_preprocessing_info,
            device=args.device,
            save_dir=args.save_dir
        )
        
        print(f"\n=== 绘制基因表达相位图 ===")
        
        if args.custom_genes:
            custom_gene_expressions, custom_gene_names = get_original_gene_expressions(
                args.test_file, args.custom_genes, test_preprocessing_info,
                None, None
            )
            
            if custom_gene_expressions is not None:
                plot_gene_expression_with_custom_data(
                    model=model,
                    test_loader=test_loader,
                    preprocessing_info=test_preprocessing_info,
                    custom_gene_expressions=custom_gene_expressions,
                    custom_gene_names=custom_gene_names,
                    device=args.device,
                    save_dir=args.save_dir
                )
            else:
                print("无法获取自定义基因数据，使用SVD选择的基因")
                plot_gene_expression_by_phase(
                    model=model,
                    test_loader=test_loader,
                    preprocessing_info=test_preprocessing_info,
                    device=args.device,
                    save_dir=args.save_dir,
                    n_genes_to_plot=args.n_genes_plot,
                    custom_genes=None
                )
        else:
            plot_gene_expression_by_phase(
                model=model,
                test_loader=test_loader,
                preprocessing_info=test_preprocessing_info,
                device=args.device,
                save_dir=args.save_dir,
                n_genes_to_plot=args.n_genes_plot,
                custom_genes=None
            )
        
        print(f"\n=== 测试完成 ===")
        print(f"主要输出文件:")
        print(f"  - 模型权重: {args.save_dir}/final_model.pth")
        print(f"  - 详细预测: {args.save_dir}/phase_predictions.csv")
        print(f"  - 简化预测: {args.save_dir}/phase_predictions_simple.csv")
        print(f"  - 训练曲线: {args.save_dir}/training_curves.png")
        print(f"  - 预测分析: {args.save_dir}/prediction_analysis.png")
        print(f"  - 基因表达相位图: {args.save_dir}/gene_expression_phase_*.png")
        print(f"  - 细胞类型对比图: {args.save_dir}/gene_expression_celltype_comparison.png")
    else:
        print(f"\n未提供测试文件，只完成训练阶段")
        print(f"主要输出文件:")
        print(f"  - 模型权重: {args.save_dir}/final_model.pth")
        print(f"  - 训练曲线: {args.save_dir}/training_curves.png")

if __name__ == "__main__":
    main()