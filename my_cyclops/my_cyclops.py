import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from AE import PhaseAutoEncoder
from utils import (
    time_to_phase, coords_to_phase, predict_and_save_phases,
    plot_eigengenes_2d_with_phase_gradient, plot_gene_expression_with_custom_data,
    get_original_gene_expressions
)
from dataset import (
    load_and_preprocess_train_data, load_and_preprocess_test_data
)

def time_supervision_loss(phase_coords, true_times, lambda_time=1.0, period_hours=24.0):
    if true_times is None:
        return torch.tensor(0.0, device=phase_coords.device)
    
    true_phases = time_to_phase(true_times, period_hours)
    
    pred_phases = coords_to_phase(phase_coords)
    
    phase_diff = torch.abs(pred_phases - true_phases)
    phase_diff = torch.min(phase_diff, 2*np.pi - phase_diff)
    
    return lambda_time * torch.mean(phase_diff)


def train_model(model, train_dataset, preprocessing_info, 
                num_epochs=100, lr=0.001, device='cuda',
                lambda_recon=1.0, lambda_time=0.5,
                period_hours=24.0, save_dir='./model_checkpoints'):
    
    if 'train_has_time' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_time'] = 'time' in sample
    
    if 'train_has_celltype' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_celltype'] = 'celltype' in sample
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
    
    recon_criterion = nn.MSELoss()
    
    train_losses = []
    
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
    
    celltype_indices_tensor = None
    if preprocessing_info['train_has_celltype'] and celltypes_array is not None:
        celltype_to_idx = preprocessing_info['celltype_to_idx']
        celltype_indices = []
        for ct in celltypes_array:
            if ct == 'PADDING':
                celltype_indices.append(0)
            else:
                celltype_indices.append(celltype_to_idx.get(ct, 0))
        celltype_indices_tensor = torch.tensor(celltype_indices, device=device)
    
    print(f"训练数据准备完成:")
    print(f"  - 总样本数: {len(expressions_tensor)}")
    print(f"  - 有效样本数: {valid_mask_tensor.sum().item()}")
    if celltype_indices_tensor is not None:
        print(f"  - 细胞类型数量: {preprocessing_info['n_celltypes']}")
    
    print(f"\n=== 开始端到端联合训练 ({num_epochs} epochs) ===")
    
    with tqdm(total=num_epochs, desc="Joint Training Progress") as pbar:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            phase_coords, reconstructed = model(expressions_tensor, celltype_indices_tensor)
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

            total_loss = lambda_recon * recon_loss + lambda_time * time_loss
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'Train loss': f'{total_loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'Time': f'{time_loss.item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            pbar.update(1)
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'preprocessing_info': preprocessing_info,
        'train_losses': train_losses
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    
    return train_losses


def main():
    parser = argparse.ArgumentParser(description="训练相位自编码器模型")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_time", type=float, default=0.5)
    parser.add_argument("--lambda_sine", type=float, default=0.5)
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--save_dir", default='./phase_autoencoder_results')
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_genes_plot", type=int, default=10)
    parser.add_argument("--custom_genes", nargs='*', default=None, required=True)
    parser.add_argument("--sine_predictor_hidden", type=int, default=64)
    parser.add_argument("--metadata", default=None)

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

    df_tmp = pd.read_csv(args.train_file, low_memory=False)
    sample_columns_tmp = [col for col in df_tmp.columns if col != 'Gene_Symbol']
    max_samples = len(sample_columns_tmp)
    print(f"自动设置最大样本数量为训练数据样本数: {max_samples}")

    print(f"设备: {args.device}")
    
    if args.custom_genes:
        print(f"用户指定基因: {args.custom_genes}")
    else:
        print(f"将绘制前 {args.n_genes_plot} 个重要基因")
    
    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        args.train_file, args.n_components, max_samples, args.random_seed
    )
    
    preprocessing_info['period_hours'] = args.period_hours
    
    model = PhaseAutoEncoder(
        input_dim=args.n_components,
        n_celltypes=preprocessing_info.get('n_celltypes', 0),
        dropout=args.dropout
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    if preprocessing_info.get('n_celltypes', 0) > 0:
        print(f"细胞类型数量: {preprocessing_info['n_celltypes']}")
    
    print("使用神经网络正弦损失进行端到端训练...")
    train_losses = train_model(
        model=model,
        train_dataset=train_dataset,
        preprocessing_info=preprocessing_info,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        lambda_recon=args.lambda_recon,
        lambda_time=args.lambda_time,
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
        
        results_df = predict_and_save_phases(
            model=model,
            test_loader=test_loader,
            preprocessing_info=test_preprocessing_info,
            device=args.device,
            save_dir=args.save_dir
        )

        if getattr(args, 'metadata', None):
            if os.path.isfile(args.metadata):
                from utils import generate_phase_metadata_comparison
                generate_phase_metadata_comparison(results_df, args.metadata, args.save_dir)
            else:
                print(f"[WARN] 指定的 metadata 文件不存在: {args.metadata}")

        print(f"\n=== 绘制Eigengenes 2D关系图（相位渐变色）===")
        predicted_phases = results_df['Predicted_Phase_Radians'].values
        celltypes_data = results_df['Cell_Type'].values if 'Cell_Type' in results_df.columns else None
        
        plot_eigengenes_2d_with_phase_gradient(
            test_file=args.test_file,
            preprocessing_info=test_preprocessing_info,
            predicted_phases=predicted_phases,
            celltypes_data=celltypes_data,
            save_dir=args.save_dir
        )

        print(f"\n=== 绘制基因表达相位图 ===")
        
        custom_gene_expressions, custom_gene_names = get_original_gene_expressions(
            args.test_file, args.custom_genes, test_preprocessing_info,
            None, None
        )
        plot_gene_expression_with_custom_data(
            model=model,
            test_loader=test_loader,
            preprocessing_info=test_preprocessing_info,
            custom_gene_expressions=custom_gene_expressions,
            custom_gene_names=custom_gene_names,
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
        print(f"  - 基因表达相位图: {args.save_dir}/gene_expression_phase_*.png")
        print(f"  - 细胞类型对比图: {args.save_dir}/gene_expression_celltype_comparison.png")
    else:
        print(f"\n未提供测试文件，只完成训练阶段")
        print(f"主要输出文件:")
        print(f"  - 模型权重: {args.save_dir}/final_model.pth")
        print(f"  - 训练曲线: {args.save_dir}/training_curves.png")

if __name__ == "__main__":
    main()