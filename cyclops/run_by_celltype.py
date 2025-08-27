import os
import sys
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_GENES = "PER1,PER2,CRY1,CRY2,CLOCK,ARNTL,NR1D1,NR1D2,DBP"


def find_expression_file(data_dir: str) -> str:
    cand = [
        os.path.join(data_dir, 'expression.csv'),
        os.path.join(data_dir, 'filtered_expression.csv'),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No expression.csv or filtered_expression.csv under {data_dir}")


def find_metadata_file(base_dir: str, dataset_name: str, specified_path: str = None) -> str:
    """Find metadata.csv file with auto-detection fallback"""
    if specified_path:
        if os.path.isfile(specified_path):
            return specified_path
        else:
            print(f"Warning: Specified metadata file not found: {specified_path}")
    
    # Auto-detect metadata
    candidates = [
        # First priority: zeitzeiger dataset metadata
        os.path.join(base_dir, '..', 'data', 'zeitzeiger', 'metadata.csv'),
        os.path.join(base_dir, 'data', 'zeitzeiger', 'metadata.csv'),
        # Second priority: current dataset directory
        os.path.join(base_dir, '..', 'data', dataset_name, 'metadata.csv'),
        os.path.join(base_dir, 'data', dataset_name, 'metadata.csv'),
        # Third priority: search in parent directories
        os.path.join(base_dir, '..', 'data', 'zeitzeiger', 'metadata.csv'),
        os.path.join(base_dir, '..', '..', 'data', 'zeitzeiger', 'metadata.csv'),
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if abs_path not in seen:
            seen.add(abs_path)
            unique_candidates.append(candidate)
    
    for candidate in unique_candidates:
        if os.path.isfile(candidate):
            print(f"Auto-detected metadata: {candidate}")
            return candidate
    
    # If still not found, do a recursive search for any metadata.csv
    print("Searching recursively for metadata.csv...")
    search_roots = [
        os.path.join(base_dir, '..', 'data'),
        os.path.join(base_dir, 'data'),
        os.path.join(base_dir, '..'),
    ]
    
    for search_root in search_roots:
        if os.path.isdir(search_root):
            for root, dirs, files in os.walk(search_root):
                if 'metadata.csv' in files:
                    found_path = os.path.join(root, 'metadata.csv')
                    print(f"Found metadata.csv at: {found_path}")
                    return found_path
    
    print("No metadata.csv found, phase-vs-metadata comparison will be skipped")
    return None


def sanitize_filename(s: str) -> str:
    return ''.join(c if (c.isalnum() or c in ('-', '_')) else '_' for c in str(s))


def split_by_celltype(input_csv: str, celltype_key: str, out_root: str) -> dict:
    df = pd.read_csv(input_csv, low_memory=False)
    groups = {}

    if 'Gene_Symbol' in df.columns and (df['Gene_Symbol'] == celltype_key).any():
        # Row-based format
        sample_cols = [c for c in df.columns if c != 'Gene_Symbol']
        ct_row = df[df['Gene_Symbol'] == celltype_key].iloc[0]
        # Build mapping sample_col -> celltype
        sample_to_ct = {col: str(ct_row[col]) for col in sample_cols}
        # For each unique celltype, select the corresponding sample columns and drop the celltype row
        # Keep time_C row if present; my_cyclops can use it.
        for ct in sorted(set(sample_to_ct.values())):
            cols_for_ct = [col for col, val in sample_to_ct.items() if val == ct]
            if not cols_for_ct:
                continue
            sub = df[df['Gene_Symbol'] != celltype_key].copy()
            sub = sub[['Gene_Symbol'] + cols_for_ct]
            ct_safe = sanitize_filename(ct)
            ct_dir = os.path.join(out_root, ct_safe)
            os.makedirs(ct_dir, exist_ok=True)
            out_csv = os.path.join(ct_dir, 'expression.csv')
            sub.to_csv(out_csv, index=False)
            groups[ct] = out_csv
        return groups

    # Fallback: column-based split
    if celltype_key not in df.columns:
        raise ValueError(
            f"Neither a celltype row (Gene_Symbol == '{celltype_key}') nor a column '{celltype_key}' was found in {input_csv}"
        )
    for ct, sub in df.groupby(celltype_key):
        sub = sub.copy()
        sub = sub.drop(columns=[celltype_key])
        ct_safe = sanitize_filename(ct)
        ct_dir = os.path.join(out_root, ct_safe)
        os.makedirs(ct_dir, exist_ok=True)
        out_csv = os.path.join(ct_dir, 'expression.csv')
        sub.to_csv(out_csv, index=False)
        groups[ct] = out_csv
    return groups


def run_cyclops_for_file(train_csv: str, save_dir: str, n_components: int, epochs: int, lr: float, device: str, genes_csv: str, metadata_file: str = None) -> int:
    # Convert comma-separated genes to argument list
    genes_list = [g.strip() for g in genes_csv.split(',') if g.strip()]
    cmd = [
        sys.executable, 'my_cyclops.py',
        '--train_file', train_csv,
        '--test_file', train_csv,
        '--n_components', str(n_components),
        '--num_epochs', str(epochs),
        '--lr', str(lr),
        '--device', device,
        '--save_dir', save_dir,
        '--custom_genes', *genes_list,
    ]
    
    # Add metadata if available
    if metadata_file and os.path.isfile(metadata_file):
        cmd.extend(['--metadata', metadata_file])
    
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def collect_all_metrics(results_root: str, output_file: str):
    """收集所有细胞类型的 metrics.csv 并合并为一个文件"""
    print(f"\n=== 收集所有 metrics 文件 ===")
    
    all_metrics = []
    metrics_files = []
    
    # 搜索所有 metrics.csv 文件
    for root, dirs, files in os.walk(results_root):
        if 'metrics.csv' in files:
            metrics_path = os.path.join(root, 'metrics.csv')
            metrics_files.append(metrics_path)
    
    print(f"找到 {len(metrics_files)} 个 metrics.csv 文件")
    
    for metrics_path in metrics_files:
        try:
            df = pd.read_csv(metrics_path)
            if not df.empty:
                # 从路径中提取真正的细胞类型名称
                path_parts = os.path.normpath(metrics_path).split(os.sep)
                # 查找包含 metrics.csv 的文件夹的上级文件夹名（应该是细胞类型）
                for i, part in enumerate(path_parts):
                    if part == 'phase_vs_metadata' and i > 0:
                        celltype_folder = path_parts[i-1]
                        # 更新 celltype 列
                        df['celltype'] = celltype_folder
                        break
                
                all_metrics.append(df)
                print(f"  加载: {metrics_path} ({len(df)} 行) - celltype: {df['celltype'].iloc[0] if not df.empty else 'Unknown'}")
        except Exception as e:
            print(f"  [WARN] 无法读取 {metrics_path}: {e}")
    
    if all_metrics:
        # 合并所有 metrics
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        
        # 去重（基于 celltype）
        combined_metrics = combined_metrics.drop_duplicates(subset=['celltype'], keep='first')
        
        # 排序
        combined_metrics = combined_metrics.sort_values('celltype')
        
        # 保存合并后的文件
        combined_metrics.to_csv(output_file, index=False)
        print(f"合并的 metrics 保存到: {output_file}")
        print(f"总共包含 {len(combined_metrics)} 个细胞类型")
        
        # 显示统计信息
        print("\n=== 所有细胞类型的 metrics 汇总 ===")
        print(combined_metrics.to_string(index=False))
        
        return combined_metrics
    else:
        print("没有找到有效的 metrics 数据")
        return None

def main():
    ap = argparse.ArgumentParser(description='Split expression.csv by celltype and run my_cyclops for each split')
    ap.add_argument('--dataset', help='Dataset name under my_cyclops/data/<dataset> (use this OR --input-csv)')
    ap.add_argument('--input-csv', help='Path to expression.csv to split (use this OR --dataset)')
    ap.add_argument('--celltype-col', default='celltype_D', help="Column name to split on (default: celltype_D)")
    ap.add_argument('--n-components', type=int, default=5, help='PCA components (default: 5)')
    ap.add_argument('--epochs', type=int, default=2000, help='Training epochs (default: 2000)')
    ap.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    ap.add_argument('--device', default='cuda', help='Device (default: cuda)')
    ap.add_argument('--genes', default=DEFAULT_GENES, help=f'Comma-separated gene list (default: {DEFAULT_GENES})')
    ap.add_argument('--metadata', default=None, help='Path to metadata.csv (optional, auto-detect if not specified)')
    ap.add_argument('--out-root', default=None, help='Output root for results; default: my_cyclops/result/<dataset>_by_celltype_<timestamp>')
    ap.add_argument('--splits-root', default=None, help='Where to write split CSVs; default: my_cyclops/data/<dataset>_splits_<timestamp>')
    ap.add_argument('--keep-splits', action='store_true', help='Keep split CSVs after run (default: delete)')

    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not args.dataset and not args.input_csv:
        ap.error('Please provide --dataset or --input-csv')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.input_csv:
        dataset_name = Path(args.input_csv).stem
        input_csv = os.path.abspath(args.input_csv)
    else:
        dataset_name = args.dataset
        data_dir = os.path.join(script_dir, '..', 'data', dataset_name)
        input_csv = find_expression_file(data_dir)

    # Find metadata file
    metadata_file = find_metadata_file(script_dir, dataset_name, args.metadata)

    splits_root = args.splits_root or os.path.join(script_dir, 'data', f'{dataset_name}_splits_{timestamp}')
    results_root = args.out_root or os.path.join(script_dir, 'result', f'{dataset_name}_by_celltype_{timestamp}')

    os.makedirs(splits_root, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    print(f'Input CSV: {input_csv}')
    print(f'Splitting by column: {args.celltype_col}')
    print(f'Splits root: {splits_root}')
    print(f'Results root: {results_root}')
    if metadata_file:
        print(f'Metadata file: {metadata_file}')

    groups = split_by_celltype(input_csv, args.celltype_col, splits_root)
    print(f'Total cell types: {len(groups)}')

    failures = []
    for ct, ct_csv in groups.items():
        ct_safe = sanitize_filename(ct)
        save_dir = os.path.join(results_root, ct_safe)
        os.makedirs(save_dir, exist_ok=True)
        ret = run_cyclops_for_file(ct_csv, save_dir, args.n_components, args.epochs, args.lr, args.device, args.genes, metadata_file)
        if ret != 0:
            print(f'[ERROR] Training failed for cell type {ct} (exit {ret})')
            failures.append(ct)
        else:
            print(f'[OK] Finished cell type {ct}')

    if failures:
        print('Failed cell types:', ', '.join(map(str, failures)))
    else:
        print('All cell types finished successfully.')
        
        # If metadata was used, print summary of where phase-vs-metadata plots are located
        if metadata_file:
            print(f'\nPhase-vs-metadata comparison plots for each cell type saved in:')
            for ct in groups.keys():
                ct_safe = sanitize_filename(ct)
                save_dir = os.path.join(results_root, ct_safe)
                print(f'  {ct}: {save_dir}/phase_vs_metadata/')
            
            # 收集并合并所有 metrics.csv 文件
            combined_metrics_file = os.path.join(results_root, 'combined_metrics.csv')
            combined_metrics = collect_all_metrics(results_root, combined_metrics_file)
            
            if combined_metrics is not None:
                print(f"\n=== 最终 Metrics 汇总 ===")
                print(f"文件位置: {combined_metrics_file}")
                print("内容预览:")
                print(combined_metrics.head(10).to_string(index=False))
                
                # 计算一些统计信息
                if len(combined_metrics) > 0:
                    avg_pearson = combined_metrics['Pearson_R'].mean()
                    avg_r2 = combined_metrics['R_spuare'].mean()
                    avg_spearman = combined_metrics['Spearman_R'].mean()
                    print(f"\n平均指标:")
                    print(f"  Pearson R: {avg_pearson:.3f}")
                    print(f"  R²: {avg_r2:.3f}")
                    print(f"  Spearman ρ: {avg_spearman:.3f}")
                    
                    # 找出表现最好的细胞类型
                    best_pearson = combined_metrics.loc[combined_metrics['Pearson_R'].idxmax()]
                    best_r2 = combined_metrics.loc[combined_metrics['R_spuare'].idxmax()]
                    best_spearman = combined_metrics.loc[combined_metrics['Spearman_R'].idxmax()]
                    
                    print(f"\n表现最佳的细胞类型:")
                    print(f"  Pearson R: {best_pearson['celltype']} ({best_pearson['Pearson_R']:.3f})")
                    print(f"  R²: {best_r2['celltype']} ({best_r2['R_spuare']:.3f})")
                    print(f"  Spearman ρ: {best_spearman['celltype']} ({best_spearman['Spearman_R']:.3f})")

    if not args.keep_splits:
        try:
            shutil.rmtree(splits_root)
            print(f'Removed splits: {splits_root}')
        except Exception as e:
            print(f'[WARN] Could not remove splits folder: {e}')


if __name__ == '__main__':
    main()
