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


def sanitize_filename(s: str) -> str:
    return ''.join(c if (c.isalnum() or c in ('-', '_')) else '_' for c in str(s))


def split_by_celltype(input_csv: str, celltype_key: str, out_root: str) -> dict:
    """Split expression matrix by cell type.
    Supports two formats:
      1) Row-based (CYCLOPS/my_cyclops expected): 'Gene_Symbol' first column, with a special row where Gene_Symbol == celltype_key.
         Samples are columns; the celltype row values assign each sample to a cell type. We will drop that row in outputs.
      2) Column-based (tabular samples as rows): a column named celltype_key used to group rows. We will drop that column in outputs.
    Returns mapping {cell_type: output_csv}.
    """
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


def run_cyclops_for_file(train_csv: str, save_dir: str, n_components: int, epochs: int, lr: float, device: str, genes_csv: str) -> int:
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
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description='Split expression.csv by celltype and run my_cyclops for each split')
    ap.add_argument('--dataset', help='Dataset name under my_cyclops/data/<dataset> (use this OR --input-csv)')
    ap.add_argument('--input-csv', help='Path to expression.csv to split (use this OR --dataset)')
    ap.add_argument('--celltype-col', default='celltype_D', help="Column name to split on (default: celltype_D)")
    ap.add_argument('--n-components', type=int, default=5, help='PCA components (default: 50)')
    ap.add_argument('--epochs', type=int, default=2000, help='Training epochs (default: 2000)')
    ap.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    ap.add_argument('--device', default='cuda', help='Device (default: cuda)')
    ap.add_argument('--genes', default=DEFAULT_GENES, help=f'Comma-separated gene list (default: {DEFAULT_GENES})')
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

    splits_root = args.splits_root or os.path.join(script_dir, 'data', f'{dataset_name}_splits_{timestamp}')
    results_root = args.out_root or os.path.join(script_dir, 'result', f'{dataset_name}_by_celltype_{timestamp}')

    os.makedirs(splits_root, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    print(f'Input CSV: {input_csv}')
    print(f'Splitting by column: {args.celltype_col}')
    print(f'Splits root: {splits_root}')
    print(f'Results root: {results_root}')

    groups = split_by_celltype(input_csv, args.celltype_col, splits_root)
    print(f'Total cell types: {len(groups)}')

    failures = []
    for ct, ct_csv in groups.items():
        ct_safe = sanitize_filename(ct)
        save_dir = os.path.join(results_root, ct_safe)
        os.makedirs(save_dir, exist_ok=True)
        ret = run_cyclops_for_file(ct_csv, save_dir, args.n_components, args.epochs, args.lr, args.device, args.genes)
        if ret != 0:
            print(f'[ERROR] Training failed for cell type {ct} (exit {ret})')
            failures.append(ct)
        else:
            print(f'[OK] Finished cell type {ct}')

    if failures:
        print('Failed cell types:', ', '.join(map(str, failures)))
    else:
        print('All cell types finished successfully.')

    if not args.keep_splits:
        try:
            shutil.rmtree(splits_root)
            print(f'Removed splits: {splits_root}')
        except Exception as e:
            print(f'[WARN] Could not remove splits folder: {e}')


if __name__ == '__main__':
    main()
