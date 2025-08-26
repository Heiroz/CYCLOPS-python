import argparse
import os
import re
import pandas as pd


def norm_celltype(s: str) -> str:
    if pd.isna(s):
        return ''
    s = str(s)
    s = s.replace('_', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()


def load_metrics(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: celltype, R_spuare, Pearson_R
    missing = [c for c in ['celltype', 'R_spuare', 'Pearson_R'] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    df = df.copy()
    df['ct_norm'] = df['celltype'].apply(norm_celltype)
    # Rename metric columns with label suffix to disambiguate
    df = df.rename(columns={
        'R_spuare': f'R_spuare_{label}',
        'Pearson_R': f'Pearson_R_{label}',
        'celltype': f'celltype_{label}',
    })
    return df


def main():
    ap = argparse.ArgumentParser(description='Merge two metrics.csv files by cell type (normalized).')
    ap.add_argument('--left', required=True, help='Path to first metrics.csv')
    ap.add_argument('--right', required=True, help='Path to second metrics.csv')
    ap.add_argument('--left-label', default='left', help='Label suffix for left metrics (default: left)')
    ap.add_argument('--right-label', default='right', help='Label suffix for right metrics (default: right)')
    ap.add_argument('--out', default=None, help='Output CSV path (default: alongside --left as merged_metrics.csv)')
    args = ap.parse_args()

    left = load_metrics(args.left, args.left_label)
    right = load_metrics(args.right, args.right_label)

    merged = pd.merge(left, right, on='ct_norm', how='outer')

    # Choose a display celltype: prefer left, then right; if both exist and differ, keep left
    def choose_name(row):
        a = row.get(f'celltype_{args.left_label}')
        b = row.get(f'celltype_{args.right_label}')
        return a if pd.notna(a) and str(a).strip() else b

    merged['celltype'] = merged.apply(choose_name, axis=1)

    # Select and order columns
    cols = ['celltype',
            f'R_spuare_{args.left_label}', f'Pearson_R_{args.left_label}',
            f'R_spuare_{args.right_label}', f'Pearson_R_{args.right_label}']
    # Ensure missing columns exist
    for c in cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    merged_out = merged[cols].copy()
    merged_out = merged_out.sort_values('celltype', key=lambda s: s.str.lower() if s.dtype == 'object' else s)

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.left)), 'merged_metrics.csv')
    merged_out.to_csv(out_path, index=False)
    print(f'Saved merged metrics: {out_path}')


if __name__ == '__main__':
    main()
