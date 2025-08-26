import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
from scipy.stats import pearsonr


def sanitize_filename(s: str) -> str:
    if s is None:
        return 'ALL'
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(s))


def auto_detect_metadata(base_dir: str) -> str:
    base_dir = os.path.abspath(base_dir)
    preferred = os.path.join(base_dir, 'data', 'zeitzeiger', 'metadata.csv')
    if os.path.isfile(preferred):
        return preferred
    matches = [str(p) for p in Path(base_dir).rglob('metadata.csv')]
    if not matches:
        return ''
    z = [m for m in matches if 'zeitzeiger' in m.replace('\\', '/').lower()]
    if z:
        return z[0]
    return sorted(matches, key=lambda s: len(s))[0]


def load_metadata(metadata_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    required = {'study', 'sample', 'time'}
    if not required.issubset(meta.columns):
        raise ValueError(f"metadata must contain columns: {required}")
    meta['study'] = meta['study'].astype(str)
    meta['sample'] = meta['sample'].astype(str)
    meta['study_sample'] = meta['study'] + '_' + meta['sample']
    meta['time'] = pd.to_numeric(meta['time'], errors='coerce')
    meta['time_mod24'] = meta['time'] % 24
    return meta


def load_predictions(pred_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    # Infer study_sample
    if 'study_sample' in df.columns:
        df['study_sample'] = df['study_sample'].astype(str)
    elif {'study', 'sample'}.issubset(df.columns):
        df['study'] = df['study'].astype(str)
        df['sample'] = df['sample'].astype(str)
        df['study_sample'] = df['study'] + '_' + df['sample']
    elif 'Sample' in df.columns:
        df['study_sample'] = df['Sample'].astype(str)
    elif 'Sample_ID' in df.columns:
        # my_cyclops phase_predictions_simple.csv uses Sample_ID
        df['study_sample'] = df['Sample_ID'].astype(str)
    elif 'sample' in df.columns:
        df['study_sample'] = df['sample'].astype(str)
    else:
        raise ValueError(f'{pred_csv}: cannot infer study_sample (expected study+sample, Sample, or study_sample)')

    # Infer predicted phase
    # Prefer hours if present; else convert from radians/degrees; else fall back to common names
    if 'Predicted_Phase_Hours' in df.columns:
        pred_hours = pd.to_numeric(df['Predicted_Phase_Hours'], errors='coerce')
    elif 'Predicted_Phase_Radians' in df.columns:
        rad = pd.to_numeric(df['Predicted_Phase_Radians'], errors='coerce')
        pred_hours = rad * 24.0 / (2 * np.pi)
    elif 'Predicted_Phase_Degrees' in df.columns:
        deg = pd.to_numeric(df['Predicted_Phase_Degrees'], errors='coerce')
        pred_hours = deg * (24.0 / 360.0)
    else:
        phase_candidates = [
            'phase', 'Phase', 'phase_hours', 'pred_phase', 'predicted_phase',
            'theta', 'Theta', 'predicted_time', 'pred_time', 'phase_prediction',
        ]
        phase_col = None
        for c in phase_candidates:
            if c in df.columns:
                phase_col = c
                break
        if phase_col is None:
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if num_cols:
                phase_col = num_cols[-1]
            else:
                raise ValueError(f'{pred_csv}: cannot locate predicted phase column')
        pred_hours = pd.to_numeric(df[phase_col], errors='coerce')

    df['pred_phase'] = pred_hours % 24
    return df[['study_sample', 'pred_phase']]


def compute_metrics(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    try:
        r = float(pearsonr(x, y)[0])
    except Exception:
        r = np.nan
    r2 = float(r * r) if np.isfinite(r) else np.nan
    return r, r2


def best_align_phase(x_hours: np.ndarray, y_hours: np.ndarray, step: float = 0.1) -> Tuple[np.ndarray, float, float, float, bool]:
    """
    Search circular shifts in [0,24) and optional flip (x -> 24 - x) to maximize Pearson r.
    Returns: (aligned_x, best_r, best_r2, best_shift, flipped)
    """
    x_hours = np.asarray(x_hours, dtype=float)
    y_hours = np.asarray(y_hours, dtype=float)
    shifts = np.arange(0.0, 24.0, step, dtype=float)

    best_r = -np.inf
    best = {
        'aligned': x_hours,
        'r': np.nan,
        'r2': np.nan,
        'shift': 0.0,
        'flipped': False,
    }

    for flipped in (False, True):
        x0 = (24.0 - x_hours) % 24.0 if flipped else x_hours
        for s in shifts:
            xs = (x0 + s) % 24.0
            try:
                r = float(pearsonr(xs, y_hours)[0])
            except Exception:
                r = np.nan
            if not np.isfinite(r):
                continue
            # Maximize positive correlation after allowing flip
            if r > best_r:
                best_r = r
                best.update(aligned=xs, r=r, r2=r*r, shift=float(s), flipped=flipped)

    return best['aligned'], best['r'], best['r2'], best['shift'], best['flipped']


def plot_one(pred_csv: str, celltype: str, meta: pd.DataFrame, out_dir: str) -> Optional[Tuple[str, float, float]]:
    preds = load_predictions(pred_csv)
    joined = preds.merge(meta[['study_sample', 'time_mod24']], on='study_sample', how='left')
    joined = joined.dropna(subset=['pred_phase', 'time_mod24'])
    # Fallback: if no matches, try matching predictions' identifier to metadata.sample
    if joined.empty and 'sample' in meta.columns:
        joined_alt = preds.merge(meta[['sample', 'time_mod24']], left_on='study_sample', right_on='sample', how='left')
        joined_alt = joined_alt.dropna(subset=['pred_phase', 'time_mod24'])
        if not joined_alt.empty:
            joined = joined_alt[['pred_phase', 'time_mod24']]
    if joined.empty:
        print(f"[WARN] No matches for {celltype} ({pred_csv})")
        return None

    x_raw = joined['pred_phase'].astype(float).values
    y = joined['time_mod24'].astype(float).values
    # Force alignment by cyclic shift and optional flip to maximize Pearson r
    x_aligned, r, r2, best_shift, flipped = best_align_phase(x_raw, y, step=0.1)

    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(x_aligned, y, s=28, alpha=0.85, edgecolors='white', linewidths=0.4, color='tab:blue')
    plt.xlabel('Predicted Phase (hours)')
    plt.ylabel('Metadata Time (mod 24 h)')
    plt.title(f'Phase vs Metadata — {celltype}')
    info = (
        f"Pearson r = {r:.3f}\nR^2 = {r2:.3f}\n"
        f"Shift = {best_shift:.2f} h\nFlip = {'Yes' if flipped else 'No'}\nN = {len(x_aligned)}"
    )
    plt.gca().text(0.02, 0.98, info, transform=plt.gca().transAxes,
                   va='top', ha='left', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))
    plt.xlim(0, 24)
    plt.ylim(0, 24)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe_ct = sanitize_filename(celltype)
    out_path = os.path.join(out_dir, f'phase_vs_time_{safe_ct}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
    return out_path, r, r2


def find_prediction_files(results_root: str, filename: str = 'phase_predictions_simple.csv') -> List[Tuple[str, str]]:
    paths = []
    for p in Path(results_root).glob(f'*/*{filename}'):
        # The above glob might not match correctly; use rglob
        pass
    for p in Path(results_root).rglob(filename):
        # Infer cell type from parent folder name
        celltype = p.parent.name
        paths.append((str(p), celltype))
    return paths


def main():
    ap = argparse.ArgumentParser(description='Batch compare per-celltype phase predictions vs metadata; one plot per cell type')
    ap.add_argument('--results_root', required=True, help='Root folder containing <celltype>/phase_predictions_simple.csv')
    ap.add_argument('--metadata', default=None, help='Path to metadata.csv; default: auto-detect')
    ap.add_argument('--out_dir', default=None, help='Output folder; default: <results-root>/phase_vs_metadata')
    ap.add_argument('--filename', default='phase_predictions_simple.csv', help='Predictions filename (default: phase_predictions_simple.csv)')
    args = ap.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    metadata_csv = args.metadata or auto_detect_metadata(base_dir)
    if not metadata_csv:
        raise SystemExit('Could not auto-detect metadata.csv; please pass --metadata')
    meta = load_metadata(metadata_csv)

    pairs = find_prediction_files(args.results_root, args.filename)
    if not pairs:
        raise SystemExit(f'No {args.filename} found under {args.results_root}')

    out_dir = args.out_dir or os.path.join(args.results_root, 'phase_vs_metadata')
    print(f'Using metadata: {metadata_csv}')
    print(f'Output directory: {out_dir}')
    print(f'Found {len(pairs)} prediction files')

    rows = []
    for pred_csv, celltype in pairs:
        try:
            res = plot_one(pred_csv, celltype, meta, out_dir)
            if res is not None:
                _, r, r2 = res
                # Write exactly as requested: celltype, R_spuare, Pearson_R
                rows.append({'celltype': celltype, 'R_spuare': r2, 'Pearson_R': r})
        except Exception as e:
            print(f'[ERROR] {celltype}: {e}')

    # Save summary table
    if rows:
        summary_df = pd.DataFrame(rows)
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, 'metrics.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'Saved metrics table: {summary_path}')


if __name__ == '__main__':
    main()
