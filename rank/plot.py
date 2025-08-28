import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import spearmanr, pearsonr
from typing import Tuple
from dataclasses import dataclass

CONFIG_SLUGS = {
    'Pure Smoothness': 'Pure_Smoothness',
    'Mostly Smooth': 'Mostly_Smooth',
    'Balanced': 'Balanced',
    'More Variation': 'More_Variation',
    'Equal Balance': 'Equal_Balance',
}
SLUG_TO_NAME = {v: k for k, v in CONFIG_SLUGS.items()}


def find_rank_csvs(root_dir: str):
    pattern = os.path.join(root_dir, '**', 'sample_ranks_*.csv')
    files = glob(pattern, recursive=True)
    return files

def parse_file_info(filepath: str):
    base = os.path.basename(filepath)
    name = os.path.splitext(base)[0]
    prefix = 'sample_ranks_'
    if not name.startswith(prefix):
        return None, None
    rest = name[len(prefix):]
    for slug in SLUG_TO_NAME.keys():
        suffix = f'_{slug}'
        if rest.endswith(suffix):
            celltype = rest[: -len(suffix)]
            config_slug = slug
            return celltype if celltype else None, config_slug
        if rest == slug:
            return None, slug
    return None, None


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

def sanitize_filename(s: str) -> str:
    if s is None:
        return 'ALL'
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)


def join_single_file(rank_csv: str, meta: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    celltype, slug = parse_file_info(rank_csv)
    df = pd.read_csv(rank_csv)
    if not {'Sample', 'Rank'}.issubset(df.columns):
        raise ValueError(f"Rank file missing required columns: {rank_csv}")
    joined = df.merge(meta[['study_sample', 'time_mod24']], left_on='Sample', right_on='study_sample', how='left')
    joined['Cell_Type'] = celltype if celltype is not None else 'ALL'
    joined['Config_Slug'] = slug if slug is not None else 'Unknown'
    return joined[['Sample', 'Rank', 'time_mod24', 'Cell_Type', 'Config_Slug']], celltype, slug


def map_rank_to_24(rank_series: pd.Series) -> np.ndarray:
    r = pd.to_numeric(rank_series, errors='coerce').values.astype(float)
    r = r[~np.isnan(r)] if isinstance(rank_series, pd.Series) else r
    rmin = np.nanmin(r) if r.size else 0.0
    rmax = np.nanmax(r) if r.size else 1.0
    span = max(rmax - rmin, 1e-9)
    x = (rank_series.astype(float) - rmin) / span * 24.0
    x = np.mod(x, 24.0)
    return x.values if isinstance(x, pd.Series) else x


@dataclass
class ShiftFitResult:
    slope: float
    shift: float
    r2: float
    corr: float
    n: int
    orientation: str


def compute_best_shift_no_intercept(x_base: np.ndarray, y: np.ndarray, step: float = 0.05) -> ShiftFitResult:
    def eval_orientation(x_in: np.ndarray, orient_name: str) -> ShiftFitResult:
        best = ShiftFitResult(slope=np.nan, shift=np.nan, r2=-np.inf, corr=-np.inf, n=len(y), orientation=orient_name)
        if len(x_in) == 0 or len(y) == 0:
            return best
        y_var = float(np.var(y))
        if y_var <= 1e-12:
            return best
        shifts = np.arange(0.0, 24.0, step)
        for s in shifts:
            x = (x_in + s) % 24.0
            try:
                corr = float(pearsonr(x, y)[0])
            except Exception:
                corr = -np.inf
            if not np.isfinite(corr):
                continue
            if corr > best.corr:
                xx = float(np.dot(x, x))
                if xx <= 1e-12:
                    continue
                a = float(np.dot(x, y) / xx)
                if a <= 0:
                    continue
                y_hat = a * x
                sst0 = float(np.sum(y ** 2))
                sse = float(np.sum((y - y_hat) ** 2))
                r2 = 1.0 - sse / sst0 if sst0 > 1e-12 else -np.inf
                best = ShiftFitResult(slope=a, shift=s, r2=r2, corr=corr, n=len(y), orientation=orient_name)
        return best

    best_normal = eval_orientation(x_base, 'normal')
    best_flipped = eval_orientation((24.0 - x_base) % 24.0, 'flipped')
    if best_normal.corr > best_flipped.corr:
        return best_normal
    if best_flipped.corr > best_normal.corr:
        return best_flipped
    return best_normal if best_normal.r2 >= best_flipped.r2 else best_flipped


def plot_single_file_shifted(joined_df: pd.DataFrame, celltype: str, slug: str, out_dir: str,
                             step: float = 0.05, make_plot: bool = True) -> Tuple[str, ShiftFitResult]:
    df = joined_df.copy()
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['Rank', 'time_mod24'])
    if df.empty:
        print(f"[WARN] No matched rows for {celltype} / {slug}")
        return '', ShiftFitResult(np.nan, np.nan, np.nan, np.nan, 0, 'normal')

    x0 = map_rank_to_24(df['Rank'])
    y = df['time_mod24'].astype(float).values
    res = compute_best_shift_no_intercept(x0, y, step=step)
    x_use = x0 if res.orientation == 'normal' else (24.0 - x0) % 24.0
    x_shift = (x_use + res.shift) % 24.0

    out_path = ''
    if make_plot:
        plt.figure(figsize=(7.5, 5.5))
        plt.scatter(x_shift, y, s=28, alpha=0.85, edgecolors='white', linewidths=0.4, color='tab:blue')
        try:
            sr_val = float(spearmanr(x_shift, y)[0])
        except Exception:
            sr_val = float('nan')

        info = (
            f"Best shift = {res.shift:.2f} h\n"
            f"Pearson r (max) = {res.corr:.3f}\n"
            f"Spearman ρ = {sr_val:.3f}\n"
            f"Slope (no intercept) = {res.slope:.3f}\n"
            f"R^2 (no intercept) = {res.r2:.3f}\n"
            f"N = {res.n}\n"
            f"Orientation = {res.orientation}"
        )
        plt.gca().text(0.02, 0.98, info, transform=plt.gca().transAxes,
                       va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

        plt.xlabel('Shifted Predicted Rank mapped to [0, 24) (hours)')
        plt.ylabel('Metadata Time (mod 24 h)')
        title = f"{SLUG_TO_NAME.get(slug, slug or 'Unknown')}: Shift-aligned (zero-intercept) — {celltype or 'ALL'}"
        plt.title(title)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        safe_cell = sanitize_filename(celltype)
        safe_slug = sanitize_filename(slug)
        out_path = os.path.join(out_dir, f"rank_vs_time_shifted_{safe_cell}_{safe_slug}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved shifted per-file plot: {out_path}")
    return out_path, res