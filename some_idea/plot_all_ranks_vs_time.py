import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path

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


def auto_detect_ranks_dir(base_dir: str) -> str:
    """Try to pick a sensible default ranks root.
    Strategy:
      1) If base_dir already contains sample_ranks_*.csv (recursively), use base_dir.
      2) Else, scan subfolders like result_* and choose the one with most rank CSVs; tie-break by latest mtime.
      3) If still none, return '' to indicate not found.
    """
    base_dir = os.path.abspath(base_dir)
    found_here = find_rank_csvs(base_dir)
    if found_here:
        return base_dir
    candidates = []
    for p in Path(base_dir).glob('result_*'):
        if p.is_dir():
            files = find_rank_csvs(str(p))
            if files:
                latest_mtime = max(os.path.getmtime(f) for f in files)
                candidates.append((len(files), latest_mtime, str(p)))
    if candidates:
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return candidates[0][2]
    return ''


def auto_detect_metadata(base_dir: str) -> str:
    """Pick metadata default:
      1) Prefer data/zeitzeiger/metadata.csv if exists under base_dir.
      2) Else, search recursively for a file named metadata.csv (prefer paths containing 'zeitzeiger').
    """
    base_dir = os.path.abspath(base_dir)
    preferred = os.path.join(base_dir, 'data', 'zeitzeiger', 'metadata.csv')
    if os.path.isfile(preferred):
        return preferred
    matches = [str(p) for p in Path(base_dir).rglob('metadata.csv')]
    if not matches:
        return ''
    # Prefer paths containing 'zeitzeiger', else shortest path
    z = [m for m in matches if 'zeitzeiger' in m.replace('\\', '/').lower()]
    if z:
        return z[0]
    return sorted(matches, key=lambda s: len(s))[0]


def parse_file_info(filepath: str):
    base = os.path.basename(filepath)
    name = os.path.splitext(base)[0]
    # name is like: sample_ranks_<celltype>_<slug> OR sample_ranks_<slug>
    prefix = 'sample_ranks_'
    if not name.startswith(prefix):
        return None, None
    rest = name[len(prefix):]
    # Try to detect which config slug suffix matches
    for slug in SLUG_TO_NAME.keys():
        suffix = f'_{slug}'
        if rest.endswith(suffix):
            celltype = rest[: -len(suffix)]  # may be empty if no celltype
            config_slug = slug
            return celltype if celltype else None, config_slug
        if rest == slug:
            return None, slug
    # Fallback: cannot parse config
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


def aggregate_by_config(rank_files: list, meta: pd.DataFrame):
    data_by_config = defaultdict(list)
    stats = defaultdict(lambda: {'total': 0, 'matched': 0})

    for fp in rank_files:
        celltype, slug = parse_file_info(fp)
        if slug is None:
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if not {'Sample', 'Rank'}.issubset(df.columns):
            continue
        # Merge with metadata
        joined = df.merge(meta[['study_sample', 'time_mod24']], left_on='Sample', right_on='study_sample', how='left')
        matched = joined['time_mod24'].notna().sum()
        total = len(joined)
        stats[slug]['total'] += total
        stats[slug]['matched'] += matched
        joined['Cell_Type'] = celltype if celltype is not None else 'ALL'
        joined['Config_Slug'] = slug
        data_by_config[slug].append(joined[['Sample', 'Rank', 'time_mod24', 'Cell_Type', 'Config_Slug']])

    combined = {slug: (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['Sample','Rank','time_mod24','Cell_Type','Config_Slug']))
                for slug, dfs in data_by_config.items()}
    return combined, stats


def plot_config(df: pd.DataFrame, slug: str, out_dir: str):
    if df.empty:
        print(f"[WARN] No data for config {slug}")
        return None
    df = df.copy()
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['Rank', 'time_mod24'])
    df['Rank'] = df['Rank'].astype(int)

    # Color by cell type
    celltypes = sorted(df['Cell_Type'].fillna('NA').unique().tolist())
    n_ct = len(celltypes)
    cmap = plt.get_cmap('tab20')
    colors = {ct: cmap(i % 20) for i, ct in enumerate(celltypes)}

    plt.figure(figsize=(9, 6))
    for ct in celltypes:
        sub = df[df['Cell_Type'] == ct]
        plt.scatter(sub['Rank'], sub['time_mod24'], s=24, alpha=0.75, edgecolors='white', linewidths=0.4, color=colors[ct], label=ct)

    # Correlations
    try:
        pr, pp = pearsonr(df['Rank'], df['time_mod24'])
        sr, sp = spearmanr(df['Rank'], df['time_mod24'])
        corr_txt = f"Pearson r={pr:.2f} (p={pp:.2g})\nSpearman ρ={sr:.2f} (p={sp:.2g})"
    except Exception:
        corr_txt = None

    plt.xlabel('Predicted Rank')
    plt.ylabel('Metadata Time (mod 24 h)')
    title = f"{SLUG_TO_NAME.get(slug, slug)}: Rank vs Time (All Cell Types)"
    plt.title(title)

    if corr_txt:
        plt.gca().text(0.02, 0.98, corr_txt, transform=plt.gca().transAxes,
                       va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    # Legend handling
    if n_ct <= 15:
        plt.legend(loc='best', fontsize=8, ncol=1, framealpha=0.8)
    else:
        plt.legend(loc='upper center', fontsize=7, ncol=3, bbox_to_anchor=(0.5, -0.15), framealpha=0.8)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rank_vs_time_{slug}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def sanitize_filename(s: str) -> str:
    if s is None:
        return 'ALL'
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)


def join_single_file(rank_csv: str, meta: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """Join one rank csv with metadata; return df and parsed (celltype, slug)."""
    celltype, slug = parse_file_info(rank_csv)
    df = pd.read_csv(rank_csv)
    if not {'Sample', 'Rank'}.issubset(df.columns):
        raise ValueError(f"Rank file missing required columns: {rank_csv}")
    joined = df.merge(meta[['study_sample', 'time_mod24']], left_on='Sample', right_on='study_sample', how='left')
    joined['Cell_Type'] = celltype if celltype is not None else 'ALL'
    joined['Config_Slug'] = slug if slug is not None else 'Unknown'
    return joined[['Sample', 'Rank', 'time_mod24', 'Cell_Type', 'Config_Slug']], celltype, slug


def plot_single_file(joined_df: pd.DataFrame, celltype: str, slug: str, out_dir: str) -> str:
    """Plot rank vs time(mod24) for one file (one celltype)."""
    df = joined_df.copy()
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['Rank', 'time_mod24'])
    if df.empty:
        print(f"[WARN] No matched rows for {celltype} / {slug}")
        return ''
    df['Rank'] = df['Rank'].astype(int)

    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(df['Rank'], df['time_mod24'], s=28, alpha=0.8, edgecolors='white', linewidths=0.4, color='tab:blue')

    # Correlations
    try:
        pr, pp = pearsonr(df['Rank'], df['time_mod24'])
        sr, sp = spearmanr(df['Rank'], df['time_mod24'])
        corr_txt = f"Pearson r={pr:.2f} (p={pp:.2g})\nSpearman ρ={sr:.2f} (p={sp:.2g})"
    except Exception:
        corr_txt = None

    plt.xlabel('Predicted Rank')
    plt.ylabel('Metadata Time (mod 24 h)')
    title = f"{SLUG_TO_NAME.get(slug, slug or 'Unknown')}: Rank vs Time ({celltype or 'ALL'})"
    plt.title(title)

    if corr_txt:
        plt.gca().text(0.02, 0.98, corr_txt, transform=plt.gca().transAxes,
                       va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe_cell = sanitize_filename(celltype)
    safe_slug = sanitize_filename(slug)
    out_path = os.path.join(out_dir, f"rank_vs_time_{safe_cell}_{safe_slug}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-file plot: {out_path}")
    return out_path


def map_rank_to_24(rank_series: pd.Series) -> np.ndarray:
    """Linearly map ranks to [0, 24). Preserve relative scale.
    If all ranks equal, returns zeros.
    """
    r = pd.to_numeric(rank_series, errors='coerce').values.astype(float)
    r = r[~np.isnan(r)] if isinstance(rank_series, pd.Series) else r
    rmin = np.nanmin(r) if r.size else 0.0
    rmax = np.nanmax(r) if r.size else 1.0
    span = max(rmax - rmin, 1e-9)
    x = (rank_series.astype(float) - rmin) / span * 24.0
    # Clamp tiny numerical issues
    x = np.mod(x, 24.0)
    return x.values if isinstance(x, pd.Series) else x


@dataclass
class ShiftFitResult:
    slope: float
    shift: float
    r2: float
    corr: float
    n: int
    orientation: str  # 'normal' or 'flipped'


def compute_best_shift_no_intercept(x_base: np.ndarray, y: np.ndarray, step: float = 0.05) -> ShiftFitResult:
    """Given x_base in [0,24) and y in [0,24), find shift s in [0,24) that maximizes
    positive Pearson correlation between x and y. For the chosen shift, also compute
    the zero-intercept slope a>0 and corresponding R^2 for reporting. Try both orientations.
    """
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
            # Pearson correlation as primary objective
            try:
                corr = float(pearsonr(x, y)[0])
            except Exception:
                corr = -np.inf
            if not np.isfinite(corr):
                continue
            # Compute zero-intercept slope and R^2 for reporting if this is current best
            if corr > best.corr:
                xx = float(np.dot(x, x))
                if xx <= 1e-12:
                    continue
                a = float(np.dot(x, y) / xx)
                if a <= 0:
                    # Enforce positive linear relation requirement
                    continue
                y_hat = a * x
                sst0 = float(np.sum(y ** 2))
                sse = float(np.sum((y - y_hat) ** 2))
                r2 = 1.0 - sse / sst0 if sst0 > 1e-12 else -np.inf
                best = ShiftFitResult(slope=a, shift=s, r2=r2, corr=corr, n=len(y), orientation=orient_name)
        return best

    # Evaluate normal and flipped orientation to allow wrap direction
    best_normal = eval_orientation(x_base, 'normal')
    best_flipped = eval_orientation((24.0 - x_base) % 24.0, 'flipped')
    # Prefer the one with higher Pearson correlation; break ties by R^2
    if best_normal.corr > best_flipped.corr:
        return best_normal
    if best_flipped.corr > best_normal.corr:
        return best_flipped
    return best_normal if best_normal.r2 >= best_flipped.r2 else best_flipped


def plot_single_file_shifted(joined_df: pd.DataFrame, celltype: str, slug: str, out_dir: str,
                             step: float = 0.05, make_plot: bool = True) -> Tuple[str, ShiftFitResult]:
    """Compute best zero-intercept shift fit and optionally save plot; return path (or '') and fit result."""
    df = joined_df.copy()
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['Rank', 'time_mod24'])
    if df.empty:
        print(f"[WARN] No matched rows for {celltype} / {slug}")
        return '', ShiftFitResult(np.nan, np.nan, np.nan, np.nan, 0, 'normal')

    x0 = map_rank_to_24(df['Rank'])
    y = df['time_mod24'].astype(float).values
    # Center NaNs removed already
    res = compute_best_shift_no_intercept(x0, y, step=step)
    # Prepare shifted x according to best orientation and shift
    x_use = x0 if res.orientation == 'normal' else (24.0 - x0) % 24.0
    x_shift = (x_use + res.shift) % 24.0

    out_path = ''
    if make_plot:
        plt.figure(figsize=(7.5, 5.5))
        plt.scatter(x_shift, y, s=28, alpha=0.85, edgecolors='white', linewidths=0.4, color='tab:blue')

        info = (
            f"Best shift = {res.shift:.2f} h\n"
            f"Pearson r (max) = {res.corr:.3f}\n"
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


def main():
    parser = argparse.ArgumentParser(description='Aggregate rank CSVs and plot Rank vs Time per config (all cell types) and per file (cell type)')
    parser.add_argument('--ranks-dir', default="../result_20250821_203100_components50_genes8_celltypes26", help='Directory to search for sample_ranks_*.csv (recursively). Default: auto-detect')
    parser.add_argument('--metadata', default="D:\\CriticalFile\\Preprocess_CIRCADIA\\CYCLOPS-python\\data\\zeitzeiger\\metadata.csv", help='Path to metadata.csv. Default: auto-detect')
    parser.add_argument('--out-dir', required=False, default=None, help='Output directory for plots; default: <ranks-dir>/rank_vs_time')
    parser.add_argument('--shift-step', type=float, default=0.05, help='Shift step (hours) for search in [0,24)')
    parser.add_argument('--keep-all-per-file', action='store_true', help='Also keep original per-file plots (unshifted)')
    args = parser.parse_args()

    # Fill defaults if not provided
    base_dir = os.getcwd()
    ranks_dir = args.ranks_dir or auto_detect_ranks_dir(base_dir)
    if not ranks_dir:
        print('ERROR: Could not auto-detect ranks directory. Please provide --ranks-dir.')
        return
    metadata_csv = args.metadata or auto_detect_metadata(base_dir)
    if not metadata_csv:
        print('ERROR: Could not auto-detect metadata.csv. Please provide --metadata.')
        return
    out_dir = args.out_dir or os.path.join(ranks_dir, 'rank_vs_time')
    print(f"Using ranks-dir: {ranks_dir}")
    print(f"Using metadata: {metadata_csv}")
    print(f"Output directory: {out_dir}")

    files = find_rank_csvs(ranks_dir)
    if not files:
        print(f"No rank files found under: {ranks_dir}")
        return

    print(f"Found {len(files)} rank files.")
    meta = load_metadata(metadata_csv)

    # Per-file processing
    per_file_dir = os.path.join(out_dir, 'per_file')
    os.makedirs(per_file_dir, exist_ok=True)
    # Compute best-per-celltype and only keep those plots in a dedicated folder
    shift_best_dir = os.path.join(out_dir, 'per_file_shifted_best')
    best_per_celltype = {}

    for fp in files:
        try:
            joined, celltype, slug = join_single_file(fp, meta)
        except Exception as e:
            print(f"Skip {fp}: {e}")
            continue
        # compute shifted fit (no plot for non-best)
        _, res = plot_single_file_shifted(joined, celltype, slug, out_dir='', step=args.shift_step, make_plot=False)
        # Track best per cell type by Pearson r
        key = celltype or 'ALL'
        cur_best = best_per_celltype.get(key)
        if (cur_best is None) or (res.corr > cur_best['result'].corr) or (
            np.isclose(res.corr, cur_best['result'].corr) and res.r2 > cur_best['result'].r2
        ):
            best_per_celltype[key] = {'file': fp, 'joined': joined, 'celltype': celltype, 'slug': slug, 'result': res}
        # Optionally keep original per-file plot as well
        if args.keep_all_per_file:
            plot_single_file(joined, celltype, slug, per_file_dir)

    # Write best-only shifted plots per cell type (no CSVs or aggregates)
    os.makedirs(shift_best_dir, exist_ok=True)
    for ct, info in best_per_celltype.items():
        joined = info['joined']
        celltype = info['celltype']
        slug = info['slug']
        _path, _ = plot_single_file_shifted(joined, celltype, slug, out_dir=shift_best_dir, step=args.shift_step, make_plot=True)
    print(f"Saved best-only shifted plots per cell type to: {shift_best_dir}")

    # Save metrics table per cell type: Pearson R and R^2 (as Pearson r squared) for the best result
    if best_per_celltype:
        rows = []
        for ct, info in best_per_celltype.items():
            res = info['result']
            r = float(res.corr) if np.isfinite(res.corr) else np.nan
            r2 = float(r * r) if np.isfinite(r) else np.nan
            rows.append({'celltype': ct, 'R_spuare': r2, 'Pearson_R': r})
        metrics_df = pd.DataFrame(rows)
        metrics_path = os.path.join(shift_best_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics table: {metrics_path}")


if __name__ == '__main__':
    main()
