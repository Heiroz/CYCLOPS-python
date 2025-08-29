import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import importlib.util
import re

# --- config ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK_DIR = os.path.join(REPO_ROOT, 'rank')
DATA_ZEIT = os.path.join(REPO_ROOT, 'data', 'zeitzeiger')
EXPRESSION_FILE = os.path.join(DATA_ZEIT, 'expression.csv')
METADATA_FILE = os.path.join(DATA_ZEIT, 'metadata.csv')
CELLTYPES = ['macrophages', 'hypothalamus', 'MEF', 'aorta']
N_EIGEN = 5
OUT_DIR = os.path.join(RANK_DIR, 'result_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# --- helper functions (copied/simplified) ---

def fit_sine_curve(x_data, y_data):
    def sine_func(x, amplitude, phase, offset):
        period = len(x_data)
        return amplitude * np.sin(2 * np.pi * x / period + phase) + offset

    try:
        amplitude_guess = (np.nanmax(y_data) - np.nanmin(y_data)) / 2
        offset_guess = np.nanmean(y_data)
        phase_guess = 0.0
        popt, _ = curve_fit(sine_func, x_data, y_data, p0=[amplitude_guess, phase_guess, offset_guess], maxfev=2000)
        amplitude, phase, offset = popt
        x_smooth = np.linspace(x_data[0], x_data[-1], 200)
        y_smooth = sine_func(x_smooth, amplitude, phase, offset)
        y_pred = sine_func(x_data, amplitude, phase, offset)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.nanmean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return x_smooth, y_smooth, r_squared, popt
    except Exception:
        x_smooth = np.linspace(1, len(x_data), 200)
        y_smooth = np.zeros_like(x_smooth)
        return x_smooth, y_smooth, 0.0, (0, 0, 0)


def create_eigengenes_from_expression(expr_matrix, n_components):
    # expr_matrix: samples x genes (already standardized)
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(expr_matrix)
    # normalize components
    scaler = StandardScaler()
    comps_norm = scaler.fit_transform(comps)
    return comps_norm, pca, pca.explained_variance_ratio_

# --- load optimizer (greedy) from rank/optimizer.py ---
optimizer_path = os.path.join(RANK_DIR, 'optimizer.py')
spec = importlib.util.spec_from_file_location('rank_optimizer', optimizer_path)
rank_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank_optimizer)
MultiScaleOptimizer = getattr(rank_optimizer, 'MultiScaleOptimizer')

# --- load expression.csv in the same format as rank.py expects ---
print('Loading expression file:', EXPRESSION_FILE)
df = pd.read_csv(EXPRESSION_FILE, low_memory=False)
sample_columns = [c for c in df.columns if c != 'Gene_Symbol']
celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
if not celltype_row.empty:
    celltypes_arr = celltype_row.iloc[0][sample_columns].values
else:
    celltypes_arr = None

# build expression_data matrix samples x genes
gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
gene_names = gene_df['Gene_Symbol'].values
expression_data_raw = gene_df[sample_columns].values
n_samples = len(sample_columns)
expression_data = np.zeros((n_samples, len(gene_names)), dtype=float)
for i, sample in enumerate(sample_columns):
    for j in range(len(gene_names)):
        v = expression_data_raw[j, i]
        expression_data[i, j] = float(v) if pd.notna(v) else 0.0

# load metadata
meta = pd.read_csv(METADATA_FILE)
# detect which columns contain sample ids and time (be lenient to different datasets)
sample_col_candidates = ['study_sample', 'sample', 'sample_id', 'study:sample', 'sampleName', 'gsm', 'GSM', 'sampleName']
time_col_candidates = ['time_mod24', 'time', 'zeitgeber', 'zeitgeber_time', 'zeit', 'ZT', 'timepoint']

sample_col = next((c for c in sample_col_candidates if c in meta.columns), None)
time_col = next((c for c in time_col_candidates if c in meta.columns), None)

if sample_col is None or time_col is None:
    # try fuzzy matches if exact candidates not found
    if sample_col is None:
        sample_col = next((c for c in meta.columns if 'sample' in c.lower() or 'gsm' in c.lower()), None)
    if time_col is None:
        time_col = next((c for c in meta.columns if 'time' in c.lower() or 'zeit' in c.lower() or 'zt' in c.lower()), None)

if sample_col is None:
    print('Warning: could not detect a sample id column in metadata; defaulting to first column')
    sample_col = meta.columns[0]

if time_col is None:
    print('Warning: could not detect a time column in metadata; time values will be NaN')
    meta['__time_missing__'] = np.nan
    time_col = '__time_missing__'

print(f"Using metadata columns: sample='{sample_col}', time='{time_col}'")

# For each requested celltype, compute eigengenes (PCA), predicted rank via greedy, and metadata rank
for ct in CELLTYPES:
    print('\nProcessing celltype:', ct)
    if celltypes_arr is None:
        print('No celltype information in expression file; skipping')
        continue
    mask = (celltypes_arr == ct)
    if np.sum(mask) == 0:
        print(f'No samples for celltype {ct} - skipping')
        continue
    sample_names = np.array(sample_columns)[mask]
    sub_expr = expression_data[mask]
    n_sub = sub_expr.shape[0]
    n_comp = min(N_EIGEN, max(1, n_sub))

    # standardize genes for this celltype
    scaler = StandardScaler()
    try:
        sub_scaled = scaler.fit_transform(sub_expr)
    except Exception:
        sub_scaled = scaler.fit_transform(sub_expr.astype(float))

    eigs, pca_model, ev = create_eigengenes_from_expression(sub_scaled, n_components=n_comp)

    # predicted rank: neural optimizer
    opt = MultiScaleOptimizer(method='neural')
    try:
        pred_ranks = opt.optimize(eigs, None).flatten()
    except Exception as e:
        print('Optimizer failed:', e)
        # fallback: sort by first component
        pred_ranks = np.argsort(eigs[:, 0])

    # metadata ranks: try to map sample_names to metadata time column robustly
    def map_samples_to_times(sample_names, meta_df, sample_col_name, time_col_name):
        # tries exact match, contains, reverse-contains, and prefix-stripped matches
        times = []
        matched = 0
        meta_series = meta_df[sample_col_name].astype(str)
        for s in sample_names:
            s_str = str(s)
            # exact
            hits = meta_df[meta_series == s_str]
            if not hits.empty:
                times.append(hits.iloc[0][time_col_name])
                matched += 1
                continue
            # meta contains sample
            hits = meta_df[meta_series.str.contains(re.escape(s_str), case=False, na=False)]
            if not hits.empty:
                times.append(hits.iloc[0][time_col_name])
                matched += 1
                continue
            # sample contains meta (reverse)
            hits = meta_df[meta_series.apply(lambda x: str(x).lower() in s_str.lower())]
            if not hits.empty:
                times.append(hits.iloc[0][time_col_name])
                matched += 1
                continue
            # strip common prefixes from sample name and try contains
            s_norm = re.sub(r'^(NiG_|GSM_|sample_|SAM_)', '', s_str, flags=re.I)
            hits = meta_df[meta_series.str.contains(re.escape(s_norm), case=False, na=False)]
            if not hits.empty:
                times.append(hits.iloc[0][time_col_name])
                matched += 1
                continue
            times.append(np.nan)
        return np.array(times, dtype=float), matched

    times, matched = map_samples_to_times(sample_names, meta, sample_col, time_col)
    if matched == 0:
        print(f'Warning: no metadata times found for samples in {ct} (matched=0); metadata plot will use sample index order')
        order_meta = np.arange(len(sample_names))
    else:
        # convert times to ranks (handle nan by placing at end)
        order_meta = np.argsort(np.nan_to_num(times, nan=np.nanmax(times) + 1))
    meta_ranks = np.empty_like(order_meta)
    meta_ranks[order_meta] = np.arange(len(order_meta))

    # prepare plotting
    fig, axes = plt.subplots(N_EIGEN, 2, figsize=(10, 3 * N_EIGEN))
    for i_e in range(N_EIGEN):
        if i_e >= eigs.shape[1]:
            axes[i_e, 0].axis('off')
            axes[i_e, 1].axis('off')
            continue
        # predicted ordering
        order_pred = np.argsort(pred_ranks)
        vals_pred = eigs[order_pred, i_e]
        x_pred = np.arange(1, len(vals_pred) + 1)
        axes[i_e, 0].scatter(x_pred, vals_pred, c=np.arange(len(vals_pred)), cmap='viridis', s=25, edgecolors='white')
        xs, ys, r2, _ = fit_sine_curve(x_pred, vals_pred)
        axes[i_e, 0].plot(xs, ys, color='C1')
        axes[i_e, 0].set_title(f'{ct} - Eigengene {i_e+1} (Predicted rank)')
        axes[i_e, 0].set_xlabel('Rank')
        axes[i_e, 0].set_ylabel('Value')

        # metadata ordering
        order_m = order_meta
        vals_meta = eigs[order_m, i_e]
        x_meta = np.arange(1, len(vals_meta) + 1)
        axes[i_e, 1].scatter(x_meta, vals_meta, c=times[order_m], cmap='plasma', s=25, edgecolors='white')
        xs2, ys2, r22, _ = fit_sine_curve(x_meta, vals_meta)
        axes[i_e, 1].plot(xs2, ys2, color='C2')
        axes[i_e, 1].set_title(f'{ct} - Eigengene {i_e+1} (Metadata time)')
        axes[i_e, 1].set_xlabel('Time-sorted index')
        axes[i_e, 1].set_ylabel('Value')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{ct}_eigengenes_compare.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print('Saved:', out_path)

print('\nDone')
