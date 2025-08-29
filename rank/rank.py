import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import sys
import os
import plot as rank_plot
from optimizer import MultiScaleOptimizer, create_optimizer_configs

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'my_cyclops'))

class Config:
    
    N_COMPONENTS = 5
    MIN_SAMPLES_PER_CELLTYPE = 10
    
    DEFAULT_SMOOTHNESS_FACTOR = 0.7
    DEFAULT_LOCAL_VARIATION_FACTOR = 0.3
    DEFAULT_WINDOW_SIZE = 10
    
    MAX_ITERATIONS_RATIO = 5
    VARIATION_TOLERANCE_RATIO = 0.5
    
    LOCAL_VARIATION_WINDOW_DIVISOR = 5
    BALANCE_SCORE_VARIATION_WEIGHT = 0.1
    
    SINE_FIT_MAX_ITERATIONS = 2000
    SINE_FIT_SMOOTH_POINTS = 200
    
    # METHOD = "neural"
    METHOD = "greedy"
    CIRCADIAN_GENES = [
        'PER1', 'PER2', 'CRY1', 'CRY2', 
        'CLOCK', 'NR1D1', 'NR1D2', 'DBP'
    ]
    
    SMOOTHNESS_CONFIGS = [
        {'smoothness_factor': 1.0, 'local_variation_factor': 0.0, 'name': 'Pure Smoothness'},
        {'smoothness_factor': 0.8, 'local_variation_factor': 0.2, 'name': 'Mostly Smooth'},
        {'smoothness_factor': 0.7, 'local_variation_factor': 0.3, 'name': 'Balanced'},
        {'smoothness_factor': 0.6, 'local_variation_factor': 0.4, 'name': 'More Variation'},
        {'smoothness_factor': 0.5, 'local_variation_factor': 0.5, 'name': 'Equal Balance'}
    ]
    
    @staticmethod
    def get_eigengene_weights(n_components):
        """Generate eigengene weights based on component importance"""
        if n_components <= 50:
            weights = []
            weights.extend([1.0])
            weights.extend([0.8] * min(10, n_components - 1))
            if n_components > 11:
                weights.extend([0.6] * min(15, n_components - 11))
            if n_components > 26:
                weights.extend([0.4] * (n_components - 26))
            return weights[:n_components]
        else:
            weights = []
            weights.extend([1.0])
            weights.extend([0.8] * 10)
            weights.extend([0.6] * 15)
            weights.extend([0.4] * 24)
            remaining = n_components - 50
            if remaining > 0:
                weights.extend([0.2] * min(25, remaining))
                if remaining > 25:
                    weights.extend([0.1] * (remaining - 25))
            return weights[:n_components]
    
    FIGURE_DPI = 300
    SUBPLOT_WIDTH_PER_GENE = 3
    SUBPLOT_HEIGHT_PER_CELLTYPE = 3
    EIGENGENE_SUBPLOT_WIDTH = 4
    EIGENGENE_SUBPLOT_HEIGHT = 3
    MAX_EIGENGENES_PLOT = 5
    
    SCATTER_SIZE_MAIN = 25
    SCATTER_SIZE_EIGENGENE = 20
    SCATTER_ALPHA = 0.7
    LINE_WIDTH_MAIN = 2.5
    LINE_WIDTH_EIGENGENE = 2.0
    LINE_ALPHA = 0.8
    EDGE_LINE_WIDTH_MAIN = 0.5
    EDGE_LINE_WIDTH_EIGENGENE = 0.3
    GRID_ALPHA = 0.3
    
    COLORMAP = 'viridis'
    EDGE_COLOR = 'white'
    
    DEFAULT_EXPRESSION_FILE = "../data/zeitzeiger/expression.csv"
    DEFAULT_METADATA_FILE = "../data/zeitzeiger/metadata.csv"

    RESULT_DIR_PREFIX = "result"
    OUTPUT_FIGURE_FORMAT = "png"
    
    INVALID_VALUE_REPLACEMENT = 0.0
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        print(f"Data Processing:")
        print(f"  - PCA Components: {cls.N_COMPONENTS}")
        print(f"  - Min samples per cell type: {cls.MIN_SAMPLES_PER_CELLTYPE}")
        print(f"Optimization:")
        print(f"  - Default smoothness factor: {cls.DEFAULT_SMOOTHNESS_FACTOR}")
        print(f"  - Default local variation factor: {cls.DEFAULT_LOCAL_VARIATION_FACTOR}")
        print(f"  - Default window size: {cls.DEFAULT_WINDOW_SIZE}")
        print(f"  - Max iterations ratio: {cls.MAX_ITERATIONS_RATIO}")
        print(f"Visualization:")
        print(f"  - Figure DPI: {cls.FIGURE_DPI}")
        print(f"  - Scatter size: {cls.SCATTER_SIZE_MAIN}")
        print(f"  - Line width: {cls.LINE_WIDTH_MAIN}")
        print(f"Genes of Interest:")
        print(f"  - Circadian genes: {', '.join(cls.CIRCADIAN_GENES)}")
        print(f"Configurations: {len(cls.SMOOTHNESS_CONFIGS)} optimization strategies")
        print("=" * 60)

def create_eigengenes(expression_scaled, n_components=None):
    if n_components is None:
        n_components = Config.N_COMPONENTS
        
    print("Performing PCA transformation...")
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(expression_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    scaler = StandardScaler()
    components_normalized = scaler.fit_transform(components)
    
    return components_normalized, pca, explained_variance

def fit_sine_curve(x_data, y_data):
    def sine_func(x, amplitude, phase, offset):
        period = len(x_data)
        return amplitude * np.sin(2 * np.pi * x / period + phase) + offset
    
    amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2
    offset_guess = np.mean(y_data)
    phase_guess = 0.0
    
    popt, _ = curve_fit(sine_func, x_data, y_data, 
                       p0=[amplitude_guess, phase_guess, offset_guess],
                       maxfev=Config.SINE_FIT_MAX_ITERATIONS)
    
    amplitude, phase, offset = popt
    
    x_smooth = np.linspace(x_data[0], x_data[-1], Config.SINE_FIT_SMOOTH_POINTS)
    y_smooth = sine_func(x_smooth, amplitude, phase, offset)
    
    y_pred = sine_func(x_data, amplitude, phase, offset)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return x_smooth, y_smooth, r_squared, popt

def load_and_process_expression_data(expression_file, n_components=None):
    if n_components is None:
        n_components = Config.N_COMPONENTS
    print("=== Loading Expression Data ===")
    print(f"File: {expression_file}")
    
    df = pd.read_csv(expression_file, low_memory=False)
    print(f"Data shape: {df.shape}")
    
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    
    print(f"Contains celltype info: {has_celltype}")
    print(f"Contains time info: {has_time}")
    
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    n_samples = len(sample_columns)
    print(f"Number of samples: {n_samples}")
    
    celltypes = None
    times = None
    
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
        unique_celltypes = np.unique(celltypes)
        print(f"Cell types found: {unique_celltypes}")
    
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"Time range: {times.min():.2f} - {times.max():.2f} hours")
    
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    gene_names = gene_df['Gene_Symbol'].values
    
    expression_data_raw = gene_df[sample_columns].values
    
    print("Converting expression data to numeric...")
    expression_data = np.zeros((len(sample_columns), len(gene_names)), dtype=float)
    
    for i, sample in enumerate(sample_columns):
        for j, gene in enumerate(gene_names):
            value = float(expression_data_raw[j, i]) if pd.notna(expression_data_raw[j, i]) else Config.INVALID_VALUE_REPLACEMENT
            expression_data[i, j] = value
    
    print(f"Number of genes: {len(gene_names)}")
    print(f"Expression data shape: {expression_data.shape}")
    print(f"Expression data type: {expression_data.dtype}")
    print(f"Expression data range: {expression_data.min():.4f} to {expression_data.max():.4f}")
    
    print("Loaded expression matrix; deferring PCA to per-celltype or single-dataset processing.")
    return {
        'original_expression': expression_data,
        'gene_names': gene_names,
        'sample_columns': sample_columns,
        'celltypes': celltypes,
        'times': times,
        'pca_model': None,
        'scaler': None,
        'explained_variance': None,
        'n_components': n_components
    }

def get_circadian_gene_expressions(data_info, circadian_genes):
    gene_names = data_info['gene_names']
    original_expression = data_info['original_expression']
    
    found_genes = []
    gene_expressions = []
    
    for gene in circadian_genes:
        if gene in gene_names:
            gene_idx = np.where(gene_names == gene)[0][0]
            gene_expr = original_expression[:, gene_idx]
            
            if gene_expr.dtype == 'object' or not np.issubdtype(gene_expr.dtype, np.number):
                gene_expr = pd.to_numeric(gene_expr, errors='coerce')
                gene_expr = np.nan_to_num(gene_expr, nan=Config.INVALID_VALUE_REPLACEMENT)
            
            gene_expressions.append(gene_expr)
            found_genes.append(gene)
        else:
            print(f"Warning: Gene {gene} not found in data")
    
    if len(found_genes) == 0:
        print("Error: No circadian genes found in data!")
        return None, None
    
    gene_expressions = np.array(gene_expressions, dtype=float).T
    print(f"Found {len(found_genes)} circadian genes: {found_genes}")
    print(f"Circadian expression data shape: {gene_expressions.shape}")
    print(f"Circadian expression data type: {gene_expressions.dtype}")
    
    return gene_expressions, found_genes


def create_circadian_gene_visualization(all_results, circadian_genes, celltypes, output_dir):
    print("Creating circadian gene visualization...")
    
    n_celltypes = len(celltypes)
    n_genes = len(circadian_genes)
    
    _, axes = plt.subplots(n_celltypes, n_genes, figsize=(3*n_genes, 3*n_celltypes))
    
    if n_celltypes == 1:
        axes = axes.reshape(1, -1)
    if n_genes == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_genes))
    
    for ct_idx, celltype in enumerate(celltypes):
        result = all_results[celltype]['Balanced']
        ranks = result['ranks']
        circadian_data = result['circadian_expressions']
        
        order_indices = np.argsort(ranks.flatten())
        ordered_circadian = circadian_data[order_indices]
        n_samples = len(ordered_circadian)
        
        for gene_idx, gene_name in enumerate(circadian_genes):
            ax = axes[ct_idx, gene_idx] if n_celltypes > 1 else axes[gene_idx]
            
            x_range = np.array(range(1, n_samples + 1))
            gene_values = ordered_circadian[:, gene_idx]
            
            if gene_values.dtype == 'object' or not np.issubdtype(gene_values.dtype, np.number):
                gene_values = pd.to_numeric(gene_values, errors='coerce')
                gene_values = np.nan_to_num(gene_values, nan=0.0)
            
            ax.scatter(x_range, gene_values, 
                               c=range(n_samples), cmap='viridis', s=25, alpha=0.7, 
                               edgecolors='white', linewidth=0.5, zorder=3)
            
            x_smooth, y_smooth, r_squared, _ = fit_sine_curve(x_range, gene_values)
            ax.plot(x_smooth, y_smooth, '-', color=colors[gene_idx], 
                   linewidth=2.5, alpha=0.8, label=f'Sine fit (R²={r_squared:.3f})', zorder=2)
            
            balance_score = result['metrics']['balance_score']
            ax.set_title(f'CT {celltype}: {gene_name}\nBalance={balance_score:.3f}, R²={r_squared:.3f}', 
                        fontsize=10, fontweight='bold', pad=8)
            ax.set_xlabel('Sample Rank', fontsize=9)
            ax.set_ylabel('Expression', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
            ax.set_xlim(0, n_samples + 1)
            
            y_min, y_max = float(np.min(gene_values)), float(np.max(gene_values))
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ax.set_ylim(-0.1, 0.1)
            
            if ct_idx == 0 and gene_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Circadian Gene Expression Ordered by Eigengene Optimization', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    output_path = os.path.join(output_dir, 'circadian_genes_eigengene_optimization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Circadian gene visualization saved as: {output_path}")
    plt.close()

def create_eigengene_comparison_visualization(all_results, representative_celltype, n_components, output_dir):
    print(f"Creating eigengene comparison for cell type {representative_celltype}...")
    
    celltype_results = all_results[representative_celltype]
    
    n_eigengenes_plot = min(5, n_components)
    
    _, axes = plt.subplots(len(celltype_results), n_eigengenes_plot, 
                            figsize=(4*n_eigengenes_plot, 3*len(celltype_results)))
    # Ensure axes is at least 2D for consistent indexing
    axes = np.atleast_2d(axes)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for config_idx, (config_name, result) in enumerate(celltype_results.items()):
        ranks = result['ranks']
        eigengenes_data = result['eigengenes']
        metrics = result['metrics']
        
        order_indices = np.argsort(ranks.flatten())
        ordered_eigengenes = eigengenes_data[order_indices]
        n_samples = len(ordered_eigengenes)
        available = ordered_eigengenes.shape[1]

        for eigen_idx in range(n_eigengenes_plot):
            ax = axes[config_idx, eigen_idx]
            if eigen_idx >= available:
                ax.axis('off')
                continue
            
            x_range = np.array(range(1, n_samples + 1))
            eigen_values = ordered_eigengenes[:, eigen_idx]
            
            if eigen_values.dtype == 'object' or not np.issubdtype(eigen_values.dtype, np.number):
                eigen_values = pd.to_numeric(eigen_values, errors='coerce')
                eigen_values = np.nan_to_num(eigen_values, nan=0.0)
            
            ax.scatter(x_range, eigen_values, 
                        c=range(n_samples), cmap='viridis', s=20, alpha=0.7, 
                        edgecolors='white', linewidth=0.3, zorder=3)
        
            x_smooth, y_smooth, r_squared, _ = fit_sine_curve(x_range, eigen_values)
            ax.plot(x_smooth, y_smooth, '-', color=colors[eigen_idx], 
                   linewidth=2.0, alpha=0.8, label=f'Sine fit (R²={r_squared:.3f})', zorder=2)
            
            balance_score = metrics['balance_score']
            ax.set_title(f'{config_name}: Eigengene {eigen_idx+1}\nBalance={balance_score:.3f}, R²={r_squared:.3f}', 
                        fontsize=10, fontweight='bold', pad=8)
            ax.set_xlabel('Sample Rank', fontsize=9)
            ax.set_ylabel('Eigengene Value', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
            ax.set_xlim(0, n_samples + 1)
            
            y_min, y_max = float(np.min(eigen_values)), float(np.max(eigen_values))
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ax.set_ylim(-0.1, 0.1)
            
            if config_idx == 0 and eigen_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'Eigengene Optimization Comparison (Cell Type {representative_celltype})', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    output_path = os.path.join(output_dir, 'eigengene_optimization_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Eigengene comparison saved as: {output_path}")
    plt.close()

def create_single_dataset_visualization(results, circadian_genes, output_dir):
    """Create visualization for single dataset without cell type grouping"""
    print("Creating single dataset visualization...")
    
    n_genes = len(circadian_genes)
    n_configs = len(results)
    
    _, axes = plt.subplots(n_configs, n_genes, figsize=(3*n_genes, 3*n_configs))
    
    if n_configs == 1:
        axes = axes.reshape(1, -1)
    if n_genes == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_genes))
    
    for config_idx, (config_name, result) in enumerate(results.items()):
        ranks = result['ranks']
        circadian_data = result['circadian_expressions']
        metrics = result['metrics']
        
        order_indices = np.argsort(ranks.flatten())
        ordered_circadian = circadian_data[order_indices]
        n_samples = len(ordered_circadian)
        
        for gene_idx, gene_name in enumerate(circadian_genes):
            ax = axes[config_idx, gene_idx] if n_configs > 1 else axes[gene_idx]
            
            x_range = np.array(range(1, n_samples + 1))
            gene_values = ordered_circadian[:, gene_idx]
            
            if gene_values.dtype == 'object' or not np.issubdtype(gene_values.dtype, np.number):
                gene_values = pd.to_numeric(gene_values, errors='coerce')
                gene_values = np.nan_to_num(gene_values, nan=0.0)
            
            ax.scatter(x_range, gene_values, 
                    c=range(n_samples), cmap='viridis', s=25, alpha=0.7, 
                    edgecolors='white', linewidth=0.5, zorder=3)
            
            x_smooth, y_smooth, r_squared, _ = fit_sine_curve(x_range, gene_values)
            ax.plot(x_smooth, y_smooth, '-', color=colors[gene_idx], 
                   linewidth=2.5, alpha=0.8, label=f'Sine fit (R²={r_squared:.3f})', zorder=2)
            
            balance_score = metrics['balance_score']
            ax.set_title(f'{config_name}: {gene_name}\nBalance={balance_score:.3f}, R²={r_squared:.3f}', 
                        fontsize=10, fontweight='bold', pad=8)
            ax.set_xlabel('Sample Rank', fontsize=9)
            ax.set_ylabel('Expression', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
            ax.set_xlim(0, n_samples + 1)
            
            y_min, y_max = float(np.min(gene_values)), float(np.max(gene_values))
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ax.set_ylim(-0.1, 0.1)
            
            if config_idx == 0 and gene_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Circadian Gene Expression with Different Optimization Strategies', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    output_path = os.path.join(output_dir, 'circadian_genes_single_dataset.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Single dataset visualization saved as: {output_path}")
    plt.close()

def print_summary_statistics(all_results, output_dir):
    print("\n=== Summary Statistics ===")
    
    summary_file = os.path.join(output_dir, 'optimization_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("=== Eigengene-Based Multi-Scale Optimization Summary ===\n\n")
        
        print(f"{'Cell Type':<10} {'Samples':<8} {'Best Config':<15} {'Balance Score':<12}")
        print("-" * 55)
        f.write(f"{'Cell Type':<10} {'Samples':<8} {'Best Config':<15} {'Balance Score':<12}\n")
        f.write("-" * 55 + "\n")
        
        overall_best_configs = {}
        for celltype, celltype_results in all_results.items():
            best_config = max(celltype_results.items(), key=lambda x: x[1]['metrics']['balance_score'])
            config_name = best_config[0]
            balance_score = best_config[1]['metrics']['balance_score']
            n_samples = len(best_config[1]['eigengenes'])
            
            line = f"{celltype:<10} {n_samples:<8} {config_name:<15} {balance_score:<12.4f}"
            print(line)
            f.write(line + "\n")
            
            if config_name not in overall_best_configs:
                overall_best_configs[config_name] = []
            overall_best_configs[config_name].append(celltype)
        
        print(f"\n=== Configuration Usage Summary ===")
        f.write(f"\n=== Configuration Usage Summary ===\n")
        for config_name, celltypes in overall_best_configs.items():
            line = f"  {config_name}: {len(celltypes)} cell types {celltypes}"
            print(line)
            f.write(line + "\n")
        
        most_popular = max(overall_best_configs.items(), key=lambda x: len(x[1]))
        print(f"\nMost successful configuration: {most_popular[0]}")
        print(f"Works best for {len(most_popular[1])} out of {len(all_results)} cell types")
        f.write(f"\nMost successful configuration: {most_popular[0]}\n")
        f.write(f"Works best for {len(most_popular[1])} out of {len(all_results)} cell types\n")
        
        algorithm_summary = [
            "\nAlgorithm Summary:",
            "- Uses PCA-derived eigengenes for optimization",
            "- Visualizes results using original circadian genes",
            "- Balances global smoothness with local variation preservation",
            "- Processes each cell type independently for optimal results"
        ]
        
        for line in algorithm_summary:
            print(line)
            f.write(line + "\n")
    
    print(f"\nSummary statistics saved to: {summary_file}")
    
    csv_file = os.path.join(output_dir, 'detailed_results.csv')
    detailed_data = []
    
    for celltype, celltype_results in all_results.items():
        for config_name, result in celltype_results.items():
            detailed_data.append({
                'Cell_Type': celltype,
                'Configuration': config_name,
                'Smoothness_Factor': result['config']['smoothness_factor'],
                'Local_Variation_Factor': result['config']['local_variation_factor'],
                'N_Samples': len(result['eigengenes']),
                'Global_Smoothness': result['metrics']['global_smoothness'],
                'Local_Variation': result['metrics']['local_variation'],
                'Trend_Smoothness': result['metrics']['trend_smoothness'],
                'Balance_Score': result['metrics']['balance_score']
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to: {csv_file}")
    
    print(f"\nAll results saved to directory: {os.path.abspath(output_dir)}")

def main(expression_file: str = None, metadata_file: str = None, use_eigengene_weights: bool = True, weights_file: str = None):
    if expression_file is None:
        expression_file = Config.DEFAULT_EXPRESSION_FILE
    if metadata_file is None:
        metadata_file = Config.DEFAULT_METADATA_FILE
    n_components = Config.N_COMPONENTS

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"result_{timestamp}_components{n_components}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    circadian_genes = ['PER1', 'PER2', 'CRY1', 'CRY2', 'CLOCK', 'ARNTL', 'NR1D1', 'NR1D2', 'DBP']

    smoothness_configs = create_optimizer_configs()
    eigengene_weights = Config.get_eigengene_weights(n_components) if use_eigengene_weights else None

    weights_to_use = eigengene_weights
    if weights_file:
        try:
            wdf = pd.read_csv(weights_file, header=None)
            row = wdf.values.flatten()
            row = row[~pd.isna(row)]
            if len(row) > 0:
                weights_to_use = row.astype(float).tolist()
                print(f"Loaded weights from {weights_file}")
        except Exception as e:
            print(f"Could not read weights file {weights_file}: {e}")

    print("=== Eigengene-Based Multi-Scale Optimization ===")

    data_info = load_and_process_expression_data(expression_file, n_components)
    celltypes = data_info['celltypes']

    circadian_expressions, found_circadian_genes = get_circadian_gene_expressions(data_info, circadian_genes)

    if circadian_expressions is None:
        print("Cannot proceed without circadian genes")
        return

    actual_output_dir = f"{output_dir}_genes{len(found_circadian_genes)}"
    if output_dir != actual_output_dir:
        os.rename(output_dir, actual_output_dir)
        output_dir = actual_output_dir
        print(f"Updated output directory to: {os.path.abspath(output_dir)}")

    if celltypes is not None:
        unique_celltypes = np.unique(celltypes)
        eligible_celltypes = [ct for ct in unique_celltypes if np.sum(celltypes == ct) >= 10]
        print(f"\nCell types with >=10 samples: {eligible_celltypes}")

        final_output_dir = f"{output_dir}_celltypes{len(eligible_celltypes)}"
        if output_dir != final_output_dir:
            os.rename(output_dir, final_output_dir)
            output_dir = final_output_dir
            print(f"Final output directory: {os.path.abspath(output_dir)}")

        all_results = {}

        for celltype in eligible_celltypes:
            print(f"\n=== Processing Cell Type {celltype} ===")

            # Build a per-celltype PCA from the original expression matrix
            celltype_mask = celltypes == celltype
            # original_expression is samples x genes
            sub_expression = data_info['original_expression'][celltype_mask]
            celltype_circadian = circadian_expressions[celltype_mask]
            n_samples = sub_expression.shape[0]

            print(f"Samples: {n_samples}")

            # Determine components for this celltype (cannot exceed n_samples)
            n_components_ct = min(n_components, max(1, n_samples))

            # Standardize genes for this celltype and run PCA to get eigengenes
            if n_samples > 0:
                local_scaler = StandardScaler()
                try:
                    sub_scaled = local_scaler.fit_transform(sub_expression)
                except Exception:
                    # fallback: cast to float then scale
                    sub_scaled = local_scaler.fit_transform(sub_expression.astype(float))

                celltype_eigengenes, pca_model_ct, explained_variance_ct = create_eigengenes(sub_scaled, n_components=n_components_ct)
            else:
                celltype_eigengenes = np.empty((0, n_components_ct))
                pca_model_ct = None
                explained_variance_ct = np.array([])

            if weights_to_use is None:
                weights_for_ct = None
            else:
                try:
                    weights_for_ct = list(weights_to_use)[:celltype_eigengenes.shape[1]]
                except Exception:
                    weights_for_ct = weights_to_use

            celltype_results = {}

            for config in smoothness_configs:
                print(f"  Testing {config['name']}...")

                optimizer = MultiScaleOptimizer(
                    smoothness_factor=config['smoothness_factor'],
                    local_variation_factor=config['local_variation_factor'],
                    window_size=Config.DEFAULT_WINDOW_SIZE,
                    max_iterations_ratio=Config.MAX_ITERATIONS_RATIO,
                    variation_tolerance_ratio=Config.VARIATION_TOLERANCE_RATIO,
                    method=Config.METHOD
                )

                ranks = optimizer.optimize(celltype_eigengenes, weights_for_ct)
                
                metrics = optimizer.analyze_metrics(celltype_eigengenes, ranks)

                celltype_results[config['name']] = {
                    'ranks': ranks,
                    'metrics': metrics,
                    'config': config,
                    'eigengenes': celltype_eigengenes,
                    'circadian_expressions': celltype_circadian,
                    'pca_model': pca_model_ct,
                    'explained_variance': explained_variance_ct
                }

                print(f"    Balance Score: {metrics['balance_score']:.4f}")

            all_results[celltype] = celltype_results

            best_config = max(celltype_results.items(), key=lambda x: x[1]['metrics']['balance_score'])
            print(f"  Best config: {best_config[0]} (Score: {best_config[1]['metrics']['balance_score']:.4f})")

        print("\n=== Creating Visualizations ===")

        create_circadian_gene_visualization(all_results, found_circadian_genes, eligible_celltypes, output_dir)

        if len(eligible_celltypes) > 0:
            create_eigengene_comparison_visualization(all_results, eligible_celltypes[0], n_components, output_dir)

        print_summary_statistics(all_results, output_dir)

    else:
        print("\nNo cell type information available, processing all samples together...")

        final_output_dir = f"{output_dir}_single_dataset"
        if output_dir != final_output_dir:
            os.rename(output_dir, final_output_dir)
            output_dir = final_output_dir
            print(f"Final output directory: {os.path.abspath(output_dir)}")

        results = {}
        # Compute PCA for the entire dataset once (single-dataset mode)
        original_expression = data_info['original_expression']
        n_samples = original_expression.shape[0]
        print(f"Total samples: {n_samples}")

        n_components_sd = min(n_components, max(1, n_samples))
        scaler_sd = StandardScaler()
        try:
            scaled_sd = scaler_sd.fit_transform(original_expression)
        except Exception:
            scaled_sd = scaler_sd.fit_transform(original_expression.astype(float))

        eigengenes_sd, pca_model_sd, explained_variance_sd = create_eigengenes(scaled_sd, n_components=n_components_sd)

        # Trim weights for single-dataset PCA
        if weights_to_use is None:
            weights_for_sd = None
        else:
            try:
                weights_for_sd = list(weights_to_use)[:eigengenes_sd.shape[1]]
            except Exception:
                weights_for_sd = weights_to_use

        for config in smoothness_configs:
            print(f"Testing {config['name']}...")

            optimizer = MultiScaleOptimizer(
                smoothness_factor=config['smoothness_factor'],
                local_variation_factor=config['local_variation_factor'],
                window_size=Config.DEFAULT_WINDOW_SIZE,
                max_iterations_ratio=Config.MAX_ITERATIONS_RATIO,
                variation_tolerance_ratio=Config.VARIATION_TOLERANCE_RATIO
            )

            ranks = optimizer.optimize(eigengenes_sd, weights_for_sd)
            
            metrics = optimizer.analyze_metrics(eigengenes_sd, ranks)

            results[config['name']] = {
                'ranks': ranks,
                'metrics': metrics,
                'config': config,
                'eigengenes': eigengenes_sd,
                'circadian_expressions': circadian_expressions,
                'pca_model': pca_model_sd,
                'explained_variance': explained_variance_sd
            }

            print(f"  Balance Score: {metrics['balance_score']:.4f}")

        create_single_dataset_visualization(results, found_circadian_genes, output_dir)

    if metadata_file:
        if not os.path.isfile(metadata_file):
            print(f"Provided metadata file not found: {metadata_file}. Skipping rank-time plots.")
            return

        ranks_root = os.path.abspath(output_dir)
        meta = rank_plot.load_metadata(metadata_file)

        shift_best_dir = os.path.join(ranks_root, 'rank_vs_time', 'per_file_shifted_best')
        os.makedirs(shift_best_dir, exist_ok=True)

        best_per_celltype = {}

        # Always use in-memory ranks (no CSV files expected)
        print(f"Using in-memory ranks for rank-vs-time plotting for results in: {ranks_root}")

        # Use per-celltype results if available, otherwise the single-dataset results
        if celltypes is not None:
            # iterate over all_results built earlier
            for ct, celltype_results in all_results.items():
                # sample names for this cell type
                mask = (data_info['celltypes'] == ct)
                sample_names = np.array(data_info['sample_columns'])[mask]
                for config_name, resdict in celltype_results.items():
                    ranks_arr = resdict['ranks'].flatten()
                    df = pd.DataFrame({'Sample': sample_names, 'Rank': ranks_arr})
                    # merge with metadata to emulate join_single_file output
                    joined = df.merge(meta[['study_sample', 'time_mod24']], left_on='Sample', right_on='study_sample', how='left')
                    joined['Cell_Type'] = ct if ct is not None else 'ALL'
                    # use slug consistent with plot module
                    slug = rank_plot.CONFIG_SLUGS.get(config_name, config_name)

                    _, res = rank_plot.plot_single_file_shifted(joined, ct, slug, out_dir='', step=0.05, make_plot=False)
                    key = ct or 'ALL'
                    cur_best = best_per_celltype.get(key)
                    if (cur_best is None) or (res.corr > cur_best['result'].corr) or (
                        np.isclose(res.corr, cur_best['result'].corr) and res.r2 > cur_best['result'].r2
                    ):
                        best_per_celltype[key] = {'file': f'in-memory:{ct}:{slug}', 'joined': joined, 'celltype': ct, 'slug': slug, 'result': res}
        else:
            # single dataset: use `results` built earlier
            sample_names = np.array(data_info['sample_columns'])
            for config_name, resdict in results.items():
                ranks_arr = resdict['ranks'].flatten()
                df = pd.DataFrame({'Sample': sample_names, 'Rank': ranks_arr})
                joined = df.merge(meta[['study_sample', 'time_mod24']], left_on='Sample', right_on='study_sample', how='left')
                joined['Cell_Type'] = 'ALL'
                slug = rank_plot.CONFIG_SLUGS.get(config_name, config_name)

                _, res = rank_plot.plot_single_file_shifted(joined, None, slug, out_dir='', step=0.05, make_plot=False)
                key = 'ALL'
                cur_best = best_per_celltype.get(key)
                if (cur_best is None) or (res.corr > cur_best['result'].corr) or (
                    np.isclose(res.corr, cur_best['result'].corr) and res.r2 > cur_best['result'].r2
                ):
                    best_per_celltype[key] = {'file': f'in-memory:ALL:{slug}', 'joined': joined, 'celltype': None, 'slug': slug, 'result': res}

        for ct, info in best_per_celltype.items():
            joined = info['joined']
            celltype = info['celltype']
            slug = info['slug']
            rank_plot.plot_single_file_shifted(joined, celltype, slug, out_dir=shift_best_dir, step=0.05, make_plot=True)
        print(f"Saved best-only shifted plots per cell type to: {shift_best_dir}")

        if best_per_celltype:
            rows = []
            for ct, info in best_per_celltype.items():
                res = info['result']
                r = float(res.corr) if np.isfinite(res.corr) else np.nan
                r2 = float(r * r) if np.isfinite(r) else np.nan
                joined = info['joined']
                x0 = rank_plot.map_rank_to_24(joined['Rank'])
                y = joined['time_mod24'].astype(float).values
                x_use = x0 if res.orientation == 'normal' else (24.0 - x0) % 24.0
                x_shift = (x_use + res.shift) % 24.0
                from scipy.stats import spearmanr
                sr_val = float(spearmanr(x_shift, y)[0])
                rows.append({'celltype': ct, 'R_spuare': r2, 'Pearson_R': r, 'Spearman_R': sr_val})
            metrics_df = pd.DataFrame(rows)
            metrics_path = os.path.join(shift_best_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Saved metrics table: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eigengene optimization with optional rank-vs-time plotting')
    parser.add_argument('--expression', '-e', default=None, help='Path to expression.csv')
    parser.add_argument('--metadata', '-m', default=None, help='Optional path to metadata.csv for rank-vs-time plots')
    parser.add_argument('--no-eigengene-weights', dest='use_eigengene_weights', action='store_false',
                        help='Disable automatic use of generated eigengene weights (default: enabled)')
    parser.add_argument('--weights-file', dest='weights_file', default=None,
                        help='Optional path to a CSV or whitespace-separated file with eigengene weights (one row)')
    args = parser.parse_args()
    # Pass weight-related CLI args into main
    main(expression_file=args.expression, metadata_file=args.metadata, use_eigengene_weights=args.use_eigengene_weights, weights_file=args.weights_file)