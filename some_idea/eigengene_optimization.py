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

# Add the my_cyclops directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'my_cyclops'))

def create_eigengenes(expression_scaled, n_components=50):
    """Create eigengenes using PCA"""
    print("Performing PCA transformation...")
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(expression_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    scaler = StandardScaler()
    components_normalized = scaler.fit_transform(components)
    
    return components_normalized, pca, explained_variance

def fit_sine_curve(x_data, y_data):
    """
    Fit a single-period sine curve to the data
    Function: y = A * sin(2*pi*x/period + phase) + offset
    """
    def sine_func(x, amplitude, phase, offset):
        # Fixed period to span the entire data range (one complete cycle)
        period = len(x_data)
        return amplitude * np.sin(2 * np.pi * x / period + phase) + offset
    
    try:
        # Initial parameter guesses
        amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2
        offset_guess = np.mean(y_data)
        phase_guess = 0.0
        
        # Fit the curve
        popt, _ = curve_fit(sine_func, x_data, y_data, 
                           p0=[amplitude_guess, phase_guess, offset_guess],
                           maxfev=2000)
        
        amplitude, phase, offset = popt
        
        # Generate smooth curve for plotting
        x_smooth = np.linspace(x_data[0], x_data[-1], 200)
        y_smooth = sine_func(x_smooth, amplitude, phase, offset)
        
        # Calculate R-squared
        y_pred = sine_func(x_data, amplitude, phase, offset)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return x_smooth, y_smooth, r_squared, popt
        
    except Exception as e:
        print(f"Warning: Sine fitting failed: {e}")
        # Return a flat line if fitting fails
        x_smooth = np.linspace(x_data[0], x_data[-1], 200)
        y_smooth = np.full_like(x_smooth, np.mean(y_data))
        return x_smooth, y_smooth, 0.0, [0, 0, np.mean(y_data)]

def multi_scale_optimize(x, weights=None, smoothness_factor=0.7, local_variation_factor=0.3, window_size=10):
    """
    Multi-scale optimization: balance global smoothness with local variation
    
    Parameters:
    x: data matrix (n_samples, n_dimensions)
    weights: dimension weights
    smoothness_factor: weight for global smoothness (0-1)
    local_variation_factor: weight for preserving local patterns (0-1)
    window_size: size of local windows for variation analysis
    """
    n_samples = x.shape[0]
    n_dims = x.shape[1]
    
    if weights is None:
        weights = np.ones(n_dims)
    else:
        weights = np.array(weights)
    
    # Step 1: Global smoothness optimization (TSP-based)
    def weighted_distance(xi, xj):
        return np.sum(weights * np.abs(xi - xj))
    
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance_matrix[i, j] = weighted_distance(x[i], x[j])
    
    # Enhanced distance function considering local variation
    def enhanced_distance(xi, xj, i, j):
        base_dist = weighted_distance(xi, xj)
        
        # Add local variation penalty - penalize too much smoothing in naturally varying regions
        local_variation_penalty = 0
        for dim in range(n_dims):
            # Calculate local standard deviation around these points
            window_start_i = max(0, i - window_size//2)
            window_end_i = min(n_samples, i + window_size//2)
            window_start_j = max(0, j - window_size//2)
            window_end_j = min(n_samples, j + window_size//2)
            
            local_std_i = np.std(x[window_start_i:window_end_i, dim]) if window_end_i > window_start_i + 1 else 0
            local_std_j = np.std(x[window_start_j:window_end_j, dim]) if window_end_j > window_start_j + 1 else 0
            
            # If both regions have high variation, allow more difference
            avg_local_std = (local_std_i + local_std_j) / 2
            if avg_local_std > 0:
                variation_tolerance = min(avg_local_std * local_variation_factor, base_dist * 0.5)
                local_variation_penalty -= variation_tolerance * weights[dim]
        
        return base_dist + local_variation_penalty
    
    # Build enhanced distance matrix
    enhanced_distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            enhanced_distance_matrix[i, j] = enhanced_distance(x[i], x[j], i, j)
    
    # Greedy + 2-opt with enhanced distances
    def greedy_tsp_enhanced():
        visited = [False] * n_samples
        path = [0]
        visited[0] = True
        current = 0
        
        for _ in range(n_samples - 1):
            min_dist = float('inf')
            next_node = -1
            
            for i in range(n_samples):
                if not visited[i]:
                    # Use combination of original and enhanced distance
                    combined_dist = (smoothness_factor * distance_matrix[current, i] + 
                                   local_variation_factor * enhanced_distance_matrix[current, i])
                    if combined_dist < min_dist:
                        min_dist = combined_dist
                        next_node = i
            
            if next_node != -1:
                path.append(next_node)
                visited[next_node] = True
                current = next_node
        
        return path
    
    def two_opt_improve_enhanced(path):
        def calculate_path_cost(p):
            cost = 0
            for i in range(len(p) - 1):
                cost += (smoothness_factor * distance_matrix[p[i], p[i + 1]] + 
                        local_variation_factor * enhanced_distance_matrix[p[i], p[i + 1]])
            return cost
        
        best_path = path[:]
        best_cost = calculate_path_cost(best_path)
        improved = True
        
        iterations = 0
        max_iterations = min(50, n_samples)  # Limit iterations to prevent over-optimization
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path)):
                    if j - i == 1: continue
                    
                    new_path = path[:i] + path[i:j][::-1] + path[j:]
                    new_cost = calculate_path_cost(new_path)
                    
                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost
                        improved = True
                        path = new_path
                        break
                if improved:
                    break
            iterations += 1
        
        return best_path
    
    # Execute optimization
    initial_path = greedy_tsp_enhanced()
    optimized_path = two_opt_improve_enhanced(initial_path)
    
    ranks = np.zeros(n_samples, dtype=int)
    ranks[np.array(optimized_path)] = np.arange(n_samples)
    
    return ranks.reshape(-1, 1)

def analyze_smoothness_and_variation(data, ranks):
    """Analyze both global smoothness and local variation characteristics"""
    ordered_data = data[ranks.flatten()]
    
    # Global smoothness metrics
    global_diff = np.mean(np.abs(ordered_data[1:] - ordered_data[:-1]))
    
    # Local variation metrics
    window_size = min(10, len(ordered_data) // 5)
    local_variations = []
    
    for i in range(len(ordered_data) - window_size + 1):
        window = ordered_data[i:i+window_size]
        local_std = np.std(window, axis=0)
        local_variations.append(np.mean(local_std))
    
    avg_local_variation = np.mean(local_variations)
    
    # Trend smoothness (using gradient)
    trends = []
    for dim in range(data.shape[1]):
        gradient = np.gradient(ordered_data[:, dim])
        trend_smoothness = 1.0 / (1.0 + np.std(gradient))  # Higher = smoother trend
        trends.append(trend_smoothness)
    
    avg_trend_smoothness = np.mean(trends)
    
    return {
        'global_smoothness': 1.0 / (1.0 + global_diff),  # Higher = smoother
        'local_variation': avg_local_variation,
        'trend_smoothness': avg_trend_smoothness,
        'balance_score': avg_trend_smoothness * (1.0 + 0.1 * avg_local_variation)  # Good balance score
    }

def load_and_process_expression_data(expression_file, n_components=50):
    """
    Load expression data and perform PCA to get eigengenes
    Based on the my_cyclops approach
    """
    print("=== Loading Expression Data ===")
    print(f"File: {expression_file}")
    
    # Load the data
    df = pd.read_csv(expression_file, low_memory=False)
    print(f"Data shape: {df.shape}")
    
    # Check for metadata rows
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    
    print(f"Contains celltype info: {has_celltype}")
    print(f"Contains time info: {has_time}")
    
    # Get sample columns (exclude Gene_Symbol)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    n_samples = len(sample_columns)
    print(f"Number of samples: {n_samples}")
    
    # Extract metadata if available
    celltypes = None
    times = None
    
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
        unique_celltypes = np.unique(celltypes)
        print(f"Cell types found: {unique_celltypes}")
    
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"Time range: {times.min():.2f} - {times.max():.2f} hours")
    
    # Get gene expression data (exclude metadata rows)
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    gene_names = gene_df['Gene_Symbol'].values
    
    # Convert expression data to numeric, handling any non-numeric values
    expression_data_raw = gene_df[sample_columns].values
    
    # Ensure all expression data is numeric
    print("Converting expression data to numeric...")
    expression_data = np.zeros((len(sample_columns), len(gene_names)), dtype=float)
    
    for i, sample in enumerate(sample_columns):
        for j, gene in enumerate(gene_names):
            try:
                value = float(expression_data_raw[j, i])
                expression_data[i, j] = value
            except (ValueError, TypeError):
                expression_data[i, j] = 0.0  # Set invalid values to 0
    
    print(f"Number of genes: {len(gene_names)}")
    print(f"Expression data shape: {expression_data.shape}")
    print(f"Expression data type: {expression_data.dtype}")
    print(f"Expression data range: {expression_data.min():.4f} to {expression_data.max():.4f}")
    
    # Standardize the data
    print("Standardizing expression data...")
    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_data)
    
    # Perform PCA to get eigengenes
    print(f"Performing PCA to get {n_components} eigengenes...")
    eigengenes, pca_model, explained_variance = create_eigengenes(
        expression_scaled, n_components
    )
    
    print(f"Eigengenes shape: {eigengenes.shape}")
    print(f"Explained variance ratio (first 10): {explained_variance[:10]}")
    print(f"Total explained variance: {explained_variance.sum():.4f}")
    
    return {
        'eigengenes': eigengenes,
        'original_expression': expression_data,
        'gene_names': gene_names,
        'sample_columns': sample_columns,
        'celltypes': celltypes,
        'times': times,
        'pca_model': pca_model,
        'scaler': scaler,
        'explained_variance': explained_variance,
        'n_components': n_components
    }

def get_circadian_gene_expressions(data_info, circadian_genes):
    """
    Extract expression data for specific circadian genes
    """
    gene_names = data_info['gene_names']
    original_expression = data_info['original_expression']
    
    # Find indices of circadian genes that exist in the data
    found_genes = []
    gene_expressions = []
    
    for gene in circadian_genes:
        if gene in gene_names:
            gene_idx = np.where(gene_names == gene)[0][0]
            gene_expr = original_expression[:, gene_idx]
            
            # Ensure numeric data
            if gene_expr.dtype == 'object' or not np.issubdtype(gene_expr.dtype, np.number):
                try:
                    gene_expr = pd.to_numeric(gene_expr, errors='coerce')
                    gene_expr = np.nan_to_num(gene_expr, nan=0.0)
                except:
                    gene_expr = np.zeros_like(gene_expr, dtype=float)
            
            gene_expressions.append(gene_expr)
            found_genes.append(gene)
        else:
            print(f"Warning: Gene {gene} not found in data")
    
    if len(found_genes) == 0:
        print("Error: No circadian genes found in data!")
        return None, None
    
    gene_expressions = np.array(gene_expressions, dtype=float).T  # samples x genes
    print(f"Found {len(found_genes)} circadian genes: {found_genes}")
    print(f"Circadian expression data shape: {gene_expressions.shape}")
    print(f"Circadian expression data type: {gene_expressions.dtype}")
    
    return gene_expressions, found_genes

def main():
    expression_file = r"d:/CriticalFile/Preprocess_CIRCADIA/CYCLOPS-python/data/Zhang_CancerCell_2025.Sample_SubCluster/expression.csv"
    n_components = 50
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"result_{timestamp}_components{n_components}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")
    
    circadian_genes = ['PER1', 'PER2', 'PER3', 'CRY1', 'CRY2', 'CLOCK', 'ARNTL', 'NR1D1', 'NR1D2', 'DBP']
    
    smoothness_configs = [
        {'smoothness_factor': 1.0, 'local_variation_factor': 0.0, 'name': 'Pure Smoothness'},
        {'smoothness_factor': 0.8, 'local_variation_factor': 0.2, 'name': 'Mostly Smooth'},
        {'smoothness_factor': 0.7, 'local_variation_factor': 0.3, 'name': 'Balanced'},
        {'smoothness_factor': 0.6, 'local_variation_factor': 0.4, 'name': 'More Variation'},
        {'smoothness_factor': 0.5, 'local_variation_factor': 0.5, 'name': 'Equal Balance'}
    ]
    
    # Component weights (give more weight to first few eigengenes)
    eigengene_weights = [1.0] + [0.8] * 10 + [0.6] * 15 + [0.4] * 24  # Total 50 components
    
    print("=== Eigengene-Based Multi-Scale Optimization ===")
    
    # Load and process data
    data_info = load_and_process_expression_data(expression_file, n_components)
    eigengenes = data_info['eigengenes']
    celltypes = data_info['celltypes']
    
    # Get circadian gene expressions for visualization
    circadian_expressions, found_circadian_genes = get_circadian_gene_expressions(data_info, circadian_genes)
    
    if circadian_expressions is None:
        print("Cannot proceed without circadian genes")
        return
    
    # Update output directory name with actual data info
    actual_output_dir = f"{output_dir}_genes{len(found_circadian_genes)}"
    if output_dir != actual_output_dir:
        os.rename(output_dir, actual_output_dir)
        output_dir = actual_output_dir
        print(f"Updated output directory to: {os.path.abspath(output_dir)}")
    
    # Process by cell type if available
    if celltypes is not None:
        unique_celltypes = np.unique(celltypes)
        eligible_celltypes = [ct for ct in unique_celltypes if np.sum(celltypes == ct) >= 10]
        print(f"\nCell types with >=10 samples: {eligible_celltypes}")
        
        # Further update output directory name with celltype info
        final_output_dir = f"{output_dir}_celltypes{len(eligible_celltypes)}"
        if output_dir != final_output_dir:
            os.rename(output_dir, final_output_dir)
            output_dir = final_output_dir
            print(f"Final output directory: {os.path.abspath(output_dir)}")
        
        # Process each cell type
        all_results = {}
        
        for celltype in eligible_celltypes:
            print(f"\n=== Processing Cell Type {celltype} ===")
            
            # Filter data for this cell type
            celltype_mask = celltypes == celltype
            celltype_eigengenes = eigengenes[celltype_mask]
            celltype_circadian = circadian_expressions[celltype_mask]
            n_samples = len(celltype_eigengenes)
            
            print(f"Samples: {n_samples}")
            
            # Test different optimization configurations
            celltype_results = {}
            
            for config in smoothness_configs:
                print(f"  Testing {config['name']}...")
                
                # Optimize using eigengenes
                ranks = multi_scale_optimize(
                    celltype_eigengenes, 
                    weights=eigengene_weights[:n_components],
                    smoothness_factor=config['smoothness_factor'],
                    local_variation_factor=config['local_variation_factor']
                )
                
                # Analyze the optimization results
                metrics = analyze_smoothness_and_variation(celltype_eigengenes, ranks)
                
                celltype_results[config['name']] = {
                    'ranks': ranks,
                    'metrics': metrics,
                    'config': config,
                    'eigengenes': celltype_eigengenes,
                    'circadian_expressions': celltype_circadian
                }
                
                print(f"    Balance Score: {metrics['balance_score']:.4f}")
            
            all_results[celltype] = celltype_results
            
            # Show best configuration for this cell type
            best_config = max(celltype_results.items(), key=lambda x: x[1]['metrics']['balance_score'])
            print(f"  Best config: {best_config[0]} (Score: {best_config[1]['metrics']['balance_score']:.4f})")
        
        # Create visualizations
        print("\n=== Creating Visualizations ===")
        
        # 1. Visualization using circadian genes (final result)
        create_circadian_gene_visualization(all_results, found_circadian_genes, eligible_celltypes, output_dir)
        
        # 2. Eigengene optimization comparison for one representative cell type
        if len(eligible_celltypes) > 0:
            create_eigengene_comparison_visualization(all_results, eligible_celltypes[0], n_components, output_dir)
        
        # 3. Summary statistics
        print_summary_statistics(all_results, output_dir)
        
    else:
        print("\nNo cell type information available, processing all samples together...")
        
        # Update output directory for single dataset
        final_output_dir = f"{output_dir}_single_dataset"
        if output_dir != final_output_dir:
            os.rename(output_dir, final_output_dir)
            output_dir = final_output_dir
            print(f"Final output directory: {os.path.abspath(output_dir)}")
        
        # Process all samples together
        results = {}
        n_samples = len(eigengenes)
        print(f"Total samples: {n_samples}")
        
        for config in smoothness_configs:
            print(f"Testing {config['name']}...")
            
            ranks = multi_scale_optimize(
                eigengenes,
                weights=eigengene_weights[:n_components],
                smoothness_factor=config['smoothness_factor'],
                local_variation_factor=config['local_variation_factor']
            )
            
            metrics = analyze_smoothness_and_variation(eigengenes, ranks)
            
            results[config['name']] = {
                'ranks': ranks,
                'metrics': metrics,
                'config': config,
                'eigengenes': eigengenes,
                'circadian_expressions': circadian_expressions
            }
            
            print(f"  Balance Score: {metrics['balance_score']:.4f}")
        
        # Create visualization for single dataset
        create_single_dataset_visualization(results, found_circadian_genes, output_dir)

def create_circadian_gene_visualization(all_results, circadian_genes, celltypes, output_dir):
    """Create visualization using original circadian genes"""
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
        # Use balanced configuration
        result = all_results[celltype]['Balanced']
        ranks = result['ranks']
        circadian_data = result['circadian_expressions']
        
        # Order samples according to optimization
        order_indices = np.argsort(ranks.flatten())
        ordered_circadian = circadian_data[order_indices]
        n_samples = len(ordered_circadian)
        
        for gene_idx, gene_name in enumerate(circadian_genes):
            ax = axes[ct_idx, gene_idx] if n_celltypes > 1 else axes[gene_idx]
            
            x_range = np.array(range(1, n_samples + 1))
            gene_values = ordered_circadian[:, gene_idx]
            
            # Ensure the data is numeric
            if gene_values.dtype == 'object' or not np.issubdtype(gene_values.dtype, np.number):
                try:
                    gene_values = pd.to_numeric(gene_values, errors='coerce')
                    gene_values = np.nan_to_num(gene_values, nan=0.0)
                except:
                    gene_values = np.zeros_like(gene_values, dtype=float)
            
            # Plot scatter points only (no connecting lines)
            scatter = ax.scatter(x_range, gene_values, 
                               c=range(n_samples), cmap='viridis', s=25, alpha=0.7, 
                               edgecolors='white', linewidth=0.5, zorder=3)
            
            # Fit and plot sine curve
            x_smooth, y_smooth, r_squared, fit_params = fit_sine_curve(x_range, gene_values)
            ax.plot(x_smooth, y_smooth, '-', color=colors[gene_idx], 
                   linewidth=2.5, alpha=0.8, label=f'Sine fit (R²={r_squared:.3f})', zorder=2)
            
            balance_score = result['metrics']['balance_score']
            ax.set_title(f'CT {celltype}: {gene_name}\nBalance={balance_score:.3f}, R²={r_squared:.3f}', 
                        fontsize=10, fontweight='bold', pad=8)
            ax.set_xlabel('Sample Rank', fontsize=9)
            ax.set_ylabel('Expression', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
            ax.set_xlim(0, n_samples + 1)
            
            # Set y-limits based on data range
            y_min, y_max = float(np.min(gene_values)), float(np.max(gene_values))
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ax.set_ylim(-0.1, 0.1)
            
            # Add legend for the first subplot
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
    """Create comparison visualization showing different optimization strategies"""
    print(f"Creating eigengene comparison for cell type {representative_celltype}...")
    
    celltype_results = all_results[representative_celltype]
    
    # Use first 5 eigengenes for visualization
    n_eigengenes_plot = min(5, n_components)
    
    _, axes = plt.subplots(len(celltype_results), n_eigengenes_plot, 
                            figsize=(4*n_eigengenes_plot, 3*len(celltype_results)))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for config_idx, (config_name, result) in enumerate(celltype_results.items()):
        ranks = result['ranks']
        eigengenes_data = result['eigengenes']
        metrics = result['metrics']
        
        order_indices = np.argsort(ranks.flatten())
        ordered_eigengenes = eigengenes_data[order_indices]
        n_samples = len(ordered_eigengenes)
        
        for eigen_idx in range(n_eigengenes_plot):
            ax = axes[config_idx, eigen_idx]
            
            x_range = np.array(range(1, n_samples + 1))
            eigen_values = ordered_eigengenes[:, eigen_idx]
            
            # Ensure the data is numeric
            if eigen_values.dtype == 'object' or not np.issubdtype(eigen_values.dtype, np.number):
                try:
                    eigen_values = pd.to_numeric(eigen_values, errors='coerce')
                    eigen_values = np.nan_to_num(eigen_values, nan=0.0)
                except:
                    eigen_values = np.zeros_like(eigen_values, dtype=float)
            
            # Plot scatter points only (no connecting lines)
            ax.scatter(x_range, eigen_values, 
                        c=range(n_samples), cmap='viridis', s=20, alpha=0.7, 
                        edgecolors='white', linewidth=0.3, zorder=3)
        
            # Fit and plot sine curve
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
            
            # Set y-limits based on data range
            y_min, y_max = float(np.min(eigen_values)), float(np.max(eigen_values))
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ax.set_ylim(-0.1, 0.1)
            
            # Add legend for the first subplot
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
            
            # Ensure the data is numeric
            if gene_values.dtype == 'object' or not np.issubdtype(gene_values.dtype, np.number):
                try:
                    gene_values = pd.to_numeric(gene_values, errors='coerce')
                    gene_values = np.nan_to_num(gene_values, nan=0.0)
                except:
                    gene_values = np.zeros_like(gene_values, dtype=float)
            
            # Plot scatter points only (no connecting lines)
            scatter = ax.scatter(x_range, gene_values, 
                               c=range(n_samples), cmap='viridis', s=25, alpha=0.7, 
                               edgecolors='white', linewidth=0.5, zorder=3)
            
            # Fit and plot sine curve
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
            
            # Add legend for the first subplot
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
    """Print summary statistics for all cell types and configurations"""
    print("\n=== Summary Statistics ===")
    
    # Also save summary to a text file
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
    
    # Save detailed results as CSV
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
    print("Generated files:")
    print("  - circadian_genes_eigengene_optimization.png: Main visualization with circadian genes")
    print("  - eigengene_optimization_comparison.png: Optimization strategy comparison")
    print("  - optimization_summary.txt: Summary statistics and recommendations")
    print("  - detailed_results.csv: Complete numerical results")

if __name__ == "__main__":
    main()
