import numpy as np
from typing import Optional, Dict
from neural_network import neural_multi_scale_optimize

class MultiScaleOptimizer:
    
    def __init__(self, 
                 smoothness_factor: float = 0.7,
                 local_variation_factor: float = 0.3,
                 window_size: int = 10,
                 max_iterations_ratio: int = 50,
                 variation_tolerance_ratio: float = 0.5,
                 method: str = 'greedy'):
        self.smoothness_factor = smoothness_factor
        self.local_variation_factor = local_variation_factor
        self.window_size = window_size
        self.max_iterations_ratio = max_iterations_ratio
        self.variation_tolerance_ratio = variation_tolerance_ratio
        self.method = method
    
    def weighted_distance(self, xi: np.ndarray, xj: np.ndarray, weights: np.ndarray) -> float:
        return np.sum(weights * np.abs(xi - xj))
    
    def compute_distance_matrix(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = self.weighted_distance(x[i], x[j], weights)
        
        return distance_matrix
    
    def enhanced_distance(self, xi: np.ndarray, xj: np.ndarray, 
                         i: int, j: int, x: np.ndarray, weights: np.ndarray) -> float:
        base_dist = self.weighted_distance(xi, xj, weights)
        n_samples, n_dims = x.shape
        
        local_variation_penalty = 0
        for dim in range(n_dims):
            window_start_i = max(0, i - self.window_size//2)
            window_end_i = min(n_samples, i + self.window_size//2)
            window_start_j = max(0, j - self.window_size//2)
            window_end_j = min(n_samples, j + self.window_size//2)
            
            local_std_i = np.std(x[window_start_i:window_end_i, dim]) if window_end_i > window_start_i + 1 else 0
            local_std_j = np.std(x[window_start_j:window_end_j, dim]) if window_end_j > window_start_j + 1 else 0
            
            avg_local_std = (local_std_i + local_std_j) / 2
            if avg_local_std > 0:
                variation_tolerance = min(
                    avg_local_std * self.local_variation_factor, 
                    base_dist * self.variation_tolerance_ratio
                )
                local_variation_penalty -= variation_tolerance * weights[dim]
        
        return base_dist + local_variation_penalty
    
    def compute_enhanced_distance_matrix(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        enhanced_distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                enhanced_distance_matrix[i, j] = self.enhanced_distance(
                    x[i], x[j], i, j, x, weights
                )
        
        return enhanced_distance_matrix
    
    def greedy_tsp_construction(self, distance_matrix: np.ndarray, 
                               enhanced_distance_matrix: np.ndarray) -> list:
        n_samples = distance_matrix.shape[0]
        visited = [False] * n_samples
        path = [0]
        visited[0] = True
        current = 0
        
        for _ in range(n_samples - 1):
            min_dist = float('inf')
            next_node = -1
            
            for i in range(n_samples):
                if not visited[i]:
                    combined_dist = (
                        self.smoothness_factor * distance_matrix[current, i] + 
                        self.local_variation_factor * enhanced_distance_matrix[current, i]
                    )
                    
                    if combined_dist < min_dist:
                        min_dist = combined_dist
                        next_node = i
            
            if next_node != -1:
                path.append(next_node)
                visited[next_node] = True
                current = next_node
        
        return path
    
    def calculate_path_cost(self, path: list, distance_matrix: np.ndarray, 
                           enhanced_distance_matrix: np.ndarray) -> float:
        cost = 0
        for i in range(len(path) - 1):
            cost += (
                self.smoothness_factor * distance_matrix[path[i], path[i + 1]] + 
                self.local_variation_factor * enhanced_distance_matrix[path[i], path[i + 1]]
            )
        return cost
    
    def two_opt_improvement(self, path: list, distance_matrix: np.ndarray, 
                           enhanced_distance_matrix: np.ndarray) -> list:
        best_path = path[:]
        best_cost = self.calculate_path_cost(best_path, distance_matrix, enhanced_distance_matrix)
        improved = True
        
        iterations = 0
        max_iterations = min(self.max_iterations_ratio, len(path))
        
        while improved and iterations < max_iterations:
            improved = False
            
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path)):
                    if j - i == 1:
                        continue
                    
                    new_path = path[:i] + path[i:j][::-1] + path[j:]
                    new_cost = self.calculate_path_cost(new_path, distance_matrix, enhanced_distance_matrix)
                    
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
    
    def optimize(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        if self.method == 'neural':
            return self._neural_optimize(x, weights)
        else:
            return self._greedy_optimize(x, weights)
    
    def _greedy_optimize(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        n_samples, n_dims = x.shape
        
        if weights is None:
            weights = np.ones(n_dims)
        else:
            weights = np.array(weights)
        
        print("Computing distance matrices...")
        distance_matrix = self.compute_distance_matrix(x, weights)
        enhanced_distance_matrix = self.compute_enhanced_distance_matrix(x, weights)
        
        print("Constructing initial path...")
        initial_path = self.greedy_tsp_construction(distance_matrix, enhanced_distance_matrix)
        
        print("Improving path with 2-opt...")
        optimized_path = self.two_opt_improvement(initial_path, distance_matrix, enhanced_distance_matrix)
        
        ranks = np.zeros(n_samples, dtype=int)
        ranks[np.array(optimized_path)] = np.arange(n_samples)
        
        return ranks.reshape(-1, 1)
    
    def _neural_optimize(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        return neural_multi_scale_optimize(x, weights, n_epochs=50)

    def analyze_metrics(self, data: np.ndarray, ranks: np.ndarray) -> Dict[str, float]:
        ordered_data = data[ranks.flatten()]
        
        global_diff = np.mean(np.abs(ordered_data[1:] - ordered_data[:-1]))
        global_smoothness = 1.0 / (1.0 + global_diff)
        
        window_size = min(self.window_size, len(ordered_data) // 5)
        local_variations = []
        
        for i in range(len(ordered_data) - window_size + 1):
            window = ordered_data[i:i+window_size]
            local_std = np.std(window, axis=0)
            local_variations.append(np.mean(local_std))
        
        avg_local_variation = np.mean(local_variations)
        
        trends = []
        for dim in range(data.shape[1]):
            gradient = np.gradient(ordered_data[:, dim])
            trend_smoothness = 1.0 / (1.0 + np.std(gradient))
            trends.append(trend_smoothness)
        
        avg_trend_smoothness = np.mean(trends)
        
        balance_score = avg_trend_smoothness * (1.0 + 0.1 * avg_local_variation)
        
        return {
            'global_smoothness': global_smoothness,
            'local_variation': avg_local_variation,
            'trend_smoothness': avg_trend_smoothness,
            'balance_score': balance_score
        }


def create_optimizer_configs() -> list:
    """
    Create predefined optimizer configurations for different strategies
    """
    configs = [
        {
            'name': 'Pure Smoothness',
            'smoothness_factor': 1.0,
            'local_variation_factor': 0.0,
            'method': 'greedy'
        },
        {
            'name': 'Mostly Smooth',
            'smoothness_factor': 0.8,
            'local_variation_factor': 0.2,
            'method': 'greedy'
        },
        {
            'name': 'Balanced',
            'smoothness_factor': 0.7,
            'local_variation_factor': 0.3,
            'method': 'greedy'
        },
        {
            'name': 'More Variation',
            'smoothness_factor': 0.6,
            'local_variation_factor': 0.4,
            'method': 'greedy'
        },
        {
            'name': 'Neural Balanced',  # 新增神经网络配置
            'smoothness_factor': 0.7,
            'local_variation_factor': 0.3,
            'method': 'neural'
        }
    ]
    
    return configs


def multi_scale_optimize(x: np.ndarray, 
                        weights: Optional[np.ndarray] = None,
                        smoothness_factor: float = 0.7,
                        local_variation_factor: float = 0.3,
                        window_size: int = 10) -> np.ndarray:
    """
    Convenience function for backward compatibility
    """
    optimizer = MultiScaleOptimizer(
        smoothness_factor=smoothness_factor,
        local_variation_factor=local_variation_factor,
        window_size=window_size
    )
    
    return optimizer.optimize(x, weights)


def analyze_smoothness_and_variation(data: np.ndarray, ranks: np.ndarray) -> Dict[str, float]:
    """
    Convenience function for backward compatibility
    """
    optimizer = MultiScaleOptimizer()
    return optimizer.analyze_metrics(data, ranks)