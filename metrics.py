import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict

import scipy.stats
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class ConceptEmergenceMetrics:
    """Comprehensive metrics for measuring concept emergence in ZD-CBE"""
    
    def __init__(self, vocab_size: int = 28, vq_codebook_size: int = 64):
        self.vocab_size = vocab_size
        self.vq_codebook_size = vq_codebook_size
        
        # History buffers
        self.loss_history = deque(maxlen=10000)
        self.perplexity_history = deque(maxlen=10000)
        self.sequence_history = deque(maxlen=1000)
        self.vq_usage_history = deque(maxlen=1000)
        self.embedding_history = deque(maxlen=1000)
        
        # Computed metrics
        self.entropy_history = deque(maxlen=10000)
        self.stability_history = deque(maxlen=10000)
        self.repetition_history = deque(maxlen=10000)
        self.fitness_history = deque(maxlen=10000)
        self.complexity_history = deque(maxlen=10000)

    @staticmethod
    def _to_numpy(data):
        if data is None:
            return np.array([])
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            data = np.asarray(data)
        return data

    def update_basic_metrics(self, loss: float, perplexity: float, 
                           sequence: str, vq_usage: np.ndarray, 
                           embeddings: Optional[np.ndarray] = None):
        """Update basic metrics from model outputs"""
        self.loss_history.append(loss)
        self.perplexity_history.append(perplexity)
        self.sequence_history.append(sequence)
        vq_usage_array = self._to_numpy(vq_usage).astype(np.float64, copy=False).reshape(-1)
        self.vq_usage_history.append(vq_usage_array.copy())
        
        if embeddings is not None:
            embeddings_array = self._to_numpy(embeddings)
            self.embedding_history.append(np.array(embeddings_array, copy=True))
    
    def calculate_entropy_reduction(self, predictions: torch.Tensor) -> float:
        """Calculate Shannon entropy reduction in predictions"""
        # Convert logits to probabilities
        probs = torch.softmax(predictions, dim=-1)
        
        # Calculate entropy for each position
        entropies = []
        for prob_dist in probs:
            # Avoid log(0)
            prob_dist = prob_dist + 1e-10
            entropy = -torch.sum(prob_dist * torch.log2(prob_dist))
            entropies.append(entropy.item())
        
        mean_entropy = np.mean(entropies)
        self.entropy_history.append(mean_entropy)
        
        # Calculate reduction from maximum entropy
        max_entropy = np.log2(self.vocab_size)
        entropy_reduction = max_entropy - mean_entropy
        
        return entropy_reduction
    
    def calculate_vq_usage_concentration(self, vq_usage: np.ndarray) -> Dict[str, float]:
        """Calculate VQ code usage concentration metrics"""
        vq_usage_array = self._to_numpy(vq_usage)
        if vq_usage_array.size == 0:
            return {'concentration': 0.0, 'diversity': 0.0, 'gini': 0.0}
        vq_usage_array = np.asarray(vq_usage_array, dtype=np.float64).reshape(-1)

        total_usage = np.sum(vq_usage_array)
        if total_usage <= 0:
            return {'concentration': 0.0, 'diversity': 0.0, 'gini': 0.0}
        
        # Normalize usage counts
        usage_probs = vq_usage_array / total_usage

        # Calculate concentration (inverse of entropy)
        entropy = -np.sum(usage_probs * np.log2(usage_probs + 1e-10))
        max_entropy = np.log2(self.vq_codebook_size)
        concentration = 1.0 - (entropy / max_entropy)
        
        # Calculate diversity (effective number of codes)
        diversity = np.exp(entropy)
        
        # Calculate Gini coefficient for inequality
        sorted_usage = np.sort(usage_probs)
        n = sorted_usage.size
        cumulative = np.cumsum(sorted_usage)
        if n > 0:
            gini = 1.0 - (2.0 / n) * np.sum(cumulative) + (1.0 / n)
            gini = float(np.clip(gini, 0.0, 1.0))
        else:
            gini = 0.0
        
        return {
            'concentration': concentration,
            'diversity': diversity / self.vq_codebook_size,  # Normalize to [0,1]
            'gini': gini
        }
    
    def calculate_repetition_index(self, sequence: str, window_sizes: List[int] = [3, 5, 7]) -> float:
        """Calculate repetition index across multiple window sizes"""
        if len(sequence) < max(window_sizes):
            return 0.0
        
        total_repetitions = 0
        total_patterns = 0
        
        for window_size in window_sizes:
            if len(sequence) >= window_size:
                patterns = defaultdict(int)
                
                # Count all patterns of this window size
                for i in range(len(sequence) - window_size + 1):
                    pattern = sequence[i:i+window_size]
                    patterns[pattern] += 1
                
                # Calculate repetitions
                repetitions = sum(count - 1 for count in patterns.values() if count > 1)
                total_repetitions += repetitions
                total_patterns += len(patterns)
        
        repetition_index = total_repetitions / (total_patterns + 1e-10)
        self.repetition_history.append(repetition_index)
        
        return repetition_index
    
    def calculate_conceptual_stability(self, current_embeddings: Optional[np.ndarray] = None) -> float:
        """Calculate conceptual stability over time"""
        if len(self.embedding_history) < 10 or current_embeddings is None:
            return 0.0
        
        # Calculate stability based on embedding consistency
        recent_embeddings = list(self.embedding_history)[-10:]
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(recent_embeddings) - 1):
            emb1 = recent_embeddings[i].flatten()
            emb2 = recent_embeddings[i + 1].flatten()
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
            similarities.append(similarity)
        
        stability = np.mean(similarities)
        self.stability_history.append(stability)
        
        return stability
    
    def calculate_emergent_complexity(self, sequence: str, vq_codes: np.ndarray) -> float:
        """Calculate emergent complexity metric"""
        # Sequence complexity (normalized unique subsequences)
        unique_subsequences = len(set(sequence[i:i+3] for i in range(len(sequence) - 2)))
        
        max_possible = min(len(sequence) - 2, 100)  # Cap at reasonable limit
        sequence_complexity = unique_subsequences / max_possible if max_possible > 0 else 0
        
        # VQ code complexity (clustering coefficient)
        vq_codes_array = self._to_numpy(vq_codes)

        if vq_codes_array.size > 1:
            if vq_codes_array.ndim == 1:
                samples = vq_codes_array.reshape(-1, 1)
            else:
                samples = vq_codes_array
            # Simplified clustering based on code distances
            distances = []
            max_samples = min(samples.shape[0], 20)
            for i in range(max_samples):  # Limit for efficiency
                for j in range(i+1, max_samples):
                    dist = np.linalg.norm(samples[i] - samples[j])
                    distances.append(dist)

            if distances:
                vq_complexity = np.std(distances) / (np.mean(distances) + 1e-10)
            else:
                vq_complexity = 0.0
        else:
            vq_complexity = 0.0
        
        # Combined complexity
        complexity = (sequence_complexity + vq_complexity) / 2
        self.complexity_history.append(complexity)
        
        return complexity
    
    def calculate_evolutionary_fitness(self, current_step: int) -> float:
        """Calculate evolutionary fitness score"""
        if len(self.loss_history) < 10:
            return 0.0
        
        # Fitness based on multiple factors
        recent_losses = list(self.loss_history)[-10:]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Normalize loss (lower is better)
        normalized_loss = 1.0 / (1.0 + np.mean(recent_losses))
        
        # Stability component (higher stability is better)
        stability = self.stability_history[-1] if self.stability_history else 0.0
        
        # Complexity component (moderate complexity is optimal)
        complexity = self.complexity_history[-1] if self.complexity_history else 0.0
        optimal_complexity = 0.5
        complexity_score = 1.0 - abs(complexity - optimal_complexity)
        
        # Combined fitness
        fitness = (normalized_loss + max(0, -loss_trend) + stability + complexity_score) / 4
        self.fitness_history.append(fitness)
        
        return fitness
    
    def calculate_mutual_information_matrix(self, sequences: List[str]) -> np.ndarray:
        """Calculate mutual information between positions in sequences"""
        if not sequences or len(sequences[0]) < 2:
            return np.array([])
        
        seq_length = min(len(seq) for seq in sequences)
        mi_matrix = np.zeros((seq_length, seq_length))
        
        for i in range(seq_length):
            for j in range(i+1, seq_length):
                # Get character distributions at positions i and j
                chars_i = [seq[i] for seq in sequences if len(seq) > i]
                chars_j = [seq[j] for seq in sequences if len(seq) > j]
                
                if chars_i and chars_j:
                    # Convert to numerical for MI calculation
                    unique_chars = list(set(chars_i + chars_j))
                    char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
                    
                    nums_i = [char_to_num[char] for char in chars_i]
                    nums_j = [char_to_num[char] for char in chars_j]
                    
                    mi = mutual_info_score(nums_i, nums_j)
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
        
        return mi_matrix
    
    def analyze_concept_clusters(self, embeddings: np.ndarray, n_clusters: int = 8) -> Dict:
        """Analyze concept clusters in embedding space"""
        if len(embeddings) < n_clusters:
            return {}
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate cluster metrics
        cluster_counts = np.bincount(clusters, minlength=n_clusters)
        cluster_sizes = cluster_counts[cluster_counts > 0]
        
        # Silhouette score (simplified)
        from sklearn.metrics import silhouette_score
        if len(set(clusters)) > 1:
            silhouette = silhouette_score(embeddings, clusters)
        else:
            silhouette = 0.0
        
        return {
            'n_clusters': len(cluster_sizes),
            'cluster_sizes': cluster_sizes.tolist(),
            'silhouette_score': silhouette,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'embeddings_2d': embeddings_2d.tolist()
        }
    
    def generate_comprehensive_report(self) -> Dict[str, any]:
        """Generate comprehensive emergence report"""
        report = {
            'basic_metrics': {
                'final_loss': self.loss_history[-1] if self.loss_history else 0.0,
                'loss_trend': np.polyfit(range(len(self.loss_history)), list(self.loss_history), 1)[0] if len(self.loss_history) > 1 else 0.0,
                'final_perplexity': self.perplexity_history[-1] if self.perplexity_history else 0.0,
                'final_entropy': self.entropy_history[-1] if self.entropy_history else 0.0,
            },
            'concept_emergence': {
                'vq_concentration': self.calculate_vq_usage_concentration(self.vq_usage_history[-1]) if self.vq_usage_history else {},
                'repetition_index': self.repetition_history[-1] if self.repetition_history else 0.0,
                'conceptual_stability': self.stability_history[-1] if self.stability_history else 0.0,
                'emergent_complexity': self.complexity_history[-1] if self.complexity_history else 0.0,
            },
            'evolution_metrics': {
                'final_fitness': self.fitness_history[-1] if self.fitness_history else 0.0,
                'fitness_trend': np.polyfit(range(len(self.fitness_history)), list(self.fitness_history), 1)[0] if len(self.fitness_history) > 1 else 0.0,
            }
        }
        
        # Add sequence analysis if we have sequences
        if self.sequence_history:
            recent_sequences = list(self.sequence_history)[-100:]
            mi_matrix = self.calculate_mutual_information_matrix(recent_sequences)
            report['sequence_analysis'] = {
                'mutual_information_mean': np.mean(mi_matrix) if mi_matrix.size > 0 else 0.0,
                'sequence_diversity': len(set(recent_sequences)) / len(recent_sequences),
                'average_sequence_length': np.mean([len(seq) for seq in recent_sequences]),
            }
        
        # Add embedding analysis if we have embeddings
        if self.embedding_history:
            recent_embeddings = np.array(list(self.embedding_history)[-10:])
            cluster_analysis = self.analyze_concept_clusters(recent_embeddings.reshape(-1, recent_embeddings.shape[-1]))
            report['embedding_analysis'] = cluster_analysis
        
        return report
    
    def plot_emergence_evolution(self, save_path: str = None):
        """Create comprehensive emergence evolution plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Concept Emergence Evolution', fontsize=16, fontweight='bold')
        
        # Loss evolution
        axes[0, 0].plot(list(self.loss_history), color='red', alpha=0.7, linewidth=2)
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Entropy reduction
        if self.entropy_history:
            axes[0, 1].plot(list(self.entropy_history), color='blue', alpha=0.7, linewidth=2)
            axes[0, 1].set_title('Entropy Evolution')
            axes[0, 1].set_ylabel('Entropy (bits)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Perplexity evolution
        if self.perplexity_history:
            axes[0, 2].plot(list(self.perplexity_history), color='green', alpha=0.7, linewidth=2)
            axes[0, 2].set_title('VQ Perplexity Evolution')
            axes[0, 2].set_ylabel('Perplexity')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Repetition index
        if self.repetition_history:
            axes[1, 0].plot(list(self.repetition_history), color='orange', alpha=0.7, linewidth=2)
            axes[1, 0].set_title('Repetition Index')
            axes[1, 0].set_ylabel('Repetition Index')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Conceptual stability
        if self.stability_history:
            axes[1, 1].plot(list(self.stability_history), color='purple', alpha=0.7, linewidth=2)
            axes[1, 1].set_title('Conceptual Stability')
            axes[1, 1].set_ylabel('Stability')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Fitness evolution
        if self.fitness_history:
            axes[1, 2].plot(list(self.fitness_history), color='brown', alpha=0.7, linewidth=2)
            axes[1, 2].set_title('Evolutionary Fitness')
            axes[1, 2].set_ylabel('Fitness Score')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_metrics(self, filepath: str):
        """Save all metrics to file"""
        import pickle
        
        metrics_data = {
            'loss_history': list(self.loss_history),
            'perplexity_history': list(self.perplexity_history),
            'entropy_history': list(self.entropy_history),
            'repetition_history': list(self.repetition_history),
            'stability_history': list(self.stability_history),
            'fitness_history': list(self.fitness_history),
            'complexity_history': list(self.complexity_history),
            'sequence_history': list(self.sequence_history),
            'vq_usage_history': list(self.vq_usage_history),
            'embedding_history': list(self.embedding_history),
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metrics_data, f)
    
    def load_metrics(self, filepath: str):
        """Load metrics from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            metrics_data = pickle.load(f)
        
        # Restore histories
        for key, values in metrics_data.items():
            if hasattr(self, key):
                history = getattr(self, key)
                history.clear()
                for value in values:
                    history.append(value)