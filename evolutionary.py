import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import random
from collections import deque, defaultdict

class EvolutionaryAgent:
    """Evolutionary algorithm for weight perturbation and fitness selection"""
    
    def __init__(self, model: nn.Module, population_size: int = 10, 
                 mutation_rate: float = 0.01, elite_ratio: float = 0.2):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.elite_size = max(1, int(population_size * elite_ratio))
        
        # Population management
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_individual = None
        
        # History tracking
        self.fitness_history = deque(maxlen=1000)
        self.generation_history = deque(maxlen=1000)
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with slightly mutated versions of original model"""
        self.population = []
        
        # Original model as first individual
        original_state = copy.deepcopy(self.model.state_dict())
        self.population.append(original_state)
        
        # Create mutated versions
        for _ in range(self.population_size - 1):
            mutated_state = self._mutate_weights(original_state, mutation_factor=0.1)
            self.population.append(mutated_state)
        
        self.fitness_scores = [0.0] * self.population_size
    
    def _mutate_weights(self, state_dict: Dict, mutation_factor: float = None) -> Dict:
        """Apply Gaussian noise mutation to model weights"""
        if mutation_factor is None:
            mutation_factor = self.mutation_rate
            
        mutated_state = {}
        
        for key, param in state_dict.items():
            if param.dtype in [torch.float32, torch.float64]:
                # Apply Gaussian noise
                noise = torch.randn_like(param) * mutation_factor * torch.std(param)
                mutated_state[key] = param + noise
            else:
                mutated_state[key] = param.clone()
        
        return mutated_state
    
    def _crossover(self, parent1: Dict, parent2: Dict, crossover_rate: float = 0.5) -> Dict:
        """Perform uniform crossover between two parents"""
        child = {}
        
        for key in parent1.keys():
            if random.random() < crossover_rate:
                child[key] = parent1[key].clone()
            else:
                child[key] = parent2[key].clone()
        
        return child
    
    def evaluate_fitness(self, individual_idx: int, metrics: Dict) -> float:
        """Evaluate fitness of an individual based on multiple criteria"""
        
        # Extract relevant metrics
        loss = metrics.get('loss', 1.0)
        perplexity = metrics.get('perplexity', 1.0)
        entropy = metrics.get('entropy', np.log2(28))  # Max entropy for 28 tokens
        repetition = metrics.get('repetition', 0.0)
        stability = metrics.get('stability', 0.0)
        complexity = metrics.get('complexity', 0.0)
        
        # Fitness components (higher is better)
        loss_fitness = 1.0 / (1.0 + loss)  # Inverse loss
        perplexity_fitness = 1.0 / (1.0 + perplexity)  # Inverse perplexity
        entropy_fitness = (np.log2(28) - entropy) / np.log2(28)  # Entropy reduction
        repetition_fitness = min(1.0, repetition * 10)  # Moderate repetition is good
        stability_fitness = stability  # Direct stability score
        complexity_fitness = 1.0 - abs(complexity - 0.5)  # Optimal complexity around 0.5
        
        # Combined fitness with weights
        weights = {
            'loss': 0.3,
            'perplexity': 0.2,
            'entropy': 0.2,
            'repetition': 0.1,
            'stability': 0.15,
            'complexity': 0.05
        }
        
        fitness = (
            weights['loss'] * loss_fitness +
            weights['perplexity'] * perplexity_fitness +
            weights['entropy'] * entropy_fitness +
            weights['repetition'] * repetition_fitness +
            weights['stability'] * stability_fitness +
            weights['complexity'] * complexity_fitness
        )
        
        return fitness
    
    def select_parents(self) -> List[int]:
        """Select parents using tournament selection"""
        tournament_size = 3
        parents = []
        
        for _ in range(2):  # Select 2 parents
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            tournament_fitness = [(idx, self.fitness_scores[idx]) for idx in tournament_indices]
            winner = max(tournament_fitness, key=lambda x: x[1])[0]
            parents.append(winner)
        
        return parents
    
    def evolve_population(self, current_metrics: Dict) -> Tuple[nn.Module, float]:
        """Evolve population for one generation"""
        self.generation += 1
        
        # Evaluate fitness for all individuals
        for i in range(self.population_size):
            # Load individual into model for evaluation
            self.model.load_state_dict(self.population[i])
            self.fitness_scores[i] = self.evaluate_fitness(i, current_metrics)
        
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Update best individual
        if self.fitness_scores[sorted_indices[0]] > self.best_fitness:
            self.best_fitness = self.fitness_scores[sorted_indices[0]]
            self.best_individual = copy.deepcopy(self.population[sorted_indices[0]])
        
        # Create new population
        new_population = []
        
        # Keep elite individuals
        for i in range(self.elite_size):
            elite_idx = sorted_indices[i]
            new_population.append(copy.deepcopy(self.population[elite_idx]))
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            parents = self.select_parents()
            
            # Crossover
            if random.random() < 0.8:  # 80% crossover probability
                child = self._crossover(self.population[parents[0]], self.population[parents[1]])
            else:
                child = copy.deepcopy(self.population[parents[0]])
            
            # Mutation
            child = self._mutate_weights(child)
            
            new_population.append(child)
        
        # Replace population
        self.population = new_population[:self.population_size]
        
        # Update history
        mean_fitness = np.mean(self.fitness_scores)
        self.fitness_history.append(mean_fitness)
        self.generation_history.append(self.generation)
        
        # Return best individual
        self.model.load_state_dict(self.best_individual)
        return self.model, self.best_fitness
    
    def adaptive_mutation_rate(self) -> float:
        """Adapt mutation rate based on population diversity"""
        if len(self.fitness_history) < 10:
            return self.mutation_rate
        
        # Calculate fitness diversity
        recent_fitness = list(self.fitness_history)[-10:]
        fitness_std = np.std(recent_fitness)
        fitness_mean = np.mean(recent_fitness)
        
        # Adapt mutation rate based on diversity
        if fitness_std < 0.01:  # Low diversity - increase mutation
            return min(0.1, self.mutation_rate * 1.5)
        elif fitness_std > 0.1:  # High diversity - decrease mutation
            return max(0.001, self.mutation_rate * 0.5)
        else:
            return self.mutation_rate
    
    def get_population_statistics(self) -> Dict:
        """Get statistics about current population"""
        if not self.fitness_scores:
            return {}
        
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'mean_fitness': np.mean(self.fitness_scores),
            'fitness_std': np.std(self.fitness_scores),
            'population_diversity': np.std(self.fitness_scores),
            'mutation_rate': self.mutation_rate,
            'elite_size': self.elite_size
        }
    
    def save_checkpoint(self, filepath: str):
        """Save evolutionary state"""
        checkpoint = {
            'population': self.population,
            'fitness_scores': self.fitness_scores,
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'mutation_rate': self.mutation_rate,
            'population_size': self.population_size
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load evolutionary state"""
        checkpoint = torch.load(filepath)
        
        self.population = checkpoint['population']
        self.fitness_scores = checkpoint['fitness_scores']
        self.generation = checkpoint['generation']
        self.best_fitness = checkpoint['best_fitness']
        self.best_individual = checkpoint['best_individual']
        self.mutation_rate = checkpoint['mutation_rate']
        self.population_size = checkpoint['population_size']
        
        # Load best individual into model
        if self.best_individual is not None:
            self.model.load_state_dict(self.best_individual)

class FitnessScheduler:
    """Adaptive fitness scheduling for evolutionary optimization"""
    
    def __init__(self, initial_weights: Dict[str, float], adaptation_rate: float = 0.01):
        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.fitness_history = deque(maxlen=100)
        self.performance_history = defaultdict(deque)
        
    def update_weights(self, performance_improvement: Dict[str, float]):
        """Update fitness weights based on performance improvements"""
        for metric, improvement in performance_improvement.items():
            if metric in self.weights:
                # Increase weight for improving metrics, decrease for declining
                if improvement > 0:
                    self.weights[metric] *= (1 + self.adaptation_rate)
                else:
                    self.weights[metric] *= (1 - self.adaptation_rate)
                
                # Normalize weights
                total_weight = sum(self.weights.values())
                for key in self.weights:
                    self.weights[key] /= total_weight
                
                self.performance_history[metric].append(improvement)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current fitness weights"""
        return self.weights.copy()
    
    def get_weight_history(self) -> Dict[str, List[float]]:
        """Get weight adaptation history"""
        return {metric: list(history) for metric, history in self.performance_history.items()}

class EvolutionaryTracker:
    """Track evolutionary progress and generate reports"""
    
    def __init__(self):
        self.generation_data = []
        self.fitness_trends = defaultdict(list)
        self.diversity_metrics = defaultdict(list)
        
    def log_generation(self, generation: int, fitness_scores: List[float], 
                      population_stats: Dict, diversity_metrics: Dict):
        """Log data for one generation"""
        generation_data = {
            'generation': generation,
            'fitness_scores': fitness_scores.copy(),
            'best_fitness': max(fitness_scores),
            'mean_fitness': np.mean(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'diversity': diversity_metrics.get('diversity', 0.0),
            'population_stats': population_stats.copy()
        }
        
        self.generation_data.append(generation_data)
        
        # Update trends
        self.fitness_trends['best'].append(max(fitness_scores))
        self.fitness_trends['mean'].append(np.mean(fitness_scores))
        self.fitness_trends['std'].append(np.std(fitness_scores))
        
        for key, value in diversity_metrics.items():
            self.diversity_metrics[key].append(value)
    
    def get_evolution_report(self) -> Dict:
        """Generate comprehensive evolution report"""
        if not self.generation_data:
            return {}
        
        return {
            'total_generations': len(self.generation_data),
            'fitness_improvement': (
                self.generation_data[-1]['best_fitness'] - 
                self.generation_data[0]['best_fitness']
            ),
            'convergence_rate': self._calculate_convergence_rate(),
            'diversity_trends': dict(self.diversity_metrics),
            'fitness_trends': dict(self.fitness_trends),
            'final_statistics': self.generation_data[-1] if self.generation_data else {}
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate rate of convergence"""
        if len(self.fitness_trends['best']) < 10:
            return 0.0
        
        recent_improvements = []
        for i in range(len(self.fitness_trends['best']) - 10, len(self.fitness_trends['best']) - 1):
            improvement = self.fitness_trends['best'][i + 1] - self.fitness_trends['best'][i]
            recent_improvements.append(improvement)
        
        return np.mean(recent_improvements) if recent_improvements else 0.0
    
    def plot_evolution_progress(self, save_path: str = None):
        """Plot evolutionary progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evolutionary Algorithm Progress', fontsize=16)
        
        generations = range(len(self.generation_data))
        
        # Fitness evolution
        axes[0, 0].plot(generations, self.fitness_trends['best'], 'g-', label='Best Fitness', linewidth=2)
        axes[0, 0].plot(generations, self.fitness_trends['mean'], 'b-', label='Mean Fitness', linewidth=2)
        axes[0, 0].fill_between(generations, 
                               np.array(self.fitness_trends['mean']) - np.array(self.fitness_trends['std']),
                               np.array(self.fitness_trends['mean']) + np.array(self.fitness_trends['std']),
                               alpha=0.3, color='blue')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Diversity metrics
        if self.diversity_metrics:
            for key, values in self.diversity_metrics.items():
                axes[0, 1].plot(generations, values, label=key, linewidth=2)
            axes[0, 1].set_title('Population Diversity')
            axes[0, 1].set_ylabel('Diversity Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Fitness distribution over generations
        fitness_matrix = [gen['fitness_scores'] for gen in self.generation_data]
        axes[1, 0].imshow(np.array(fitness_matrix).T, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Fitness Distribution Heatmap')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Individual')
        
        # Convergence analysis
        if len(self.fitness_trends['best']) > 1:
            improvement_rates = np.diff(self.fitness_trends['best'])
            axes[1, 1].plot(generations[1:], improvement_rates, 'r-', linewidth=2)
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Fitness Improvement Rate')
            axes[1, 1].set_ylabel('Improvement per Generation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()