import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import List, Dict, Optional
import time
from collections import deque
import seaborn as sns

class RealTimeVisualizer:
    """Real-time visualization for the dreaming and awakening phases"""
    
    def __init__(self, num_vq_codes: int = 64, max_sequence_length: int = 100):
        self.num_vq_codes = num_vq_codes
        self.max_sequence_length = max_sequence_length
        
        # Data storage
        self.loss_history = deque(maxlen=1000)
        self.vq_usage_history = deque(maxlen=100)
        self.sequence_history = deque(maxlen=max_sequence_length)
        self.perplexity_history = deque(maxlen=1000)
        
        # Styling - scientific journal aesthetic
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'primary': '#2C3E50',      # Deep blue-gray
            'secondary': '#34495E',     # Darker blue-gray
            'accent': '#E74C3C',       # Vibrant red
            'success': '#27AE60',      # Green
            'warning': '#F39C12',      # Orange
            'background': '#FDFEFE',   # Clean white
            'grid': '#ECF0F1'          # Light gray
        }
        
        # Initialize plots
        self.fig = None
        self.axes = {}
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize the matplotlib figure and subplots"""
        self.fig = plt.figure(figsize=(16, 12), facecolor=self.colors['background'])
        self.fig.suptitle('Zero-Data Cognitive Bootstrap Engine - Real-Time Visualization', 
                         fontsize=20, fontweight='bold', color=self.colors['primary'])
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1])
        
        # 1. Generated Sequence (top, spans 2 columns)
        self.axes['sequence'] = self.fig.add_subplot(gs[0, :2])
        self.axes['sequence'].set_title('Generated Dream Sequence', fontsize=14, fontweight='bold', 
                                       color=self.colors['primary'])
        self.axes['sequence'].set_xlabel('Position', fontsize=12)
        self.axes['sequence'].set_ylabel('Token', fontsize=12)
        
        # 2. VQ Code Usage Heatmap (top right)
        self.axes['vq_heatmap'] = self.fig.add_subplot(gs[0, 2])
        self.axes['vq_heatmap'].set_title('VQ Code Usage', fontsize=14, fontweight='bold', 
                                         color=self.colors['primary'])
        
        # 3. Loss Curve (middle left)
        self.axes['loss'] = self.fig.add_subplot(gs[1, 0])
        self.axes['loss'].set_title('Prediction Loss Over Time', fontsize=14, fontweight='bold', 
                                   color=self.colors['primary'])
        self.axes['loss'].set_xlabel('Steps', fontsize=12)
        self.axes['loss'].set_ylabel('Loss', fontsize=12)
        
        # 4. Perplexity Curve (middle center)
        self.axes['perplexity'] = self.fig.add_subplot(gs[1, 1])
        self.axes['perplexity'].set_title('VQ Perplexity', fontsize=14, fontweight='bold', 
                                         color=self.colors['primary'])
        self.axes['perplexity'].set_xlabel('Steps', fontsize=12)
        self.axes['perplexity'].set_ylabel('Perplexity', fontsize=12)
        
        # 5. Concept Stability (middle right)
        self.axes['stability'] = self.fig.add_subplot(gs[1, 2])
        self.axes['stability'].set_title('Concept Stability', fontsize=14, fontweight='bold', 
                                        color=self.colors['primary'])
        
        # 6. Entropy Reduction (bottom left)
        self.axes['entropy'] = self.fig.add_subplot(gs[2, 0])
        self.axes['entropy'].set_title('Entropy Reduction', fontsize=14, fontweight='bold', 
                                      color=self.colors['primary'])
        self.axes['entropy'].set_xlabel('Steps', fontsize=12)
        self.axes['entropy'].set_ylabel('Entropy', fontsize=12)
        
        # 7. Repetition Index (bottom center)
        self.axes['repetition'] = self.fig.add_subplot(gs[2, 1])
        self.axes['repetition'].set_title('Repetition Index', fontsize=14, fontweight='bold', 
                                         color=self.colors['primary'])
        
        # 8. Fitness Evolution (bottom right)
        self.axes['fitness'] = self.fig.add_subplot(gs[2, 2])
        self.axes['fitness'].set_title('Evolutionary Fitness', fontsize=14, fontweight='bold', 
                                      color=self.colors['primary'])
        
        # Style all subplots
        for ax_name, ax in self.axes.items():
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.set_facecolor(self.colors['background'])
            for spine in ax.spines.values():
                spine.set_color(self.colors['secondary'])
                spine.set_linewidth(1.5)
        
        plt.tight_layout()
        plt.ion()
        
    def update_sequence(self, sequence: str, recent_tokens: List[int]):
        """Update the generated sequence visualization"""
        ax = self.axes['sequence']
        ax.clear()
        
        # Convert sequence to numerical representation for visualization
        numerical_seq = []
        for char in sequence[-50:]:  # Show last 50 characters
            if char.isalpha():
                numerical_seq.append(ord(char) - ord('A'))
            elif char == '0':
                numerical_seq.append(26)
            elif char == '1':
                numerical_seq.append(27)
        
        # Create color map for tokens
        colors = []
        for val in numerical_seq:
            if val < 26:  # Letters
                colors.append(plt.cm.Set3(val % 12))
            else:  # Numbers
                colors.append(self.colors['accent'])
        
        # Plot as colored bars
        bars = ax.bar(range(len(numerical_seq)), [1] * len(numerical_seq), 
                     color=colors, alpha=0.7, edgecolor=self.colors['secondary'])
        
        # Add character labels
        for i, (bar, char) in enumerate(zip(bars, sequence[-50:])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   char, ha='center', va='bottom', fontsize=10, 
                   fontweight='bold', color=self.colors['primary'])
        
        ax.set_title('Generated Dream Sequence', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'])
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Token', fontsize=12)
        ax.set_ylim(0, 2)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
    def update_vq_heatmap(self, vq_usage: np.ndarray):
        """Update VQ code usage heatmap"""
        ax = self.axes['vq_heatmap']
        ax.clear()
        
        if vq_usage is None:
            return

        usage_array = np.asarray(vq_usage, dtype=np.float64)

        if usage_array.ndim > 1:
            # Collapse extra dimensions assuming last axis corresponds to codebook size
            if usage_array.shape[-1] == self.num_vq_codes:
                usage_array = usage_array.reshape(-1, self.num_vq_codes).sum(axis=0)
            else:
                usage_array = usage_array.reshape(-1)

        if usage_array.size == 0:
            return

        if usage_array.size != self.num_vq_codes:
            if usage_array.size > self.num_vq_codes:
                usage_array = usage_array[:self.num_vq_codes]
            else:
                usage_array = np.pad(usage_array, (0, self.num_vq_codes - usage_array.size))

        usage_matrix = usage_array.reshape(8, 8)
        
        # Create heatmap
        if len(vq_usage) > 0:
            im = ax.imshow(usage_matrix, cmap='YlOrRd', aspect='auto', 
                          interpolation='nearest')
            
            # Add text annotations
            for i in range(8):
                for j in range(8):
                    text = ax.text(j, i, f'{usage_matrix[i, j]:.2f}',
                                 ha='center', va='center', fontsize=8,
                                 color='white' if usage_matrix[i, j] > usage_matrix.max()/2 else 'black')
            
            ax.set_title('VQ Code Usage', fontsize=14, fontweight='bold', 
                        color=self.colors['primary'])
            ax.set_xlabel('Code Index', fontsize=10)
            ax.set_ylabel('Code Index', fontsize=10)
            
            # Add colorbar
            if not hasattr(ax, 'colorbar'):
                ax.colorbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    def update_loss_curve(self, loss: float):
        """Update loss curve"""
        self.loss_history.append(loss)
        ax = self.axes['loss']
        ax.clear()
        
        steps = range(len(self.loss_history))
        losses = list(self.loss_history)
        
        # Plot loss curve
        ax.plot(steps, losses, color=self.colors['accent'], linewidth=2, alpha=0.8)
        ax.fill_between(steps, losses, alpha=0.3, color=self.colors['accent'])
        
        # Add trend line
        if len(losses) > 10:
            z = np.polyfit(steps, losses, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), '--', color=self.colors['success'], 
                   linewidth=2, alpha=0.8, label=f'Trend: {z[0]:.4f}')
            ax.legend()
        
        ax.set_title('Prediction Loss Over Time', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'])
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
    def update_perplexity(self, perplexity: float):
        """Update perplexity visualization"""
        self.perplexity_history.append(perplexity)
        ax = self.axes['perplexity']
        ax.clear()
        
        steps = range(len(self.perplexity_history))
        perplexities = list(self.perplexity_history)
        
        ax.plot(steps, perplexities, color=self.colors['success'], linewidth=2, alpha=0.8)
        ax.fill_between(steps, perplexities, alpha=0.3, color=self.colors['success'])
        
        ax.set_title('VQ Perplexity', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'])
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
    def update_concept_stability(self, stability: float):
        """Update concept stability visualization"""
        ax = self.axes['stability']
        ax.clear()
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, 'k-', linewidth=8, alpha=0.3)
        
        # Stability arc
        stability_theta = theta[:int(stability * 100)]
        stability_r = r[:int(stability * 100)]
        ax.plot(stability_theta, stability_r, color=self.colors['success'], 
               linewidth=8, alpha=0.8)
        
        # Add stability text
        ax.text(np.pi/2, 0.5, f'{stability:.3f}', ha='center', va='center',
               fontsize=20, fontweight='bold', color=self.colors['primary'])
        
        ax.set_xlim(-0.2, np.pi + 0.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_title('Concept Stability', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'])
        
    def update_entropy_reduction(self, entropy: float):
        """Update entropy reduction visualization"""
        ax = self.axes['entropy']
        ax.clear()
        
        # Create entropy bars
        categories = ['Input Entropy', 'Current Entropy', 'Reduction']
        values = [np.log(28), entropy, np.log(28) - entropy]  # Assuming 28 possible tokens
        colors = [self.colors['secondary'], self.colors['accent'], self.colors['success']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, 
                     edgecolor=self.colors['secondary'], linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10,
                   fontweight='bold', color=self.colors['primary'])
        
        ax.set_title('Entropy Reduction', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'])
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
    def update_repetition_index(self, repetition: float):
        """Update repetition index visualization"""
        ax = self.axes['repetition']
        ax.clear()
        
        # Create circular progress indicator
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False, 
                           edgecolor=self.colors['secondary'], linewidth=3)
        ax.add_patch(circle)
        
        # Add repetition wedge
        if repetition > 0:
            wedge = patches.Wedge((0.5, 0.5), 0.4, 0, repetition * 360, 
                                facecolor=self.colors['warning'], alpha=0.7)
            ax.add_patch(wedge)
        
        # Add text
        ax.text(0.5, 0.5, f'{repetition:.3f}', ha='center', va='center',
               fontsize=20, fontweight='bold', color=self.colors['primary'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_title('Repetition Index', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'])
        
    def update_fitness_evolution(self, fitness: float):
        """Update fitness evolution visualization"""
        ax = self.axes['fitness']
        ax.clear()
        
        # Simple fitness indicator
        ax.bar(['Current Fitness'], [fitness], color=self.colors['success'], 
               alpha=0.7, edgecolor=self.colors['secondary'], linewidth=1.5)
        
        # Add value label
        ax.text(0, fitness + 0.01, f'{fitness:.4f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold', color=self.colors['primary'])
        
        ax.set_title('Evolutionary Fitness', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'])
        ax.set_ylabel('Fitness Score', fontsize=12)
        ax.set_ylim(0, max(1.0, fitness * 1.2))
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
    def update_all(self, data: Dict):
        """Update all visualizations with new data"""
        if 'sequence' in data:
            self.update_sequence(data['sequence'], data.get('recent_tokens', []))
        if 'vq_usage' in data:
            self.update_vq_heatmap(data['vq_usage'])
        if 'loss' in data:
            self.update_loss_curve(data['loss'])
        if 'perplexity' in data:
            self.update_perplexity(data['perplexity'])
        if 'stability' in data:
            self.update_concept_stability(data['stability'])
        if 'entropy' in data:
            self.update_entropy_reduction(data['entropy'])
        if 'repetition' in data:
            self.update_repetition_index(data['repetition'])
        if 'fitness' in data:
            self.update_fitness_evolution(data['fitness'])
            
        plt.pause(0.001)  # Small pause to update the plot
        
    def save_snapshot(self, filename: str):
        """Save current visualization as image"""
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        
    def close(self):
        """Close the visualization"""
        plt.ioff()
        plt.close(self.fig)

class MetricsCollector:
    """Collect and analyze metrics for concept emergence"""
    
    def __init__(self):
        self.metrics = {
            'losses': [],
            'perplexities': [],
            'vq_usage_counts': np.zeros(64),
            'sequences': [],
            'entropies': [],
            'stabilities': [],
            'repetition_indices': [],
            'fitness_scores': []
        }
        
    def add_metric(self, metric_type: str, value):
        """Add a metric value"""
        if metric_type in self.metrics:
            self.metrics[metric_type].append(value)
            
    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        # Avoid log(0)
        probabilities = probabilities + 1e-10
        return -np.sum(probabilities * np.log2(probabilities))
    
    def calculate_repetition_index(self, sequence: str, window_size: int = 10) -> float:
        """Calculate repetition index for a sequence"""
        if len(sequence) < window_size:
            return 0.0
            
        repetitions = 0
        total_patterns = len(sequence) - window_size + 1
        
        for i in range(total_patterns):
            pattern = sequence[i:i+window_size]
            # Count occurrences of this pattern
            count = sequence.count(pattern)
            if count > 1:
                repetitions += count - 1
                
        return repetitions / total_patterns if total_patterns > 0 else 0.0
    
    def calculate_stability(self, vq_codes: np.ndarray, history_length: int = 100) -> float:
        """Calculate concept stability over time"""
        if len(self.metrics['vq_usage_counts']) == 0:
            return 0.0
            
        # Calculate variance in VQ code usage
        recent_usage = self.metrics['vq_usage_counts'][-history_length:]
        if len(recent_usage) < 2:
            return 0.0
            
        stability = 1.0 / (1.0 + np.var(recent_usage))
        return stability
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {}
        
        for key, values in self.metrics.items():
            if len(values) > 0:
                if isinstance(values[0], (int, float)):
                    summary[f'{key}_mean'] = np.mean(values)
                    summary[f'{key}_std'] = np.std(values)
                    summary[f'{key}_trend'] = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                elif isinstance(values[0], np.ndarray):
                    summary[f'{key}_mean'] = np.mean(values, axis=0)
                    summary[f'{key}_std'] = np.std(values, axis=0)
                    
        return summary
    
    def save_metrics(self, filename: str):
        """Save metrics to file"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.metrics, f)
            
    def load_metrics(self, filename: str):
        """Load metrics from file"""
        import pickle
        with open(filename, 'rb') as f:
            self.metrics = pickle.load(f)