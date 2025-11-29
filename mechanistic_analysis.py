"""
Mechanistic Interpretability Analysis for ZD-CBE

Research Questions:
1. What representations does DFCB learn during the dream phase?
2. How do attention patterns differ: Random vs Dreamer?
3. What task-relevant information is encoded in hidden states?
4. Which layers/heads are most important for transfer?

Components:
1. Attention Pattern Analysis - Visualize attention heads
2. Representation Similarity Analysis (RSA) - Compare internal representations
3. Probing Classifiers - Test what information is encoded

Usage:
    python mechanistic_analysis.py --mode attention --model_path dreamer_addition.pth
    python mechanistic_analysis.py --mode rsa --task addition
    python mechanistic_analysis.py --mode probe --task sorting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

from mini_transformer import MiniTransformerWithVQ
from task_generators import (
    BinaryAdditionTask,
    SequenceReverseTask,
    PatternCompletionTask,
    SortingTask,
    ParityCheckingTask,
    NextInSequenceTask
)


TASK_MAP = {
    'addition': BinaryAdditionTask,
    'reverse': SequenceReverseTask,
    'pattern': PatternCompletionTask,
    'sorting': SortingTask,
    'parity': ParityCheckingTask,
    'sequence': NextInSequenceTask
}


class AttentionAnalyzer:
    """Analyze attention patterns in Transformer models"""

    def __init__(self, model: MiniTransformerWithVQ):
        self.model = model
        self.model.eval()
        self.attention_maps = []

    def extract_attention_patterns(
        self,
        task,
        num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """Extract attention patterns from model"""
        print(f"Extracting attention patterns from {num_samples} samples...")

        all_attentions = {
            f'layer_{i}_head_{j}': []
            for i in range(len(self.model.blocks))
            for j in range(self.model.blocks[0].attention.num_heads)
        }

        with torch.no_grad():
            for _ in range(num_samples // 32 + 1):
                inputs, _ = task.generate_batch(32)

                # Extract attention weights from each layer
                x = self.model.embedding(inputs)

                for layer_idx, block in enumerate(self.model.blocks):
                    # Get attention weights
                    attention = block.attention

                    # Compute attention scores
                    B, T, C = x.shape
                    q = attention.q_proj(x)
                    k = attention.k_proj(x)

                    q = q.view(B, T, attention.num_heads, C // attention.num_heads).transpose(1, 2)
                    k = k.view(B, T, attention.num_heads, C // attention.num_heads).transpose(1, 2)

                    # Attention scores
                    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(C // attention.num_heads)
                    attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)

                    # Store per head
                    for head_idx in range(attention.num_heads):
                        head_attn = attn_weights[:, head_idx, :, :].cpu().numpy()  # (B, T, T)
                        all_attentions[f'layer_{layer_idx}_head_{head_idx}'].append(head_attn)

                    # Continue forward pass
                    x = block(x)

        # Average attention patterns
        averaged_attentions = {}
        for key, attn_list in all_attentions.items():
            if attn_list:
                # Concatenate and average
                all_attn = np.concatenate(attn_list, axis=0)  # (N, T, T)
                averaged_attentions[key] = all_attn.mean(axis=0)  # (T, T)

        return averaged_attentions

    def visualize_attention_patterns(
        self,
        attention_patterns: Dict[str, np.ndarray],
        save_dir: str = "attention_analysis"
    ):
        """Visualize attention patterns as heatmaps"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        num_layers = len(self.model.blocks)
        num_heads = self.model.blocks[0].attention.num_heads

        # Create grid of attention heatmaps
        fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))

        if num_layers == 1:
            axes = axes.reshape(1, -1)

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = f'layer_{layer_idx}_head_{head_idx}'
                attn_map = attention_patterns[key]

                ax = axes[layer_idx, head_idx]
                sns.heatmap(
                    attn_map,
                    cmap='viridis',
                    ax=ax,
                    cbar=True,
                    square=True,
                    vmin=0,
                    vmax=1
                )
                ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=10)
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')

        plt.tight_layout()
        plt.savefig(save_path / 'attention_patterns.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention patterns to {save_path / 'attention_patterns.png'}")
        plt.close()

    def compare_attention_patterns(
        self,
        random_model: MiniTransformerWithVQ,
        dreamer_model: MiniTransformerWithVQ,
        task,
        save_dir: str = "attention_analysis"
    ):
        """Compare attention patterns: Random vs Dreamer"""
        print("\nComparing attention patterns: Random vs Dreamer")

        # Extract patterns from both models
        print("Random model:")
        self.model = random_model
        random_attn = self.extract_attention_patterns(task, num_samples=100)

        print("Dreamer model:")
        self.model = dreamer_model
        dreamer_attn = self.extract_attention_patterns(task, num_samples=100)

        # Compute differences
        diff_attn = {}
        for key in random_attn.keys():
            diff_attn[key] = dreamer_attn[key] - random_attn[key]

        # Visualize
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        num_layers = len(random_model.blocks)
        num_heads = random_model.blocks[0].attention.num_heads

        fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))

        if num_layers == 1:
            axes = axes.reshape(1, -1)

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = f'layer_{layer_idx}_head_{head_idx}'
                diff_map = diff_attn[key]

                ax = axes[layer_idx, head_idx]
                sns.heatmap(
                    diff_map,
                    cmap='RdBu_r',
                    ax=ax,
                    cbar=True,
                    square=True,
                    center=0,
                    vmin=-0.3,
                    vmax=0.3
                )
                ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=10)
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')

        plt.suptitle('Attention Difference: Dreamer - Random', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path / 'attention_difference.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention difference to {save_path / 'attention_difference.png'}")
        plt.close()


class RepresentationAnalyzer:
    """Representation Similarity Analysis (RSA)"""

    def __init__(self, model: MiniTransformerWithVQ):
        self.model = model
        self.model.eval()

    def extract_representations(
        self,
        task,
        num_samples: int = 200,
        layer_idx: int = -1
    ) -> np.ndarray:
        """Extract hidden representations from a specific layer"""
        print(f"Extracting representations from layer {layer_idx}...")

        all_representations = []

        with torch.no_grad():
            for _ in range(num_samples // 32 + 1):
                inputs, _ = task.generate_batch(32)

                # Forward pass to extract representations
                x = self.model.embedding(inputs)

                # Pass through layers
                for idx, block in enumerate(self.model.blocks):
                    x = block(x)
                    if idx == layer_idx or (layer_idx == -1 and idx == len(self.model.blocks) - 1):
                        # Extract representations
                        representations = x.mean(dim=1)  # Average over sequence (B, C)
                        all_representations.append(representations.cpu().numpy())
                        break

        # Concatenate all representations
        all_reps = np.concatenate(all_representations, axis=0)[:num_samples]
        print(f"  Extracted {all_reps.shape[0]} representations of dimension {all_reps.shape[1]}")
        return all_reps

    def compute_rsa_matrix(
        self,
        representations: np.ndarray,
        metric: str = 'correlation'
    ) -> np.ndarray:
        """Compute Representation Similarity Analysis matrix"""
        print(f"Computing RSA matrix with {metric} metric...")

        # Compute pairwise distances/similarities
        if metric == 'correlation':
            # Compute correlation between representations
            rsa_matrix = np.corrcoef(representations)
        elif metric == 'euclidean':
            # Compute Euclidean distances
            distances = squareform(pdist(representations, metric='euclidean'))
            # Convert to similarity (inverse distance)
            rsa_matrix = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return rsa_matrix

    def compare_task_representations(
        self,
        model: MiniTransformerWithVQ,
        tasks: Dict[str, any],
        num_samples: int = 100,
        save_dir: str = "rsa_analysis"
    ):
        """Compare representations across different tasks"""
        print("\nComparing representations across tasks...")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Extract representations for each task
        task_representations = {}
        for task_name, task in tasks.items():
            self.model = model
            reps = self.extract_representations(task, num_samples=num_samples)
            task_representations[task_name] = reps

        # Compute cross-task RSA
        task_names = list(task_representations.keys())
        num_tasks = len(task_names)

        # Compute pairwise similarity between task representations
        cross_task_rsa = np.zeros((num_tasks, num_tasks))

        for i, task_i in enumerate(task_names):
            for j, task_j in enumerate(task_names):
                # Compute similarity between average representations
                rep_i = task_representations[task_i].mean(axis=0)
                rep_j = task_representations[task_j].mean(axis=0)

                # Cosine similarity
                similarity = np.dot(rep_i, rep_j) / (np.linalg.norm(rep_i) * np.linalg.norm(rep_j))
                cross_task_rsa[i, j] = similarity

        # Visualize
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            cross_task_rsa,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            xticklabels=[t[:3].upper() for t in task_names],
            yticklabels=[t[:3].upper() for t in task_names],
            vmin=-1,
            vmax=1,
            center=0,
            square=True
        )
        plt.title('Cross-Task Representation Similarity (RSA)', fontsize=14)
        plt.xlabel('Task', fontsize=12)
        plt.ylabel('Task', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path / 'cross_task_rsa.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved cross-task RSA to {save_path / 'cross_task_rsa.png'}")
        plt.close()

        return cross_task_rsa


class ProbingClassifier:
    """Probing classifiers to test what information is encoded"""

    def __init__(self, model: MiniTransformerWithVQ):
        self.model = model
        self.model.eval()

    def extract_features_and_labels(
        self,
        task,
        num_samples: int = 500,
        layer_idx: int = -1,
        label_type: str = 'task_type'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels for probing"""
        print(f"Extracting features and labels (label_type={label_type})...")

        all_features = []
        all_labels = []

        with torch.no_grad():
            for _ in range(num_samples // 32 + 1):
                inputs, targets = task.generate_batch(32)

                # Forward pass to extract features
                x = self.model.embedding(inputs)

                # Pass through layers
                for idx, block in enumerate(self.model.blocks):
                    x = block(x)
                    if idx == layer_idx or (layer_idx == -1 and idx == len(self.model.blocks) - 1):
                        # Extract features
                        features = x.mean(dim=1)  # Average over sequence (B, C)
                        all_features.append(features.cpu().numpy())

                        # Create labels based on label_type
                        if label_type == 'sequence_length':
                            # Label: sequence length
                            labels = (inputs != 0).sum(dim=1).cpu().numpy()
                        elif label_type == 'first_token':
                            # Label: first non-zero token
                            labels = inputs[:, 0].cpu().numpy()
                        else:
                            # Default: just use batch index
                            labels = np.arange(inputs.shape[0])

                        all_labels.append(labels)
                        break

        # Concatenate
        features = np.concatenate(all_features, axis=0)[:num_samples]
        labels = np.concatenate(all_labels, axis=0)[:num_samples]

        print(f"  Extracted {features.shape[0]} samples with {features.shape[1]} features")
        return features, labels

    def train_probe(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.3
    ) -> Dict[str, float]:
        """Train linear probe and evaluate"""
        print("Training linear probe...")

        # Split train/test
        split_idx = int(len(features) * (1 - test_size))
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        # Train logistic regression probe
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, probe.predict(X_train))
        test_acc = accuracy_score(y_test, probe.predict(X_test))

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'probe': probe
        }

    def compare_probing_accuracy(
        self,
        random_model: MiniTransformerWithVQ,
        dreamer_model: MiniTransformerWithVQ,
        task,
        label_type: str = 'sequence_length',
        save_dir: str = "probing_analysis"
    ):
        """Compare probing accuracy: Random vs Dreamer"""
        print(f"\nComparing probing accuracy for: {label_type}")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Random model
        print("\nRandom model:")
        self.model = random_model
        features_random, labels = self.extract_features_and_labels(
            task, num_samples=500, label_type=label_type
        )
        results_random = self.train_probe(features_random, labels)

        # Dreamer model
        print("\nDreamer model:")
        self.model = dreamer_model
        features_dreamer, labels = self.extract_features_and_labels(
            task, num_samples=500, label_type=label_type
        )
        results_dreamer = self.train_probe(features_dreamer, labels)

        # Visualize comparison
        fig, ax = plt.subplots(figsize=(8, 6))

        models = ['Random', 'Dreamer']
        train_accs = [results_random['train_accuracy'], results_dreamer['train_accuracy']]
        test_accs = [results_random['test_accuracy'], results_dreamer['test_accuracy']]

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, train_accs, width, label='Train', color='steelblue')
        ax.bar(x + width/2, test_accs, width, label='Test', color='coral')

        ax.set_ylabel('Probing Accuracy', fontsize=12)
        ax.set_title(f'Probing Accuracy: {label_type.replace("_", " ").title()}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(save_path / f'probing_{label_type}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved probing results to {save_path / f'probing_{label_type}.png'}")
        plt.close()

        return {
            'random': results_random,
            'dreamer': results_dreamer
        }


def create_task(task_name: str):
    """Create task generator"""
    task_class = TASK_MAP.get(task_name)
    if task_class is None:
        raise ValueError(f"Unknown task: {task_name}")

    if task_name == 'addition':
        return task_class(max_digits=3)
    elif task_name == 'reverse':
        return task_class(min_length=3, max_length=5)
    elif task_name == 'sorting':
        return task_class(min_length=3, max_length=6)
    elif task_name == 'parity':
        return task_class(min_length=3, max_length=8)
    else:
        return task_class()


def main():
    parser = argparse.ArgumentParser(description='Mechanistic Analysis for ZD-CBE')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['attention', 'rsa', 'probe', 'all'],
                       help='Analysis mode')
    parser.add_argument('--task', type=str, default='addition',
                       choices=list(TASK_MAP.keys()),
                       help='Task to analyze')
    parser.add_argument('--dreamer_path', type=str, default=None,
                       help='Path to dreamer model weights')
    parser.add_argument('--vocab_size', type=int, default=36)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='mechanistic_analysis')

    args = parser.parse_args()

    # Create task
    task = create_task(args.task)

    # Create models
    random_model = MiniTransformerWithVQ(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        use_vq=False
    )

    dreamer_model = MiniTransformerWithVQ(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        use_vq=False
    )

    # Load dreamer weights
    if args.dreamer_path:
        dreamer_path = Path(args.dreamer_path)
    else:
        dreamer_path = Path(f"dreamer_{args.task}.pth")

    if dreamer_path.exists():
        checkpoint = torch.load(dreamer_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            dreamer_model.load_state_dict(checkpoint['model_state'], strict=False)
            print(f"✓ Loaded dreamer weights from {dreamer_path}")
        else:
            print(f"Warning: Could not load dreamer weights from {dreamer_path}")
    else:
        print(f"Warning: Dreamer weights not found at {dreamer_path}")
        print("Using random initialization for dreamer model (for testing)")

    # Run analysis
    print("="*70)
    print(f"MECHANISTIC ANALYSIS: {args.mode.upper()}")
    print(f"Task: {args.task}")
    print("="*70)

    if args.mode == 'attention' or args.mode == 'all':
        analyzer = AttentionAnalyzer(random_model)
        analyzer.compare_attention_patterns(
            random_model, dreamer_model, task, save_dir=args.output_dir
        )

    if args.mode == 'rsa' or args.mode == 'all':
        analyzer = RepresentationAnalyzer(dreamer_model)

        # Create all tasks for cross-task comparison
        tasks = {name: create_task(name) for name in TASK_MAP.keys()}

        analyzer.compare_task_representations(
            dreamer_model, tasks, num_samples=args.num_samples,
            save_dir=args.output_dir
        )

    if args.mode == 'probe' or args.mode == 'all':
        prober = ProbingClassifier(random_model)
        prober.compare_probing_accuracy(
            random_model, dreamer_model, task,
            label_type='sequence_length',
            save_dir=args.output_dir
        )

    print("\n" + "="*70)
    print("✅ MECHANISTIC ANALYSIS COMPLETED")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
