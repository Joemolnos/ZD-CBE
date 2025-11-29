"""
Transfer Matrix Experiment for ZD-CBE

This is a NOVEL contribution: 6×6 Transfer Learning Matrix showing cross-task transfer.

Matrix Structure:
- Rows: Dream task (which task the model dreams on)
- Columns: Fine-tune task (which task the model is fine-tuned on)
- Cells: Sample efficiency improvement (or final accuracy)

Example:
                Fine-tune Task →
Dream Task ↓  | Add | Rev | Pat | Sort | Par | Seq |
---------------------------------------------------------
Addition      | 4.2× | 2.1× | 1.8× | 2.5× | 3.0× | 2.2× |
Reverse       | 1.9× | 3.5× | 2.0× | 2.8× | 1.6× | 2.4× |
Pattern       | 2.1× | 1.8× | 3.8× | 2.0× | 1.7× | 2.5× |
Sorting       | 2.3× | 2.6× | 1.9× | 4.0× | 2.2× | 2.7× |
Parity        | 2.5× | 1.5× | 1.6× | 2.1× | 3.6× | 2.0× |
Sequence      | 2.0× | 2.3× | 2.4× | 2.6× | 1.9× | 3.9× |

Research Questions:
1. Do models dream best on the task they'll be tested on? (diagonal should be highest)
2. Which tasks provide the most general representations?
3. Are there task clusters that transfer well to each other?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from mini_transformer import MiniTransformerWithVQ
from task_generators import (
    BinaryAdditionTask,
    SequenceReverseTask,
    PatternCompletionTask,
    SortingTask,
    ParityCheckingTask,
    NextInSequenceTask
)
from simple_dream import simple_dream_phase


# Task mapping
TASK_MAP = {
    'addition': BinaryAdditionTask,
    'reverse': SequenceReverseTask,
    'pattern': PatternCompletionTask,
    'sorting': SortingTask,
    'parity': ParityCheckingTask,
    'sequence': NextInSequenceTask
}

TASK_NAMES = list(TASK_MAP.keys())


class TransferExperiment:
    """Single transfer experiment: dream on task A, fine-tune on task B"""

    def __init__(
        self,
        dream_task: Optional[str],  # None for random baseline
        finetune_task: str,
        config: Dict,
        seed: int = 42
    ):
        self.dream_task = dream_task
        self.finetune_task = finetune_task
        self.config = config
        self.seed = seed

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create task generator for fine-tuning
        self.task = self._create_task(finetune_task)

        # Create model
        self.model = self._create_model(config)

        # Load dreamer weights if applicable
        if dream_task is not None:
            self._load_dreamer_weights(dream_task)

        # Training state
        self.learning_curve = []  # (examples_seen, accuracy)

    def _create_task(self, task_name: str):
        """Create task generator"""
        task_class = TASK_MAP.get(task_name)
        if task_class is None:
            raise ValueError(f"Unknown task: {task_name}")

        # Create task with appropriate parameters
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

    def _create_model(self, config: Dict):
        """Create model"""
        model = MiniTransformerWithVQ(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            num_vq_codes=config['num_vq_codes'],
            use_vq=config.get('use_vq', False),  # Default: no VQ (based on v2 findings)
            dropout=config.get('dropout', 0.1)
        )
        return model

    def _load_dreamer_weights(self, dream_task: str):
        """Load weights from dream phase"""
        dreamer_path = Path(f"dreamer_{dream_task}.pth")

        if not dreamer_path.exists():
            print(f"Warning: Dreamer weights not found at {dreamer_path}")
            print(f"Run dream phase first with: python transfer_matrix_experiment.py --mode dream")
            return

        try:
            checkpoint = torch.load(dreamer_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'], strict=False)
                print(f"✓ Loaded dreamer weights from {dream_task}")
            else:
                print(f"Warning: Invalid checkpoint format at {dreamer_path}")
        except Exception as e:
            print(f"Warning: Could not load dreamer weights: {e}")

    def evaluate(self, num_batches: int = 10) -> float:
        """Evaluate current model accuracy"""
        self.model.eval()
        total_correct = 0
        total_chars = 0

        with torch.no_grad():
            for _ in range(num_batches):
                inputs, targets = self.task.generate_batch(32)

                logits, _ = self.model(inputs)
                predictions = torch.argmax(logits, dim=-1)

                # Character-level accuracy (ignore padding)
                mask = (targets != 0)
                correct = (predictions == targets) & mask
                total_correct += correct.sum().item()
                total_chars += mask.sum().item()

        accuracy = total_correct / total_chars if total_chars > 0 else 0.0
        self.model.train()
        return accuracy

    def train_with_sample_tracking(
        self,
        max_examples: int = 5000,
        batch_size: int = 32,
        eval_interval: int = 100
    ):
        """Train while tracking sample efficiency"""
        dream_str = self.dream_task if self.dream_task else "random"
        print(f"\nTraining: Dream({dream_str}) → Finetune({self.finetune_task})")

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        examples_seen = 0
        start_time = time.time()

        while examples_seen < max_examples:
            # Generate batch
            inputs, targets = self.task.generate_batch(batch_size)

            # Forward pass
            logits, vq_info = self.model(inputs)

            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.config['vocab_size']),
                targets.reshape(-1),
                ignore_index=0
            )

            # Add VQ loss if applicable
            if 'loss' in vq_info:
                total_loss = loss + 0.1 * vq_info['loss']
            else:
                total_loss = loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            examples_seen += batch_size

            # Evaluate periodically
            if examples_seen % eval_interval == 0 or examples_seen >= max_examples:
                accuracy = self.evaluate()
                self.learning_curve.append((examples_seen, accuracy))

                if examples_seen % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Examples: {examples_seen:5d} | Accuracy: {accuracy:.4f} | Time: {elapsed:.1f}s")

        return self.learning_curve

    def find_convergence_point(self, target_accuracy: float = 0.90) -> int:
        """Find number of examples needed to reach target accuracy"""
        for examples, accuracy in self.learning_curve:
            if accuracy >= target_accuracy:
                return examples
        return -1  # Never converged

    def save_results(self, output_dir: Path):
        """Save learning curve and metrics"""
        output_dir.mkdir(parents=True, exist_ok=True)

        dream_str = self.dream_task if self.dream_task else "random"

        results = {
            'dream_task': self.dream_task,
            'finetune_task': self.finetune_task,
            'seed': self.seed,
            'learning_curve': self.learning_curve,
            'final_accuracy': self.learning_curve[-1][1] if self.learning_curve else 0.0,
            'convergence_90': self.find_convergence_point(0.90),
            'convergence_80': self.find_convergence_point(0.80),
            'convergence_70': self.find_convergence_point(0.70),
        }

        filename = f"transfer_{dream_str}_to_{self.finetune_task}_seed{self.seed}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)

        return results


def run_dream_phase_all_tasks(
    config: Dict,
    dream_steps: int = 1000,
    seed: int = 42
):
    """Run dream phase on all 6 tasks separately"""
    print("="*70)
    print("PHASE 1: DREAM PHASE ON ALL 6 TASKS")
    print("="*70)

    torch.manual_seed(seed)
    np.random.seed(seed)

    for task_name in TASK_NAMES:
        print(f"\n{'='*70}")
        print(f"Dreaming on: {task_name.upper()}")
        print(f"{'='*70}")

        # Create model
        model = MiniTransformerWithVQ(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            num_vq_codes=config['num_vq_codes'],
            use_vq=config.get('use_vq', False)
        )

        # Run dream phase
        save_path = f"dreamer_{task_name}.pth"
        simple_dream_phase(
            model=model,
            vocab_size=config['vocab_size'],
            dream_steps=dream_steps,
            batch_size=32,
            seq_len=20,
            learning_rate=config['learning_rate'],
            save_path=save_path
        )

    print("\n" + "="*70)
    print("✅ DREAM PHASE COMPLETED FOR ALL 6 TASKS")
    print("="*70)


def run_transfer_matrix_experiment(
    config: Dict,
    max_examples: int = 5000,
    seeds: List[int] = [0, 1, 2],
    output_dir: str = "transfer_matrix_results"
):
    """Run complete 6×6 transfer matrix experiment"""
    print("="*70)
    print("PHASE 2: TRANSFER MATRIX EXPERIMENT (6×6 + baseline)")
    print("="*70)
    print(f"Total experiments per seed: 6×6 + 6 baseline = 42")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {42 * len(seeds)}")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        # Run baseline (random initialization on all tasks)
        for finetune_task in TASK_NAMES:
            exp = TransferExperiment(
                dream_task=None,  # Random baseline
                finetune_task=finetune_task,
                config=config,
                seed=seed
            )

            exp.train_with_sample_tracking(
                max_examples=max_examples,
                batch_size=32,
                eval_interval=100
            )

            results = exp.save_results(output_path)
            all_results.append(results)

        # Run transfer experiments (6×6 matrix)
        for dream_task in TASK_NAMES:
            for finetune_task in TASK_NAMES:
                exp = TransferExperiment(
                    dream_task=dream_task,
                    finetune_task=finetune_task,
                    config=config,
                    seed=seed
                )

                exp.train_with_sample_tracking(
                    max_examples=max_examples,
                    batch_size=32,
                    eval_interval=100
                )

                results = exp.save_results(output_path)
                all_results.append(results)

    print("\n" + "="*70)
    print("✅ TRANSFER MATRIX EXPERIMENT COMPLETED")
    print(f"Results saved to: {output_path}")
    print("="*70)

    return all_results


def analyze_transfer_matrix(results_dir: str = "transfer_matrix_results"):
    """Analyze transfer matrix results and create heatmap"""
    print("="*70)
    print("PHASE 3: TRANSFER MATRIX ANALYSIS")
    print("="*70)

    results_path = Path(results_dir)

    # Load all results
    all_results = []
    for json_file in results_path.glob("transfer_*.json"):
        with open(json_file, 'r') as f:
            all_results.append(json.load(f))

    if not all_results:
        print("No results found!")
        return

    # Create matrices for different metrics
    metrics = ['final_accuracy', 'convergence_90']

    for metric in metrics:
        print(f"\n{'='*70}")
        print(f"Metric: {metric}")
        print(f"{'='*70}")

        # Initialize matrix
        matrix = np.zeros((len(TASK_NAMES), len(TASK_NAMES)))
        baseline = np.zeros(len(TASK_NAMES))
        counts = np.zeros((len(TASK_NAMES), len(TASK_NAMES)))
        baseline_counts = np.zeros(len(TASK_NAMES))

        # Aggregate results
        for result in all_results:
            dream_task = result['dream_task']
            finetune_task = result['finetune_task']
            value = result[metric]

            finetune_idx = TASK_NAMES.index(finetune_task)

            if dream_task is None:
                # Baseline
                if value > 0:  # Only count if converged
                    baseline[finetune_idx] += value
                    baseline_counts[finetune_idx] += 1
            else:
                # Transfer
                dream_idx = TASK_NAMES.index(dream_task)
                if value > 0:  # Only count if converged
                    matrix[dream_idx, finetune_idx] += value
                    counts[dream_idx, finetune_idx] += 1

        # Average
        for i in range(len(TASK_NAMES)):
            if baseline_counts[i] > 0:
                baseline[i] /= baseline_counts[i]
            for j in range(len(TASK_NAMES)):
                if counts[i, j] > 0:
                    matrix[i, j] /= counts[i, j]

        # Calculate improvement ratio (for convergence metrics)
        if 'convergence' in metric:
            improvement_matrix = np.zeros_like(matrix)
            for i in range(len(TASK_NAMES)):
                for j in range(len(TASK_NAMES)):
                    if matrix[i, j] > 0 and baseline[j] > 0:
                        improvement_matrix[i, j] = baseline[j] / matrix[i, j]
                    else:
                        improvement_matrix[i, j] = 0.0

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                improvement_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                xticklabels=[t[:3].upper() for t in TASK_NAMES],
                yticklabels=[t[:3].upper() for t in TASK_NAMES],
                cbar_kws={'label': 'Sample Efficiency Improvement (×)'},
                vmin=0.5,
                vmax=5.0
            )
            plt.title(f'Transfer Matrix: {metric.replace("_", " ").title()}', fontsize=14)
            plt.xlabel('Fine-tune Task →', fontsize=12)
            plt.ylabel('Dream Task ↓', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'transfer_matrix_{metric}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved heatmap: transfer_matrix_{metric}.png")

            # Print matrix
            print("\nSample Efficiency Improvement Matrix (×):")
            print("Dream Task ↓ | " + " | ".join([f"{t[:3].upper():>4}" for t in TASK_NAMES]) + " |")
            print("-" * 70)
            for i, dream_task in enumerate(TASK_NAMES):
                row_str = f"{dream_task[:8]:<12} | "
                row_str += " | ".join([f"{improvement_matrix[i, j]:4.1f}" for j in range(len(TASK_NAMES))])
                row_str += " |"
                print(row_str)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETED")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Transfer Matrix Experiment')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['dream', 'transfer', 'analyze', 'all'],
                       help='Experiment mode')
    parser.add_argument('--vocab_size', type=int, default=36)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=256)
    parser.add_argument('--num_vq_codes', type=int, default=32)
    parser.add_argument('--use_vq', action='store_true', help='Use VQ bottleneck')
    parser.add_argument('--dream_steps', type=int, default=1000)
    parser.add_argument('--max_examples', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='transfer_matrix_results')

    args = parser.parse_args()

    config = {
        'vocab_size': args.vocab_size,
        'embed_dim': args.embed_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'ff_dim': args.ff_dim,
        'num_vq_codes': args.num_vq_codes,
        'use_vq': args.use_vq,
        'learning_rate': args.learning_rate,
        'dropout': 0.1
    }

    if args.mode == 'dream' or args.mode == 'all':
        run_dream_phase_all_tasks(config, args.dream_steps, seed=args.seeds[0])

    if args.mode == 'transfer' or args.mode == 'all':
        run_transfer_matrix_experiment(
            config,
            max_examples=args.max_examples,
            seeds=args.seeds,
            output_dir=args.output_dir
        )

    if args.mode == 'analyze' or args.mode == 'all':
        analyze_transfer_matrix(args.output_dir)


if __name__ == "__main__":
    main()
