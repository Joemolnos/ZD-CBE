"""
Sample Efficiency Experiment for ZD-CBE v2

This is the KEY METRIC: How many examples needed to reach X% accuracy?

Measures:
- Learning curves (accuracy vs. training examples)
- Convergence speed
- Sample efficiency ratio

Compares:
1. Random initialization
2. Dreamer initialization (with VQ)
3. Dreamer without VQ (ablation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from mini_transformer import MiniTransformerWithVQ
from task_generators import BinaryAdditionTask, SequenceReverseTask, PatternCompletionTask


class SampleEfficiencyExperiment:
    """Measure sample efficiency across different initializations"""

    def __init__(
        self,
        task_name: str,
        model_type: str,  # 'random', 'dreamer', 'dreamer_no_vq'
        config: Dict,
        seed: int = 42
    ):
        self.task_name = task_name
        self.model_type = model_type
        self.config = config
        self.seed = seed

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create task generator
        self.task = self._create_task(task_name)

        # Create model
        self.model = self._create_model(model_type, config)

        # Training state
        self.training_examples_seen = 0
        self.learning_curve = []  # (examples_seen, accuracy)

    def _create_task(self, task_name: str):
        """Create task generator"""
        if task_name == 'addition':
            return BinaryAdditionTask(max_digits=3)
        elif task_name == 'reverse':
            return SequenceReverseTask(min_length=3, max_length=5)
        elif task_name == 'pattern':
            return PatternCompletionTask()
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def _create_model(self, model_type: str, config: Dict):
        """Create model based on type"""
        use_vq = 'no_vq' not in model_type

        model = MiniTransformerWithVQ(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            num_vq_codes=config['num_vq_codes'],
            use_vq=use_vq,
            dropout=config.get('dropout', 0.1)
        )

        # Load dreamer weights if applicable
        if 'dreamer' in model_type and 'dreamer_model.pth' in os.listdir('.'):
            try:
                checkpoint = torch.load('dreamer_model.pth', map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                    # Try to load compatible weights
                    model.load_state_dict(checkpoint['model_state'], strict=False)
                    print(f"Loaded dreamer weights (partial match)")
            except Exception as e:
                print(f"Could not load dreamer weights: {e}")

        return model

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
        return accuracy

    def train_with_sample_tracking(
        self,
        max_examples: int = 5000,
        batch_size: int = 32,
        eval_interval: int = 100  # Evaluate every N examples
    ):
        """Train while tracking sample efficiency"""
        print(f"\nTraining {self.model_type} on {self.task_name}...")
        print(f"Target: {max_examples} examples, eval every {eval_interval}")

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

                elapsed = time.time() - start_time
                print(f"  Examples: {examples_seen:5d} | Accuracy: {accuracy:.4f} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

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

        results = {
            'task': self.task_name,
            'model_type': self.model_type,
            'seed': self.seed,
            'learning_curve': self.learning_curve,
            'final_accuracy': self.learning_curve[-1][1] if self.learning_curve else 0.0,
            'convergence_90': self.find_convergence_point(0.90),
            'convergence_80': self.find_convergence_point(0.80),
            'convergence_70': self.find_convergence_point(0.70),
        }

        filename = f"{self.task_name}_{self.model_type}_seed{self.seed}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)

        return results


def run_sample_efficiency_experiment(
    task_name: str,
    max_examples: int = 3000,
    seeds: List[int] = [0, 1, 2],
    output_dir: str = "sample_efficiency_results"
):
    """Run complete sample efficiency experiment"""
    print("="*70)
    print(f"SAMPLE EFFICIENCY EXPERIMENT: {task_name.upper()}")
    print("="*70)

    config = {
        'vocab_size': 28,
        'embed_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'ff_dim': 256,
        'num_vq_codes': 32,
        'learning_rate': 0.003,
        'dropout': 0.1
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_types = ['random', 'dreamer', 'dreamer_no_vq']
    all_results = []

    for model_type in model_types:
        for seed in seeds:
            exp = SampleEfficiencyExperiment(
                task_name=task_name,
                model_type=model_type,
                config=config,
                seed=seed
            )

            # Train with sample tracking
            exp.train_with_sample_tracking(
                max_examples=max_examples,
                batch_size=32,
                eval_interval=100
            )

            # Save results
            results = exp.save_results(output_path)
            all_results.append(results)

    # Aggregate results
    aggregated = aggregate_results(all_results, task_name)

    with open(output_path / f"{task_name}_aggregated.json", 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n✅ {task_name} experiment complete!")
    print(f"Results saved to: {output_path}/")

    return aggregated


def aggregate_results(results: List[Dict], task_name: str) -> Dict:
    """Aggregate results across seeds"""
    from collections import defaultdict

    by_model = defaultdict(list)
    for r in results:
        by_model[r['model_type']].append(r)

    aggregated = {
        'task': task_name,
        'models': {}
    }

    for model_type, model_results in by_model.items():
        # Extract convergence points
        conv_90 = [r['convergence_90'] for r in model_results if r['convergence_90'] > 0]
        conv_80 = [r['convergence_80'] for r in model_results if r['convergence_80'] > 0]
        conv_70 = [r['convergence_70'] for r in model_results if r['convergence_70'] > 0]

        aggregated['models'][model_type] = {
            'final_accuracy_mean': np.mean([r['final_accuracy'] for r in model_results]),
            'final_accuracy_std': np.std([r['final_accuracy'] for r in model_results]),
            'convergence_90_mean': np.mean(conv_90) if conv_90 else -1,
            'convergence_90_std': np.std(conv_90) if conv_90 else 0,
            'convergence_80_mean': np.mean(conv_80) if conv_80 else -1,
            'convergence_80_std': np.std(conv_80) if conv_80 else 0,
            'num_runs': len(model_results)
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Sample Efficiency Experiment')
    parser.add_argument('--task', type=str, default='addition',
                       choices=['addition', 'reverse', 'pattern'],
                       help='Task to evaluate')
    parser.add_argument('--max_examples', type=int, default=3000,
                       help='Maximum training examples')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='Random seeds')
    parser.add_argument('--output_dir', type=str, default='sample_efficiency_results',
                       help='Output directory')

    args = parser.parse_args()

    # Import os here (needed for loading dreamer weights)
    import os
    globals()['os'] = os

    results = run_sample_efficiency_experiment(
        task_name=args.task,
        max_examples=args.max_examples,
        seeds=args.seeds,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for model_type, stats in results['models'].items():
        print(f"\n{model_type.upper()}:")
        print(f"  Final accuracy: {stats['final_accuracy_mean']:.4f} ± {stats['final_accuracy_std']:.4f}")
        if stats['convergence_90_mean'] > 0:
            print(f"  90% convergence: {stats['convergence_90_mean']:.0f} ± {stats['convergence_90_std']:.0f} examples")
        else:
            print(f"  90% convergence: Did not converge")


if __name__ == "__main__":
    main()
