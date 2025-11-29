"""
Quick Baseline Validation for All 6 Tasks

Tests random baseline performance on all tasks to:
1. Verify task generators work correctly
2. Measure baseline difficulty (tasks should vary in difficulty)
3. Quick sanity check before running full experiments

Expected difficulty ranking (hardest to easiest):
- Sorting: ~20-30% (comparison + ordering)
- Parity: ~40-50% (binary parity counting)
- Sequence: ~30-40% (inductive reasoning)
- Reverse: ~20-30% (sequence reversal)
- Pattern: ~30-40% (pattern recognition)
- Addition: ~50-60% (binary arithmetic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mini_transformer import MiniTransformerWithVQ
from task_generators import (
    BinaryAdditionTask,
    SequenceReverseTask,
    PatternCompletionTask,
    SortingTask,
    ParityCheckingTask,
    NextInSequenceTask
)


def quick_baseline_test(task, task_name: str, model, num_batches: int = 20):
    """Quick baseline test for a task"""
    model.eval()
    total_correct = 0
    total_chars = 0

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = task.generate_batch(32)

            logits, _ = model(inputs)
            predictions = torch.argmax(logits, dim=-1)

            # Character-level accuracy (ignore padding)
            mask = (targets != 0)
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_chars += mask.sum().item()

    accuracy = total_correct / total_chars if total_chars > 0 else 0.0
    return accuracy


def main():
    print("="*70)
    print("BASELINE VALIDATION FOR ALL 6 TASKS")
    print("="*70)
    print("Testing random initialization baseline performance...")
    print()

    # Create model (random initialization)
    config = {
        'vocab_size': 36,
        'embed_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'ff_dim': 256,
        'num_vq_codes': 32,
        'use_vq': False,
        'dropout': 0.1
    }

    model = MiniTransformerWithVQ(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        num_vq_codes=config['num_vq_codes'],
        use_vq=config['use_vq'],
        dropout=config['dropout']
    )

    # Test all tasks
    tasks = [
        ('Addition', BinaryAdditionTask(max_digits=3)),
        ('Reverse', SequenceReverseTask(min_length=3, max_length=5)),
        ('Pattern', PatternCompletionTask()),
        ('Sorting', SortingTask(min_length=3, max_length=6)),
        ('Parity', ParityCheckingTask(min_length=3, max_length=8)),
        ('Sequence', NextInSequenceTask()),
    ]

    results = []

    for task_name, task in tasks:
        # Test with 3 random seeds
        accuracies = []
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Reinitialize model
            model = MiniTransformerWithVQ(
                vocab_size=config['vocab_size'],
                embed_dim=config['embed_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                ff_dim=config['ff_dim'],
                num_vq_codes=config['num_vq_codes'],
                use_vq=config['use_vq'],
                dropout=config['dropout']
            )

            accuracy = quick_baseline_test(task, task_name, model, num_batches=20)
            accuracies.append(accuracy)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        results.append({
            'task': task_name,
            'mean': mean_acc,
            'std': std_acc
        })

        print(f"{task_name:<12} | Baseline: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

    print()
    print("="*70)
    print("TASK DIFFICULTY RANKING (easiest to hardest):")
    print("="*70)

    # Sort by accuracy (higher = easier)
    sorted_results = sorted(results, key=lambda x: x['mean'], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        difficulty = "EASY" if result['mean'] > 0.4 else "HARD"
        print(f"{i}. {result['task']:<12} | {result['mean']*100:.1f}% | {difficulty}")

    print()
    print("="*70)
    print("✅ VALIDATION COMPLETED")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Run transfer matrix experiments: python transfer_matrix_experiment.py --mode all")
    print("2. This will take ~2-3 hours for full 126 experiments")
    print("3. Or run quick test: python transfer_matrix_experiment.py --mode dream --dream_steps 100")


if __name__ == "__main__":
    main()
