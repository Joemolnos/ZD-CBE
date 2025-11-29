"""
Strategic Transfer Matrix - Publication-Quality Results Fast

Runs carefully selected experiments that demonstrate key findings:
1. Diagonal (6): Dream(task) → Finetune(same task) - Should be best
2. Baseline (6): Random → Finetune(task) - For comparison
3. Selected off-diagonal (12): Cross-task transfer examples

Total: 24 experiments (vs 84 full suite)
Still scientifically rigorous, just focused on key results.
"""
import torch
import numpy as np
import json
from pathlib import Path
from transfer_matrix_experiment import TransferExperiment, TASK_NAMES

def run_strategic_transfer():
    """Run strategic subset of transfer experiments"""

    print("="*70)
    print("STRATEGIC TRANSFER MATRIX")
    print("="*70)
    print("Running 24 key experiments:")
    print("  - 6 diagonal (dream → same task)")
    print("  - 6 baseline (random → task)")
    print("  - 12 off-diagonal (selected cross-task)")
    print("="*70)

    config = {
        'vocab_size': 36,
        'embed_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'ff_dim': 256,
        'num_vq_codes': 32,
        'use_vq': False,
        'learning_rate': 0.003,
        'dropout': 0.1
    }

    results_dir = Path("strategic_transfer_results")
    results_dir.mkdir(exist_ok=True)

    all_results = []
    experiment_count = 0

    # 1. Baseline experiments (6)
    print("\n" + "="*70)
    print("PHASE 1: BASELINE (Random → Task)")
    print("="*70)

    for task in TASK_NAMES:
        experiment_count += 1
        print(f"\n[{experiment_count}/24] Random → {task}")

        exp = TransferExperiment(
            dream_task=None,
            finetune_task=task,
            config=config,
            seed=0
        )

        exp.train_with_sample_tracking(max_examples=3000, batch_size=32, eval_interval=200)
        results = exp.save_results(results_dir)
        all_results.append(results)

    # 2. Diagonal experiments (6)
    print("\n" + "="*70)
    print("PHASE 2: DIAGONAL (Dream → Same Task)")
    print("="*70)

    for task in TASK_NAMES:
        experiment_count += 1
        print(f"\n[{experiment_count}/24] {task} → {task}")

        exp = TransferExperiment(
            dream_task=task,
            finetune_task=task,
            config=config,
            seed=0
        )

        exp.train_with_sample_tracking(max_examples=3000, batch_size=32, eval_interval=200)
        results = exp.save_results(results_dir)
        all_results.append(results)

    # 3. Selected off-diagonal (12 key pairs)
    print("\n" + "="*70)
    print("PHASE 3: OFF-DIAGONAL (Selected Cross-Task)")
    print("="*70)

    cross_task_pairs = [
        ('addition', 'parity'),  # Both binary operations
        ('reverse', 'sorting'),  # Both sequence operations
        ('pattern', 'sequence'),  # Both inductive reasoning
        ('sorting', 'reverse'),  # Reverse pair
        ('parity', 'addition'),  # Reverse pair
        ('sequence', 'pattern'),  # Reverse pair
        ('addition', 'sorting'),  # Different domains
        ('reverse', 'pattern'),  # Different domains
        ('parity', 'sorting'),  # Different domains
        ('pattern', 'addition'),  # Different domains
        ('sorting', 'parity'),  # Different domains
        ('sequence', 'reverse'),  # Different domains
    ]

    for dream_task, finetune_task in cross_task_pairs:
        experiment_count += 1
        print(f"\n[{experiment_count}/24] {dream_task} → {finetune_task}")

        exp = TransferExperiment(
            dream_task=dream_task,
            finetune_task=finetune_task,
            config=config,
            seed=0
        )

        exp.train_with_sample_tracking(max_examples=3000, batch_size=32, eval_interval=200)
        results = exp.save_results(results_dir)
        all_results.append(results)

    print("\n" + "="*70)
    print("✅ STRATEGIC TRANSFER COMPLETED")
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {results_dir}")
    print("="*70)

    # Save summary
    summary = {
        'total_experiments': len(all_results),
        'config': config,
        'results': all_results
    }

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return all_results

if __name__ == "__main__":
    run_strategic_transfer()
