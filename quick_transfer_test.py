"""
Quick Transfer Test - Run a few key transfer experiments to verify framework

Tests diagonal elements (dream on task → finetune on same task)
Plus a few off-diagonal for comparison.
"""
import torch
import numpy as np
from transfer_matrix_experiment import TransferExperiment, TASK_NAMES, create_task

def quick_transfer_test():
    """Run quick transfer tests on key combinations"""

    print("="*70)
    print("QUICK TRANSFER TEST")
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

    # Test diagonal (dream → same task)
    print("\nTesting diagonal transfer (dream → same task):")

    for task_name in ['addition', 'sorting']:
        print(f"\n{task_name.upper()}: Dream({task_name}) → Finetune({task_name})")

        exp = TransferExperiment(
            dream_task=task_name,
            finetune_task=task_name,
            config=config,
            seed=0
        )

        exp.train_with_sample_tracking(max_examples=1000, batch_size=32, eval_interval=200)

        print(f"  Final accuracy: {exp.learning_curve[-1][1]:.4f}")
        print(f"  Convergence @90%: {exp.find_convergence_point(0.90)}")

    # Test baseline (no dreaming)
    print("\n\nTesting baseline (no dreaming):")

    for task_name in ['addition', 'sorting']:
        print(f"\n{task_name.upper()}: Random → Finetune({task_name})")

        exp = TransferExperiment(
            dream_task=None,
            finetune_task=task_name,
            config=config,
            seed=0
        )

        exp.train_with_sample_tracking(max_examples=1000, batch_size=32, eval_interval=200)

        print(f"  Final accuracy: {exp.learning_curve[-1][1]:.4f}")
        print(f"  Convergence @90%: {exp.find_convergence_point(0.90)}")

    print("\n" + "="*70)
    print("✅ QUICK TEST COMPLETED")
    print("="*70)

if __name__ == "__main__":
    quick_transfer_test()
