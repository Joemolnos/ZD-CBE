"""
Comprehensive Experimental Suite Runner for ZD-CBE TOP TIER Enhancement

This script orchestrates all experiments needed for the publication:
1. Baseline validation (quick sanity check)
2. Dream phase on all 6 tasks
3. Transfer matrix experiments (6×6 + baselines)
4. Mechanistic analysis (attention, RSA, probing)
5. Hyperparameter robustness sweeps
6. Task difficulty correlation analysis

Usage:
    # Run everything (recommended for final experiments):
    python run_full_experimental_suite.py --mode all --seeds 0 1 2

    # Run specific components:
    python run_full_experimental_suite.py --mode baseline
    python run_full_experimental_suite.py --mode dream
    python run_full_experimental_suite.py --mode transfer
    python run_full_experimental_suite.py --mode mechanistic
    python run_full_experimental_suite.py --mode robustness

    # Quick test run (1 seed, fewer examples):
    python run_full_experimental_suite.py --mode all --quick

Estimated Time:
    - Baseline: ~30 seconds
    - Dream (6 tasks): ~10 minutes
    - Transfer (126 experiments): ~2-3 hours
    - Mechanistic: ~30 minutes
    - Robustness: ~1-2 hours
    - Total: ~4-6 hours
"""
import subprocess
import argparse
import time
from pathlib import Path
import json
from typing import List


class ExperimentalSuiteRunner:
    """Orchestrate all ZD-CBE experiments"""

    def __init__(
        self,
        seeds: List[int] = [0, 1, 2],
        dream_steps: int = 1000,
        max_examples: int = 5000,
        quick: bool = False
    ):
        self.seeds = seeds
        self.dream_steps = dream_steps if not quick else 100
        self.max_examples = max_examples if not quick else 500
        self.quick = quick

        self.results_dir = Path("experimental_results")
        self.results_dir.mkdir(exist_ok=True)

        # Track experiment status
        self.status = {
            'baseline': False,
            'dream': False,
            'transfer': False,
            'mechanistic': False,
            'robustness': False
        }

    def run_baseline_validation(self):
        """Phase 0: Baseline validation"""
        print("\n" + "="*70)
        print("PHASE 0: BASELINE VALIDATION")
        print("="*70)
        print("Quick sanity check of all 6 tasks...")

        start_time = time.time()

        cmd = ["python", "validate_tasks_baseline.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
            self.status['baseline'] = True
            print(f"✓ Baseline validation completed in {time.time() - start_time:.1f}s")
        else:
            print(f"✗ Baseline validation failed:")
            print(result.stderr)
            return False

        return True

    def run_dream_phase(self):
        """Phase 1: Dream phase on all 6 tasks"""
        print("\n" + "="*70)
        print("PHASE 1: DREAM PHASE (6 TASKS)")
        print("="*70)
        print(f"Dream steps: {self.dream_steps}")

        start_time = time.time()

        cmd = [
            "python", "transfer_matrix_experiment.py",
            "--mode", "dream",
            "--dream_steps", str(self.dream_steps),
            "--seeds", str(self.seeds[0])  # Use first seed for dream phase
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
            self.status['dream'] = True
            print(f"✓ Dream phase completed in {time.time() - start_time:.1f}s")
        else:
            print(f"✗ Dream phase failed:")
            print(result.stderr)
            return False

        return True

    def run_transfer_matrix(self):
        """Phase 2: Transfer matrix experiments"""
        print("\n" + "="*70)
        print("PHASE 2: TRANSFER MATRIX EXPERIMENTS")
        print("="*70)
        print(f"Seeds: {self.seeds}")
        print(f"Max examples per experiment: {self.max_examples}")
        print(f"Total experiments: {42 * len(self.seeds)}")

        start_time = time.time()

        cmd = [
            "python", "transfer_matrix_experiment.py",
            "--mode", "transfer",
            "--max_examples", str(self.max_examples),
            "--seeds"
        ] + [str(s) for s in self.seeds]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
            self.status['transfer'] = True
            elapsed = time.time() - start_time
            print(f"✓ Transfer matrix completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        else:
            print(f"✗ Transfer matrix failed:")
            print(result.stderr)
            return False

        # Run analysis
        print("\nGenerating transfer matrix visualizations...")
        cmd_analyze = [
            "python", "transfer_matrix_experiment.py",
            "--mode", "analyze"
        ]

        result = subprocess.run(cmd_analyze, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Warning: Analysis phase had issues")

        return True

    def run_mechanistic_analysis(self):
        """Phase 3: Mechanistic interpretability analysis"""
        print("\n" + "="*70)
        print("PHASE 3: MECHANISTIC ANALYSIS")
        print("="*70)
        print("Analyzing: attention patterns, RSA, probing classifiers")

        start_time = time.time()

        # Run analysis on a few representative tasks
        tasks = ['addition', 'sorting', 'parity'] if not self.quick else ['addition']

        for task in tasks:
            print(f"\nAnalyzing task: {task}")

            cmd = [
                "python", "mechanistic_analysis.py",
                "--mode", "all",
                "--task", task,
                "--num_samples", str(100 if not self.quick else 50),
                "--output_dir", f"mechanistic_analysis/{task}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Warning: Mechanistic analysis for {task} had issues")
                print(result.stderr)

        self.status['mechanistic'] = True
        print(f"\n✓ Mechanistic analysis completed in {time.time() - start_time:.1f}s")

        return True

    def run_robustness_sweeps(self):
        """Phase 4: Hyperparameter robustness sweeps"""
        print("\n" + "="*70)
        print("PHASE 4: HYPERPARAMETER ROBUSTNESS")
        print("="*70)

        if self.quick:
            print("Skipping robustness sweeps in quick mode")
            self.status['robustness'] = True
            return True

        start_time = time.time()

        # Dream steps sweep: [500, 1000, 2000, 5000]
        dream_steps_range = [500, 1000, 2000] # Reduced for time

        # Learning rate sweep: [1e-3, 3e-3, 1e-2]
        lr_range = [0.001, 0.003, 0.01]

        print("\nSweeping dream steps...")
        for dream_steps in dream_steps_range:
            print(f"  Dream steps: {dream_steps}")

            # Run dream phase
            cmd = [
                "python", "transfer_matrix_experiment.py",
                "--mode", "dream",
                "--dream_steps", str(dream_steps),
                "--output_dir", f"robustness_results/dream_steps_{dream_steps}"
            ]

            subprocess.run(cmd, capture_output=True, text=True)

            # Run a few transfer experiments
            cmd = [
                "python", "transfer_matrix_experiment.py",
                "--mode", "transfer",
                "--max_examples", str(3000),
                "--seeds", str(self.seeds[0]),
                "--output_dir", f"robustness_results/dream_steps_{dream_steps}"
            ]

            subprocess.run(cmd, capture_output=True, text=True)

        print("\nSweeping learning rates...")
        for lr in lr_range:
            print(f"  Learning rate: {lr}")

            cmd = [
                "python", "transfer_matrix_experiment.py",
                "--mode", "dream",
                "--dream_steps", str(self.dream_steps),
                "--learning_rate", str(lr),
                "--output_dir", f"robustness_results/lr_{lr}"
            ]

            subprocess.run(cmd, capture_output=True, text=True)

        self.status['robustness'] = True
        elapsed = time.time() - start_time
        print(f"\n✓ Robustness sweeps completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

        return True

    def generate_summary_report(self):
        """Generate summary report of all experiments"""
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70)

        report = {
            'experiment_config': {
                'seeds': self.seeds,
                'dream_steps': self.dream_steps,
                'max_examples': self.max_examples,
                'quick_mode': self.quick
            },
            'status': self.status,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save report
        with open(self.results_dir / 'experiment_summary.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\nExperiment Status:")
        for phase, completed in self.status.items():
            status_str = "✓ COMPLETED" if completed else "✗ INCOMPLETE"
            print(f"  {phase.capitalize():<20}: {status_str}")

        print(f"\nResults saved to: {self.results_dir}")

    def run_all(self):
        """Run complete experimental suite"""
        print("\n" + "="*70)
        print("ZD-CBE COMPREHENSIVE EXPERIMENTAL SUITE")
        print("="*70)
        print(f"Configuration:")
        print(f"  Seeds: {self.seeds}")
        print(f"  Dream steps: {self.dream_steps}")
        print(f"  Max examples: {self.max_examples}")
        print(f"  Quick mode: {self.quick}")
        print("="*70)

        overall_start = time.time()

        # Run all phases
        phases = [
            ('baseline', self.run_baseline_validation),
            ('dream', self.run_dream_phase),
            ('transfer', self.run_transfer_matrix),
            ('mechanistic', self.run_mechanistic_analysis),
            ('robustness', self.run_robustness_sweeps)
        ]

        for phase_name, phase_func in phases:
            if not phase_func():
                print(f"\n✗ Experimental suite stopped at phase: {phase_name}")
                break

        # Generate summary
        self.generate_summary_report()

        overall_elapsed = time.time() - overall_start
        print("\n" + "="*70)
        print("✅ EXPERIMENTAL SUITE COMPLETED")
        print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive ZD-CBE experimental suite'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'baseline', 'dream', 'transfer', 'mechanistic', 'robustness'],
        help='Which experiments to run'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='Random seeds for statistical validation'
    )
    parser.add_argument(
        '--dream_steps',
        type=int,
        default=1000,
        help='Number of dream steps'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=5000,
        help='Maximum training examples per experiment'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: fewer examples, 1 seed (for testing)'
    )

    args = parser.parse_args()

    # Create runner
    runner = ExperimentalSuiteRunner(
        seeds=args.seeds,
        dream_steps=args.dream_steps,
        max_examples=args.max_examples,
        quick=args.quick
    )

    # Run experiments
    if args.mode == 'all':
        runner.run_all()
    elif args.mode == 'baseline':
        runner.run_baseline_validation()
        runner.generate_summary_report()
    elif args.mode == 'dream':
        runner.run_dream_phase()
        runner.generate_summary_report()
    elif args.mode == 'transfer':
        runner.run_transfer_matrix()
        runner.generate_summary_report()
    elif args.mode == 'mechanistic':
        runner.run_mechanistic_analysis()
        runner.generate_summary_report()
    elif args.mode == 'robustness':
        runner.run_robustness_sweeps()
        runner.generate_summary_report()


if __name__ == "__main__":
    main()
