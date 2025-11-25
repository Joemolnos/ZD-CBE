"""
Run multiple experiments with different seeds for statistical validation
"""
import os
import json
import argparse
import subprocess
import time
from typing import List, Dict
import numpy as np
from pathlib import Path


class MultiRunExperiment:
    """Run experiments multiple times with different seeds"""

    def __init__(self, config_path: str, num_runs: int = 10):
        self.config_path = config_path
        self.num_runs = num_runs
        self.results_dir = Path("multi_run_results")
        self.results_dir.mkdir(exist_ok=True)

        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)

    def run_single_experiment(self, seed: int) -> Dict:
        """Run a single experiment with given seed"""
        print(f"\n{'='*60}")
        print(f"Running experiment {seed+1}/{self.num_runs} (seed={seed})")
        print(f"{'='*60}")

        # Create seed-specific config
        config = self.base_config.copy()
        config['seed'] = seed

        seed_config_path = self.results_dir / f"config_seed_{seed}.json"
        with open(seed_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Run experiment
        cmd = [
            'python', 'run_experiment.py',
            '--config', str(seed_config_path),
            '--no_visualization'
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            runtime = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Seed {seed} completed in {runtime:.1f}s")

                # Load results
                with open('output/final_report.json', 'r') as f:
                    results = json.load(f)

                # Add seed and runtime
                results['seed'] = seed
                results['runtime'] = runtime

                # Save seed-specific results
                output_path = self.results_dir / f"results_seed_{seed}.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                return results
            else:
                print(f"❌ Seed {seed} failed!")
                print(f"Error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"⏱️ Seed {seed} timed out after 10 minutes")
            return None
        except Exception as e:
            print(f"❌ Seed {seed} error: {e}")
            return None

    def run_all_experiments(self) -> List[Dict]:
        """Run all experiments"""
        results = []

        for seed in range(self.num_runs):
            result = self.run_single_experiment(seed)
            if result is not None:
                results.append(result)

            # Clean up intermediate files
            if os.path.exists('dreamer_model.pth'):
                os.rename('dreamer_model.pth',
                         self.results_dir / f'dreamer_model_seed_{seed}.pth')
            if os.path.exists('awakening_results.json'):
                os.rename('awakening_results.json',
                         self.results_dir / f'awakening_results_seed_{seed}.json')

        # Save aggregated results
        aggregated_path = self.results_dir / "all_results.json"
        with open(aggregated_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"Completed {len(results)}/{self.num_runs} experiments successfully")
        print(f"{'='*60}")

        return results


class StatisticalAnalyzer:
    """Analyze results from multiple runs"""

    def __init__(self, results: List[Dict]):
        self.results = [r for r in results if r is not None]
        self.n = len(self.results)

    def extract_metric(self, path: List[str]) -> List[float]:
        """Extract metric from nested dict using path"""
        values = []
        for result in self.results:
            value = result
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break

            if value is not None:
                # Handle tensor strings
                if isinstance(value, str) and 'tensor' in value:
                    try:
                        value = float(value.split('(')[1].split(')')[0])
                    except:
                        value = None

                if value is not None:
                    values.append(float(value))

        return values

    def compute_statistics(self, values: List[float]) -> Dict:
        """Compute statistical metrics"""
        if not values:
            return {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'n': 0
            }

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'sem': float(np.std(values) / np.sqrt(len(values))),  # Standard error
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'n': len(values)
        }

    def t_test(self, values1: List[float], values2: List[float]) -> Dict:
        """Perform independent t-test"""
        from scipy import stats

        if len(values1) < 2 or len(values2) < 2:
            return {'t_statistic': None, 'p_value': None, 'significant': False}

        t_stat, p_value = stats.ttest_ind(values1, values2)

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_p05': p_value < 0.05,
            'significant_p01': p_value < 0.01,
            'interpretation': 'significant' if p_value < 0.05 else 'not significant'
        }

    def analyze_all_metrics(self) -> Dict:
        """Analyze all key metrics"""
        analysis = {
            'metadata': {
                'num_runs': self.n,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dream_phase': {},
            'awakening_phase': {},
            'statistical_tests': {}
        }

        # Dream phase metrics
        dream_metrics = {
            'final_loss': ['dream_phase_summary', 'final_loss'],
            'final_perplexity': ['dream_phase_summary', 'final_perplexity'],
            'repetition_index': ['dream_phase_summary', 'concept_emergence', 'repetition_index'],
        }

        for name, path in dream_metrics.items():
            values = self.extract_metric(path)
            analysis['dream_phase'][name] = self.compute_statistics(values)

        # Awakening phase metrics - Dreamer
        dreamer_accuracy = self.extract_metric([
            'awakening_phase_summary',
            'final_performance_comparison',
            'dreamer',
            'character_accuracy'
        ])
        dreamer_loss = self.extract_metric([
            'awakening_phase_summary',
            'final_performance_comparison',
            'dreamer',
            'loss'
        ])

        # Awakening phase metrics - Random
        random_accuracy = self.extract_metric([
            'awakening_phase_summary',
            'final_performance_comparison',
            'tabula_rasa',
            'character_accuracy'
        ])
        random_loss = self.extract_metric([
            'awakening_phase_summary',
            'final_performance_comparison',
            'tabula_rasa',
            'loss'
        ])

        analysis['awakening_phase']['dreamer'] = {
            'accuracy': self.compute_statistics(dreamer_accuracy),
            'loss': self.compute_statistics(dreamer_loss)
        }

        analysis['awakening_phase']['random'] = {
            'accuracy': self.compute_statistics(random_accuracy),
            'loss': self.compute_statistics(random_loss)
        }

        # Statistical tests
        if dreamer_accuracy and random_accuracy:
            analysis['statistical_tests']['accuracy'] = self.t_test(
                dreamer_accuracy, random_accuracy
            )

        if dreamer_loss and random_loss:
            analysis['statistical_tests']['loss'] = self.t_test(
                dreamer_loss, random_loss
            )

        # Calculate advantage
        if dreamer_accuracy and random_accuracy:
            advantages = [d - r for d, r in zip(dreamer_accuracy, random_accuracy)]
            analysis['awakening_phase']['advantage'] = self.compute_statistics(advantages)

        return analysis

    def generate_report(self) -> str:
        """Generate human-readable report"""
        analysis = self.analyze_all_metrics()

        report = []
        report.append("="*60)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append(f"Number of successful runs: {analysis['metadata']['num_runs']}")
        report.append("="*60)

        # Dream phase
        report.append("\n## Dream Phase Results")
        for metric, stats in analysis['dream_phase'].items():
            if stats['mean'] is not None:
                report.append(f"\n{metric}:")
                report.append(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                report.append(f"  SEM: {stats['sem']:.4f}")
                report.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        # Awakening phase
        report.append("\n## Awakening Phase Results")

        dreamer_acc = analysis['awakening_phase']['dreamer']['accuracy']
        random_acc = analysis['awakening_phase']['random']['accuracy']

        if dreamer_acc['mean'] is not None and random_acc['mean'] is not None:
            report.append(f"\nDreamer Accuracy: {dreamer_acc['mean']:.4f} ± {dreamer_acc['std']:.4f}")
            report.append(f"Random Accuracy:  {random_acc['mean']:.4f} ± {random_acc['std']:.4f}")

            advantage = analysis['awakening_phase'].get('advantage', {})
            if advantage.get('mean') is not None:
                report.append(f"\nAdvantage: {advantage['mean']:.4f} ± {advantage['std']:.4f}")

        # Statistical tests
        report.append("\n## Statistical Significance Tests")

        acc_test = analysis['statistical_tests'].get('accuracy', {})
        if acc_test.get('p_value') is not None:
            report.append(f"\nAccuracy t-test:")
            report.append(f"  t-statistic: {acc_test['t_statistic']:.4f}")
            report.append(f"  p-value: {acc_test['p_value']:.6f}")
            report.append(f"  Result: {acc_test['interpretation']}")

            if acc_test['significant_p05']:
                report.append("  ✅ Significant at p < 0.05")
            else:
                report.append("  ❌ NOT significant at p < 0.05")

        report.append("\n" + "="*60)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple experiments for statistical validation'
    )
    parser.add_argument('--config', type=str, default='config_fast.json',
                       help='Base configuration file')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs with different seeds')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze existing results')

    args = parser.parse_args()

    if not args.analyze_only:
        # Run experiments
        experiment = MultiRunExperiment(args.config, args.num_runs)
        results = experiment.run_all_experiments()
    else:
        # Load existing results
        results_dir = Path("multi_run_results")
        results_file = results_dir / "all_results.json"

        if not results_file.exists():
            print("No results file found! Run experiments first.")
            return

        with open(results_file, 'r') as f:
            results = json.load(f)

    # Analyze results
    analyzer = StatisticalAnalyzer(results)
    analysis = analyzer.analyze_all_metrics()

    # Save analysis
    results_dir = Path("multi_run_results")
    analysis_path = results_dir / "statistical_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Generate and save report
    report = analyzer.generate_report()
    print("\n" + report)

    report_path = results_dir / "statistical_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nResults saved to {results_dir}/")
    print(f"  - all_results.json")
    print(f"  - statistical_analysis.json")
    print(f"  - statistical_report.txt")


if __name__ == "__main__":
    main()
