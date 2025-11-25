"""
Generate publication-quality plots for ZD-CBE paper
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import argparse

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("husl")


class PublicationPlotter:
    """Generate publication-quality plots"""

    def __init__(self, results_dir: str = "multi_run_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("publication_figures")
        self.output_dir.mkdir(exist_ok=True)

        # Load statistical analysis
        analysis_path = self.results_dir / "statistical_analysis.json"
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                self.analysis = json.load(f)
        else:
            self.analysis = None

        # Load all individual results
        all_results_path = self.results_dir / "all_results.json"
        if all_results_path.exists():
            with open(all_results_path, 'r') as f:
                self.all_results = json.load(f)
        else:
            self.all_results = []

    def plot_accuracy_comparison(self):
        """Plot dreamer vs random accuracy with error bars"""
        if not self.analysis:
            print("No analysis data available")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        dreamer_stats = self.analysis['awakening_phase']['dreamer']['accuracy']
        random_stats = self.analysis['awakening_phase']['random']['accuracy']

        models = ['Dreamer', 'Random']
        means = [dreamer_stats['mean'], random_stats['mean']]
        stds = [dreamer_stats['std'], random_stats['std']]

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                     color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Character-Level Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Mathematical Task Performance\n(Addition, Binary Encoding)',
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=13)
        ax.set_ylim(0, max(means) * 1.3)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add significance marker if significant
        acc_test = self.analysis.get('statistical_tests', {}).get('accuracy', {})
        if acc_test.get('significant_p05'):
            y_max = max(means) + max(stds) + 0.05
            ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=2)
            ax.plot([0, 0], [y_max-0.01, y_max], 'k-', linewidth=2)
            ax.plot([1, 1], [y_max-0.01, y_max], 'k-', linewidth=2)
            ax.text(0.5, y_max + 0.01, f'p = {acc_test["p_value"]:.4f}',
                   ha='center', fontsize=11, fontweight='bold')

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Generated: accuracy_comparison.pdf/png")

    def plot_loss_comparison(self):
        """Plot loss comparison with error bars"""
        if not self.analysis:
            print("No analysis data available")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        dreamer_stats = self.analysis['awakening_phase']['dreamer']['loss']
        random_stats = self.analysis['awakening_phase']['random']['loss']

        models = ['Dreamer', 'Random']
        means = [dreamer_stats['mean'], random_stats['mean']]
        stds = [dreamer_stats['std'], random_stats['std']]

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                     color=['#2ecc71', '#f39c12'], edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Final Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
        ax.set_title('Training Loss Comparison\n(Lower is Better)',
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=13)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.05, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Generated: loss_comparison.pdf/png")

    def plot_dream_phase_metrics(self):
        """Plot dream phase self-organization metrics"""
        if not self.analysis:
            print("No analysis data available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Final loss
        loss_stats = self.analysis['dream_phase'].get('final_loss', {})
        if loss_stats.get('mean') is not None:
            ax = axes[0]
            ax.bar([0], [loss_stats['mean']], yerr=[loss_stats['std']],
                  capsize=5, alpha=0.7, color='#9b59b6', edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Final Loss', fontsize=14, fontweight='bold')
            ax.set_title('Dream Phase Self-Organization\n(1000 Steps)',
                        fontsize=14, fontweight='bold')
            ax.set_xticks([0])
            ax.set_xticklabels(['Final Loss'], fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.text(0, loss_stats['mean'] + loss_stats['std'] + 0.05,
                   f"{loss_stats['mean']:.3f}±{loss_stats['std']:.3f}",
                   ha='center', fontsize=11, fontweight='bold')

        # Repetition index
        rep_stats = self.analysis['dream_phase'].get('repetition_index', {})
        if rep_stats.get('mean') is not None:
            ax = axes[1]
            ax.bar([0], [rep_stats['mean']], yerr=[rep_stats['std']],
                  capsize=5, alpha=0.7, color='#1abc9c', edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Repetition Index', fontsize=14, fontweight='bold')
            ax.set_title('Emergent Pattern Formation\n(Higher = More Structure)',
                        fontsize=14, fontweight='bold')
            ax.set_xticks([0])
            ax.set_xticklabels(['Repetition Index'], fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.text(0, rep_stats['mean'] + rep_stats['std'] + 0.5,
                   f"{rep_stats['mean']:.2f}±{rep_stats['std']:.2f}",
                   ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'dream_phase_metrics.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'dream_phase_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Generated: dream_phase_metrics.pdf/png")

    def plot_advantage_distribution(self):
        """Plot distribution of advantage across runs"""
        if not self.all_results:
            print("No individual results available")
            return

        advantages = []
        for result in self.all_results:
            try:
                dreamer_acc = result['awakening_phase_summary']['final_performance_comparison']['dreamer']['character_accuracy']
                random_acc = result['awakening_phase_summary']['final_performance_comparison']['tabula_rasa']['character_accuracy']
                advantages.append(dreamer_acc - random_acc)
            except:
                continue

        if not advantages:
            print("Could not extract advantages")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        n, bins, patches = ax.hist(advantages, bins=10, alpha=0.7, color='#3498db',
                                   edgecolor='black', linewidth=1.5)

        # Add mean line
        mean_adv = np.mean(advantages)
        ax.axvline(mean_adv, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_adv:.3f}')

        # Add zero line
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        ax.set_xlabel('Accuracy Advantage (Dreamer - Random)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'Distribution of Dreamer Advantage\n({len(advantages)} runs)',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'advantage_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'advantage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Generated: advantage_distribution.pdf/png")

    def plot_all_metrics_summary(self):
        """Create a comprehensive summary figure"""
        if not self.analysis:
            print("No analysis data available")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        dreamer_acc = self.analysis['awakening_phase']['dreamer']['accuracy']
        random_acc = self.analysis['awakening_phase']['random']['accuracy']
        models = ['Dreamer', 'Random']
        means = [dreamer_acc['mean'], random_acc['mean']]
        stds = [dreamer_acc['std'], random_acc['std']]
        ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7,
               color=['#3498db', '#e74c3c'], edgecolor='black')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('A) Awakening Accuracy')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Loss comparison
        ax2 = fig.add_subplot(gs[0, 1])
        dreamer_loss = self.analysis['awakening_phase']['dreamer']['loss']
        random_loss = self.analysis['awakening_phase']['random']['loss']
        means = [dreamer_loss['mean'], random_loss['mean']]
        stds = [dreamer_loss['std'], random_loss['std']]
        ax2.bar(models, means, yerr=stds, capsize=5, alpha=0.7,
               color=['#2ecc71', '#f39c12'], edgecolor='black')
        ax2.set_ylabel('Loss')
        ax2.set_title('B) Training Loss')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Dream loss
        ax3 = fig.add_subplot(gs[0, 2])
        dream_loss = self.analysis['dream_phase'].get('final_loss', {})
        if dream_loss.get('mean') is not None:
            ax3.bar(['Dream\nLoss'], [dream_loss['mean']], yerr=[dream_loss['std']],
                   capsize=5, alpha=0.7, color='#9b59b6', edgecolor='black')
            ax3.set_ylabel('Loss')
            ax3.set_title('C) Dream Phase')
            ax3.grid(axis='y', alpha=0.3)

        # 4. Advantage
        ax4 = fig.add_subplot(gs[1, :2])
        if self.all_results:
            advantages = []
            for result in self.all_results:
                try:
                    dreamer = result['awakening_phase_summary']['final_performance_comparison']['dreamer']['character_accuracy']
                    random = result['awakening_phase_summary']['final_performance_comparison']['tabula_rasa']['character_accuracy']
                    advantages.append(dreamer - random)
                except:
                    continue

            if advantages:
                ax4.hist(advantages, bins=10, alpha=0.7, color='#3498db', edgecolor='black')
                ax4.axvline(np.mean(advantages), color='red', linestyle='--', linewidth=2,
                           label=f'Mean = {np.mean(advantages):.3f}')
                ax4.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                ax4.set_xlabel('Accuracy Advantage')
                ax4.set_ylabel('Frequency')
                ax4.set_title('D) Advantage Distribution')
                ax4.legend()
                ax4.grid(alpha=0.3)

        # 5. Statistical significance
        ax5 = fig.add_subplot(gs[1, 2])
        acc_test = self.analysis.get('statistical_tests', {}).get('accuracy', {})
        if acc_test.get('p_value') is not None:
            p_value = acc_test['p_value']
            is_sig = p_value < 0.05

            colors = ['green' if is_sig else 'red']
            ax5.bar(['p-value'], [p_value], color=colors, alpha=0.7, edgecolor='black')
            ax5.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
            ax5.set_ylabel('p-value')
            ax5.set_title('E) Statistical Significance')
            ax5.legend()
            ax5.set_ylim(0, max(0.1, p_value * 1.2))
            ax5.grid(axis='y', alpha=0.3)

            if is_sig:
                ax5.text(0, p_value + 0.005, '✓ Significant',
                        ha='center', fontsize=10, fontweight='bold', color='green')
            else:
                ax5.text(0, p_value + 0.005, '✗ Not Significant',
                        ha='center', fontsize=10, fontweight='bold', color='red')

        fig.suptitle('Zero-Data Cognitive Bootstrap Engine - Results Summary',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / 'summary_figure.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'summary_figure.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Generated: summary_figure.pdf/png")

    def generate_all_plots(self):
        """Generate all publication plots"""
        print("\n" + "="*60)
        print("GENERATING PUBLICATION-QUALITY PLOTS")
        print("="*60 + "\n")

        self.plot_accuracy_comparison()
        self.plot_loss_comparison()
        self.plot_dream_phase_metrics()
        self.plot_advantage_distribution()
        self.plot_all_metrics_summary()

        print("\n" + "="*60)
        print(f"All plots saved to: {self.output_dir}/")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate publication plots')
    parser.add_argument('--results_dir', type=str, default='multi_run_results',
                       help='Directory containing multi-run results')

    args = parser.parse_args()

    plotter = PublicationPlotter(args.results_dir)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
