"""
Visualization script for ZD-CBE v2 results
Generates the "killer chart": Learning curves for all tasks
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_all_results(results_dir: str = "results_v2_final") -> Dict:
    """Load all experimental results"""
    results_path = Path(results_dir)

    all_results = {
        'addition': {'random': [], 'dreamer': [], 'dreamer_no_vq': []},
        'reverse': {'random': [], 'dreamer': [], 'dreamer_no_vq': []},
        'pattern': {'random': [], 'dreamer': [], 'dreamer_no_vq': []}
    }

    for task in ['addition', 'reverse', 'pattern']:
        for model in ['random', 'dreamer', 'dreamer_no_vq']:
            for seed in [0, 1, 2]:
                filename = f"{task}_{model}_seed{seed}.json"
                filepath = results_path / filename

                if filepath.exists():
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        all_results[task][model].append(data)

    return all_results


def plot_learning_curves(all_results: Dict, output_dir: str = "plots_v2"):
    """Generate learning curve plots"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tasks = ['addition', 'reverse', 'pattern']
    task_titles = ['Binary Addition', 'Sequence Reverse', 'Pattern Completion']

    colors = {
        'random': '#e74c3c',  # Red
        'dreamer': '#3498db',  # Blue
        'dreamer_no_vq': '#2ecc71'  # Green
    }

    labels = {
        'random': 'Random Init',
        'dreamer': 'Dreamer (VQ)',
        'dreamer_no_vq': 'Dreamer (no VQ)'
    }

    for idx, (task, title) in enumerate(zip(tasks, task_titles)):
        ax = axes[idx]

        for model in ['random', 'dreamer', 'dreamer_no_vq']:
            runs = all_results[task][model]

            if not runs:
                continue

            # Extract learning curves from all seeds
            all_curves = []
            for run in runs:
                curve = run['learning_curve']  # List of (examples, accuracy)
                all_curves.append(curve)

            # Find common x-axis (examples)
            if all_curves:
                examples = [point[0] for point in all_curves[0]]

                # Extract accuracies for each seed
                accuracies = []
                for curve in all_curves:
                    acc = [point[1] for point in curve]
                    accuracies.append(acc)

                # Calculate mean and std
                accuracies = np.array(accuracies)
                mean_acc = np.mean(accuracies, axis=0)
                std_acc = np.std(accuracies, axis=0)

                # Plot
                ax.plot(examples, mean_acc, label=labels[model],
                       color=colors[model], linewidth=2)
                ax.fill_between(examples, mean_acc - std_acc, mean_acc + std_acc,
                               alpha=0.2, color=colors[model])

        # Formatting
        ax.set_xlabel('Training Examples', fontsize=12)
        ax.set_ylabel('Character Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / 'learning_curves_all_tasks.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'learning_curves_all_tasks.pdf', bbox_inches='tight')
    print(f"✅ Learning curves saved to {output_path}/learning_curves_all_tasks.png/pdf")
    plt.close()


def plot_convergence_comparison(all_results: Dict, output_dir: str = "plots_v2"):
    """Plot convergence speed comparison"""
    output_path = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = ['addition', 'reverse', 'pattern']
    task_labels = ['Addition', 'Reverse', 'Pattern']

    models = ['random', 'dreamer', 'dreamer_no_vq']
    model_labels = ['Random', 'Dreamer (VQ)', 'Dreamer (no VQ)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    x = np.arange(len(tasks))
    width = 0.25

    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        convergence_points = []
        errors = []

        for task in tasks:
            runs = all_results[task][model]
            conv_points = [r['convergence_90'] for r in runs if r['convergence_90'] > 0]

            if conv_points:
                convergence_points.append(np.mean(conv_points))
                errors.append(np.std(conv_points))
            else:
                convergence_points.append(3500)  # Above max if no convergence
                errors.append(0)

        ax.bar(x + i * width, convergence_points, width, label=label,
               color=color, yerr=errors, capsize=5, alpha=0.8)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Examples to 90% Accuracy', fontsize=12)
    ax.set_title('Sample Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 3500])

    plt.tight_layout()
    plt.savefig(output_path / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'convergence_comparison.pdf', bbox_inches='tight')
    print(f"✅ Convergence comparison saved to {output_path}/convergence_comparison.png/pdf")
    plt.close()


def plot_final_accuracy_comparison(all_results: Dict, output_dir: str = "plots_v2"):
    """Plot final accuracy comparison"""
    output_path = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = ['addition', 'reverse', 'pattern']
    task_labels = ['Addition', 'Reverse', 'Pattern']

    models = ['random', 'dreamer', 'dreamer_no_vq']
    model_labels = ['Random', 'Dreamer (VQ)', 'Dreamer (no VQ)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    x = np.arange(len(tasks))
    width = 0.25

    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        accuracies = []
        errors = []

        for task in tasks:
            runs = all_results[task][model]
            final_accs = [r['final_accuracy'] for r in runs]

            accuracies.append(np.mean(final_accs))
            errors.append(np.std(final_accs))

        ax.bar(x + i * width, accuracies, width, label=label,
               color=color, yerr=errors, capsize=5, alpha=0.8)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='90% Target')

    plt.tight_layout()
    plt.savefig(output_path / 'final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'final_accuracy_comparison.pdf', bbox_inches='tight')
    print(f"✅ Final accuracy comparison saved to {output_path}/final_accuracy_comparison.png/pdf")
    plt.close()


def generate_summary_table(all_results: Dict, output_dir: str = "plots_v2"):
    """Generate LaTeX summary table"""
    output_path = Path(output_dir)

    tasks = ['addition', 'reverse', 'pattern']
    models = ['random', 'dreamer', 'dreamer_no_vq']

    latex_table = r"""\begin{table}[h]
\centering
\caption{Sample Efficiency Results Across Three Symbolic Reasoning Tasks}
\label{tab:results}
\begin{tabular}{l|ccc|ccc}
\hline
\multirow{2}{*}{Task} & \multicolumn{3}{c|}{Final Accuracy (\%)} & \multicolumn{3}{c}{Examples to 90\%} \\
& Random & Dreamer & Dreamer-NoVQ & Random & Dreamer & Dreamer-NoVQ \\
\hline
"""

    for task in tasks:
        task_name = task.capitalize()
        row = [task_name]

        # Final accuracies
        for model in models:
            runs = all_results[task][model]
            final_accs = [r['final_accuracy'] * 100 for r in runs]
            mean_acc = np.mean(final_accs)
            std_acc = np.std(final_accs)
            row.append(f"{mean_acc:.1f} $\\pm$ {std_acc:.1f}")

        # Convergence points
        for model in models:
            runs = all_results[task][model]
            conv_points = [r['convergence_90'] for r in runs if r['convergence_90'] > 0]

            if conv_points:
                mean_conv = np.mean(conv_points)
                std_conv = np.std(conv_points)
                row.append(f"{mean_conv:.0f} $\\pm$ {std_conv:.0f}")
            else:
                row.append("DNC")  # Did Not Converge

        latex_table += " & ".join(row) + " \\\\\n"

    latex_table += r"""\hline
\end{tabular}
\end{table}
"""

    with open(output_path / 'results_table.tex', 'w') as f:
        f.write(latex_table)

    print(f"✅ LaTeX table saved to {output_path}/results_table.tex")


def main():
    print("="*70)
    print("GENERATING V2 VISUALIZATIONS")
    print("="*70)

    # Load results
    print("\n📂 Loading experimental results...")
    all_results = load_all_results("results_v2_final")

    # Generate plots
    print("\n📊 Generating learning curves...")
    plot_learning_curves(all_results)

    print("\n📊 Generating convergence comparison...")
    plot_convergence_comparison(all_results)

    print("\n📊 Generating final accuracy comparison...")
    plot_final_accuracy_comparison(all_results)

    print("\n📝 Generating LaTeX summary table...")
    generate_summary_table(all_results)

    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("Output directory: plots_v2/")


if __name__ == "__main__":
    main()
