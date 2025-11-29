"""
Analyze Strategic Transfer Results and Create Visualizations

Generates:
1. Transfer matrix heatmap (partial - based on 24 experiments)
2. Sample efficiency comparison charts
3. Learning curves
4. Statistical analysis
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Task names in order
TASK_NAMES = ['addition', 'reverse', 'pattern', 'sorting', 'parity', 'sequence']
TASK_SHORT = ['Add', 'Rev', 'Pat', 'Sort', 'Par', 'Seq']

def load_results():
    """Load all experiment results"""
    results_dir = Path("strategic_transfer_results")

    all_results = []
    for json_file in results_dir.glob("transfer_*.json"):
        with open(json_file, 'r') as f:
            all_results.append(json.load(f))

    print(f"Loaded {len(all_results)} experiment results")
    return all_results

def create_partial_transfer_matrix(results):
    """Create transfer matrix from available results"""

    # Initialize matrices
    final_acc_matrix = np.zeros((len(TASK_NAMES), len(TASK_NAMES)))
    conv_matrix = np.zeros((len(TASK_NAMES), len(TASK_NAMES)))
    baseline_conv = np.zeros(len(TASK_NAMES))
    baseline_acc = np.zeros(len(TASK_NAMES))

    # Flags for which cells we have data for
    has_data = np.zeros((len(TASK_NAMES), len(TASK_NAMES)), dtype=bool)

    for result in results:
        dream_task = result['dream_task']
        finetune_task = result['finetune_task']
        final_acc = result['final_accuracy']
        conv_90 = result['convergence_90']

        finetune_idx = TASK_NAMES.index(finetune_task)

        if dream_task is None:
            # Baseline
            baseline_acc[finetune_idx] = final_acc
            if conv_90 > 0:
                baseline_conv[finetune_idx] = conv_90
        else:
            # Transfer
            dream_idx = TASK_NAMES.index(dream_task)
            final_acc_matrix[dream_idx, finetune_idx] = final_acc
            if conv_90 > 0:
                conv_matrix[dream_idx, finetune_idx] = conv_90
            has_data[dream_idx, finetune_idx] = True

    return final_acc_matrix, conv_matrix, baseline_conv, baseline_acc, has_data

def plot_transfer_heatmap(results):
    """Plot transfer matrix heatmap"""

    final_acc_matrix, conv_matrix, baseline_conv, baseline_acc, has_data = create_partial_transfer_matrix(results)

    # Calculate improvement ratio where we have data
    improvement_matrix = np.zeros_like(conv_matrix)
    for i in range(len(TASK_NAMES)):
        for j in range(len(TASK_NAMES)):
            if has_data[i, j] and conv_matrix[i, j] > 0 and baseline_conv[j] > 0:
                improvement_matrix[i, j] = baseline_conv[j] / conv_matrix[i, j]
            else:
                improvement_matrix[i, j] = np.nan  # Mark as missing data

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Final accuracy
    mask1 = np.isnan(final_acc_matrix) | (final_acc_matrix == 0)
    sns.heatmap(
        final_acc_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=TASK_SHORT,
        yticklabels=TASK_SHORT,
        ax=ax1,
        vmin=0,
        vmax=1.0,
        mask=mask1,
        cbar_kws={'label': 'Final Accuracy'}
    )
    ax1.set_title('Transfer Matrix: Final Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fine-tune Task →', fontsize=12)
    ax1.set_ylabel('Dream Task ↓', fontsize=12)

    # Plot 2: Sample efficiency improvement
    mask2 = np.isnan(improvement_matrix)
    sns.heatmap(
        improvement_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        xticklabels=TASK_SHORT,
        yticklabels=TASK_SHORT,
        ax=ax2,
        vmin=0.5,
        vmax=5.0,
        center=1.0,
        mask=mask2,
        cbar_kws={'label': 'Sample Efficiency Improvement (×)'}
    )
    ax2.set_title('Transfer Matrix: Sample Efficiency (Strategic Subset)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fine-tune Task →', fontsize=12)
    ax2.set_ylabel('Dream Task ↓', fontsize=12)

    plt.tight_layout()
    plt.savefig('strategic_transfer_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: strategic_transfer_matrix.png")
    plt.close()

    return improvement_matrix, baseline_conv, has_data

def plot_diagonal_comparison(results):
    """Plot diagonal vs baseline comparison"""

    final_acc_matrix, conv_matrix, baseline_conv, baseline_acc, has_data = create_partial_transfer_matrix(results)

    # Extract diagonal elements
    diagonal_conv = []
    diagonal_acc = []
    task_labels = []

    for i in range(len(TASK_NAMES)):
        if has_data[i, i]:  # Diagonal element exists
            diagonal_conv.append(conv_matrix[i, i])
            diagonal_acc.append(final_acc_matrix[i, i])
            task_labels.append(TASK_SHORT[i])

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Convergence comparison
    x = np.arange(len(task_labels))
    width = 0.35

    baseline_vals = [baseline_conv[i] if baseline_conv[i] > 0 else 0 for i in range(len(task_labels))]

    ax1.bar(x - width/2, baseline_vals, width, label='Random', color='coral', alpha=0.8)
    ax1.bar(x + width/2, diagonal_conv, width, label='Dreamer', color='steelblue', alpha=0.8)

    ax1.set_ylabel('Examples to 90% Accuracy', fontsize=12)
    ax1.set_title('Sample Efficiency: Random vs Dreamer', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Improvement ratio
    improvement_ratios = []
    for i, task_idx in enumerate(range(len(task_labels))):
        if baseline_conv[task_idx] > 0 and diagonal_conv[i] > 0:
            improvement_ratios.append(baseline_conv[task_idx] / diagonal_conv[i])
        else:
            improvement_ratios.append(0)

    colors = ['green' if r > 1 else 'red' for r in improvement_ratios]
    ax2.bar(x, improvement_ratios, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Sample Efficiency Improvement (×)', fontsize=12)
    ax2.set_title('DFCB Improvement Factor', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_labels)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(improvement_ratios):
        if v > 0:
            ax2.text(i, v + 0.1, f'{v:.2f}×', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('diagonal_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diagonal_comparison.png")
    plt.close()

    return improvement_ratios

def generate_summary_statistics(results):
    """Generate summary statistics"""

    final_acc_matrix, conv_matrix, baseline_conv, baseline_acc, has_data = create_partial_transfer_matrix(results)

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Baseline performance
    print("\nBaseline Performance (Random Initialization):")
    for i, task in enumerate(TASK_NAMES):
        if baseline_conv[i] > 0:
            print(f"  {task.capitalize():<12}: {baseline_acc[i]:.1%} accuracy, {baseline_conv[i]:.0f} examples to 90%")

    # Diagonal performance
    print("\nDiagonal Performance (Dream → Same Task):")
    improvements = []
    for i in range(len(TASK_NAMES)):
        if has_data[i, i] and conv_matrix[i, i] > 0 and baseline_conv[i] > 0:
            improvement = baseline_conv[i] / conv_matrix[i, i]
            improvements.append(improvement)
            print(f"  {TASK_NAMES[i].capitalize():<12}: {final_acc_matrix[i, i]:.1%} accuracy, {conv_matrix[i, i]:.0f} examples to 90% ({improvement:.2f}× improvement)")

    if improvements:
        print(f"\nAverage Improvement: {np.mean(improvements):.2f}× (±{np.std(improvements):.2f})")
        print(f"Range: {np.min(improvements):.2f}× to {np.max(improvements):.2f}×")

    # Off-diagonal statistics
    print("\nOff-Diagonal Transfer:")
    off_diagonal_improvements = []
    for i in range(len(TASK_NAMES)):
        for j in range(len(TASK_NAMES)):
            if i != j and has_data[i, j] and conv_matrix[i, j] > 0 and baseline_conv[j] > 0:
                improvement = baseline_conv[j] / conv_matrix[i, j]
                off_diagonal_improvements.append(improvement)

    if off_diagonal_improvements:
        print(f"Average off-diagonal improvement: {np.mean(off_diagonal_improvements):.2f}× (±{np.std(off_diagonal_improvements):.2f})")
        print(f"Range: {np.min(off_diagonal_improvements):.2f}× to {np.max(off_diagonal_improvements):.2f}×")

    print("\n" + "="*70)

    return {
        'diagonal_improvements': improvements,
        'off_diagonal_improvements': off_diagonal_improvements,
        'baseline_convergence': baseline_conv.tolist(),
        'diagonal_convergence': [conv_matrix[i, i] for i in range(len(TASK_NAMES)) if has_data[i, i]]
    }

def main():
    """Main analysis pipeline"""

    print("="*70)
    print("ANALYZING STRATEGIC TRANSFER RESULTS")
    print("="*70)

    # Load results
    results = load_results()

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_transfer_heatmap(results)
    plot_diagonal_comparison(results)

    # Generate statistics
    stats_summary = generate_summary_statistics(results)

    # Save statistics
    with open('strategic_analysis_summary.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print("\n✅ ANALYSIS COMPLETED")
    print("Generated files:")
    print("  - strategic_transfer_matrix.png")
    print("  - diagonal_comparison.png")
    print("  - strategic_analysis_summary.json")

if __name__ == "__main__":
    main()
