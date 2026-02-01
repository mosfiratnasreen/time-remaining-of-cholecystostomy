"""
visualisation and statistical analysis for MPHY0043
generates 2 plots and runs statistical significance tests.

loads results from:
- outputs/task_a/results.json
- outputs/task_b/results.json
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel
import warnings
warnings.filterwarnings('ignore')

# plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# constants
TOOLS = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']

# colours
COLOURS = {
    'baseline': '#4575b4',
    'task_a': '#fdae61',
    'oracle': '#1a9850',
    'weak_baseline': '#d73027',
    'weak_timed': '#f46d43',
}

###################################################################################################################################
def load_results(base_dir: Path):
    """load results from JSON files."""
    task_a_path = base_dir / "outputs" / "task_a" / "results.json"
    task_b_path = base_dir / "outputs" / "task_b" / "results.json"
    
    task_a_results = None
    task_b_results = None
    
    if task_a_path.exists():
        with open(task_a_path, 'r') as f:
            task_a_results = json.load(f)
        print(f"loaded task A results from {task_a_path}")
    else:
        print(f"warning: task A results not found at {task_a_path}")
    
    if task_b_path.exists():
        with open(task_b_path, 'r') as f:
            task_b_results = json.load(f)
        print(f"loaded task B results from {task_b_path}")
    else:
        print(f"warning: task B results not found at {task_b_path}")
    
    return task_a_results, task_b_results

###################################################################################################################################
def plot_timing_quality_gap(task_b_results: dict, output_dir: Path):
    """
    plot 1: shows the "quality gap" between oracle and predicted timing.
    """
    baseline_map = task_b_results['baseline']['mAP']
    task_a_map = task_b_results['task_a_timed']['mAP']
    oracle_map = task_b_results['oracle_timed']['mAP']
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    models = ['Baseline\n(No Timing)', 'Task A Timed\n(Predicted)', 'Oracle Timed\n(Ground Truth)']
    values = [baseline_map, task_a_map, oracle_map]
    colours = [COLOURS['baseline'], COLOURS['task_a'], COLOURS['oracle']]
    
    bars = ax.bar(models, values, color=colours, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('mAP')
    ax.set_title('Timing Quality Gap: Predicted vs Ground Truth')
    ax.set_ylim(0, max(values) * 1.25)
    
    # value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                   ha='center', fontsize=11, fontweight='bold')
    
    # gaps
    actual_gain = task_a_map - baseline_map
    potential_gain = oracle_map - baseline_map
    lost_benefit = oracle_map - task_a_map
    
    # actual gain annotations
    ax.annotate('', xy=(1, task_a_map), xytext=(1, baseline_map),
               arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.annotate(f'Actual: +{actual_gain:.4f}\n(+{actual_gain/baseline_map*100:.1f}%)', 
               xy=(1.35, (baseline_map + task_a_map)/2), fontsize=9, color='green')
    
    # quality gap annotations
    ax.annotate('', xy=(2, oracle_map), xytext=(2, task_a_map),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    pct_lost = (lost_benefit/potential_gain*100) if potential_gain > 0 else 0
    ax.annotate(f'Gap: {lost_benefit:.4f}\n({pct_lost:.0f}% lost)', 
               xy=(2.35, (task_a_map + oracle_map)/2), fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_quality_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: timing_quality_gap.png")
    return baseline_map, task_a_map, oracle_map

###################################################################################################################################
def plot_weak_model_ablation(task_b_results: dict, output_dir: Path):
    """
    plot 2: shows timing becomes essential when visual signals fail (weak models)
    """
    weak_baseline = task_b_results['weak_baseline']['mAP']
    weak_timed = task_b_results['weak_timed']['mAP']
    baseline = task_b_results['baseline']['mAP']
    oracle = task_b_results['oracle_timed']['mAP']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Standard Models\n(with current tools)', 'Weak Models\n(no current tools)']
    x = np.arange(len(categories))
    width = 0.35
    
    baseline_vals = [baseline, weak_baseline]
    timed_vals = [oracle, weak_timed]
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Without Timing', 
                   color=COLOURS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, timed_vals, width, label='With Oracle Timing', 
                   color=COLOURS['oracle'], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('mAP')
    ax.set_title('Timing Value During Visual Occlusion')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(baseline_vals), max(timed_vals)) * 1.2)
    
    # value labels and improvement percentages
    for i, (b1, b2, v1, v2) in enumerate(zip(bars1, bars2, baseline_vals, timed_vals)):
        ax.annotate(f'{v1:.4f}', xy=(b1.get_x() + b1.get_width()/2, v1 + 0.01),
                   ha='center', fontsize=10)
        ax.annotate(f'{v2:.4f}', xy=(b2.get_x() + b2.get_width()/2, v2 + 0.01),
                   ha='center', fontsize=10)
        
        delta = (v2 - v1) / v1 * 100
        ax.annotate(f'+{delta:.1f}%', xy=(x[i], max(v1, v2) + 0.04),
                   ha='center', fontsize=11, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weak_model_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: weak_model_ablation.png")
    return baseline_vals, timed_vals

###################################################################################################################################
def run_statistical_tests(task_b_results: dict, output_dir: Path):
    """
    statistical significance tests on per-tool AP values.
    """
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)
    
    # per-tool AP values
    baseline_ap = np.array([task_b_results['baseline']['ap_per_tool'][t] for t in TOOLS])
    task_a_ap = np.array([task_b_results['task_a_timed']['ap_per_tool'][t] for t in TOOLS])
    oracle_ap = np.array([task_b_results['oracle_timed']['ap_per_tool'][t] for t in TOOLS])
    weak_baseline_ap = np.array([task_b_results['weak_baseline']['ap_per_tool'][t] for t in TOOLS])
    weak_timed_ap = np.array([task_b_results['weak_timed']['ap_per_tool'][t] for t in TOOLS])
    
    results = {}
    
    # test 1: Baseline vs Oracle
    print("\n1. Baseline vs Oracle Timed:")
    t_stat, p_value = ttest_rel(baseline_ap, oracle_ap)
    w_stat, w_pvalue = wilcoxon(baseline_ap, oracle_ap)
    d = cohens_d(oracle_ap, baseline_ap)
    results['baseline_vs_oracle'] = {'t': t_stat, 'p_ttest': p_value, 'p_wilcoxon': w_pvalue, 'd': d}
    print(f"   Wilcoxon: W={w_stat:.1f}, p={w_pvalue:.4f}")
    print(f"   Cohen's d: {d:.3f} ({effect_size_label(d)})")
    print(f"   Significant (α=0.05): {'Yes' if w_pvalue < 0.05 else 'No'}")
    
    # test 2: Baseline vs Task A Timed
    print("\n2. Baseline vs Task A Timed:")
    t_stat, p_value = ttest_rel(baseline_ap, task_a_ap)
    w_stat, w_pvalue = wilcoxon(baseline_ap, task_a_ap)
    d = cohens_d(task_a_ap, baseline_ap)
    results['baseline_vs_task_a'] = {'t': t_stat, 'p_ttest': p_value, 'p_wilcoxon': w_pvalue, 'd': d}
    print(f"   Wilcoxon: W={w_stat:.1f}, p={w_pvalue:.4f}")
    print(f"   Cohen's d: {d:.3f} ({effect_size_label(d)})")
    print(f"   Significant (α=0.05): {'Yes' if w_pvalue < 0.05 else 'No'}")
    
    # test 3: Weak Baseline vs Weak Timed
    print("\n3. Weak Baseline vs Weak Timed:")
    t_stat, p_value = ttest_rel(weak_baseline_ap, weak_timed_ap)
    w_stat, w_pvalue = wilcoxon(weak_baseline_ap, weak_timed_ap)
    d = cohens_d(weak_timed_ap, weak_baseline_ap)
    results['weak_comparison'] = {'t': t_stat, 'p_ttest': p_value, 'p_wilcoxon': w_pvalue, 'd': d}
    print(f"   Wilcoxon: W={w_stat:.1f}, p={w_pvalue:.4f}")
    print(f"   Cohen's d: {d:.3f} ({effect_size_label(d)})")
    print(f"   Significant (α=0.05): {'Yes' if w_pvalue < 0.05 else 'No'}")
    
    # results
    with open(output_dir / 'statistical_tests.json', 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {output_dir / 'statistical_tests.json'}")
    
    # print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Comparison':<30} {'p (Wilcoxon)':<12} {'Cohen d':<12} {'Significant':<12}")
    print("-"*66)
    print(f"{'Baseline vs Oracle':<30} {results['baseline_vs_oracle']['p_wilcoxon']:<12.4f} {results['baseline_vs_oracle']['d']:<12.3f} {'Yes' if results['baseline_vs_oracle']['p_wilcoxon'] < 0.05 else 'No':<12}")
    print(f"{'Baseline vs Task A':<30} {results['baseline_vs_task_a']['p_wilcoxon']:<12.4f} {results['baseline_vs_task_a']['d']:<12.3f} {'Yes' if results['baseline_vs_task_a']['p_wilcoxon'] < 0.05 else 'No':<12}")
    print(f"{'Weak: Baseline vs Timed':<30} {results['weak_comparison']['p_wilcoxon']:<12.4f} {results['weak_comparison']['d']:<12.3f} {'Yes' if results['weak_comparison']['p_wilcoxon'] < 0.05 else 'No':<12}")
    return results

###################################################################################################################################
def cohens_d(x, y):
    """calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx+ny-2))
    if pooled_std == 0:
        return 0
    return (np.mean(x) - np.mean(y)) / pooled_std

###################################################################################################################################
def effect_size_label(d):
    """return effect size interpretation."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'

###################################################################################################################################
###################################################################################################################################
def main():
    """main execution."""
    base_dir = Path.cwd().parent
    output_dir = base_dir / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("VISUALISATION AND STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # load results
    task_a_results, task_b_results = load_results(base_dir)
    
    if task_b_results is None:
        print("\nerror: task B results required. exiting.")
        return
    
    # plots
    print("\n--- generating plots ---")
    plot_timing_quality_gap(task_b_results, output_dir)
    plot_weak_model_ablation(task_b_results, output_dir)
    
    # stats tests
    run_statistical_tests(task_b_results, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print("="*60)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
if __name__ == "__main__":
    main()