#!/usr/bin/env python3
"""
Run Selective Classification Experiments
==========================================

Main experiment runner for OOD-aware fairness-constrained selective classification.

This script:
1. Loads predictions from trained models (or synthetic data)
2. Computes OOD separability and ODM for all dataset pairs
3. Runs abstention experiments with all baselines
4. Generates publication-ready figures and tables

Usage:
    python run_selective_classification.py \
        --data_dir data \
        --experiments_dir experiments \
        --output_dir experiments/selective \
        --fairness_epsilon 0.05 \
        --min_coverage 0.80

For synthetic data testing:
    python run_selective_classification.py --synthetic --n_samples 1000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our modules
from ood_inversion_analysis import (
    compute_ood_separability_metrics,
    compute_ood_matrix,
    plot_ood_heatmaps,
    plot_inversion_taxonomy,
    generate_inversion_report
)
from ood_algorithms import (
    MaxSoftmaxOOD, ODIN_OOD, EnergyOOD,
    OODDetectabilityMetric,
    validate_odm_correlation
)
from abstention_policy import (
    OODAwareAbstentionPolicy,
    ConfidenceOnlyPolicy,
    NaiveOODPolicy,
    ConformalSelectiveClassifier,
    FairnessConstrainedThresholdSolver,
    SelectiveClassificationEvaluator,
    compute_fairness_metrics,
    bootstrap_fairness_gap,
    run_abstention_experiment
)

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions(
    pred_file: Path,
    group_file: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Load predictions and extract required arrays.
    
    Returns dict with: labels, predictions, pos_probs, logits, energy, group_masks
    """
    df = pd.read_csv(pred_file)
    
    result = {
        'labels': df['label'].values,
        'predictions': df['pred'].values if 'pred' in df.columns else (df['pos_prob'].values >= 0.5).astype(int),
        'pos_probs': df['pos_prob'].values if 'pos_prob' in df.columns else np.ones(len(df)) * 0.5,
    }
    
    # Confidences
    result['confidences'] = np.maximum(result['pos_probs'], 1 - result['pos_probs'])
    
    # Logits
    if 'logit_0' in df.columns and 'logit_1' in df.columns:
        result['logits'] = df[['logit_0', 'logit_1']].values
    else:
        # Derive from probabilities
        eps = 1e-6
        result['logits'] = np.column_stack([
            np.log(1 - result['pos_probs'] + eps),
            np.log(result['pos_probs'] + eps)
        ])
    
    # Energy scores
    if 'energy' in df.columns:
        result['energy'] = df['energy'].values
    else:
        result['energy'] = EnergyOOD().compute_scores(result['logits'])
    
    # Group masks
    group_cols = [c for c in df.columns if c.startswith('g_')]
    result['group_masks'] = {c: df[c].values.astype(bool) for c in group_cols}
    
    if not result['group_masks']:
        result['group_masks'] = {'all': np.ones(len(df), dtype=bool)}
    
    return result


def split_dev_test(
    data: Dict[str, np.ndarray],
    dev_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split data into dev (threshold tuning) and test (evaluation) sets.
    
    This is critical to avoid test leakage!
    """
    np.random.seed(seed)
    n = len(data['labels'])
    indices = np.random.permutation(n)
    
    n_dev = int(n * dev_ratio)
    dev_idx = indices[:n_dev]
    test_idx = indices[n_dev:]
    
    def subset(data_dict, idx):
        result = {}
        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                result[key] = val[idx]
            elif isinstance(val, dict):  # group_masks
                result[key] = {k: v[idx] for k, v in val.items()}
            else:
                result[key] = val
        return result
    
    return subset(data, dev_idx), subset(data, test_idx)


# =============================================================================
# OOD ANALYSIS
# =============================================================================

def run_ood_analysis(
    datasets: List[str],
    experiments_dir: Path,
    output_dir: Path
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Run complete OOD separability analysis across all dataset pairs.
    
    Returns:
        separability_dfs: Dict mapping OOD method to DataFrame of results
        odm_results: Dict mapping (source, target) to ODM result
    """
    print("\n" + "=" * 80)
    print("PHASE 1: OOD Separability Analysis")
    print("=" * 80)
    
    ood_methods = {
        'maxsoftmax': MaxSoftmaxOOD(),
        'energy': EnergyOOD(),
    }
    
    separability_results = {method: [] for method in ood_methods}
    odm_results = {}
    odm = OODDetectabilityMetric(method='wasserstein')
    
    for source in tqdm(datasets, desc="Analyzing OOD separability"):
        # Load source data
        source_file = experiments_dir / f'preds_{source}_test.csv'
        if not source_file.exists():
            source_file = experiments_dir / f'{source}_test.csv'
        if not source_file.exists():
            print(f"[WARN] Source file not found: {source_file}")
            continue
        
        source_data = load_predictions(source_file)
        
        for target in datasets:
            if source == target:
                # In-domain case
                for method in ood_methods:
                    separability_results[method].append({
                        'source': source,
                        'target': target,
                        'method': method,
                        'raw_auroc': 0.5,
                        'separability': 0.5,
                        'direction': 'n/a',
                        'is_usable': False,
                    })
                odm_results[(source, target)] = {'should_use_ood': False, 'separability': 0.5}
                continue
            
            # Load target data
            target_file = experiments_dir / f'preds_{source}_to_{target}.csv'
            if not target_file.exists():
                target_file = experiments_dir / f'{target}_test.csv'
            if not target_file.exists():
                print(f"[WARN] Target file not found: {target_file}")
                continue
            
            target_data = load_predictions(target_file)
            
            # Analyze each OOD method
            for method_name, detector in ood_methods.items():
                source_scores = detector.compute_scores(source_data['logits'])
                target_scores = detector.compute_scores(target_data['logits'])
                
                metrics = compute_ood_separability_metrics(source_scores, target_scores)
                metrics['source'] = source
                metrics['target'] = target
                metrics['method'] = method_name
                separability_results[method_name].append(metrics)
            
            # Compute ODM (using energy by default)
            source_energy = source_data['energy']
            target_energy = target_data['energy']
            odm_result = odm.compute(source_energy, target_energy)
            odm_results[(source, target)] = odm_result
            
            print(f"  {source} → {target}: separability={odm_result['separability']:.3f}, "
                  f"direction={odm_result['direction']}")
    
    # Convert to DataFrames
    separability_dfs = {method: pd.DataFrame(results) 
                        for method, results in separability_results.items()}
    
    # Save results
    for method, df in separability_dfs.items():
        df.to_csv(output_dir / f'ood_separability_{method}.csv', index=False)
    
    # Generate visualizations
    print("\n>>> Generating OOD visualizations...")
    plot_ood_heatmaps(separability_dfs, output_dir)
    plot_inversion_taxonomy(separability_dfs, output_dir)
    
    return separability_dfs, odm_results


# =============================================================================
# ABSTENTION EXPERIMENTS
# =============================================================================

def run_abstention_experiments(
    datasets: List[str],
    experiments_dir: Path,
    output_dir: Path,
    odm_results: Dict[Tuple[str, str], Dict],
    fairness_epsilon: float = 0.05,
    min_coverage: float = 0.80
) -> pd.DataFrame:
    """
    Run abstention experiments for all source→target pairs.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: Abstention Experiments")
    print("=" * 80)
    
    all_results = []
    
    for source in tqdm(datasets, desc="Running abstention experiments"):
        for target in datasets:
            if source == target:
                continue
            
            # Load target predictions
            pred_file = experiments_dir / f'preds_{source}_to_{target}.csv'
            if not pred_file.exists():
                pred_file = experiments_dir / f'{target}_test.csv'
            if not pred_file.exists():
                print(f"[WARN] Missing predictions: {pred_file}")
                continue
            
            data = load_predictions(pred_file)
            
            # Split into dev/test to avoid leakage
            dev_data, test_data = split_dev_test(data, dev_ratio=0.5)
            
            # Get ODM result for this pair
            odm_result = odm_results.get((source, target), 
                                          {'should_use_ood': False, 'separability': 0.5, 'direction': 'normal'})
            
            # Run experiments on dev set to find thresholds
            solver = FairnessConstrainedThresholdSolver(
                fairness_epsilon=fairness_epsilon,
                min_coverage=min_coverage
            )
            
            # Find optimal thresholds on DEV set
            thresholds = solver.solve(
                confidences=dev_data['confidences'],
                ood_scores=dev_data['energy'],
                labels=dev_data['labels'],
                predictions=dev_data['predictions'],
                group_masks=dev_data['group_masks'],
                odm_result=odm_result,
                use_ood=True
            )
            
            # Evaluate on TEST set
            results = {}
            
            # 1. No abstention
            no_abstain = np.zeros(len(test_data['labels']), dtype=bool)
            results['no_abstention'] = SelectiveClassificationEvaluator.evaluate(
                test_data['labels'], test_data['predictions'], no_abstain, test_data['group_masks']
            )
            
            # 2. Confidence only
            conf_abstain = test_data['confidences'] < thresholds['conf_threshold']
            results['confidence_only'] = SelectiveClassificationEvaluator.evaluate(
                test_data['labels'], test_data['predictions'], conf_abstain, test_data['group_masks']
            )
            
            # 3. Naive OOD (no inversion correction)
            naive_ood_std = (test_data['energy'] - test_data['energy'].mean()) / (test_data['energy'].std() + 1e-8)
            naive_abstain = naive_ood_std > 0  # standard threshold
            results['naive_ood'] = SelectiveClassificationEvaluator.evaluate(
                test_data['labels'], test_data['predictions'], naive_abstain, test_data['group_masks']
            )
            
            # 4. OOD-Aware (ours)
            policy = OODAwareAbstentionPolicy(
                conf_threshold=thresholds['conf_threshold'],
                ood_threshold=thresholds.get('ood_threshold', 0.0) or 0.0,
                odm_threshold=0.6
            )
            ood_aware_abstain = policy.compute_abstention(
                test_data['confidences'], test_data['energy'], odm_result
            )
            results['ood_aware'] = SelectiveClassificationEvaluator.evaluate(
                test_data['labels'], test_data['predictions'], ood_aware_abstain, test_data['group_masks']
            )
            
            # 5. Conformal prediction
            conformal = ConformalSelectiveClassifier(alpha=0.1)
            # Calibrate on dev
            dev_probs = np.column_stack([1 - dev_data['pos_probs'], dev_data['pos_probs']])
            conformal.calibrate(dev_probs, dev_data['labels'])
            # Predict on test
            test_probs = np.column_stack([1 - test_data['pos_probs'], test_data['pos_probs']])
            conformal_abstain = conformal.compute_abstention(test_probs)
            results['conformal'] = SelectiveClassificationEvaluator.evaluate(
                test_data['labels'], test_data['predictions'], conformal_abstain, test_data['group_masks']
            )
            
            # Collect results
            for method, result in results.items():
                all_results.append({
                    'source': source,
                    'target': target,
                    'method': method,
                    'coverage': result['coverage'],
                    'accuracy': result['accuracy'],
                    'f1': result['f1'],
                    'fpr_gap': result['fpr_gap'],
                    'tpr_gap': result['tpr_gap'],
                    'abstention_gap': result['abstention_gap'],
                    'ood_separability': odm_result.get('separability', 0.5),
                    'ood_direction': odm_result.get('direction', 'n/a'),
                    'constraints_satisfied': (
                        result['coverage'] >= min_coverage and 
                        result['fpr_gap'] <= fairness_epsilon
                    ) if result['accuracy'] else False
                })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'abstention_experiment_results.csv', index=False)
    
    return results_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_publication_figures(
    separability_dfs: Dict[str, pd.DataFrame],
    results_df: pd.DataFrame,
    output_dir: Path
):
    """Generate publication-ready figures."""
    print("\n>>> Generating publication figures...")
    
    # Figure 1: Method comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Aggregate across all pairs
    method_stats = results_df.groupby('method').agg({
        'coverage': 'mean',
        'accuracy': 'mean',
        'fpr_gap': 'mean'
    }).reset_index()
    
    # Order methods
    method_order = ['no_abstention', 'confidence_only', 'naive_ood', 'conformal', 'ood_aware']
    method_labels = ['No Abstention', 'Confidence', 'Naive OOD', 'Conformal', 'OOD-Aware (Ours)']
    
    # Coverage
    ax = axes[0]
    coverage_vals = [method_stats[method_stats['method'] == m]['coverage'].values[0] 
                     for m in method_order if m in method_stats['method'].values]
    ax.bar(method_labels[:len(coverage_vals)], coverage_vals, color='steelblue')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage (higher is better)')
    ax.axhline(y=0.8, color='red', linestyle='--', label='Constraint')
    ax.set_ylim(0, 1.1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Accuracy
    ax = axes[1]
    acc_vals = [method_stats[method_stats['method'] == m]['accuracy'].values[0] 
                for m in method_order if m in method_stats['method'].values]
    ax.bar(method_labels[:len(acc_vals)], acc_vals, color='forestgreen')
    ax.set_ylabel('Accuracy on Covered')
    ax.set_title('Accuracy (higher is better)')
    ax.set_ylim(0, 1.1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # FPR Gap
    ax = axes[2]
    fpr_vals = [method_stats[method_stats['method'] == m]['fpr_gap'].values[0] 
                for m in method_order if m in method_stats['method'].values]
    ax.bar(method_labels[:len(fpr_vals)], fpr_vals, color='coral')
    ax.set_ylabel('Max FPR Gap')
    ax.set_title('FPR Gap (lower is better)')
    ax.axhline(y=0.05, color='red', linestyle='--', label='Constraint')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'method_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # Figure 2: Per-pair results heatmap
    for metric in ['accuracy', 'fpr_gap']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        key_methods = ['confidence_only', 'naive_ood', 'ood_aware']
        titles = ['Confidence Only', 'Naive OOD', 'OOD-Aware (Ours)']
        
        for ax, method, title in zip(axes, key_methods, titles):
            method_df = results_df[results_df['method'] == method]
            if len(method_df) == 0:
                continue
            
            pivot = method_df.pivot(index='source', columns='target', values=metric)
            
            cmap = 'RdYlGn' if metric == 'accuracy' else 'RdYlGn_r'
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                       vmin=0, vmax=1 if metric == 'accuracy' else 0.3)
            ax.set_title(f'{title}\n({metric})')
        
        plt.tight_layout()
        fig.savefig(output_dir / f'heatmap_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Figures saved to {output_dir}")


def generate_latex_tables(results_df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX tables for the paper."""
    
    # Main results table
    method_order = ['no_abstention', 'confidence_only', 'naive_ood', 'conformal', 'ood_aware']
    method_names = {
        'no_abstention': 'No Abstention',
        'confidence_only': 'Confidence Only',
        'naive_ood': 'Naive OOD',
        'conformal': 'Conformal',
        'ood_aware': '\\textbf{OOD-Aware (Ours)}'
    }
    
    agg = results_df.groupby('method').agg({
        'coverage': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'fpr_gap': ['mean', 'std']
    })
    
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Selective Classification Results (averaged across all 6 cross-domain pairs)}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Method & Coverage $\\uparrow$ & Accuracy $\\uparrow$ & FPR Gap $\\downarrow$ \\\\")
    latex.append("\\midrule")
    
    for method in method_order:
        if method not in agg.index:
            continue
        cov = agg.loc[method, ('coverage', 'mean')]
        cov_std = agg.loc[method, ('coverage', 'std')]
        acc = agg.loc[method, ('accuracy', 'mean')]
        acc_std = agg.loc[method, ('accuracy', 'std')]
        fpr = agg.loc[method, ('fpr_gap', 'mean')]
        fpr_std = agg.loc[method, ('fpr_gap', 'std')]
        
        latex.append(f"{method_names[method]} & {cov:.2f}$\\pm${cov_std:.2f} & "
                    f"{acc:.2f}$\\pm${acc_std:.2f} & {fpr:.2f}$\\pm${fpr_std:.2f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_dir / 'main_results_table.tex', 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"LaTeX tables saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Selective Classification Experiments')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                       help='Directory containing predictions')
    parser.add_argument('--output_dir', type=str, default='experiments/selective',
                       help='Output directory for results')
    parser.add_argument('--fairness_epsilon', type=float, default=0.05,
                       help='Maximum allowed FPR gap')
    parser.add_argument('--min_coverage', type=float, default=0.80,
                       help='Minimum required coverage')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate and use synthetic data')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for synthetic data')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("OOD-Aware Fairness-Constrained Selective Classification")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Experiments directory: {experiments_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Fairness epsilon (max FPR gap): {args.fairness_epsilon}")
    print(f"Minimum coverage: {args.min_coverage}")
    print("=" * 80)
    
    # Generate synthetic data if requested
    if args.synthetic:
        print("\n>>> Generating synthetic data...")
        from generate_synthetic_data import generate_cross_domain_data, generate_model_predictions
        
        datasets_config = {
            'jigsaw': (args.n_samples, 0.08),
            'civil': (args.n_samples, 0.08),
            'hatexplain': (args.n_samples, 0.60),
        }
        all_data = generate_cross_domain_data(datasets_config, data_dir)
        
        for source in ['jigsaw', 'civil', 'hatexplain']:
            targets = [t for t in ['jigsaw', 'civil', 'hatexplain'] if t != source]
            generate_model_predictions(all_data, experiments_dir, source, targets)
    
    datasets = ['jigsaw', 'civil', 'hatexplain']
    
    # Phase 1: OOD Analysis
    separability_dfs, odm_results = run_ood_analysis(datasets, experiments_dir, output_dir)
    
    # Phase 2: Abstention Experiments
    results_df = run_abstention_experiments(
        datasets, experiments_dir, output_dir, odm_results,
        fairness_epsilon=args.fairness_epsilon,
        min_coverage=args.min_coverage
    )
    
    # Phase 3: Generate Figures
    generate_publication_figures(separability_dfs, results_df, output_dir)
    generate_latex_tables(results_df, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    print("\nResults Summary:")
    summary = results_df.groupby('method').agg({
        'coverage': 'mean',
        'accuracy': 'mean', 
        'fpr_gap': 'mean',
        'constraints_satisfied': 'mean'
    }).round(3)
    print(summary.to_string())
    
    print(f"\nOutputs saved to: {output_dir}")
    print("  - ood_separability_*.csv")
    print("  - abstention_experiment_results.csv")
    print("  - method_comparison.png/pdf")
    print("  - heatmap_*.png")
    print("  - main_results_table.tex")


if __name__ == '__main__':
    main()
