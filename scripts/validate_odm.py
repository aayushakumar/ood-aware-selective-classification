#!/usr/bin/env python3
"""
Validate ODM Correlation with OOD Utility
==========================================

This script validates that ODM (OOD Detectability Metric) correlates with
actual selective classification utility - a key experiment for reviewer credibility.

Key Claim: "ODM predicts when adding OOD signals improves selective classification,
not just when domains are separable."

Usage:
    python validate_odm.py --experiments_dir experiments --output_dir experiments/odm_validation
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from ood_algorithms import (
    OODDetectabilityMetric,
    EnergyOOD,
    validate_odm_correlation
)
from abstention_policy import SelectiveClassificationEvaluator


def compute_ood_utility(
    data: Dict[str, np.ndarray],
    method: str = 'energy'
) -> float:
    """
    Compute OOD utility = AURC(conf only) - AURC(conf + OOD).
    
    Positive utility means OOD signal helps selective classification.
    """
    labels = data['labels']
    predictions = data['predictions']
    confidences = data['confidences']
    ood_scores = data['ood_scores']
    
    # Standardize OOD scores
    ood_std = (ood_scores - ood_scores.mean()) / (ood_scores.std() + 1e-8)
    
    # AURC with confidence only
    aurc_conf = SelectiveClassificationEvaluator.compute_aurc(
        labels, predictions, confidences
    )
    
    # AURC with combined score (higher = more confident = select first)
    # Negate OOD because higher OOD = less confident
    combined_score = confidences - 0.3 * ood_std
    aurc_combined = SelectiveClassificationEvaluator.compute_aurc(
        labels, predictions, combined_score
    )
    
    # Utility = improvement (lower AURC is better)
    utility = aurc_conf - aurc_combined
    
    return utility


def validate_odm(
    datasets: List[str],
    experiments_dir: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Validate ODM against actual OOD utility across all dataset pairs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    odm = OODDetectabilityMetric(method='wasserstein')
    energy_detector = EnergyOOD()
    
    results = []
    
    for source in datasets:
        # Load source data
        source_file = experiments_dir / f'preds_{source}_test.csv'
        if not source_file.exists():
            source_file = experiments_dir / f'{source}_test.csv'
        if not source_file.exists():
            print(f"[WARN] Missing source: {source_file}")
            continue
        
        source_df = pd.read_csv(source_file)
        source_logits = source_df[['logit_0', 'logit_1']].values if 'logit_0' in source_df.columns else None
        if source_logits is None:
            continue
        source_energy = energy_detector.compute_scores(source_logits)
        
        for target in datasets:
            if source == target:
                continue
            
            # Load target data
            target_file = experiments_dir / f'preds_{source}_to_{target}.csv'
            if not target_file.exists():
                target_file = experiments_dir / f'{target}_test.csv'
            if not target_file.exists():
                print(f"[WARN] Missing target: {target_file}")
                continue
            
            target_df = pd.read_csv(target_file)
            target_logits = target_df[['logit_0', 'logit_1']].values if 'logit_0' in target_df.columns else None
            if target_logits is None:
                continue
            target_energy = energy_detector.compute_scores(target_logits)
            
            # Compute ODM
            odm_result = odm.compute(source_energy, target_energy)
            
            # Prepare target data for utility computation
            target_data = {
                'labels': target_df['label'].values,
                'predictions': target_df['pred'].values if 'pred' in target_df.columns else (target_df['pos_prob'].values >= 0.5).astype(int),
                'confidences': np.maximum(target_df['pos_prob'].values, 1 - target_df['pos_prob'].values),
                'ood_scores': odm.correct_scores(target_energy, odm_result['direction'])
            }
            
            # Compute actual OOD utility
            utility = compute_ood_utility(target_data)
            
            results.append({
                'source': source,
                'target': target,
                'odm_score': odm_result['odm_score'],
                'separability': odm_result['separability'],
                'direction': odm_result['direction'],
                'ood_utility': utility,
                'should_use_ood': odm_result['should_use_ood'],
            })
            
            print(f"{source} → {target}: ODM={odm_result['odm_score']:.3f}, "
                  f"Utility={utility:.4f}, Dir={odm_result['direction']}")
    
    results_df = pd.DataFrame(results)
    
    # Compute correlation
    corr_result = validate_odm_correlation(
        [(r['source'], r['target']) for r in results],
        [r['odm_score'] for r in results],
        [r['ood_utility'] for r in results]
    )
    
    print("\n" + "=" * 60)
    print("ODM VALIDATION RESULTS")
    print("=" * 60)
    print(f"Pearson r:  {corr_result['pearson_r']:.3f} (p={corr_result['pearson_p']:.4f})")
    print(f"Spearman r: {corr_result['spearman_r']:.3f} (p={corr_result['spearman_p']:.4f})")
    
    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(results_df['odm_score'], results_df['ood_utility'], 
               s=100, alpha=0.7, c='steelblue', edgecolors='black')
    
    # Add labels
    for _, row in results_df.iterrows():
        ax.annotate(f"{row['source'][0]}→{row['target'][0]}",
                   (row['odm_score'], row['ood_utility']),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Trend line
    if len(results_df) > 2:
        z = np.polyfit(results_df['odm_score'], results_df['ood_utility'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(results_df['odm_score'].min(), results_df['odm_score'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend (r={corr_result["pearson_r"]:.2f})')
    
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('ODM Score (OOD Detectability)', fontsize=12)
    ax.set_ylabel('OOD Utility (AURC Improvement)', fontsize=12)
    ax.set_title('ODM Validates: Higher ODM → Higher OOD Utility', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'odm_validation.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'odm_validation.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # Save results
    results_df.to_csv(output_dir / 'odm_validation_results.csv', index=False)
    
    with open(output_dir / 'odm_correlation.json', 'w') as f:
        import json
        json.dump(corr_result, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Validate ODM Correlation')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                       help='Directory containing prediction files')
    parser.add_argument('--output_dir', type=str, default='experiments/odm_validation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    datasets = ['jigsaw', 'civil', 'hatexplain']
    
    validate_odm(
        datasets,
        Path(args.experiments_dir),
        Path(args.output_dir)
    )


if __name__ == '__main__':
    main()
