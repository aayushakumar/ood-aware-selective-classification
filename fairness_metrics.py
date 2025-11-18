#!/usr/bin/env python3
"""
Fairness Metrics Computation for Toxicity Classification
Computes group-wise fairness metrics from predictions and identity attributes.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


def compute_group_fairness(
    df: pd.DataFrame,
    group_cols: List[str],
    label_col: str = "label",
    pred_col: str = "pred",
    score_col: str = "pos_prob",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute fairness metrics across identity groups.
    
    For each group column, computes:
    - Positive prediction rate (Demographic Parity)
    - True Positive Rate (Equal Opportunity)
    - False Positive Rate (Equalized Odds component)
    
    Then computes fairness gaps:
    - DP difference: max - min positive prediction rates
    - EO difference: max - min TPRs
    - EOdds difference: max of (TPR gap, FPR gap)
    
    Args:
        df: DataFrame with predictions and group membership indicators
        group_cols: List of binary group indicator column names (0/1)
        label_col: Name of true label column
        pred_col: Name of predicted label column
        score_col: Name of score/probability column (optional, for future use)
    
    Returns:
        (summary_df, per_group_df):
            summary_df: One row per group_col with fairness gaps
            per_group_df: One row per (group_col, value) with metrics
    """
    per_group_rows = []
    summary_rows = []
    
    for group_col in group_cols:
        if group_col not in df.columns:
            print(f"[WARN] Group column '{group_col}' not found in data, skipping")
            continue
        
        # Collect metrics for each group value (0 and 1)
        group_metrics = {}
        
        for group_val in [0, 1]:
            subset = df[df[group_col] == group_val]
            
            if len(subset) == 0:
                # No data for this group
                group_metrics[group_val] = {
                    "support": 0,
                    "pos_rate": np.nan,
                    "tpr": np.nan,
                    "fpr": np.nan,
                }
                continue
            
            support = len(subset)
            
            # Positive prediction rate (for Demographic Parity)
            pos_rate = subset[pred_col].mean()
            
            # TPR: P(pred=1 | label=1, group=group_val)
            positives = subset[subset[label_col] == 1]
            if len(positives) > 0:
                tpr = positives[pred_col].mean()
            else:
                tpr = np.nan
            
            # FPR: P(pred=1 | label=0, group=group_val)
            negatives = subset[subset[label_col] == 0]
            if len(negatives) > 0:
                fpr = negatives[pred_col].mean()
            else:
                fpr = np.nan
            
            group_metrics[group_val] = {
                "support": support,
                "pos_rate": pos_rate,
                "tpr": tpr,
                "fpr": fpr,
            }
            
            # Add to per-group results
            per_group_rows.append({
                "group_col": group_col,
                "group_val": group_val,
                "support": support,
                "pos_rate": pos_rate,
                "tpr": tpr,
                "fpr": fpr,
            })
        
        # Compute fairness gaps for this group column
        pos_rates = [m["pos_rate"] for m in group_metrics.values() if not np.isnan(m["pos_rate"])]
        tprs = [m["tpr"] for m in group_metrics.values() if not np.isnan(m["tpr"])]
        fprs = [m["fpr"] for m in group_metrics.values() if not np.isnan(m["fpr"])]
        
        # Demographic Parity difference
        if len(pos_rates) >= 2:
            dp_diff = max(pos_rates) - min(pos_rates)
        else:
            dp_diff = np.nan
        
        # Equal Opportunity difference (TPR gap)
        if len(tprs) >= 2:
            eop_diff = max(tprs) - min(tprs)
        else:
            eop_diff = np.nan
        
        # Equalized Odds difference (max of TPR gap and FPR gap)
        tpr_gap = eop_diff if not np.isnan(eop_diff) else 0.0
        if len(fprs) >= 2:
            fpr_gap = max(fprs) - min(fprs)
        else:
            fpr_gap = 0.0
        
        if not np.isnan(tpr_gap) or fpr_gap > 0:
            eo_diff = max(tpr_gap if not np.isnan(tpr_gap) else 0.0, fpr_gap)
        else:
            eo_diff = np.nan
        
        summary_rows.append({
            "group_col": group_col,
            "dp_diff": dp_diff,
            "eop_diff": eop_diff,
            "eo_diff": eo_diff,
        })
    
    per_group_df = pd.DataFrame(per_group_rows)
    summary_df = pd.DataFrame(summary_rows)
    
    return summary_df, per_group_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute fairness metrics from predictions and group attributes"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., jigsaw, civil)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Data split (e.g., test, val)",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="Path to predictions CSV (must have 'id', 'label', 'pred' columns)",
    )
    parser.add_argument(
        "--full_data_file",
        type=str,
        required=True,
        help="Path to full data CSV with group attributes (must have 'id' and 'g_*' columns)",
    )
    parser.add_argument(
        "--group_prefix",
        type=str,
        default="g_",
        help="Prefix for group indicator columns",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        required=True,
        help="Output file prefix for saving results",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Computing Fairness Metrics")
    print(f"Dataset: {args.dataset} | Split: {args.split}")
    print(f"{'='*80}\n")
    
    # Load predictions
    print(f">>> Loading predictions from: {args.pred_file}")
    pred_df = pd.read_csv(args.pred_file)
    
    required_cols = ["id", "label", "pred"]
    missing = [c for c in required_cols if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")
    
    print(f"    Loaded {len(pred_df)} predictions")
    
    # Load full data with group attributes
    print(f">>> Loading full data from: {args.full_data_file}")
    full_df = pd.read_csv(args.full_data_file)
    
    if "id" not in full_df.columns:
        raise ValueError("Full data file must have 'id' column")
    
    print(f"    Loaded {len(full_df)} records")
    
    # Find group columns
    group_cols = [c for c in full_df.columns if c.startswith(args.group_prefix)]
    
    if not group_cols:
        raise ValueError(f"No group columns found with prefix '{args.group_prefix}'")
    
    print(f"    Found {len(group_cols)} group columns: {', '.join(group_cols[:5])}{'...' if len(group_cols) > 5 else ''}")
    
    # Merge predictions with group attributes
    print(f"\n>>> Merging predictions with group attributes on 'id'")
    merged_df = pred_df.merge(full_df, on="id", how="inner", suffixes=("", "_full"))
    
    # Handle duplicate label column from merge
    if "label_full" in merged_df.columns:
        merged_df = merged_df.drop(columns=["label_full"])
    
    print(f"    Merged data has {len(merged_df)} records")
    
    if len(merged_df) == 0:
        raise ValueError("No matching records found after merge. Check that 'id' columns match.")
    
    # Compute fairness metrics
    print(f"\n>>> Computing fairness metrics across {len(group_cols)} groups")
    summary_df, per_group_df = compute_group_fairness(
        merged_df,
        group_cols=group_cols,
        label_col="label",
        pred_col="pred",
        score_col="pos_prob" if "pos_prob" in merged_df.columns else "score",
    )
    
    # Save results
    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = f"{args.out_prefix}_summary.csv"
    per_group_path = f"{args.out_prefix}_per_group.csv"
    
    summary_df.to_csv(summary_path, index=False)
    per_group_df.to_csv(per_group_path, index=False)
    
    print(f"\n>>> Saved fairness summary to: {summary_path}")
    print(f">>> Saved per-group metrics to: {per_group_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("FAIRNESS METRICS SUMMARY")
    print(f"{'='*80}\n")
    
    print("Top 10 groups by Demographic Parity difference:")
    print(summary_df.nlargest(10, "dp_diff")[["group_col", "dp_diff", "eop_diff", "eo_diff"]].to_string(index=False))
    
    print("\n" + "="*80)
    print("Top 10 groups by Equal Opportunity difference:")
    print(summary_df.nlargest(10, "eop_diff")[["group_col", "dp_diff", "eop_diff", "eo_diff"]].to_string(index=False))
    
    print("\n" + "="*80)
    print("Top 10 groups by Equalized Odds difference:")
    print(summary_df.nlargest(10, "eo_diff")[["group_col", "dp_diff", "eop_diff", "eo_diff"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()