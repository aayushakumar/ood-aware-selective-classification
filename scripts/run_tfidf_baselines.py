#!/usr/bin/env python3
"""
TF-IDF Baseline Models for Toxicity Classification
Implements Logistic Regression and Linear SVM baselines with cross-domain evaluation.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, log_loss, brier_score_loss
)

# FIXED: Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Try importing from run_roberta, fallback to inline definitions
try:
    from run_roberta import SUPPORTED_DATASETS, load_dataset, set_seed
except ImportError:
    print("[WARN] Could not import from run_roberta.py, using inline definitions")
    
    # Inline definitions as fallback
    SUPPORTED_DATASETS = ["jigsaw", "civil", "hatexplain"]
    
    def set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
    
    def load_dataset(name: str, split: str, data_dir: str = "data") -> pd.DataFrame:
        """Load dataset from CSV."""
        filepath = Path(data_dir) / f"{name}_{split}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        df = pd.read_csv(filepath)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Dataset must have 'text' and 'label' columns")
        return df[["text", "label"]]

EXPERIMENTS_DIR = "experiments"

def evaluate_sklearn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model,
    model_type: str = "logreg"
) -> Dict:
    """
    Evaluate a scikit-learn classifier and return metrics.
    
    Args:
        X: Feature matrix
        y: True labels
        model: Fitted sklearn classifier
        model_type: "logreg" or "svm"
    
    Returns:
        Dictionary of metrics
    """
    predictions = model.predict(X)
    
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "f1": f1_score(y, predictions, average="binary", zero_division=0),
    }
    
    # Probabilistic metrics (only for logistic regression)
    if model_type == "logreg" and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            pos_probs = probs[:, 1]
            
            # AUROC
            try:
                metrics["auroc"] = roc_auc_score(y, pos_probs)
            except ValueError:
                metrics["auroc"] = float("nan")
            
            # PR-AUC
            try:
                metrics["pr_auc"] = average_precision_score(y, pos_probs)
            except ValueError:
                metrics["pr_auc"] = float("nan")
            
            # NLL
            try:
                metrics["nll"] = log_loss(y, probs, labels=[0, 1])
            except ValueError:
                metrics["nll"] = float("nan")
            
            # Brier score
            try:
                metrics["brier"] = brier_score_loss(y, pos_probs)
            except ValueError:
                metrics["brier"] = float("nan")
        except Exception as e:
            print(f"[WARN] Error computing probabilistic metrics: {e}")
            metrics.update({"auroc": float("nan"), "pr_auc": float("nan"), 
                          "nll": float("nan"), "brier": float("nan")})
    else:
        # For SVM, try decision_function for AUROC/PR-AUC
        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)
                try:
                    metrics["auroc"] = roc_auc_score(y, scores)
                except ValueError:
                    metrics["auroc"] = float("nan")
                try:
                    metrics["pr_auc"] = average_precision_score(y, scores)
                except ValueError:
                    metrics["pr_auc"] = float("nan")
            except Exception:
                metrics["auroc"] = float("nan")
                metrics["pr_auc"] = float("nan")
        else:
            metrics["auroc"] = float("nan")
            metrics["pr_auc"] = float("nan")
        
        metrics["nll"] = float("nan")
        metrics["brier"] = float("nan")
    
    return metrics


def train_and_evaluate_tfidf(
    source_dataset: str,
    target_datasets: List[str],
    model_type: str,
    seed: int,
    data_dir: str = "data",
    ngram_max: int = 2,
    min_df: int = 5,
    max_features: Optional[int] = None,
    save_preds: bool = False,
) -> Dict:
    """
    Train TF-IDF baseline and evaluate cross-domain.
    
    Args:
        source_dataset: Source dataset name
        target_datasets: List of target dataset names
        model_type: "logreg" or "svm"
        seed: Random seed
        data_dir: Data directory
        ngram_max: Maximum n-gram range (1, ngram_max)
        min_df: Minimum document frequency
        max_features: Maximum number of features (None for unlimited)
        save_preds: Whether to save predictions to CSV
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"TF-IDF + {model_type.upper()} | seed={seed} | source={source_dataset}")
    print(f"{'='*80}\n")
    
    set_seed(seed)
    
    # Load source data
    print(f">>> Loading source dataset: {source_dataset}")
    train_df = load_dataset(source_dataset, "train", data_dir)
    val_df = load_dataset(source_dataset, "val", data_dir)
    test_df = load_dataset(source_dataset, "test", data_dir)
    
    print(f"    Train: {len(train_df)} samples")
    print(f"    Val: {len(val_df)} samples")
    print(f"    Test: {len(test_df)} samples")
    
    # Initialize TF-IDF vectorizer
    print(f"\n>>> Fitting TF-IDF vectorizer (ngram=(1,{ngram_max}), min_df={min_df})")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        token_pattern=r"\b\w+\b",
        stop_words=None,
    )
    
    # Fit on train text only
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"].values
    
    X_val = vectorizer.transform(val_df["text"])
    y_val = val_df["label"].values
    
    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"].values
    
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"    Train matrix shape: {X_train.shape}")
    
    # Train classifier
    print(f"\n>>> Training {model_type.upper()} classifier")
    if model_type == "logreg":
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "svm":
        model = LinearSVC(
            max_iter=5000,
            random_state=seed,
            dual=False,  # Recommended when n_samples > n_features
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X_train, y_train)
    print(f"    Training complete")
    
    # Evaluate on validation
    print(f"\n>>> Evaluating on validation set")
    val_metrics = evaluate_sklearn_classifier(X_val, y_val, model, model_type)
    print(f"    Val - F1: {val_metrics['f1']:.4f}, ACC: {val_metrics['accuracy']:.4f}, "
          f"AUROC: {val_metrics.get('auroc', float('nan')):.4f}")
    
    # Evaluate on in-domain test
    print(f"\n>>> Evaluating on test set: {source_dataset}")
    test_metrics = evaluate_sklearn_classifier(X_test, y_test, model, model_type)
    print(f"    Test - F1: {test_metrics['f1']:.4f}, ACC: {test_metrics['accuracy']:.4f}, "
          f"AUROC: {test_metrics.get('auroc', float('nan')):.4f}")
    
    # Save in-domain test predictions if requested
    if save_preds:
        pred_data = {
            "id": list(range(len(test_df))),
            "text": test_df["text"].values,
            "label": y_test,
            "pred": model.predict(X_test),
        }
        
        if model_type == "logreg":
            pred_data["score"] = model.predict_proba(X_test)[:, 1]
        else:
            pred_data["score"] = model.decision_function(X_test)
        
        pred_df = pd.DataFrame(pred_data)
        Path(EXPERIMENTS_DIR).mkdir(exist_ok=True)
        pred_path = Path(EXPERIMENTS_DIR) / f"preds_tfidf_{model_type}_{source_dataset}_test.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")
        print(f"    Saved predictions to: {pred_path}")
    
    # Cross-domain evaluation
    cross_results = {}
    for target in target_datasets:
        print(f"\n>>> Cross-domain evaluation: {source_dataset} → {target}")
        try:
            target_test_df = load_dataset(target, "test", data_dir)
            X_target = vectorizer.transform(target_test_df["text"])
            y_target = target_test_df["label"].values
            
            target_metrics = evaluate_sklearn_classifier(X_target, y_target, model, model_type)
            cross_results[target] = target_metrics
            
            print(f"    {target} - F1: {target_metrics['f1']:.4f}, "
                  f"ACC: {target_metrics['accuracy']:.4f}, "
                  f"AUROC: {target_metrics.get('auroc', float('nan')):.4f}")
            
            # Save cross-domain predictions
            if save_preds:
                pred_data = {
                    "id": list(range(len(target_test_df))),
                    "text": target_test_df["text"].values,
                    "label": y_target,
                    "pred": model.predict(X_target),
                }
                
                if model_type == "logreg":
                    pred_data["score"] = model.predict_proba(X_target)[:, 1]
                else:
                    pred_data["score"] = model.decision_function(X_target)
                
                pred_df = pd.DataFrame(pred_data)
                pred_path = Path(EXPERIMENTS_DIR) / f"preds_tfidf_{model_type}_{source_dataset}_to_{target}.csv"
                pred_df.to_csv(pred_path, index=False, encoding="utf-8")
                print(f"    Saved predictions to: {pred_path}")
        
        except Exception as e:
            print(f"    [ERROR] Failed to evaluate {target}: {e}")
            cross_results[target] = {"error": str(e)}
    
    # Create summary DataFrame
    summary_rows = []
    
    # In-domain validation
    summary_rows.append({"split": "in_domain_val", **val_metrics})
    
    # In-domain test
    summary_rows.append({"split": "in_domain_test", **test_metrics})
    
    # Cross-domain results
    for target, metrics in cross_results.items():
        if "error" not in metrics:
            summary_rows.append({"split": f"cross_{target}", **metrics})
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    Path(EXPERIMENTS_DIR).mkdir(exist_ok=True)
    summary_path = Path(EXPERIMENTS_DIR) / f"summary_tfidf_{source_dataset}_{model_type}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n>>> Saved summary to: {summary_path}")
    
    return {
        "seed": seed,
        "source": source_dataset,
        "model_type": model_type,
        "in_domain_val": val_metrics,
        "in_domain_test": test_metrics,
        "cross_domain": cross_results,
    }


def run_multi_seed_tfidf(
    seeds: List[int],
    source_dataset: str,
    target_datasets: List[str],
    model_type: str,
    data_dir: str = "data",
    ngram_max: int = 2,
    min_df: int = 5,
    max_features: Optional[int] = None,
    save_preds: bool = False,
) -> List[Dict]:
    """Run TF-IDF baseline for multiple seeds and aggregate results."""
    all_results = []
    
    for seed in seeds:
        result = train_and_evaluate_tfidf(
            source_dataset=source_dataset,
            target_datasets=target_datasets,
            model_type=model_type,
            seed=seed,
            data_dir=data_dir,
            ngram_max=ngram_max,
            min_df=min_df,
            max_features=max_features,
            save_preds=save_preds,
        )
        all_results.append(result)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"MULTI-SEED SUMMARY: TF-IDF + {model_type.upper()}")
    print(f"{'='*80}\n")
    
    print(f"{'Seed':<10} {'Test F1':<15} {'Test ACC':<15}", end="")
    for target in target_datasets:
        print(f" {target} F1", end="")
    print()
    print("-" * 80)
    
    for result in all_results:
        seed = result["seed"]
        test_f1 = result["in_domain_test"]["f1"]
        test_acc = result["in_domain_test"]["accuracy"]
        print(f"{seed:<10} {test_f1:<15.4f} {test_acc:<15.4f}", end="")
        
        for target in target_datasets:
            if target in result["cross_domain"] and "f1" in result["cross_domain"][target]:
                cross_f1 = result["cross_domain"][target]["f1"]
                print(f" {cross_f1:.4f}", end="")
            else:
                print(f" N/A", end="")
        print()
    
    # Compute averages
    avg_test_f1 = np.mean([r["in_domain_test"]["f1"] for r in all_results])
    avg_test_acc = np.mean([r["in_domain_test"]["accuracy"] for r in all_results])
    print("-" * 80)
    print(f"{'AVERAGE':<10} {avg_test_f1:<15.4f} {avg_test_acc:<15.4f}", end="")
    
    for target in target_datasets:
        target_f1s = [
            r["cross_domain"][target]["f1"]
            for r in all_results
            if target in r["cross_domain"] and "f1" in r["cross_domain"][target]
        ]
        if target_f1s:
            print(f" {np.mean(target_f1s):.4f}", end="")
        else:
            print(f" N/A", end="")
    print("\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="TF-IDF baseline models for toxicity classification"
    )
    
    parser.add_argument(
        "--source_dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Source dataset for training",
    )
    parser.add_argument(
        "--target_datasets",
        type=str,
        nargs="*",
        default=[],
        help="Target datasets for cross-domain evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logreg", "svm", "both"],
        default="logreg",
        help="Model type to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used if --seeds not provided)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        help="Multiple seeds for multi-seed runs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Maximum n-gram range (1 to ngram_max)",
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=5,
        help="Minimum document frequency for TF-IDF",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Maximum number of TF-IDF features (None for unlimited)",
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        help="Save predictions to CSV files",
    )
    
    args = parser.parse_args()
    
    seeds = args.seeds if args.seeds else [args.seed]
    model_types = ["logreg", "svm"] if args.model == "both" else [args.model]
    
    print(f"\nTF-IDF Baseline Models")
    print(f"Source: {args.source_dataset}")
    print(f"Targets: {args.target_datasets if args.target_datasets else 'None'}")
    print(f"Models: {model_types}")
    print(f"Seeds: {seeds}\n")
    
    for model_type in model_types:
        if len(seeds) > 1:
            run_multi_seed_tfidf(
                seeds=seeds,
                source_dataset=args.source_dataset,
                target_datasets=args.target_datasets,
                model_type=model_type,
                data_dir=args.data_dir,
                ngram_max=args.ngram_max,
                min_df=args.min_df,
                max_features=args.max_features,
                save_preds=args.save_preds,
            )
        else:
            train_and_evaluate_tfidf(
                source_dataset=args.source_dataset,
                target_datasets=args.target_datasets,
                model_type=model_type,
                seed=seeds[0],
                data_dir=args.data_dir,
                ngram_max=args.ngram_max,
                min_df=args.min_df,
                max_features=args.max_features,
                save_preds=args.save_preds,
            )


if __name__ == "__main__":
    main()