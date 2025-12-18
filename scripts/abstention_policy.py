#!/usr/bin/env python3
"""
OOD-Aware Fairness-Constrained Abstention Policy
==================================================

Implements selective classification with:
1. OOD-aware abstention (adapts based on OOD reliability)
2. Fairness-constrained threshold optimization
3. Multiple baseline methods (confidence-only, conformal, ensemble)

Key components:
- OODAwareAbstentionPolicy: Decides abstain/classify per sample
- FairnessConstrainedThresholdSolver: Finds thresholds satisfying fairness
- SelectiveClassificationEvaluator: Computes coverage, accuracy, fairness
- ConformalSelectiveClassifier: Principled baseline with coverage guarantees

Usage:
    python abstention_policy.py \
        --predictions experiments/preds_jigsaw_to_civil.csv \
        --fairness_epsilon 0.05 \
        --min_coverage 0.80 \
        --output_dir experiments/selective
"""

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AbstentionResult:
    """Result of applying abstention policy to a dataset."""
    abstain_mask: np.ndarray  # bool array, True = abstain
    predictions: np.ndarray   # predictions (original, for non-abstained)
    coverage: float           # 1 - abstention_rate
    accuracy_on_covered: float
    f1_on_covered: float
    risk_on_covered: float    # error rate on covered
    fairness_metrics: Dict[str, float]
    thresholds_used: Dict[str, float]


@dataclass  
class FairnessMetrics:
    """Fairness metrics computed on predictions."""
    fpr_gap: float            # max - min FPR across groups
    tpr_gap: float            # max - min TPR across groups  
    abstention_gap: float     # max - min abstention rate across groups
    per_group_fpr: Dict[str, float] = field(default_factory=dict)
    per_group_tpr: Dict[str, float] = field(default_factory=dict)
    per_group_abstention: Dict[str, float] = field(default_factory=dict)
    fpr_gap_ci: Optional[Tuple[float, float]] = None  # 95% CI
    tpr_gap_ci: Optional[Tuple[float, float]] = None


# =============================================================================
# MINIMUM SUPPORT REQUIREMENTS
# =============================================================================

MIN_NEGATIVES_FOR_FPR = 100  # Require ≥100 negatives to compute FPR
MIN_POSITIVES_FOR_TPR = 50   # Require ≥50 positives to compute TPR
MIN_GROUP_SIZE = 50          # Minimum group size for any metric


# =============================================================================
# CORE ABSTENTION POLICIES
# =============================================================================

class OODAwareAbstentionPolicy:
    """
    Abstention policy that adapts based on OOD reliability.
    
    Single gating rule: use_ood = separability > τ_sep
    
    If OOD is reliable:
        abstain if (confidence < τ_conf) OR (ood_score > τ_ood)
    If OOD is unreliable:
        abstain if (confidence < τ_conf_fallback) only
    
    This prevents using inverted or non-separable OOD signals.
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.7,
        ood_threshold: float = 0.0,
        conf_threshold_fallback: float = 0.6,
        separability_threshold: float = 0.6,  # Single threshold (τ_sep)
        use_ood: bool = True
    ):
        """
        Args:
            conf_threshold: Confidence threshold when OOD is reliable
            ood_threshold: OOD score threshold (after correction)
            conf_threshold_fallback: Confidence threshold when OOD is unreliable
            separability_threshold: Single threshold for OOD reliability (τ_sep)
            use_ood: Whether to use OOD signals at all
        """
        self.conf_threshold = conf_threshold
        self.ood_threshold = ood_threshold
        self.conf_threshold_fallback = conf_threshold_fallback
        self.separability_threshold = separability_threshold
        self.use_ood = use_ood
    
    def compute_abstention(
        self,
        confidences: np.ndarray,
        ood_scores: np.ndarray,
        odm_result: Dict[str, Any],
        ood_mean: Optional[float] = None,
        ood_std: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute abstention mask for all samples.
        
        Args:
            confidences: Model confidence scores (max prob)
            ood_scores: OOD scores (higher = more likely OOD)
            odm_result: Result from OODDetectabilityMetric.compute()
                        Must contain 'source_mean' and 'source_std' if ood_mean/ood_std not passed
            ood_mean: Pre-computed mean for standardization (from source)
            ood_std: Pre-computed std for standardization (from source)
        
        Returns:
            Boolean array, True = abstain
            
        Raises:
            ValueError: If OOD is reliable but no standardization stats available
        """
        n = len(confidences)
        abstain = np.zeros(n, dtype=bool)
        
        # Single gating rule: use_ood = separability > τ_sep
        separability = odm_result.get('separability', 0)
        ood_reliable = self.use_ood and (separability > self.separability_threshold)
        
        if ood_reliable:
            # Correct OOD scores if inverted
            direction = odm_result.get('direction', 'normal')
            corrected_ood = self._correct_ood(ood_scores, direction)
            
            # Get standardization stats - MUST come from source, never from test
            if ood_mean is None:
                ood_mean = odm_result.get('source_mean')
            if ood_std is None:
                ood_std = odm_result.get('source_std')
            
            # Enforce no-leakage: stats MUST be provided
            if ood_mean is None or ood_std is None:
                raise ValueError(
                    "OOD is reliable but source_mean/source_std not provided. "
                    "Pass ood_mean/ood_std or ensure ODM returns source_mean/source_std. "
                    "This prevents test-time leakage."
                )
            
            corrected_ood = (corrected_ood - ood_mean) / (ood_std + 1e-8)
            
            # Abstain if low confidence OR high OOD score
            abstain = (confidences < self.conf_threshold) | (corrected_ood > self.ood_threshold)
        else:
            # OOD unreliable - use confidence only with fallback threshold
            abstain = confidences < self.conf_threshold_fallback
        
        return abstain
    
    def _correct_ood(self, scores: np.ndarray, direction: str) -> np.ndarray:
        """Flip OOD scores if direction is inverted."""
        if direction == 'inverted':
            return -scores
        return scores.copy()
    
    def should_abstain_single(
        self,
        confidence: float,
        ood_score: float,
        odm_result: Dict[str, Any]
    ) -> bool:
        """Decide whether to abstain on a single sample."""
        result = self.compute_abstention(
            np.array([confidence]),
            np.array([ood_score]),
            odm_result
        )
        return bool(result[0])


class ConfidenceOnlyPolicy:
    """Simple baseline: abstain based on confidence threshold only."""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def compute_abstention(
        self,
        confidences: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        return confidences < self.threshold


class NaiveOODPolicy:
    """Naive OOD baseline: abstain based on OOD score only (ignores inversion)."""
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
    
    def compute_abstention(
        self,
        ood_scores: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # Standardize scores
        std_scores = (ood_scores - ood_scores.mean()) / (ood_scores.std() + 1e-8)
        return std_scores > self.threshold


class EnsembleVariancePolicy:
    """
    Ensemble uncertainty baseline (MC Dropout style).
    
    Uses variance across multiple predictions as uncertainty measure.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def compute_abstention(
        self,
        ensemble_variances: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        return ensemble_variances > self.threshold


class ConformalSelectiveClassifier:
    """
    Conformal prediction for classification with abstention.
    
    Produces prediction sets; abstain if set size > 1 (ambiguous).
    Uses split conformal with calibration set.
    
    Reference: Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Target miscoverage rate (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.qhat = None
        self.fitted = False
    
    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray):
        """
        Fit quantile threshold on calibration set.
        
        Args:
            cal_probs: Predicted probabilities (N, num_classes)
            cal_labels: True labels (N,)
        """
        # Conformity scores: 1 - p(true class)
        n = len(cal_labels)
        if cal_probs.ndim == 1:
            # Binary case: cal_probs is P(class=1)
            true_probs = np.where(cal_labels == 1, cal_probs, 1 - cal_probs)
        else:
            true_probs = cal_probs[np.arange(n), cal_labels]
        
        scores = 1 - true_probs
        
        # Find quantile (with finite sample correction)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(scores, min(q_level, 1.0))
        self.fitted = True
    
    def predict(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with abstention for ambiguous cases.
        
        Args:
            probs: Predicted probabilities (N,) for binary or (N, num_classes)
        
        Returns:
            predictions: -1 if abstain (set size > 1), else predicted class
            set_sizes: size of prediction set per sample
        """
        if not self.fitted:
            raise RuntimeError("Must call calibrate() before predict()")
        
        if probs.ndim == 1:
            # Binary case: convert to (N, 2)
            probs = np.stack([1 - probs, probs], axis=1)
        
        # Prediction set = classes with p >= 1 - qhat
        sets = probs >= (1 - self.qhat)
        set_sizes = sets.sum(axis=1)
        
        # Abstain if set size != 1
        predictions = np.where(
            set_sizes == 1,
            sets.argmax(axis=1),  # single prediction
            -1  # abstain
        )
        
        return predictions, set_sizes
    
    def compute_abstention(self, probs: np.ndarray, **kwargs) -> np.ndarray:
        """Return abstention mask (True = abstain)."""
        predictions, set_sizes = self.predict(probs)
        return predictions == -1


# =============================================================================
# FAIRNESS METRICS COMPUTATION
# =============================================================================

def compute_group_fpr(
    labels: np.ndarray,
    predictions: np.ndarray,
    group_mask: np.ndarray
) -> Optional[float]:
    """
    Compute FPR for a single group.
    
    Returns None if insufficient support (< MIN_NEGATIVES_FOR_FPR negatives).
    """
    negatives = (labels == 0) & group_mask
    if negatives.sum() < MIN_NEGATIVES_FOR_FPR:
        return None
    return predictions[negatives].mean()


def compute_group_tpr(
    labels: np.ndarray,
    predictions: np.ndarray,
    group_mask: np.ndarray
) -> Optional[float]:
    """
    Compute TPR for a single group.
    
    Returns None if insufficient support (< MIN_POSITIVES_FOR_TPR positives).
    """
    positives = (labels == 1) & group_mask
    if positives.sum() < MIN_POSITIVES_FOR_TPR:
        return None
    return predictions[positives].mean()


def compute_fairness_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    abstain_mask: np.ndarray,
    group_masks: Dict[str, np.ndarray]
) -> FairnessMetrics:
    """
    Compute all fairness metrics across groups.
    
    Args:
        labels: True labels
        predictions: Model predictions
        abstain_mask: Boolean mask where True = abstained
        group_masks: Dict mapping group name to boolean mask
    
    Returns:
        FairnessMetrics dataclass
    """
    # Filter to covered samples for FPR/TPR
    covered = ~abstain_mask
    covered_labels = labels[covered]
    covered_preds = predictions[covered]
    
    per_group_fpr = {}
    per_group_tpr = {}
    per_group_abstention = {}
    
    for group_name, mask in group_masks.items():
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        
        # FPR on covered samples
        covered_mask = mask[covered]
        fpr = compute_group_fpr(covered_labels, covered_preds, covered_mask)
        if fpr is not None:
            per_group_fpr[group_name] = fpr
        
        # TPR on covered samples
        tpr = compute_group_tpr(covered_labels, covered_preds, covered_mask)
        if tpr is not None:
            per_group_tpr[group_name] = tpr
        
        # Abstention rate on full data
        abstention_rate = abstain_mask[mask].mean()
        per_group_abstention[group_name] = abstention_rate
    
    # Compute gaps
    fpr_values = list(per_group_fpr.values())
    tpr_values = list(per_group_tpr.values())
    abstention_values = list(per_group_abstention.values())
    
    fpr_gap = (max(fpr_values) - min(fpr_values)) if len(fpr_values) >= 2 else 0.0
    tpr_gap = (max(tpr_values) - min(tpr_values)) if len(tpr_values) >= 2 else 0.0
    abstention_gap = (max(abstention_values) - min(abstention_values)) if len(abstention_values) >= 2 else 0.0
    
    return FairnessMetrics(
        fpr_gap=fpr_gap,
        tpr_gap=tpr_gap,
        abstention_gap=abstention_gap,
        per_group_fpr=per_group_fpr,
        per_group_tpr=per_group_tpr,
        per_group_abstention=per_group_abstention
    )


def bootstrap_fairness_gap(
    labels: np.ndarray,
    predictions: np.ndarray,
    abstain_mask: np.ndarray,
    group_masks: Dict[str, np.ndarray],
    metric: str = 'fpr_gap',
    n_bootstrap: int = 1000
) -> Dict[str, float]:
    """
    Compute fairness gap with bootstrap 95% CI.
    
    Args:
        labels, predictions, abstain_mask, group_masks: As in compute_fairness_metrics
        metric: 'fpr_gap', 'tpr_gap', or 'abstention_gap'
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dict with 'mean', 'ci_low', 'ci_high'
    """
    n = len(labels)
    gaps = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        
        resampled_labels = labels[idx]
        resampled_preds = predictions[idx]
        resampled_abstain = abstain_mask[idx]
        resampled_groups = {g: m[idx] for g, m in group_masks.items()}
        
        metrics = compute_fairness_metrics(
            resampled_labels, resampled_preds, resampled_abstain, resampled_groups
        )
        gaps.append(getattr(metrics, metric))
    
    return {
        'mean': float(np.mean(gaps)),
        'ci_low': float(np.percentile(gaps, 2.5)),
        'ci_high': float(np.percentile(gaps, 97.5))
    }


# =============================================================================
# THRESHOLD SOLVER
# =============================================================================

class FairnessConstrainedThresholdSolver:
    """
    Find abstention thresholds that:
    1. Maximize accuracy on non-abstained samples
    2. Subject to: max_g FPR_gap(g) ≤ ε_fpr
    3. Subject to: abstention_gap ≤ ε_abstain (anti-burden-shifting)
    4. Subject to: coverage ≥ min_coverage
    5. Subject to: min_group_coverage ≥ min_group_cov (no group left behind)
    
    Uses grid search over threshold combinations.
    """
    
    def __init__(
        self,
        fairness_epsilon: float = 0.10,       # FPR gap constraint
        abstention_gap_epsilon: float = 0.10,  # Abstention gap constraint
        min_coverage: float = 0.70,
        min_group_coverage: float = 0.60,      # Min coverage per group
        separability_threshold: float = 0.6    # Single τ_sep
    ):
        """
        Args:
            fairness_epsilon: Maximum allowed FPR gap
            abstention_gap_epsilon: Maximum allowed abstention gap (anti-burden-shifting)
            min_coverage: Minimum fraction of samples that must be classified
            min_group_coverage: Minimum coverage for any single group
            separability_threshold: Threshold for OOD reliability
        """
        self.fairness_epsilon = fairness_epsilon
        self.abstention_gap_epsilon = abstention_gap_epsilon
        self.min_coverage = min_coverage
        self.min_group_coverage = min_group_coverage
        self.separability_threshold = separability_threshold
    
    def solve(
        self,
        confidences: np.ndarray,
        ood_scores: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        group_masks: Dict[str, np.ndarray],
        odm_result: Dict[str, Any],
        conf_grid: np.ndarray = None,
        ood_grid: np.ndarray = None,
        use_ood: bool = True,
        source_ood_mean: Optional[float] = None,
        source_ood_std: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Grid search for optimal thresholds satisfying ALL constraints.
        
        Args:
            confidences: Model confidence scores
            ood_scores: OOD scores
            labels: True labels (DEV ONLY - never use test labels)
            predictions: Model predictions (before abstention)
            group_masks: Dict mapping group name to boolean mask
            odm_result: Result from OODDetectabilityMetric
            conf_grid: Confidence threshold grid to search
            ood_grid: OOD threshold grid to search
            use_ood: Whether to use OOD in the policy
            source_ood_mean: Mean of source OOD scores (for standardization)
            source_ood_std: Std of source OOD scores (for standardization)
        
        Returns:
            Dict with optimal thresholds and achieved metrics
        """
        if conf_grid is None:
            conf_grid = np.linspace(0.5, 0.95, 20)
        if ood_grid is None:
            ood_grid = np.linspace(-2, 2, 20)
        
        # Correct OOD scores if needed
        direction = odm_result.get('direction', 'normal')
        if direction == 'inverted':
            ood_scores_corrected = -ood_scores
        else:
            ood_scores_corrected = ood_scores.copy()
        
        # Standardize using SOURCE stats (NO LEAKAGE!)
        if source_ood_mean is None:
            source_ood_mean = odm_result.get('source_mean', ood_scores_corrected.mean())
        if source_ood_std is None:
            source_ood_std = odm_result.get('source_std', ood_scores_corrected.std())
        
        ood_scores_std = (ood_scores_corrected - source_ood_mean) / (source_ood_std + 1e-8)
        
        best_result = None
        best_accuracy = -1
        
        # Single gating rule: separability > τ_sep
        separability = odm_result.get('separability', 0)
        ood_reliable = use_ood and (separability > self.separability_threshold)
        
        for conf_t in conf_grid:
            if ood_reliable:
                ood_grid_to_search = ood_grid
            else:
                # If OOD unreliable, only search confidence thresholds
                ood_grid_to_search = [None]
            
            for ood_t in ood_grid_to_search:
                # Compute abstention
                if ood_t is None:
                    abstain = confidences < conf_t
                else:
                    abstain = (confidences < conf_t) | (ood_scores_std > ood_t)
                
                # === CHECK ALL CONSTRAINTS ===
                
                # 1. Overall coverage constraint
                coverage = 1 - abstain.mean()
                if coverage < self.min_coverage:
                    continue
                
                # Compute fairness metrics
                fairness = compute_fairness_metrics(labels, predictions, abstain, group_masks)
                
                # 2. FPR gap constraint
                if fairness.fpr_gap > self.fairness_epsilon:
                    continue
                
                # 3. Abstention gap constraint (anti-burden-shifting)
                if fairness.abstention_gap > self.abstention_gap_epsilon:
                    continue
                
                # 4. Min group coverage constraint
                per_group_coverage = {
                    g: 1 - abstain[m].mean() for g, m in group_masks.items() if m.sum() > 0
                }
                min_cov = min(per_group_coverage.values()) if per_group_coverage else 0
                if min_cov < self.min_group_coverage:
                    continue
                
                # === ALL CONSTRAINTS SATISFIED ===
                
                # Compute accuracy on covered
                covered = ~abstain
                if covered.sum() == 0:
                    continue
                accuracy = accuracy_score(labels[covered], predictions[covered])
                
                # Track best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_result = {
                        'conf_threshold': float(conf_t),
                        'ood_threshold': float(ood_t) if ood_t is not None else None,
                        'achieved_coverage': float(coverage),
                        'achieved_accuracy': float(accuracy),
                        'achieved_f1': float(f1_score(labels[covered], predictions[covered])),
                        'achieved_fpr_gap': float(fairness.fpr_gap),
                        'achieved_tpr_gap': float(fairness.tpr_gap),
                        'achieved_abstention_gap': float(fairness.abstention_gap),
                        'achieved_min_group_coverage': float(min_cov),
                        'per_group_coverage': per_group_coverage,
                        'ood_reliable': ood_reliable,
                        'ood_direction': direction,
                        'source_ood_mean': float(source_ood_mean),
                        'source_ood_std': float(source_ood_std),
                        'constraints_satisfied': True,
                    }
        
        if best_result is None:
            # No valid configuration found - return least constrained
            return {
                'conf_threshold': 0.5,
                'ood_threshold': None,
                'achieved_coverage': 1.0,
                'achieved_accuracy': float(accuracy_score(labels, predictions)),
                'achieved_f1': float(f1_score(labels, predictions)),
                'achieved_fpr_gap': float('inf'),
                'achieved_tpr_gap': float('inf'),
                'achieved_abstention_gap': 0.0,
                'achieved_min_group_coverage': 1.0,
                'ood_reliable': False,
                'ood_direction': 'n/a',
                'constraints_satisfied': False,
            }
        
        return best_result


# =============================================================================
# EVALUATOR
# =============================================================================

class SelectiveClassificationEvaluator:
    """Evaluate selective classification performance with all metrics."""
    
    @staticmethod
    def evaluate(
        labels: np.ndarray,
        predictions: np.ndarray,
        abstain_mask: np.ndarray,
        group_masks: Dict[str, np.ndarray],
        compute_bootstrap_ci: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Returns:
            Dict with coverage, accuracy, F1, risk, fairness metrics
        """
        covered = ~abstain_mask
        n_covered = covered.sum()
        n_total = len(labels)
        
        if n_covered == 0:
            return {
                'coverage': 0.0,
                'abstention_rate': 1.0,
                'accuracy': None,
                'f1': None,
                'risk': 1.0,
                'fairness': None
            }
        
        covered_labels = labels[covered]
        covered_preds = predictions[covered]
        
        accuracy = accuracy_score(covered_labels, covered_preds)
        f1 = f1_score(covered_labels, covered_preds)
        risk = 1 - accuracy  # error rate
        
        fairness = compute_fairness_metrics(labels, predictions, abstain_mask, group_masks)
        
        result = {
            'coverage': float(n_covered / n_total),
            'abstention_rate': float(1 - n_covered / n_total),
            'accuracy': float(accuracy),
            'f1': float(f1),
            'risk': float(risk),
            'fpr_gap': float(fairness.fpr_gap),
            'tpr_gap': float(fairness.tpr_gap),
            'abstention_gap': float(fairness.abstention_gap),
            'per_group_fpr': fairness.per_group_fpr,
            'per_group_tpr': fairness.per_group_tpr,
            'per_group_abstention': fairness.per_group_abstention,
        }
        
        if compute_bootstrap_ci and n_covered > 100:
            fpr_ci = bootstrap_fairness_gap(
                labels, predictions, abstain_mask, group_masks, 'fpr_gap', n_bootstrap=500
            )
            result['fpr_gap_ci'] = (fpr_ci['ci_low'], fpr_ci['ci_high'])
        
        return result
    
    @staticmethod
    def compute_risk_coverage_curve(
        labels: np.ndarray,
        predictions: np.ndarray,
        selection_scores: np.ndarray,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Risk-Coverage curve.
        
        Args:
            labels: True labels
            predictions: Model predictions
            selection_scores: Higher = more confident = select first
            n_points: Number of points on curve
        
        Returns:
            coverages, risks: Arrays for plotting
        """
        # Sort by selection score descending
        sorted_idx = np.argsort(selection_scores)[::-1]
        sorted_correct = (labels[sorted_idx] == predictions[sorted_idx]).astype(float)
        
        n = len(labels)
        coverages = np.linspace(0.01, 1.0, n_points)
        risks = []
        
        for cov in coverages:
            n_selected = max(1, int(cov * n))
            selected_correct = sorted_correct[:n_selected]
            risk = 1 - selected_correct.mean()
            risks.append(risk)
        
        return coverages, np.array(risks)
    
    @staticmethod
    def compute_aurc(
        labels: np.ndarray,
        predictions: np.ndarray,
        selection_scores: np.ndarray
    ) -> float:
        """Compute Area Under Risk-Coverage Curve (lower is better)."""
        coverages, risks = SelectiveClassificationEvaluator.compute_risk_coverage_curve(
            labels, predictions, selection_scores
        )
        return float(np.trapz(risks, coverages))


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_abstention_experiment(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    ood_scores: np.ndarray,
    group_masks: Dict[str, np.ndarray],
    odm_result: Dict[str, Any],
    fairness_epsilon: float = 0.05,
    min_coverage: float = 0.80
) -> Dict[str, Any]:
    """
    Run complete abstention experiment comparing all methods.
    
    Returns:
        Dict with results for each method
    """
    results = {}
    
    # 1. No abstention baseline
    no_abstain = np.zeros(len(labels), dtype=bool)
    results['no_abstention'] = SelectiveClassificationEvaluator.evaluate(
        labels, predictions, no_abstain, group_masks
    )
    results['no_abstention']['method'] = 'No Abstention'
    
    # 2. Confidence-only baseline
    solver = FairnessConstrainedThresholdSolver(
        fairness_epsilon=fairness_epsilon,
        min_coverage=min_coverage
    )
    conf_thresholds = solver.solve(
        confidences, ood_scores, labels, predictions, group_masks, odm_result, use_ood=False
    )
    
    conf_policy = ConfidenceOnlyPolicy(threshold=conf_thresholds['conf_threshold'])
    conf_abstain = conf_policy.compute_abstention(confidences)
    results['confidence_only'] = SelectiveClassificationEvaluator.evaluate(
        labels, predictions, conf_abstain, group_masks
    )
    results['confidence_only']['method'] = 'Confidence Only'
    results['confidence_only']['thresholds'] = conf_thresholds
    
    # 3. Naive OOD baseline (ignores inversion)
    naive_abstain = NaiveOODPolicy(threshold=0.0).compute_abstention(ood_scores)
    results['naive_ood'] = SelectiveClassificationEvaluator.evaluate(
        labels, predictions, naive_abstain, group_masks
    )
    results['naive_ood']['method'] = 'Naive OOD'
    
    # 4. OOD-Aware (ours)
    ood_thresholds = solver.solve(
        confidences, ood_scores, labels, predictions, group_masks, odm_result, use_ood=True
    )
    
    ood_policy = OODAwareAbstentionPolicy(
        conf_threshold=ood_thresholds['conf_threshold'],
        ood_threshold=ood_thresholds['ood_threshold'] if ood_thresholds['ood_threshold'] else 0.0,
        odm_threshold=0.6,
        use_ood=ood_thresholds['ood_reliable']
    )
    ood_abstain = ood_policy.compute_abstention(confidences, ood_scores, odm_result)
    results['ood_aware'] = SelectiveClassificationEvaluator.evaluate(
        labels, predictions, ood_abstain, group_masks
    )
    results['ood_aware']['method'] = 'OOD-Aware (Ours)'
    results['ood_aware']['thresholds'] = ood_thresholds
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OOD-Aware Fairness-Constrained Abstention Policy'
    )
    parser.add_argument(
        '--predictions', type=str, required=True,
        help='Path to predictions CSV'
    )
    parser.add_argument(
        '--groups_file', type=str,
        help='Path to file with group membership columns'
    )
    parser.add_argument(
        '--fairness_epsilon', type=float, default=0.05,
        help='Maximum allowed FPR gap'
    )
    parser.add_argument(
        '--min_coverage', type=float, default=0.80,
        help='Minimum coverage constraint'
    )
    parser.add_argument(
        '--output_dir', type=str, default='experiments/selective',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("OOD-Aware Fairness-Constrained Abstention Policy")
    print("=" * 80)
    print(f"Predictions: {args.predictions}")
    print(f"Fairness epsilon: {args.fairness_epsilon}")
    print(f"Min coverage: {args.min_coverage}")
    print("=" * 80)
    
    # Load data
    pred_df = pd.read_csv(args.predictions)
    
    # Extract required columns
    labels = pred_df['label'].values
    predictions = pred_df['pred'].values
    
    # Confidence is max probability
    if 'pos_prob' in pred_df.columns:
        confidences = np.maximum(pred_df['pos_prob'].values, 1 - pred_df['pos_prob'].values)
    else:
        confidences = np.ones(len(labels)) * 0.5
    
    # OOD scores
    if 'energy' in pred_df.columns:
        ood_scores = pred_df['energy'].values
    else:
        ood_scores = np.zeros(len(labels))
    
    # Group masks
    group_cols = [c for c in pred_df.columns if c.startswith('g_')]
    group_masks = {c: pred_df[c].values.astype(bool) for c in group_cols}
    
    if not group_masks:
        print("[WARN] No group columns found, creating dummy group")
        group_masks = {'all': np.ones(len(labels), dtype=bool)}
    
    # Mock ODM result (in real usage, compute from source/target scores)
    odm_result = {
        'should_use_ood': True,
        'separability': 0.7,
        'direction': 'normal'
    }
    
    # Run experiment
    results = run_abstention_experiment(
        labels, predictions, confidences, ood_scores, group_masks, odm_result,
        fairness_epsilon=args.fairness_epsilon,
        min_coverage=args.min_coverage
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for method_name, result in results.items():
        print(f"\n{result.get('method', method_name)}:")
        print(f"  Coverage: {result['coverage']:.3f}")
        print(f"  Accuracy: {result['accuracy']:.3f}" if result['accuracy'] else "  Accuracy: N/A")
        print(f"  FPR Gap:  {result['fpr_gap']:.3f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'abstention_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                for kk, vv in v.items()
                if not isinstance(vv, dict)
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
