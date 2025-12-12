#!/usr/bin/env python3
"""
OOD (Out-of-Distribution) Algorithms for Toxicity Classification
==================================================================

This module implements various OOD detection and domain adaptation algorithms:

1. OOD Detection Methods:
   - MaxSoftmax: Baseline using maximum softmax probability
   - ODIN: Temperature scaling + input perturbation
   - Energy-based: Energy score for OOD detection
   - Mahalanobis: Distance-based OOD detection

2. Domain Adaptation Methods:
   - CORAL: Correlation Alignment
   - MMD: Maximum Mean Discrepancy
   - Domain Adversarial: Gradient reversal for domain invariance

3. Calibration Methods:
   - Temperature Scaling
   - Isotonic Regression
   - Platt Scaling

Author: BiasBreakers Team (CS 483)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# OOD DETECTION METHODS
# ============================================================================

class MaxSoftmaxOOD:
    """
    Baseline OOD detection using Maximum Softmax Probability (MSP).
    
    Reference: Hendrycks & Gimpel (2017) "A Baseline for Detecting 
    Misclassified and Out-of-Distribution Examples in Neural Networks"
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "MaxSoftmax"
    
    def compute_scores(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute OOD scores (higher = more likely OOD).
        
        Args:
            logits: Raw model outputs (N, num_classes)
            
        Returns:
            OOD scores (N,) - negative max softmax prob
        """
        probs = self._softmax(logits)
        max_probs = np.max(probs, axis=1)
        # Return negative because higher confidence = more likely in-distribution
        return -max_probs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def detect(self, logits: np.ndarray, threshold: float = None) -> np.ndarray:
        """Return binary OOD predictions (1 = OOD, 0 = in-distribution)."""
        if threshold is None:
            threshold = self.threshold
        scores = self.compute_scores(logits)
        return (scores > -threshold).astype(int)


class ODIN_OOD:
    """
    ODIN: Out-of-Distribution Detector for Neural Networks.
    
    Uses temperature scaling and input perturbation for better OOD detection.
    
    Reference: Liang et al. (2018) "Enhancing The Reliability of 
    Out-of-distribution Image Detection in Neural Networks"
    """
    
    def __init__(self, temperature: float = 1000.0, epsilon: float = 0.0014):
        """
        Args:
            temperature: Temperature for scaling (default: 1000)
            epsilon: Perturbation magnitude (default: 0.0014)
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.name = "ODIN"
    
    def compute_scores(self, logits: np.ndarray, temperature: float = None) -> np.ndarray:
        """
        Compute ODIN scores using temperature-scaled softmax.
        
        Args:
            logits: Raw model outputs (N, num_classes)
            temperature: Override default temperature
            
        Returns:
            OOD scores (N,) - negative max scaled softmax prob
        """
        if temperature is None:
            temperature = self.temperature
        
        # Temperature scaling
        scaled_logits = logits / temperature
        probs = self._softmax(scaled_logits)
        max_probs = np.max(probs, axis=1)
        
        return -max_probs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def compute_perturbed_scores(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        device: str = 'cuda'
    ) -> np.ndarray:
        """
        Compute ODIN scores with input perturbation (requires model).
        
        Args:
            model: PyTorch model
            inputs: Input tensor (N, seq_len)
            device: Device to use
            
        Returns:
            OOD scores with perturbation
        """
        model.eval()
        inputs = inputs.to(device)
        inputs.requires_grad = True
        
        # Forward pass
        outputs = model(inputs)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Temperature-scaled softmax
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        
        # Compute gradient
        loss = max_probs.mean()
        loss.backward()
        
        # Perturbation direction (sign of gradient)
        gradient = torch.sign(inputs.grad.data)
        
        # Perturbed input (add small perturbation in direction of gradient)
        # Note: For text embeddings, this is conceptual
        perturbed = inputs - self.epsilon * gradient
        
        # Forward pass with perturbed input
        with torch.no_grad():
            outputs_perturbed = model(perturbed)
            if hasattr(outputs_perturbed, 'logits'):
                logits_perturbed = outputs_perturbed.logits
            else:
                logits_perturbed = outputs_perturbed
        
        return self.compute_scores(logits_perturbed.cpu().numpy())


class EnergyOOD:
    """
    Energy-based Out-of-Distribution Detection.
    
    Uses the energy function E(x) = -T * log(sum(exp(f_i(x)/T))) as OOD score.
    
    Reference: Liu et al. (2020) "Energy-based Out-of-distribution Detection"
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.name = "Energy"
    
    def compute_scores(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute energy scores (higher energy = more likely OOD).
        
        Args:
            logits: Raw model outputs (N, num_classes)
            
        Returns:
            Energy scores (N,)
        """
        # Energy = -T * logsumexp(logits/T)
        scaled_logits = logits / self.temperature
        energy = -self.temperature * np.log(np.sum(np.exp(scaled_logits), axis=1))
        return energy
    
    def detect(self, logits: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Return binary OOD predictions (1 = OOD, 0 = in-distribution)."""
        scores = self.compute_scores(logits)
        return (scores > threshold).astype(int)


class MahalanobisOOD:
    """
    Mahalanobis Distance-based OOD Detection.
    
    Computes Mahalanobis distance from class-conditional Gaussian distributions
    fitted on feature representations.
    
    Reference: Lee et al. (2018) "A Simple Unified Framework for Detecting 
    Out-of-Distribution Samples and Adversarial Attacks"
    """
    
    def __init__(self):
        self.class_means = None
        self.precision = None
        self.fitted = False
        self.name = "Mahalanobis"
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> 'MahalanobisOOD':
        """
        Fit class-conditional Gaussians on training features.
        
        Args:
            features: Feature representations (N, D)
            labels: Class labels (N,)
        """
        classes = np.unique(labels)
        self.class_means = {}
        
        # Compute class means
        for c in classes:
            mask = labels == c
            self.class_means[c] = np.mean(features[mask], axis=0)
        
        # Compute tied covariance matrix
        centered_features = features.copy()
        for c in classes:
            mask = labels == c
            centered_features[mask] -= self.class_means[c]
        
        covariance = np.cov(centered_features.T)
        
        # Add small regularization for numerical stability
        covariance += 1e-5 * np.eye(covariance.shape[0])
        
        self.precision = np.linalg.inv(covariance)
        self.fitted = True
        
        return self
    
    def compute_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance scores.
        
        Args:
            features: Feature representations (N, D)
            
        Returns:
            Minimum Mahalanobis distance to any class (N,)
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before compute_scores()")
        
        n_samples = features.shape[0]
        min_distances = np.full(n_samples, np.inf)
        
        for c, mean in self.class_means.items():
            diff = features - mean
            # Mahalanobis distance: sqrt((x-μ)^T Σ^{-1} (x-μ))
            distances = np.sqrt(np.sum(diff @ self.precision * diff, axis=1))
            min_distances = np.minimum(min_distances, distances)
        
        return min_distances


# ============================================================================
# DOMAIN ADAPTATION METHODS
# ============================================================================

class CORAL:
    """
    CORAL: Correlation Alignment for Domain Adaptation.
    
    Aligns second-order statistics (covariance) between source and target domains.
    
    Reference: Sun et al. (2016) "Return of Frustratingly Easy Domain Adaptation"
    """
    
    def __init__(self):
        self.source_mean = None
        self.source_cov = None
        self.transform_matrix = None
        self.fitted = False
        self.name = "CORAL"
    
    def fit(self, source_features: np.ndarray) -> 'CORAL':
        """Fit CORAL on source domain features."""
        self.source_mean = np.mean(source_features, axis=0)
        centered = source_features - self.source_mean
        self.source_cov = np.cov(centered.T) + 1e-5 * np.eye(source_features.shape[1])
        self.fitted = True
        return self
    
    def compute_loss(
        self, 
        source_features: np.ndarray, 
        target_features: np.ndarray
    ) -> float:
        """
        Compute CORAL loss between source and target domains.
        
        CORAL_loss = (1/4d^2) * ||C_s - C_t||_F^2
        
        Args:
            source_features: Source domain features (N_s, D)
            target_features: Target domain features (N_t, D)
            
        Returns:
            CORAL loss value
        """
        d = source_features.shape[1]
        
        # Compute covariances
        source_cov = np.cov(source_features.T)
        target_cov = np.cov(target_features.T)
        
        # Frobenius norm of difference
        diff = source_cov - target_cov
        loss = np.sum(diff ** 2) / (4 * d * d)
        
        return loss
    
    def transform(self, target_features: np.ndarray) -> np.ndarray:
        """
        Transform target features to align with source distribution.
        
        This is a simplified version that whitens target and re-colors with source stats.
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Compute target statistics
        target_mean = np.mean(target_features, axis=0)
        centered = target_features - target_mean
        target_cov = np.cov(centered.T) + 1e-5 * np.eye(target_features.shape[1])
        
        # Whitening transform
        u, s, _ = np.linalg.svd(target_cov)
        whiten = u @ np.diag(1.0 / np.sqrt(s)) @ u.T
        
        # Re-coloring with source covariance
        u_s, s_s, _ = np.linalg.svd(self.source_cov)
        recolor = u_s @ np.diag(np.sqrt(s_s)) @ u_s.T
        
        # Apply transform
        whitened = centered @ whiten
        transformed = whitened @ recolor + self.source_mean
        
        return transformed


class CORALLoss(nn.Module):
    """PyTorch module for CORAL loss computation."""
    
    def __init__(self, lambda_coral: float = 1.0):
        super().__init__()
        self.lambda_coral = lambda_coral
    
    def forward(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute CORAL loss."""
        d = source_features.size(1)
        
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # CORAL loss
        loss = torch.sum((source_cov - target_cov) ** 2) / (4 * d * d)
        
        return self.lambda_coral * loss
    
    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix."""
        n = features.size(0)
        centered = features - features.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (n - 1)
        return cov


class MMD:
    """
    Maximum Mean Discrepancy for domain divergence measurement.
    
    Uses RBF kernel to measure distance between distributions.
    
    Reference: Gretton et al. (2012) "A Kernel Two-Sample Test"
    """
    
    def __init__(self, kernel: str = 'rbf', gamma: float = None):
        self.kernel = kernel
        self.gamma = gamma
        self.name = "MMD"
    
    def compute(
        self, 
        source_features: np.ndarray, 
        target_features: np.ndarray
    ) -> float:
        """
        Compute MMD between source and target distributions.
        
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        
        Args:
            source_features: Source domain features (N_s, D)
            target_features: Target domain features (N_t, D)
            
        Returns:
            MMD value (squared)
        """
        if self.gamma is None:
            # Median heuristic
            combined = np.vstack([source_features, target_features])
            pairwise_dists = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2)
            self.gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
        
        # Compute kernel matrices
        k_ss = self._rbf_kernel(source_features, source_features)
        k_tt = self._rbf_kernel(target_features, target_features)
        k_st = self._rbf_kernel(source_features, target_features)
        
        # MMD^2
        n_s = source_features.shape[0]
        n_t = target_features.shape[0]
        
        mmd_sq = (np.sum(k_ss) / (n_s * n_s) + 
                  np.sum(k_tt) / (n_t * n_t) - 
                  2 * np.sum(k_st) / (n_s * n_t))
        
        return max(0, mmd_sq)  # Ensure non-negative
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        pairwise_sq_dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * pairwise_sq_dists)


class MMDLoss(nn.Module):
    """PyTorch module for MMD loss computation."""
    
    def __init__(self, kernel: str = 'rbf', gamma: float = 1.0, lambda_mmd: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.lambda_mmd = lambda_mmd
    
    def forward(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute MMD loss."""
        k_ss = self._rbf_kernel(source_features, source_features)
        k_tt = self._rbf_kernel(target_features, target_features)
        k_st = self._rbf_kernel(source_features, target_features)
        
        n_s = source_features.size(0)
        n_t = target_features.size(0)
        
        mmd = (k_ss.sum() / (n_s * n_s) + 
               k_tt.sum() / (n_t * n_t) - 
               2 * k_st.sum() / (n_s * n_t))
        
        return self.lambda_mmd * mmd
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        pairwise_sq_dists = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-self.gamma * pairwise_sq_dists)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for Domain Adversarial Training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial domain adaptation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward pass with gradient reversal."""
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.discriminator(reversed_features)


# ============================================================================
# CALIBRATION METHODS
# ============================================================================

class TemperatureScaling:
    """
    Temperature Scaling for post-hoc calibration.
    
    Learns a single temperature parameter T to scale logits: p = softmax(z/T)
    
    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.fitted = False
        self.name = "TemperatureScaling"
    
    def fit(
        self, 
        logits: np.ndarray, 
        labels: np.ndarray, 
        bounds: Tuple[float, float] = (0.1, 10.0)
    ) -> 'TemperatureScaling':
        """
        Fit temperature parameter by minimizing NLL on validation data.
        
        Args:
            logits: Model logits (N, num_classes)
            labels: True labels (N,)
            bounds: Search bounds for temperature
        """
        def nll_loss(T):
            scaled_logits = logits / T
            probs = self._softmax(scaled_logits)
            # Negative log-likelihood
            eps = 1e-10
            log_probs = np.log(probs + eps)
            nll = -np.mean(log_probs[np.arange(len(labels)), labels])
            return nll
        
        result = minimize_scalar(nll_loss, bounds=bounds, method='bounded')
        self.temperature = result.x
        self.fitted = True
        
        return self
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to get calibrated probabilities."""
        scaled_logits = logits / self.temperature
        return self._softmax(scaled_logits)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class IsotonicCalibration:
    """
    Isotonic Regression for calibration.
    
    Non-parametric calibration using isotonic regression on validation predictions.
    """
    
    def __init__(self):
        self.calibrators = {}
        self.fitted = False
        self.name = "IsotonicRegression"
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibration':
        """
        Fit isotonic regression calibrators for each class.
        
        Args:
            probs: Predicted probabilities (N, num_classes)
            labels: True labels (N,)
        """
        n_classes = probs.shape[1]
        
        for c in range(n_classes):
            ir = IsotonicRegression(out_of_bounds='clip')
            # For class c, fit on probability vs binary label
            binary_labels = (labels == c).astype(float)
            ir.fit(probs[:, c], binary_labels)
            self.calibrators[c] = ir
        
        self.fitted = True
        return self
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to probabilities."""
        if not self.fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        
        calibrated = np.zeros_like(probs)
        for c, ir in self.calibrators.items():
            calibrated[:, c] = ir.predict(probs[:, c])
        
        # Normalize to sum to 1
        calibrated = calibrated / (calibrated.sum(axis=1, keepdims=True) + 1e-10)
        
        return calibrated


class PlattScaling:
    """
    Platt Scaling for calibration (binary classification).
    
    Fits logistic regression on model outputs to produce calibrated probabilities.
    
    Reference: Platt (1999) "Probabilistic Outputs for Support Vector Machines"
    """
    
    def __init__(self):
        self.scaler = None
        self.fitted = False
        self.name = "PlattScaling"
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """
        Fit Platt scaling on validation data.
        
        Args:
            logits: Model logits (N,) for binary or (N, 2)
            labels: True labels (N,)
        """
        # Use positive class logits for binary classification
        if logits.ndim == 2:
            scores = logits[:, 1] - logits[:, 0]  # log-odds
        else:
            scores = logits
        
        self.scaler = PlattScaler()
        self.scaler.fit(scores.reshape(-1, 1), labels)
        self.fitted = True
        
        return self
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to get calibrated probabilities."""
        if not self.fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        
        if logits.ndim == 2:
            scores = logits[:, 1] - logits[:, 0]
        else:
            scores = logits
        
        probs = self.scaler.predict_proba(scores.reshape(-1, 1))
        return probs


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def expected_calibration_error(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece


def compute_ood_metrics(
    in_scores: np.ndarray, 
    out_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Args:
        in_scores: OOD scores for in-distribution data
        out_scores: OOD scores for out-of-distribution data
        
    Returns:
        Dictionary with AUROC, FPR@95TPR, detection accuracy
    """
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    scores = np.concatenate([in_scores, out_scores])
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # FPR at 95% TPR
    sorted_out_scores = np.sort(out_scores)
    threshold_idx = int(0.05 * len(sorted_out_scores))
    threshold = sorted_out_scores[threshold_idx]
    fpr_at_95tpr = (in_scores >= threshold).mean()
    
    # Detection accuracy at optimal threshold
    thresholds = np.percentile(scores, np.arange(0, 101, 5))
    best_acc = 0
    for t in thresholds:
        predictions = (scores >= t).astype(int)
        acc = accuracy_score(labels, predictions)
        best_acc = max(best_acc, acc)
    
    return {
        'auroc': auroc,
        'fpr_at_95tpr': fpr_at_95tpr,
        'detection_accuracy': best_acc
    }


def compute_domain_divergence(
    source_features: np.ndarray, 
    target_features: np.ndarray
) -> Dict[str, float]:
    """
    Compute domain divergence metrics.
    
    Args:
        source_features: Features from source domain
        target_features: Features from target domain
        
    Returns:
        Dictionary with CORAL loss, MMD, mean/std divergence
    """
    coral = CORAL()
    coral_loss = coral.compute_loss(source_features, target_features)
    
    mmd = MMD()
    mmd_value = mmd.compute(source_features, target_features)
    
    # Simple statistics divergence
    source_mean = np.mean(source_features, axis=0)
    target_mean = np.mean(target_features, axis=0)
    mean_divergence = np.linalg.norm(source_mean - target_mean)
    
    source_std = np.std(source_features, axis=0)
    target_std = np.std(target_features, axis=0)
    std_divergence = np.linalg.norm(source_std - target_std)
    
    return {
        'coral_loss': coral_loss,
        'mmd': mmd_value,
        'mean_divergence': mean_divergence,
        'std_divergence': std_divergence
    }


# ============================================================================
# HIGH-LEVEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_ood_detection_methods(
    in_logits: np.ndarray,
    out_logits: np.ndarray,
    in_features: np.ndarray = None,
    out_features: np.ndarray = None,
    in_labels: np.ndarray = None
) -> pd.DataFrame:
    """
    Evaluate multiple OOD detection methods.
    
    Args:
        in_logits: Logits for in-distribution data
        out_logits: Logits for out-of-distribution data
        in_features: Features for Mahalanobis (optional)
        out_features: Features for Mahalanobis (optional)
        in_labels: Labels for Mahalanobis fitting (optional)
        
    Returns:
        DataFrame with metrics for each method
    """
    methods = [
        MaxSoftmaxOOD(),
        ODIN_OOD(temperature=1000),
        EnergyOOD(temperature=1.0),
    ]
    
    # Add Mahalanobis if features are provided
    if in_features is not None and out_features is not None and in_labels is not None:
        mahal = MahalanobisOOD()
        mahal.fit(in_features, in_labels)
        
    results = []
    
    for method in methods:
        in_scores = method.compute_scores(in_logits)
        out_scores = method.compute_scores(out_logits)
        
        metrics = compute_ood_metrics(in_scores, out_scores)
        metrics['method'] = method.name
        results.append(metrics)
    
    # Mahalanobis if available
    if in_features is not None and out_features is not None and in_labels is not None:
        in_scores = mahal.compute_scores(in_features)
        out_scores = mahal.compute_scores(out_features)
        metrics = compute_ood_metrics(in_scores, out_scores)
        metrics['method'] = 'Mahalanobis'
        results.append(metrics)
    
    return pd.DataFrame(results)


def evaluate_calibration_methods(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate multiple calibration methods.
    
    Args:
        val_logits: Validation logits for fitting
        val_labels: Validation labels
        test_logits: Test logits for evaluation
        test_labels: Test labels
        
    Returns:
        DataFrame with ECE, accuracy, NLL for each method
    """
    results = []
    
    # Uncalibrated baseline
    uncal_probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
    uncal_pos_probs = uncal_probs[:, 1]
    results.append({
        'method': 'Uncalibrated',
        'ece': expected_calibration_error(test_labels, uncal_pos_probs),
        'accuracy': accuracy_score(test_labels, (uncal_pos_probs >= 0.5).astype(int)),
        'nll': -np.mean(np.log(uncal_probs[np.arange(len(test_labels)), test_labels] + 1e-10))
    })
    
    # Temperature Scaling
    ts = TemperatureScaling()
    ts.fit(val_logits, val_labels)
    ts_probs = ts.calibrate(test_logits)
    ts_pos_probs = ts_probs[:, 1]
    results.append({
        'method': f'TempScale (T={ts.temperature:.2f})',
        'ece': expected_calibration_error(test_labels, ts_pos_probs),
        'accuracy': accuracy_score(test_labels, (ts_pos_probs >= 0.5).astype(int)),
        'nll': -np.mean(np.log(ts_probs[np.arange(len(test_labels)), test_labels] + 1e-10))
    })
    
    # Isotonic Regression
    val_probs = np.exp(val_logits) / np.exp(val_logits).sum(axis=1, keepdims=True)
    ir = IsotonicCalibration()
    ir.fit(val_probs, val_labels)
    ir_probs = ir.calibrate(uncal_probs)
    ir_pos_probs = ir_probs[:, 1]
    results.append({
        'method': 'IsotonicReg',
        'ece': expected_calibration_error(test_labels, ir_pos_probs),
        'accuracy': accuracy_score(test_labels, (ir_pos_probs >= 0.5).astype(int)),
        'nll': -np.mean(np.log(ir_probs[np.arange(len(test_labels)), test_labels] + 1e-10))
    })
    
    # Platt Scaling
    ps = PlattScaling()
    ps.fit(val_logits, val_labels)
    ps_probs = ps.calibrate(test_logits)
    ps_pos_probs = ps_probs[:, 1]
    results.append({
        'method': 'PlattScale',
        'ece': expected_calibration_error(test_labels, ps_pos_probs),
        'accuracy': accuracy_score(test_labels, (ps_pos_probs >= 0.5).astype(int)),
        'nll': -np.mean(np.log(ps_probs[np.arange(len(test_labels)), test_labels] + 1e-10))
    })
    
    return pd.DataFrame(results)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OOD Algorithms Evaluation")
    parser.add_argument("--mode", choices=["ood", "calibration", "domain"], default="ood",
                       help="Evaluation mode")
    parser.add_argument("--in_preds", type=str, help="Path to in-distribution predictions CSV")
    parser.add_argument("--out_preds", type=str, help="Path to OOD predictions CSV")
    parser.add_argument("--output_dir", type=str, default="experiments",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("="*60)
    print("OOD Algorithms Evaluation Tool")
    print("="*60)
    print("\nAvailable methods:")
    print("  OOD Detection: MaxSoftmax, ODIN, Energy, Mahalanobis")
    print("  Domain Adaptation: CORAL, MMD")
    print("  Calibration: Temperature Scaling, Isotonic Regression, Platt Scaling")
    print("\nRun with appropriate arguments to evaluate.")
