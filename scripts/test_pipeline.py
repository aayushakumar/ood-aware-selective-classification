#!/usr/bin/env python3
"""
Pipeline Integration Test for BiasBreakers Project
===================================================

This script tests the entire pipeline end-to-end on a 2-core CPU with sample data.
It validates all components work correctly before running full experiments.

Components tested:
1. Data loading and preprocessing
2. TF-IDF baseline training and evaluation
3. RoBERTa model training (minimal)
4. OOD detection algorithms
5. Calibration methods
6. Fairness metrics computation
7. Cross-domain evaluation

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --sample_size 100 --skip_roberta
    python scripts/test_pipeline.py --verbose --full_test

Author: BiasBreakers Team (CS 483)
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Test configuration
DEFAULT_SAMPLE_SIZE = 500  # Sample size for testing (need enough for meaningful F1)
MAX_EPOCHS = 1
MAX_LEN = 64  # Shorter sequence length for speed


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.details = {}
    
    def __str__(self):
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if self.passed else f"{Colors.RED}✗ FAIL{Colors.END}"
        return f"{status} {self.name} ({self.duration:.2f}s)"


class PipelineTester:
    """Main class for testing the entire pipeline."""
    
    def __init__(
        self,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        skip_roberta: bool = False,
        skip_fairness: bool = False,
        verbose: bool = False
    ):
        self.sample_size = sample_size
        self.skip_roberta = skip_roberta
        self.skip_fairness = skip_fairness
        self.verbose = verbose
        
        self.data_dir = ROOT_DIR / "data"
        self.experiments_dir = ROOT_DIR / "experiments"
        self.test_output_dir = ROOT_DIR / "test_output"
        
        self.results: List[TestResult] = []
        self.sample_data: Dict[str, pd.DataFrame] = {}
        
        # Create test output directory
        self.test_output_dir.mkdir(exist_ok=True)
    
    def _stratified_sample(self, df: pd.DataFrame, n: int, label_col: str = 'label') -> pd.DataFrame:
        """
        Create a stratified sample ensuring both classes are represented.
        
        Args:
            df: DataFrame to sample from
            n: Total number of samples desired
            label_col: Name of the label column
            
        Returns:
            Stratified sample DataFrame
        """
        # Get class counts
        pos_mask = df[label_col] == 1
        n_pos = pos_mask.sum()
        n_neg = (~pos_mask).sum()
        
        if n_pos == 0 or n_neg == 0:
            # Only one class present, just sample
            return df.sample(n=min(n, len(df)), random_state=42)
        
        # Ensure at least 20% of each class (minimum 10 samples each)
        min_per_class = max(10, n // 5)
        
        # Calculate samples per class
        n_pos_sample = min(max(min_per_class, int(n * n_pos / len(df))), n_pos)
        n_neg_sample = min(n - n_pos_sample, n_neg)
        
        # Ensure we don't exceed available samples
        if n_pos_sample > n_pos:
            n_pos_sample = n_pos
            n_neg_sample = min(n - n_pos_sample, n_neg)
        
        # Sample from each class
        pos_samples = df[pos_mask].sample(n=n_pos_sample, random_state=42)
        neg_samples = df[~pos_mask].sample(n=n_neg_sample, random_state=42)
        
        # Combine and shuffle
        result = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        return result
    
    def log(self, message: str, level: str = "info"):
        """Print log message."""
        if level == "info":
            print(f"  {message}")
        elif level == "success":
            print(f"  {Colors.GREEN}✓{Colors.END} {message}")
        elif level == "error":
            print(f"  {Colors.RED}✗{Colors.END} {message}")
        elif level == "warn":
            print(f"  {Colors.YELLOW}⚠{Colors.END} {message}")
        elif level == "debug" and self.verbose:
            print(f"    [DEBUG] {message}")
    
    def run_test(self, test_func, name: str) -> TestResult:
        """Run a single test and capture results."""
        result = TestResult(name)
        
        print(f"\n{Colors.BOLD}Testing: {name}{Colors.END}")
        start_time = time.time()
        
        try:
            details = test_func()
            result.passed = True
            result.details = details or {}
            self.log("Test passed", "success")
        except Exception as e:
            result.passed = False
            result.error = str(e)
            if self.verbose:
                traceback.print_exc()
            self.log(f"Test failed: {e}", "error")
        
        result.duration = time.time() - start_time
        self.results.append(result)
        
        return result
    
    # =========================================================================
    # TEST 1: Data Loading
    # =========================================================================
    
    def test_data_loading(self) -> Dict:
        """Test that all datasets can be loaded."""
        from run_roberta import load_dataset, SUPPORTED_DATASETS
        
        loaded = {}
        
        for dataset in SUPPORTED_DATASETS:
            for split in ['train', 'val', 'test']:
                try:
                    df = load_dataset(dataset, split, str(self.data_dir))
                    loaded[f"{dataset}_{split}"] = len(df)
                    self.log(f"Loaded {dataset}_{split}: {len(df)} samples", "debug")
                    
                    # Verify schema
                    assert 'text' in df.columns, f"Missing 'text' column in {dataset}_{split}"
                    assert 'label' in df.columns, f"Missing 'label' column in {dataset}_{split}"
                    
                    # Sample for later tests - use stratified sampling to ensure class balance
                    if split == 'train':
                        sample = self._stratified_sample(df, min(self.sample_size, len(df)))
                        self.sample_data[f"{dataset}_train"] = sample
                    elif split == 'test':
                        sample = self._stratified_sample(df, min(self.sample_size // 2, len(df)))
                        self.sample_data[f"{dataset}_test"] = sample
                        
                except FileNotFoundError:
                    self.log(f"File not found: {dataset}_{split} (may need preprocessing)", "warn")
                    loaded[f"{dataset}_{split}"] = -1
        
        # Check at least one dataset loaded
        valid_loads = [v for v in loaded.values() if v > 0]
        assert len(valid_loads) >= 3, f"Need at least 3 dataset splits, got {len(valid_loads)}"
        
        return {"datasets_loaded": loaded}
    
    # =========================================================================
    # TEST 2: TF-IDF Baseline
    # =========================================================================
    
    def test_tfidf_baseline(self) -> Dict:
        """Test TF-IDF vectorization and model training."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, accuracy_score
        
        # Use sample data
        if 'jigsaw_train' not in self.sample_data:
            raise RuntimeError("No training data available")
        
        train_df = self.sample_data['jigsaw_train']
        test_df = self.sample_data.get('jigsaw_test', train_df.tail(50))
        
        # Ensure balanced test set for meaningful evaluation
        if 'jigsaw_test' not in self.sample_data:
            test_df = self._stratified_sample(train_df, 50)
        
        self.log(f"Training TF-IDF on {len(train_df)} samples", "debug")
        self.log(f"Train class distribution: {train_df['label'].value_counts().to_dict()}", "debug")
        self.log(f"Test class distribution: {test_df['label'].value_counts().to_dict()}", "debug")
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_df['text'].fillna(''))
        X_test = vectorizer.transform(test_df['text'].fillna(''))
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        # Train model with class weighting to handle imbalance
        model = LogisticRegression(max_iter=200, n_jobs=2, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, predictions, zero_division=0)
        acc = accuracy_score(y_test, predictions)
        
        # Debug: check predictions distribution
        self.log(f"Predictions distribution: 0={sum(predictions==0)}, 1={sum(predictions==1)}", "debug")
        self.log(f"Actual distribution: 0={sum(y_test==0)}, 1={sum(y_test==1)}", "debug")
        
        self.log(f"TF-IDF F1: {f1:.4f}, Accuracy: {acc:.4f}", "success")
        
        # Warn if F1 is 0
        if f1 == 0:
            self.log("Warning: F1=0 suggests model predicts all same class", "warn")
        
        # Save predictions for later tests
        self.sample_data['tfidf_probs'] = probs
        self.sample_data['tfidf_preds'] = predictions
        self.sample_data['test_labels'] = y_test
        
        return {"f1": f1, "accuracy": acc, "vocab_size": len(vectorizer.vocabulary_)}
    
    # =========================================================================
    # TEST 3: RoBERTa Model (Minimal)
    # =========================================================================
    
    def test_roberta_minimal(self) -> Dict:
        """Test RoBERTa model loading and minimal training."""
        if self.skip_roberta:
            self.log("Skipping RoBERTa test (--skip_roberta)", "warn")
            return {"skipped": True}
        
        import torch
        from transformers import (
            RobertaTokenizer, 
            RobertaForSequenceClassification,
            TrainingArguments,
            Trainer
        )
        from datasets import Dataset
        
        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log(f"Using device: {device}", "debug")
        
        if 'jigsaw_train' not in self.sample_data:
            raise RuntimeError("No training data available")
        
        # Use very small sample for CPU test
        train_df = self.sample_data['jigsaw_train'].head(50)
        test_df = self.sample_data.get('jigsaw_test', train_df.tail(20)).head(20)
        
        self.log(f"Training RoBERTa on {len(train_df)} samples (minimal test)", "debug")
        
        # Load tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        
        def tokenize(examples):
            return tokenizer(
                examples['text'], 
                truncation=True, 
                padding='max_length', 
                max_length=MAX_LEN
            )
        
        # Create datasets
        train_ds = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
        test_ds = Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True))
        
        train_ds = train_ds.map(tokenize, batched=True, remove_columns=['text'])
        test_ds = test_ds.map(tokenize, batched=True, remove_columns=['text'])
        
        # Training arguments (minimal)
        training_args = TrainingArguments(
            output_dir=str(self.test_output_dir / 'roberta_test'),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_steps=10,
            save_strategy='no',
            report_to='none',
            max_steps=10,  # Only 10 steps for testing
            fp16=False,  # Disable for CPU
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
        )
        
        # Train (minimal)
        trainer.train()
        
        # Quick inference test
        predictions = trainer.predict(test_ds)
        logits = predictions.predictions
        preds = np.argmax(logits, axis=-1)
        
        # Store logits for OOD tests
        self.sample_data['roberta_logits'] = logits
        self.sample_data['roberta_preds'] = preds
        
        self.log(f"RoBERTa training completed, {len(preds)} predictions made", "success")
        
        return {"trained": True, "n_predictions": len(preds)}
    
    # =========================================================================
    # TEST 4: OOD Detection Algorithms
    # =========================================================================
    
    def test_ood_algorithms(self) -> Dict:
        """Test OOD detection algorithms."""
        from ood_algorithms import (
            MaxSoftmaxOOD, ODIN_OOD, EnergyOOD, MahalanobisOOD,
            compute_ood_metrics
        )
        
        # Create synthetic logits for testing if RoBERTa was skipped
        if 'roberta_logits' not in self.sample_data:
            self.log("Using synthetic logits for OOD testing", "debug")
            # Synthetic in-distribution: confident predictions
            in_logits = np.random.randn(100, 2) * 2
            in_logits[:, 0] += 1  # Bias toward class 0
            
            # Synthetic OOD: less confident, more uniform
            out_logits = np.random.randn(100, 2) * 0.5
        else:
            in_logits = self.sample_data['roberta_logits']
            # Simulate OOD by adding noise
            out_logits = in_logits + np.random.randn(*in_logits.shape) * 2
        
        results = {}
        
        # Test MaxSoftmax
        msp = MaxSoftmaxOOD()
        in_scores = msp.compute_scores(in_logits)
        out_scores = msp.compute_scores(out_logits)
        metrics = compute_ood_metrics(in_scores, out_scores)
        results['MaxSoftmax_AUROC'] = metrics['auroc']
        self.log(f"MaxSoftmax AUROC: {metrics['auroc']:.4f}", "debug")
        
        # Test ODIN
        odin = ODIN_OOD(temperature=1000)
        in_scores = odin.compute_scores(in_logits)
        out_scores = odin.compute_scores(out_logits)
        metrics = compute_ood_metrics(in_scores, out_scores)
        results['ODIN_AUROC'] = metrics['auroc']
        self.log(f"ODIN AUROC: {metrics['auroc']:.4f}", "debug")
        
        # Test Energy
        energy = EnergyOOD(temperature=1.0)
        in_scores = energy.compute_scores(in_logits)
        out_scores = energy.compute_scores(out_logits)
        metrics = compute_ood_metrics(in_scores, out_scores)
        results['Energy_AUROC'] = metrics['auroc']
        self.log(f"Energy AUROC: {metrics['auroc']:.4f}", "debug")
        
        # Test Mahalanobis (requires features)
        in_features = np.random.randn(100, 64)
        out_features = np.random.randn(100, 64) + 2  # Shifted
        labels = np.random.randint(0, 2, 100)
        
        mahal = MahalanobisOOD()
        mahal.fit(in_features, labels)
        in_scores = mahal.compute_scores(in_features)
        out_scores = mahal.compute_scores(out_features)
        metrics = compute_ood_metrics(in_scores, out_scores)
        results['Mahalanobis_AUROC'] = metrics['auroc']
        self.log(f"Mahalanobis AUROC: {metrics['auroc']:.4f}", "debug")
        
        return results
    
    # =========================================================================
    # TEST 5: Domain Adaptation Methods
    # =========================================================================
    
    def test_domain_adaptation(self) -> Dict:
        """Test domain adaptation methods (CORAL, MMD)."""
        from ood_algorithms import CORAL, MMD, compute_domain_divergence
        
        # Create synthetic features
        np.random.seed(42)
        source_features = np.random.randn(100, 64)
        target_features = np.random.randn(100, 64) + 1  # Domain shift
        
        # Test CORAL
        coral = CORAL()
        coral.fit(source_features)
        coral_loss = coral.compute_loss(source_features, target_features)
        self.log(f"CORAL loss: {coral_loss:.4f}", "debug")
        
        # Test CORAL transform
        transformed = coral.transform(target_features)
        coral_loss_after = coral.compute_loss(source_features, transformed)
        self.log(f"CORAL loss after transform: {coral_loss_after:.4f}", "debug")
        
        # Test MMD
        mmd = MMD()
        mmd_value = mmd.compute(source_features, target_features)
        self.log(f"MMD: {mmd_value:.4f}", "debug")
        
        # Test domain divergence computation
        divergence = compute_domain_divergence(source_features, target_features)
        self.log(f"Domain divergence metrics computed", "debug")
        
        return {
            "coral_loss": coral_loss,
            "coral_loss_reduced": coral_loss_after < coral_loss,
            "mmd": mmd_value,
            "mean_divergence": divergence['mean_divergence']
        }
    
    # =========================================================================
    # TEST 6: Calibration Methods
    # =========================================================================
    
    def test_calibration(self) -> Dict:
        """Test calibration methods."""
        from ood_algorithms import (
            TemperatureScaling, IsotonicCalibration, PlattScaling,
            expected_calibration_error
        )
        
        # Create synthetic logits and labels
        np.random.seed(42)
        n_samples = 200
        
        # Generate logits with some miscalibration
        logits = np.random.randn(n_samples, 2) * 2
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        labels = (probs[:, 1] > 0.5).astype(int)
        # Add noise to labels to create miscalibration
        labels[::10] = 1 - labels[::10]
        
        # Split into val and test
        val_logits, test_logits = logits[:100], logits[100:]
        val_labels, test_labels = labels[:100], labels[100:]
        
        results = {}
        
        # Uncalibrated ECE
        test_probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
        uncal_ece = expected_calibration_error(test_labels, test_probs[:, 1])
        results['uncalibrated_ece'] = uncal_ece
        self.log(f"Uncalibrated ECE: {uncal_ece:.4f}", "debug")
        
        # Temperature Scaling
        ts = TemperatureScaling()
        ts.fit(val_logits, val_labels)
        ts_probs = ts.calibrate(test_logits)
        ts_ece = expected_calibration_error(test_labels, ts_probs[:, 1])
        results['temp_scaling_ece'] = ts_ece
        results['temperature'] = ts.temperature
        self.log(f"Temperature Scaling ECE: {ts_ece:.4f} (T={ts.temperature:.2f})", "debug")
        
        # Isotonic Regression
        val_probs = np.exp(val_logits) / np.exp(val_logits).sum(axis=1, keepdims=True)
        ir = IsotonicCalibration()
        ir.fit(val_probs, val_labels)
        ir_probs = ir.calibrate(test_probs)
        ir_ece = expected_calibration_error(test_labels, ir_probs[:, 1])
        results['isotonic_ece'] = ir_ece
        self.log(f"Isotonic Regression ECE: {ir_ece:.4f}", "debug")
        
        # Platt Scaling
        ps = PlattScaling()
        ps.fit(val_logits, val_labels)
        ps_probs = ps.calibrate(test_logits)
        ps_ece = expected_calibration_error(test_labels, ps_probs[:, 1])
        results['platt_ece'] = ps_ece
        self.log(f"Platt Scaling ECE: {ps_ece:.4f}", "debug")
        
        return results
    
    # =========================================================================
    # TEST 7: Fairness Metrics
    # =========================================================================
    
    def test_fairness_metrics(self) -> Dict:
        """Test fairness metrics computation."""
        if self.skip_fairness:
            self.log("Skipping fairness test (--skip_fairness)", "warn")
            return {"skipped": True}
        
        # Try to load full data with identity attributes
        full_data_path = self.data_dir / "jigsaw_train_full.csv"
        
        if not full_data_path.exists():
            self.log("Full data file not found, using synthetic data", "warn")
            
            # Create synthetic data with identity groups
            n = 500
            data = {
                'text': [f'sample text {i}' for i in range(n)],
                'label': np.random.randint(0, 2, n),
                'g_male': np.random.random(n) > 0.5,
                'g_female': np.random.random(n) > 0.5,
                'g_black': np.random.random(n) > 0.8,
                'g_white': np.random.random(n) > 0.7,
            }
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(full_data_path).head(500)
        
        # Get group columns
        g_cols = [c for c in df.columns if c.startswith('g_')]
        
        if len(g_cols) == 0:
            self.log("No group columns found", "warn")
            return {"n_groups": 0}
        
        # Simulate predictions
        y_true = df['label'].values
        y_pred = (np.random.random(len(df)) > 0.5).astype(int)
        
        results = {"n_groups": len(g_cols)}
        
        # Compute fairness metrics for each group
        for col in g_cols[:3]:  # Test first 3 groups
            group_mask = (df[col].fillna(0) >= 0.5).values
            
            if group_mask.sum() < 10 or (~group_mask).sum() < 10:
                continue
            
            # Demographic Parity
            group_pos_rate = y_pred[group_mask].mean()
            non_group_pos_rate = y_pred[~group_mask].mean()
            dp_gap = abs(group_pos_rate - non_group_pos_rate)
            
            results[f'{col}_dp_gap'] = dp_gap
            self.log(f"{col} DP gap: {dp_gap:.4f}", "debug")
        
        return results
    
    # =========================================================================
    # TEST 8: Cross-Domain Evaluation
    # =========================================================================
    
    def test_cross_domain_eval(self) -> Dict:
        """Test cross-domain evaluation workflow."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        
        datasets = ['jigsaw', 'civil', 'hatexplain']
        available = [d for d in datasets if f'{d}_train' in self.sample_data]
        
        if len(available) < 2:
            self.log("Need at least 2 datasets for cross-domain test", "warn")
            return {"n_evaluated": 0}
        
        results = {}
        
        # Train on first available, test on others
        source = available[0]
        targets = available[1:]
        
        train_df = self.sample_data[f'{source}_train']
        
        # Train TF-IDF model
        vectorizer = TfidfVectorizer(max_features=500)
        X_train = vectorizer.fit_transform(train_df['text'].fillna(''))
        y_train = train_df['label'].values
        
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        
        # Evaluate on each target
        for target in targets:
            test_key = f'{target}_test'
            if test_key not in self.sample_data:
                continue
            
            test_df = self.sample_data[test_key]
            X_test = vectorizer.transform(test_df['text'].fillna(''))
            y_test = test_df['label'].values
            
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            results[f'{source}_to_{target}'] = f1
            self.log(f"{source} → {target}: F1={f1:.4f}", "debug")
        
        return results
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    def run_all_tests(self) -> bool:
        """Run all pipeline tests."""
        print(f"\n{'='*60}")
        print(f"{Colors.BOLD}BiasBreakers Pipeline Integration Tests{Colors.END}")
        print(f"{'='*60}")
        print(f"Sample size: {self.sample_size}")
        print(f"Skip RoBERTa: {self.skip_roberta}")
        print(f"Verbose: {self.verbose}")
        
        # Run tests in order
        tests = [
            (self.test_data_loading, "Data Loading"),
            (self.test_tfidf_baseline, "TF-IDF Baseline"),
            (self.test_roberta_minimal, "RoBERTa Model (Minimal)"),
            (self.test_ood_algorithms, "OOD Detection Algorithms"),
            (self.test_domain_adaptation, "Domain Adaptation Methods"),
            (self.test_calibration, "Calibration Methods"),
            (self.test_fairness_metrics, "Fairness Metrics"),
            (self.test_cross_domain_eval, "Cross-Domain Evaluation"),
        ]
        
        for test_func, name in tests:
            self.run_test(test_func, name)
        
        # Print summary
        self.print_summary()
        
        # Return True if all tests passed
        return all(r.passed for r in self.results)
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"{Colors.BOLD}Test Summary{Colors.END}")
        print(f"{'='*60}")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        for result in self.results:
            print(result)
        
        print(f"\n{'-'*60}")
        
        if passed == total:
            print(f"{Colors.GREEN}{Colors.BOLD}All {total} tests passed!{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}{passed}/{total} tests passed{Colors.END}")
            
            # Print failed test details
            failed = [r for r in self.results if not r.passed]
            if failed:
                print(f"\n{Colors.RED}Failed tests:{Colors.END}")
                for r in failed:
                    print(f"  - {r.name}: {r.error}")
        
        # Total time
        total_time = sum(r.duration for r in self.results)
        print(f"\nTotal time: {total_time:.2f}s")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save test results to JSON."""
        results_data = []
        for r in self.results:
            # Convert numpy types to Python types for JSON serialization
            details = {}
            for k, v in r.details.items():
                if isinstance(v, (np.floating, np.float64, np.float32)):
                    details[k] = float(v)
                elif isinstance(v, (np.integer, np.int64, np.int32)):
                    details[k] = int(v)
                elif isinstance(v, np.ndarray):
                    details[k] = v.tolist()
                elif isinstance(v, np.bool_):
                    details[k] = bool(v)
                else:
                    details[k] = v
                    
            results_data.append({
                'name': r.name,
                'passed': r.passed,
                'duration': float(r.duration),
                'error': r.error,
                'details': details
            })
        
        output_file = self.test_output_dir / 'test_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="BiasBreakers Pipeline Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_pipeline.py                    # Quick test
  python scripts/test_pipeline.py --sample_size 50   # Very quick test
  python scripts/test_pipeline.py --skip_roberta     # Skip slow RoBERTa test
  python scripts/test_pipeline.py --verbose          # Show detailed output
  python scripts/test_pipeline.py --full_test        # Full test (1000 samples)
"""
    )
    
    parser.add_argument(
        '--sample_size', type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f'Number of samples to use for testing (default: {DEFAULT_SAMPLE_SIZE})'
    )
    parser.add_argument(
        '--skip_roberta', action='store_true',
        help='Skip RoBERTa model test (faster)'
    )
    parser.add_argument(
        '--skip_fairness', action='store_true',
        help='Skip fairness metrics test'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--full_test', action='store_true',
        help='Run full test with more samples (1000)'
    )
    
    args = parser.parse_args()
    
    if args.full_test:
        args.sample_size = 1000
    
    tester = PipelineTester(
        sample_size=args.sample_size,
        skip_roberta=args.skip_roberta,
        skip_fairness=args.skip_fairness,
        verbose=args.verbose
    )
    
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
