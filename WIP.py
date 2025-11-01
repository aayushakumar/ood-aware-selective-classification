#!/usr/bin/env python3
"""
BiasBreakers - CS 483 Course Project
Task A: Cross-domain toxicity detection with RoBERTa

Team: BiasBreakers
Goal: Train toxicity classifiers and evaluate cross-domain generalization
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

# ============================================================================
# CONSTANTS
# ============================================================================

DATA_DIR = "data"
EXPERIMENTS_DIR = "experiments"
SUPPORTED_DATASETS = ["jigsaw", "civil", "hatexplain"]
CIVIL_TOXICITY_THRESHOLD = 0.5


# ============================================================================
# UTILITY: REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jigsaw(split: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load Jigsaw Toxic Comment dataset.
    Expected columns: comment_text, toxic (binary 0/1)
    Returns normalized DataFrame with columns: text, label
    """
    filepath = Path(data_dir) / f"jigsaw_{split}.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Jigsaw {split} file not found. Expected: {filepath}\n"
            f"Please place jigsaw_{split}.csv in {data_dir}/"
        )
    
    df = pd.read_csv(filepath)
    
    # Normalize to standard schema
    if "comment_text" in df.columns:
        df = df.rename(columns={"comment_text": "text"})
    if "toxic" in df.columns:
        df = df.rename(columns={"toxic": "label"})
    
    # Ensure binary labels
    df["label"] = df["label"].astype(int)
    
    return df[["text", "label"]]


def load_civil(split: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load Civil Comments dataset.
    Expected columns: text or comment_text, toxicity (float [0,1])
    Returns normalized DataFrame with columns: text, label
    """
    filepath = Path(data_dir) / f"civil_{split}.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Civil Comments {split} file not found. Expected: {filepath}\n"
            f"Please place civil_{split}.csv in {data_dir}/"
        )
    
    df = pd.read_csv(filepath)
    
    # Normalize text column
    if "comment_text" in df.columns:
        df = df.rename(columns={"comment_text": "text"})
    
    # Binarize toxicity at threshold
    if "toxicity" in df.columns:
        df["label"] = (df["toxicity"] >= CIVIL_TOXICITY_THRESHOLD).astype(int)
    elif "label" not in df.columns:
        raise ValueError(f"Civil Comments {split} missing toxicity or label column")
    
    return df[["text", "label"]]


def load_hatexplain(split: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load HateXplain dataset (stub for now).
    TODO: Implement JSON/JSONL loading and normalization
    """
    raise NotImplementedError(
        "HateXplain loader not yet implemented. "
        "Expected format: JSON/JSONL with text and label fields."
    )


def load_dataset(name: str, split: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Dispatcher function to load any supported dataset.
    
    Args:
        name: Dataset name (jigsaw, civil, hatexplain)
        split: Data split (train, val, test)
        data_dir: Directory containing data files
    
    Returns:
        DataFrame with columns: text, label
    """
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Choose from {SUPPORTED_DATASETS}")
    
    loaders = {
        "jigsaw": load_jigsaw,
        "civil": load_civil,
        "hatexplain": load_hatexplain,
    }
    
    return loaders[name](split, data_dir)


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class ToxicityDataset(Dataset):
    """PyTorch Dataset wrapper for toxicity classification."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128):
        """
        Args:
            df: DataFrame with columns: text, label
            tokenizer: Hugging Face tokenizer
            max_len: Maximum sequence length
        """
        self.texts = df["text"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# MODEL SETUP
# ============================================================================

def build_model(model_name: str = "roberta-base", num_labels: int = 2, device: str = "cuda"):
    """
    Build and initialize a transformer model for sequence classification.
    
    Args:
        model_name: Hugging Face model identifier
        num_labels: Number of classification labels
        device: Device to move model to
    
    Returns:
        Model moved to specified device
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(device)
    return model


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
) -> float:
    """
    Train model for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)


def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    return_probs: bool = False,
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        return_probs: If True, return probabilities and labels for calibration
    
    Returns:
        Dictionary with metrics (and optionally probs/labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    
    results = {
        "accuracy": accuracy,
        "f1": f1,
    }
    
    if return_probs:
        results["probs"] = all_probs
        results["labels"] = all_labels
    
    return results


# ============================================================================
# CALIBRATION STUBS (Week 11)
# ============================================================================

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    TODO: Implement NLL-based temperature scaling in Week 11.
    
    Args:
        logits: Model logits on validation set
        labels: True labels
    
    Returns:
        Optimal temperature parameter
    """
    raise NotImplementedError("Temperature scaling will be implemented in Week 11")


def fit_isotonic(probs: np.ndarray, labels: np.ndarray):
    """
    TODO: Implement sklearn isotonic regression in Week 11.
    
    Args:
        probs: Model probabilities on validation set
        labels: True labels
    
    Returns:
        Fitted isotonic regression model
    """
    raise NotImplementedError("Isotonic regression will be implemented in Week 11")


# ============================================================================
# CROSS-DOMAIN EVALUATION
# ============================================================================

def evaluate_cross_domain(
    model,
    tokenizer,
    source_dataset: str,
    target_datasets: List[str],
    batch_size: int,
    max_len: int,
    device: str,
    data_dir: str,
) -> Dict[str, Dict]:
    """
    Evaluate model trained on source dataset across target datasets.
    
    Returns:
        Dictionary mapping target dataset names to their metrics
    """
    results = {}
    
    for target in target_datasets:
        print(f"\n>>> Cross-domain evaluation: {source_dataset} → {target}")
        
        try:
            test_df = load_dataset(target, "test", data_dir)
            test_dataset = ToxicityDataset(test_df, tokenizer, max_len)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            metrics = evaluate(model, test_loader, device)
            results[target] = metrics
            
            print(f"[CROSS] source={source_dataset} → target={target} | "
                  f"F1={metrics['f1']:.4f} ACC={metrics['accuracy']:.4f}")
        
        except NotImplementedError as e:
            print(f"[CROSS] Skipping {target}: {e}")
            results[target] = {"error": str(e)}
        except Exception as e:
            print(f"[CROSS] Error evaluating {target}: {e}")
            results[target] = {"error": str(e)}
    
    return results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_and_evaluate(
    source_dataset: str,
    target_datasets: List[str],
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_len: int,
    seed: int,
    data_dir: str,
) -> Dict:
    """
    Main training and evaluation pipeline for a single seed.
    
    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*80}")
    print(f"Running with seed={seed}, source={source_dataset}")
    print(f"{'='*80}\n")
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")
    
    # Load data
    print(f"\n>>> Loading source dataset: {source_dataset} (train/val)")
    train_df = load_dataset(source_dataset, "train", data_dir)
    val_df = load_dataset(source_dataset, "val", data_dir)
    test_df = load_dataset(source_dataset, "test", data_dir)
    
    print(f"    Train: {len(train_df)} samples")
    print(f"    Val: {len(val_df)} samples")
    print(f"    Test: {len(test_df)} samples")
    
    # Setup tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = ToxicityDataset(train_df, tokenizer, max_len)
    val_dataset = ToxicityDataset(val_df, tokenizer, max_len)
    test_dataset = ToxicityDataset(test_df, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    print(f"\n>>> Building model: {model_name}")
    model = build_model(model_name, num_labels=2, device=device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    # Training loop
    print(f"\n>>> Training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, device)
        print(f"Validation - F1: {val_metrics['f1']:.4f}, ACC: {val_metrics['accuracy']:.4f}")
    
    # Final evaluation on source test set
    print(f"\n>>> Evaluating on source test set: {source_dataset}")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test - F1: {test_metrics['f1']:.4f}, ACC: {test_metrics['accuracy']:.4f}")
    
    # Save validation probabilities for calibration (Week 11)
    print(f"\n>>> Saving validation probabilities for calibration...")
    val_results = evaluate(model, val_loader, device, return_probs=True)
    
    Path(EXPERIMENTS_DIR).mkdir(exist_ok=True)
    probs_path = Path(EXPERIMENTS_DIR) / f"{source_dataset}_{model_name.replace('/', '_')}_seed{seed}_val_probs.npz"
    np.savez(
        probs_path,
        probs=val_results["probs"],
        labels=val_results["labels"],
    )
    print(f"    Saved to: {probs_path}")
    
    # Save model weights
    model_path = Path(EXPERIMENTS_DIR) / f"{source_dataset}_{model_name.replace('/', '_')}_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"    Model saved to: {model_path}")
    
    # Cross-domain evaluation
    cross_results = {}
    if target_datasets:
        print(f"\n>>> Cross-domain evaluation on: {', '.join(target_datasets)}")
        cross_results = evaluate_cross_domain(
            model, tokenizer, source_dataset, target_datasets,
            batch_size, max_len, device, data_dir
        )
    
    # Compile results
    results = {
        "seed": seed,
        "source": source_dataset,
        "in_domain": {
            "val": {"f1": val_results["f1"], "accuracy": val_results["accuracy"]},
            "test": {"f1": test_metrics["f1"], "accuracy": test_metrics["accuracy"]},
        },
        "cross_domain": cross_results,
    }
    
    return results


# ============================================================================
# MULTI-SEED RUNNER
# ============================================================================

def run_multi_seed(
    seeds: List[int],
    source_dataset: str,
    target_datasets: List[str],
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_len: int,
    data_dir: str,
) -> List[Dict]:
    """
    Run training and evaluation for multiple seeds.
    
    Returns:
        List of result dictionaries, one per seed
    """
    all_results = []
    
    for seed in seeds:
        results = train_and_evaluate(
            source_dataset=source_dataset,
            target_datasets=target_datasets,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_len=max_len,
            seed=seed,
            data_dir=data_dir,
        )
        all_results.append(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Seed':<10} {'In-Domain F1':<15} {'In-Domain ACC':<15}", end="")
    if target_datasets:
        for target in target_datasets:
            print(f" {target} F1", end="")
    print()
    print("-" * 80)
    
    for result in all_results:
        seed = result["seed"]
        in_f1 = result["in_domain"]["test"]["f1"]
        in_acc = result["in_domain"]["test"]["accuracy"]
        print(f"{seed:<10} {in_f1:<15.4f} {in_acc:<15.4f}", end="")
        
        for target in target_datasets:
            if target in result["cross_domain"] and "f1" in result["cross_domain"][target]:
                cross_f1 = result["cross_domain"][target]["f1"]
                print(f" {cross_f1:.4f}", end="")
            else:
                print(f" N/A", end="")
        print()
    
    # Compute averages
    avg_in_f1 = np.mean([r["in_domain"]["test"]["f1"] for r in all_results])
    avg_in_acc = np.mean([r["in_domain"]["test"]["accuracy"] for r in all_results])
    print("-" * 80)
    print(f"{'AVERAGE':<10} {avg_in_f1:<15.4f} {avg_in_acc:<15.4f}", end="")
    
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


# ============================================================================
# CLI AND MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BiasBreakers: Cross-domain toxicity detection"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train"],
        default="train",
        help="Execution mode",
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
        "--model_name",
        type=str,
        default="roberta-base",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length",
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
        default=DATA_DIR,
        help="Directory containing data files",
    )
    
    args = parser.parse_args()
    
    # Determine which seeds to use
    seeds = args.seeds if args.seeds else [args.seed]
    
    print(f"\nBiasBreakers - Cross-domain Toxicity Detection")
    print(f"Source: {args.source_dataset}")
    print(f"Targets: {args.target_datasets if args.target_datasets else 'None'}")
    print(f"Model: {args.model_name}")
    print(f"Seeds: {seeds}\n")
    
    # Run training and evaluation
    if len(seeds) > 1:
        run_multi_seed(
            seeds=seeds,
            source_dataset=args.source_dataset,
            target_datasets=args.target_datasets,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            data_dir=args.data_dir,
        )
    else:
        train_and_evaluate(
            source_dataset=args.source_dataset,
            target_datasets=args.target_datasets,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            seed=seeds[0],
            data_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()
