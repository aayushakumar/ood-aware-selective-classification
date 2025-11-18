#!/usr/bin/env python3
"""
BiasBreakers - CS 483 Course Project
Task A: Cross-domain toxicity detection with RoBERTa
Goal: Train toxicity classifiers and evaluate cross-domain generalization
"""

import argparse
import json
import os
import random
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss,
    confusion_matrix
)
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
    Load HateXplain dataset.
    Expected file: hatexplain_{split}.jsonl or .csv
    Map {hatespeech, offensive} -> 1, else -> 0
    """
    # Try JSONL first
    jsonl_path = Path(data_dir) / f"hatexplain_{split}.jsonl"
    csv_path = Path(data_dir) / f"hatexplain_{split}.csv"
    
    # Try CSV format first (most common after preprocessing)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "text" in df.columns and "label" in df.columns:
            return df[["text", "label"]]
    
    # Try JSONL format
    if jsonl_path.exists():
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    # Extract text
                    text = obj.get("text") or " ".join(obj.get("post_tokens", []))
                    # Extract label
                    lab = str(obj.get("label", "")).lower()
                    y = 1 if lab in {"hatespeech", "offensive", "offensive_language", "hate"} else 0
                    if text:
                        rows.append({"text": text, "label": y})
                except json.JSONDecodeError:
                    continue
        
        if rows:
            df = pd.DataFrame(rows)
            df["label"] = df["label"].astype(int)
            return df[["text", "label"]]
    
    # If neither exists, raise clear error
    raise FileNotFoundError(
        f"HateXplain {split} not found. Tried:\n"
        f"  - {csv_path}\n"
        f"  - {jsonl_path}\n"
        f"Please run hatexplaindata.ipynb preprocessing first."
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
# MODEL SETUP (+ PEFT)
# ============================================================================

def maybe_apply_peft(model, peft_method: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    """Wrap model with LoRA (PEFT)."""
    if peft_method is None or peft_method.lower() == "none":
        return model
    if peft_method.lower() == "lora":
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except Exception as e:
            raise RuntimeError(
                "PEFT/LoRA requested but 'peft' is not available. "
                "Install 'peft' or set --peft none."
            ) from e
        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"],  # cover RoBERTa attn/ffn
        )
        model = get_peft_model(model, cfg)
        return model
    else:
        raise ValueError(f"Unknown PEFT method: {peft_method}")


def build_model(
    model_name: str = "roberta-base",
    num_labels: int = 2,
    device: str = "cuda",
    peft_method: str = "none",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """
    Build and initialize a transformer model for sequence classification.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    # apply PEFT if requested
    model = maybe_apply_peft(model, peft_method, lora_r, lora_alpha, lora_dropout)
    model.to(device)
    return model


# ============================================================================
# TRAINING AND EVALUATION (+ CALIBRATION, + CORAL, + AMP, + THRESHOLD)
# ============================================================================

def coral_loss(source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
    """CORAL loss between two feature batches (Frobenius norm of covariance diff)."""
    def _cov(x):
        x = x - x.mean(dim=0, keepdim=True)
        n = x.size(0) - 1
        cov = (x.t() @ x) / max(n, 1)
        return cov
    cs = _cov(source_feats)
    ct = _cov(target_feats)
    d = source_feats.size(1)
    return torch.mean((cs - ct) ** 2) / (4.0 * (d ** 2))

def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    coral_target_iter: Optional[iter] = None,
    coral_lambda: float = 0.0,
    use_amp: bool = False,                       
    coral_target_loader: Optional[DataLoader] = None,  # for iterator reset
    scaler: Optional[torch.cuda.amp.GradScaler] = None 
) -> float:
    """
    Train model for one epoch.
    """
    model.train()
    total_loss = 0.0
    use_coral = (coral_target_iter is not None) and (coral_lambda > 0.0)

    autocast = torch.cuda.amp.autocast if (use_amp and device.startswith("cuda")) else torch.cpu.amp.autocast

    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=use_amp):  # mixed precision
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=use_coral,   # only when CORAL
            )
            loss = outputs.loss

            if use_coral:
                # fetch a target batch (unlabeled)
                try:
                    tgt = next(coral_target_iter)
                except StopIteration:
                    # reset with target loader, not source loader
                    if coral_target_loader is not None:
                        coral_target_iter = iter(coral_target_loader)
                        tgt = next(coral_target_iter)
                    else:
                        coral_target_iter = iter(dataloader)
                        tgt = next(coral_target_iter)
                tgt_ids = tgt["input_ids"].to(device)
                tgt_mask = tgt["attention_mask"].to(device)

                # forward target to get hidden states
                with torch.no_grad():
                    tgt_out = model(
                        input_ids=tgt_ids,
                        attention_mask=tgt_mask,
                        output_hidden_states=True,
                    )
                # CLS features from last hidden layer
                src_feat = outputs.hidden_states[-1][:, 0, :]
                tgt_feat = tgt_out.hidden_states[-1][:, 0, :]
                loss = loss + coral_lambda * coral_loss(src_feat, tgt_feat)

        if use_amp and device.startswith("cuda"):  
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def apply_calibration_from_logits(
    logits: torch.Tensor,
    temperature: Optional[float] = None,
    iso_model: Optional[object] = None,
) -> torch.Tensor:
    """Apply temperature scaling (on logits) and/or isotonic (on pos prob)."""
    if temperature is not None:
        logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    if iso_model is not None:
        pos = probs[:, 1].detach().cpu().numpy()
        cal = iso_model.predict(pos)
        cal = np.clip(cal, 0.0, 1.0)
        probs = torch.from_numpy(
            np.stack([1.0 - cal, cal], axis=-1)
        ).to(probs.device).type_as(probs)
    return probs


def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    return_probs: bool = False,
    temperature: Optional[float] = None,
    isotonic_model: Optional[object] = None,
    threshold: Optional[float] = None,               # decision threshold on pos prob
) -> Dict:
    """
    Evaluate model on a dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

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
            all_logits.append(logits.detach().cpu().numpy())

            # calibrated probs
            probs = apply_calibration_from_logits(logits, temperature, isotonic_model)
            pos = probs[:, 1]

            if threshold is not None:  
                preds = (pos >= threshold).long()
            else:
                preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_logits = np.vstack(all_logits)
    pos_probs = all_probs[:, 1]

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
    }

    # extra metrics
    try:
        metrics["auroc"] = roc_auc_score(all_labels, pos_probs)
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["nll"] = log_loss(all_labels, all_probs, labels=[0, 1])
    except Exception:
        metrics["nll"] = float("nan")
    try:
        metrics["brier"] = brier_score_loss(all_labels, pos_probs)
    except Exception:
        metrics["brier"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    
    ece, bin_stats = expected_calibration_error(all_labels, pos_probs)  
    metrics["ece"] = ece

    if return_probs:
        metrics["probs"] = all_probs
        metrics["pos_probs"] = pos_probs
        metrics["logits"] = all_logits
        metrics["labels"] = all_labels
        metrics["bin_stats"] = bin_stats  # for reliability plotting later

    if threshold is not None:
        metrics["threshold"] = threshold

    return metrics


# ============================================================================
# CALIBRATION / ECE UTILITIES
# ============================================================================

def expected_calibration_error(y_true: np.ndarray, pos_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, pd.DataFrame]:
    """Compute ECE (Expected Calibration Error) with equal-width bins."""
    y_true = np.asarray(y_true)
    pos_prob = np.asarray(pos_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(pos_prob, bins) - 1
    ece = 0.0
    rows = []
    for b in range(n_bins):
        mask = (inds == b)
        if not np.any(mask):
            rows.append([b, bins[b], bins[b+1], 0, np.nan, np.nan, 0.0])
            continue
        conf = pos_prob[mask].mean()
        acc = (y_true[mask] == (pos_prob[mask] >= 0.5)).mean()
        frac = mask.mean()
        gap = abs(acc - conf)
        ece += frac * gap
        rows.append([b, bins[b], bins[b+1], mask.sum(), acc, conf, gap])
    df = pd.DataFrame(rows, columns=["bin", "left", "right", "count", "acc", "conf", "gap"])
    return float(ece), df


# ============================================================================
# CALIBRATION (Implemented)
# ============================================================================

def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 1000, lr: float = 0.01) -> float:
    """
    NLL-based temperature scaling on validation logits.
    Returns optimal temperature > 0 (initialized as 1.0).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.ones(1, device=device, requires_grad=True)  # temperature parameter
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([t], max_iter=50, line_search_fn="strong_wolfe")

    x = torch.from_numpy(logits).to(device).float()
    y = torch.from_numpy(labels).to(device).long()

    def closure():
        opt.zero_grad()
        scaled = x / t.clamp(min=1e-6)
        loss = criterion(scaled, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = t.detach().cpu().item()
    return max(T, 1e-6)


def fit_isotonic(probs: np.ndarray, labels: np.ndarray):
    """
    sklearn Isotonic Regression on positive-class probabilities.
    Returns fitted model (pickle-able).
    """
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    return iso


# ============================================================================
# THRESHOLD TUNING (on validation)
# ============================================================================

def tune_threshold_for_f1(y_true: np.ndarray, pos_prob: np.ndarray) -> float:
    """Find threshold in [0,1] that maximizes F1 on validation."""
    # scan 1001 points (step 0.001) — cheap on CPU
    grid = np.linspace(0.0, 1.0, 1001)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (pos_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


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
    temperature: Optional[float] = None,
    isotonic_model: Optional[object] = None,
    threshold: Optional[float] = None,  
    save_preds: bool = False,          
) -> Dict[str, Dict]:
    """
    Evaluate model trained on source dataset across target datasets.
    """
    results = {}
    
    for target in target_datasets:
        print(f"\n>>> Cross-domain evaluation: {source_dataset} → {target}")
        
        try:
            test_df = load_dataset(target, "test", data_dir)
            test_dataset = ToxicityDataset(test_df, tokenizer, max_len)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            metrics = evaluate(
                model, test_loader, device,
                temperature=temperature, isotonic_model=isotonic_model,
                threshold=threshold
            )
            results[target] = metrics
            
            print(f"[CROSS] source={source_dataset} → target={target} | "
                  f"F1={metrics['f1']:.4f} ACC={metrics['accuracy']:.4f} AUROC={metrics.get('auroc', float('nan')):.4f}")
            
            # optional prediction dump
            if save_preds:
                probs = []
                labels = []
                texts = test_df["text"].tolist()
                # quick pass to get probs with calibration
                model.eval()
                with torch.no_grad():
                    for batch in DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
                        logits = model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                        ).logits
                        p = apply_calibration_from_logits(logits, temperature, isotonic_model)[:, 1].cpu().numpy()
                        probs.extend(p)
                        labels.extend(batch["labels"].numpy())
                preds = (np.array(probs) >= (0.5 if threshold is None else threshold)).astype(int)
                out = pd.DataFrame({"text": texts, "label": labels, "pos_prob": probs, "pred": preds})
                Path(EXPERIMENTS_DIR).mkdir(exist_ok=True)
                out_path = Path(EXPERIMENTS_DIR) / f"preds_{source_dataset}_to_{target}.csv"
                out.to_csv(out_path, index=False, encoding="utf-8")
                print(f"    Saved predictions to: {out_path}")
        
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
    calibration: str = "none",
    peft_method: str = "none",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    coral_target: Optional[str] = None,
    coral_lambda: float = 0.0,
    use_amp: bool = False,                    
    weighted_sampler: bool = False,           
    early_stop: bool = False,                 
    patience: int = 3,                      
    metric_for_es: str = "f1",                
    tune_threshold: bool = False,             
    save_preds: bool = False,                 
) -> Dict:
    """
    Main training and evaluation pipeline for a single seed.
    """
    print(f"\n{'='*80}")
    print(f"Running with seed={seed}, source={source_dataset}")
    print(f"{'='*80}\n")
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")
    
    # Load data
    print(f"\n>>> Loading source dataset: {source_dataset} (train/val/test)")
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

    # optional weighted sampler for imbalance
    train_loader_kwargs = {"batch_size": batch_size}
    if weighted_sampler:
        cls_counts = train_df["label"].value_counts().to_dict()
        w_map = {c: 1.0 / max(1, n) for c, n in cls_counts.items()}
        weights = train_df["label"].map(w_map).values.astype("float32")
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, **train_loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_kwargs)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # CORAL target data (unlabeled)
    coral_iter = None
    coral_loader = None
    if coral_target:
        try:
            coral_df = load_dataset(coral_target, "train", data_dir)
            coral_dataset = ToxicityDataset(coral_df, tokenizer, max_len)
            coral_loader = DataLoader(coral_dataset, batch_size=batch_size, shuffle=True)
            coral_iter = iter(coral_loader)
            print(f">>> CORAL target loaded: {coral_target} ({len(coral_df)} samples)")
        except Exception as e:
            print(f"[WARN] Failed to prepare CORAL target '{coral_target}': {e}")
            coral_iter = None

    # Build model (with optional PEFT)
    print(f"\n>>> Building model: {model_name} (peft={peft_method})")
    model = build_model(
        model_name, num_labels=2, device=device,
        peft_method=peft_method, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))

    # Training loop (+ early stopping/best ckpt)
    print(f"\n>>> Training for {epochs} epochs...")
    best_score = float("-inf") if metric_for_es == "f1" else float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            coral_target_iter=coral_iter, coral_lambda=coral_lambda,
            use_amp=use_amp, coral_target_loader=coral_loader, scaler=scaler
        )
        print(f"Training loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, device)
        print(f"Validation - F1: {val_metrics['f1']:.4f}, ACC: {val_metrics['accuracy']:.4f}, AUROC: {val_metrics.get('auroc', float('nan')):.4f}, ECE: {val_metrics.get('ece', float('nan')):.4f}")

        # early stopping 
        if early_stop:
            score = val_metrics["f1"] if metric_for_es == "f1" else val_metrics.get("nll", np.inf)
            is_better = (score > best_score) if metric_for_es == "f1" else (score < best_score)
            if is_better:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f">>> Early stopping at epoch {epoch+1}. Best {metric_for_es} = {best_score:.4f}")
                    break
    
    # Save model weights (last)
    Path(EXPERIMENTS_DIR).mkdir(exist_ok=True)
    model_path = Path(EXPERIMENTS_DIR) / f"{source_dataset}_{model_name.replace('/', '_')}_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n>>> Model saved to: {model_path}")

    # load best state if early stopping used
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
        best_path = Path(EXPERIMENTS_DIR) / f"{source_dataset}_{model_name.replace('/', '_')}_seed{seed}_best.pt"
        torch.save(model.state_dict(), best_path)
        print(f">>> Best checkpoint saved to: {best_path}")
    
    # Collect validation outputs for calibration / threshold tuning
    print(f"\n>>> Gathering validation outputs for calibration...")
    val_results = evaluate(model, val_loader, device, return_probs=True)
    probs_path = Path(EXPERIMENTS_DIR) / f"{source_dataset}_{model_name.replace('/', '_')}_seed{seed}_val_probs.npz"
    np.savez(
        probs_path,
        probs=val_results["probs"],
        pos_probs=val_results["pos_probs"],
        logits=val_results["logits"],
        labels=val_results["labels"],
    )
    # save reliability bins for val
    val_bins_path = Path(EXPERIMENTS_DIR) / f"{source_dataset}_val_reliability.csv"
    val_results["bin_stats"].to_csv(val_bins_path, index=False)
    print(f"    Saved val outputs to: {probs_path}")
    print(f"    Saved val reliability bins to: {val_bins_path}")
    
    # Fit calibration on validation set 
    temperature = None
    isotonic_model = None
    if calibration and calibration.lower() != "none":
        y_val = val_results["labels"]
        if calibration.lower() == "temperature":
            print(">>> Fitting temperature scaling on validation logits ...")
            temperature = fit_temperature(val_results["logits"], y_val)
            with open(Path(EXPERIMENTS_DIR) / f"{source_dataset}_temp_{seed}.txt", "w") as f:
                f.write(str(temperature))
            print(f"    Learned temperature: {temperature:.4f}")
        elif calibration.lower() == "isotonic":
            print(">>> Fitting isotonic regression on validation pos_probs ...")
            isotonic_model = fit_isotonic(val_results["pos_probs"], y_val)
            with open(Path(EXPERIMENTS_DIR) / f"{source_dataset}_iso_{seed}.pkl", "wb") as f:
                pickle.dump(isotonic_model, f)
            print(f"    Isotonic model saved.")
        else:
            print(f"[WARN] Unknown calibration method: {calibration}. Skip.")

    # optional threshold tuning on calibrated val probs
    tuned_threshold = None
    if tune_threshold:
        tuned_threshold = tune_threshold_for_f1(val_results["labels"], val_results["pos_probs"])
        with open(Path(EXPERIMENTS_DIR) / f"{source_dataset}_thr_{seed}.txt", "w") as f:
            f.write(str(tuned_threshold))
        print(f">>> Tuned decision threshold on val for F1: {tuned_threshold:.3f}")
    
    # Final evaluation on source test set (with calibration/threshold if any)
    print(f"\n>>> Evaluating on source test set: {source_dataset}")
    test_metrics = evaluate(
        model, test_loader, device,
        temperature=temperature, isotonic_model=isotonic_model,
        threshold=tuned_threshold
    )
    print(f"Test - F1: {test_metrics['f1']:.4f}, ACC: {test_metrics['accuracy']:.4f}, AUROC: {test_metrics.get('auroc', float('nan')):.4f}, ECE: {test_metrics.get('ece', float('nan')):.4f}")

    # dump test predictions if needed
    if save_preds:
        # calibrated pos prob with optional threshold
        probs, labels, texts = [], [], test_df["text"].tolist()
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                p = apply_calibration_from_logits(logits, temperature, isotonic_model)[:, 1].cpu().numpy()
                probs.extend(p); labels.extend(batch["labels"].numpy())
        preds = (np.array(probs) >= (0.5 if tuned_threshold is None else tuned_threshold)).astype(int)
        out = pd.DataFrame({"text": texts, "label": labels, "pos_prob": probs, "pred": preds})
        out_path = Path(EXPERIMENTS_DIR) / f"preds_{source_dataset}_test.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"    Saved test predictions to: {out_path}")
    
    # Cross-domain evaluation (with calibration/threshold if any)
    cross_results = {}
    if target_datasets:
        print(f"\n>>> Cross-domain evaluation on: {', '.join(target_datasets)}")
        cross_results = evaluate_cross_domain(
            model, tokenizer, source_dataset, target_datasets,
            batch_size, max_len, device, data_dir,
            temperature=temperature, isotonic_model=isotonic_model,
            threshold=tuned_threshold, save_preds=save_preds
        )

    # Save summary CSV
    summary_rows = []
    in_row = {"split": "in_domain_test", **test_metrics}
    summary_rows.append(in_row)
    for tgt, met in cross_results.items():
        if "error" not in met:
            summary_rows.append({"split": f"cross_{tgt}", **met})
    summary = pd.DataFrame(summary_rows)
    sum_path = Path(EXPERIMENTS_DIR) / f"summary_{source_dataset}.csv"
    summary.to_csv(sum_path, index=False)
    print(f">>> Saved summary metrics to: {sum_path}")
    
    # Compile results
    results = {
        "seed": seed,
        "source": source_dataset,
        "in_domain": {
            "val": {"f1": val_results["probs"].shape[0] and f1_score(val_results["labels"], (val_results['pos_probs']>=0.5).astype(int))},
            "test": {"f1": test_metrics["f1"], "accuracy": test_metrics["accuracy"]},
        },
        "cross_domain": cross_results,
        "calibration": {
            "method": calibration,
            "temperature": temperature,
            "has_isotonic": isotonic_model is not None,
        },
        "coral": {
            "target": coral_target,
            "lambda": coral_lambda,
        },
        "peft": peft_method,
        "amp": use_amp,
        "weighted_sampler": weighted_sampler,
        "tuned_threshold": tuned_threshold,
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
    calibration: str = "none",
    peft_method: str = "none",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    coral_target: Optional[str] = None,
    coral_lambda: float = 0.0,
    use_amp: bool = False,                  
    weighted_sampler: bool = False,         
    early_stop: bool = False,              
    patience: int = 3,                      
    metric_for_es: str = "f1",              
    tune_threshold: bool = False,           
    save_preds: bool = False,               
) -> List[Dict]:
    """
    Run training and evaluation for multiple seeds.
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
            calibration=calibration,
            peft_method=peft_method,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            coral_target=coral_target, coral_lambda=coral_lambda,
            use_amp=use_amp,
            weighted_sampler=weighted_sampler,
            early_stop=early_stop, patience=patience, metric_for_es=metric_for_es,
            tune_threshold=tune_threshold,
            save_preds=save_preds,
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
        help="Hugging Face model identifier (or local path)",
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
    # calibration
    parser.add_argument(
        "--calibration",
        type=str,
        choices=["none", "temperature", "isotonic"],
        default="none",
        help="Post-hoc calibration method trained on validation set",
    )
    # PEFT / LoRA
    parser.add_argument(
        "--peft",
        type=str,
        choices=["none", "lora"],
        default="none",
        help="Enable parameter-efficient fine-tuning via LoRA",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # CORAL
    parser.add_argument(
        "--coral_target",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Unlabeled target dataset used for CORAL alignment during training",
    )
    parser.add_argument(
        "--coral_lambda",
        type=float,
        default=0.0,
        help="Weight of CORAL loss (0 to disable)",
    )
    # misc options
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP)")
    parser.add_argument("--weighted_sampler", action="store_true", help="Use WeightedRandomSampler for class imbalance")
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping on validation")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--metric_for_es", type=str, choices=["f1", "loss"], default="f1", help="Metric to monitor for early stopping")
    parser.add_argument("--tune_threshold", action="store_true", help="Tune decision threshold on validation for best F1")
    parser.add_argument("--save_preds", action="store_true", help="Save CSV predictions for test and cross-domain")
    
    args = parser.parse_args()
    
    # ADDED: Validation checks
    print("\n" + "="*80)
    print("CONFIGURATION VALIDATION")
    print("="*80)
    
    # Check 1: Data directory exists
    if not Path(args.data_dir).exists():
        print(f"⚠️  WARNING: Data directory not found: {args.data_dir}")
        print("Creating directory...")
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Check 2: Required data files exist
    required_files = [
        f"{args.source_dataset}_train.csv",
        f"{args.source_dataset}_val.csv",
        f"{args.source_dataset}_test.csv"
    ]
    
    missing_files = []
    for fname in required_files:
        fpath = Path(args.data_dir) / fname
        if not fpath.exists():
            missing_files.append(fname)
    
    if missing_files:
        print(f"\n⚠️  ERROR: Missing required data files:")
        for fname in missing_files:
            print(f"  - {fname}")
        print(f"\nPlease run preprocessing notebooks first:")
        print(f"  - cs483_data.ipynb (for jigsaw)")
        print(f"  - civildata.ipynb (for civil)")
        print(f"  - hatexplaindata.ipynb (for hatexplain)")
        sys.exit(1)
    
    # Check 3: CORAL validation
    if args.coral_target and args.coral_lambda <= 0:
        print("⚠️  WARNING: CORAL target specified but lambda=0. CORAL will be disabled.")
        args.coral_target = None
    
    # Check 4: PEFT validation
    if args.peft == "lora":
        try:
            import peft
        except ImportError:
            print("⚠️  ERROR: LoRA requested but 'peft' library not installed")
            print("Install with: pip install peft")
            sys.exit(1)
    
    # Check 5: Calibration validation
    if args.calibration == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            print("⚠️  ERROR: Isotonic calibration requires sklearn")
            sys.exit(1)
    
    print("✓ All validation checks passed\n")
    
    seeds = args.seeds if args.seeds else [args.seed]    
    
    print(f"\nBiasBreakers - Cross-domain Toxicity Detection")
    print(f"Source: {args.source_dataset}")
    print(f"Targets: {args.target_datasets if args.target_datasets else 'None'}")
    print(f"Model: {args.model_name}")
    print(f"Seeds: {seeds}")
    print(f"Calibration: {args.calibration} | PEFT: {args.peft} | CORAL: target={args.coral_target}, lambda={args.coral_lambda}")
    print(f"AMP: {args.amp} | WeightedSampler: {args.weighted_sampler} | EarlyStop: {args.early_stop} | TuneThr: {args.tune_threshold} | SavePreds: {args.save_preds}\n")
    
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
            calibration=args.calibration,
            peft_method=args.peft,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            coral_target=args.coral_target, coral_lambda=args.coral_lambda,
            use_amp=args.amp,
            weighted_sampler=args.weighted_sampler,
            early_stop=args.early_stop, patience=args.patience, metric_for_es=args.metric_for_es,
            tune_threshold=args.tune_threshold,
            save_preds=args.save_preds,
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
            calibration=args.calibration,
            peft_method=args.peft,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            coral_target=args.coral_target, coral_lambda=args.coral_lambda,
            use_amp=args.amp,
            weighted_sampler=args.weighted_sampler,
            early_stop=args.early_stop, patience=args.patience, metric_for_es=args.metric_for_es,
            tune_threshold=args.tune_threshold,
            save_preds=args.save_preds,
        )


if __name__ == "__main__":
    main()
