#!/usr/bin/env python3
"""
Process Raw Data from raw_data/ to data/ directory.
Converts raw datasets to the format expected by the experiment scripts.

Expects:
- raw_data/civil/civil_train.csv, civil_validation.csv, civil_test.csv
- raw_data/hatexplain/dataset.json
- raw_data/jigsaw/train.csv (if available)
"""

import os
import re
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = PROJECT_DIR / "raw_data"

SEED = 42

# Identity columns for fairness analysis
IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "physical_disability", "intellectual_or_learning_disability", 
    "psychiatric_or_mental_illness", "other_disability"
]

# Toxicity threshold for binarization
TOXICITY_THRESHOLD = 0.5


# ============================================================================
# TEXT CLEANING
# ============================================================================

URL_RE = re.compile(r"http\S+")
AT_RE = re.compile(r"@\w+")

def clean_text(s) -> str:
    """Clean text by normalizing URLs, mentions, and whitespace."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = URL_RE.sub(" URL ", s)
    s = AT_RE.sub("@USER", s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================================
# CIVIL COMMENTS PROCESSING
# ============================================================================

def process_civil_comments():
    """Process Civil Comments from raw_data/civil/ to data/"""
    print("\n" + "="*80)
    print("PROCESSING CIVIL COMMENTS")
    print("="*80)
    
    civil_dir = RAW_DIR / "civil"
    
    if not civil_dir.exists():
        print(f"ERROR: {civil_dir} not found")
        return False
    
    # Map of raw file names to output split names
    file_mapping = {
        "civil_train.csv": "train",
        "civil_validation.csv": "val", 
        "civil_test.csv": "test"
    }
    
    for raw_file, split_name in file_mapping.items():
        raw_path = civil_dir / raw_file
        
        if not raw_path.exists():
            print(f"WARNING: {raw_path} not found, skipping {split_name}")
            continue
        
        print(f"\n>>> Processing {raw_file} -> {split_name}")
        
        # Load raw data
        df_raw = pd.read_csv(raw_path)
        print(f"    Loaded {len(df_raw)} records")
        print(f"    Columns: {list(df_raw.columns)}")
        
        # Determine text column name
        text_col = "text" if "text" in df_raw.columns else "comment_text"
        
        # Determine toxicity column
        if "toxicity" in df_raw.columns:
            tox_col = "toxicity"
        elif "target" in df_raw.columns:
            tox_col = "target"
        else:
            print(f"ERROR: No toxicity/target column found")
            continue
        
        # Create processed dataframe
        df = pd.DataFrame({
            "id": range(len(df_raw)),
            "text": df_raw[text_col].apply(clean_text),
            "toxicity": df_raw[tox_col].fillna(0),
            "label": (df_raw[tox_col].fillna(0) >= TOXICITY_THRESHOLD).astype(int)
        })
        
        # Add identity columns if available
        available_id_cols = [c for c in IDENTITY_COLS if c in df_raw.columns]
        for c in available_id_cols:
            df[f"g_{c}"] = (df_raw[c].fillna(0) >= TOXICITY_THRESHOLD).astype(int)
        
        # Remove empty texts and duplicates
        df = df[df["text"].str.len() > 0]
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df["id"] = range(len(df))  # Re-index after dedup
        
        print(f"    After cleaning: {len(df)} records")
        print(f"    Positive rate: {df['label'].mean():.1%}")
        print(f"    Identity columns: {len(available_id_cols)}")
        
        # Save basic version (text, label, toxicity)
        basic_path = DATA_DIR / f"civil_{split_name}.csv"
        df[["text", "label", "toxicity"]].to_csv(basic_path, index=False)
        print(f"    Saved basic: {basic_path}")
        
        # Save full version (with id and identity groups)
        full_cols = ["id", "text", "label", "toxicity"] + [f"g_{c}" for c in available_id_cols]
        full_path = DATA_DIR / f"civil_{split_name}_full.csv"
        df[full_cols].to_csv(full_path, index=False)
        print(f"    Saved full: {full_path}")
    
    print("\nCivil Comments processing complete!")
    return True


# ============================================================================
# HATEXPLAIN PROCESSING
# ============================================================================

def process_hatexplain():
    """Process HateXplain from raw_data/hatexplain/dataset.json to data/"""
    print("\n" + "="*80)
    print("PROCESSING HATEXPLAIN")
    print("="*80)
    
    hatex_dir = RAW_DIR / "hatexplain"
    json_path = hatex_dir / "dataset.json"
    
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return False
    
    print(f">>> Loading {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"    Total records: {len(data)}")
    
    # Parse records
    rows = []
    for post_id, record in data.items():
        # Get text from tokens
        tokens = record.get("post_tokens", [])
        text = " ".join(tokens)
        text = clean_text(text)
        
        if not text:
            continue
        
        # Get majority label from annotators
        labels = [a.get("label", "normal") for a in record.get("annotators", [])]
        if labels:
            majority_label = Counter(labels).most_common(1)[0][0]
        else:
            majority_label = "normal"
        
        # Binary label: hatespeech/offensive -> 1, normal -> 0
        label = 1 if majority_label.lower() in ["hatespeech", "offensive"] else 0
        
        # Get original split if available
        split = record.get("split", None)
        
        rows.append({
            "id": post_id,
            "text": text,
            "label": label,
            "original_label": majority_label,
            "split": split
        })
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    print(f"    After parsing: {len(df)} records")
    print(f"    Label distribution:")
    print(f"      - hatespeech/offensive (1): {(df['label']==1).sum()}")
    print(f"      - normal (0): {(df['label']==0).sum()}")
    
    # Check if original splits exist
    if df["split"].notna().any() and df["split"].nunique() > 1:
        print("\n>>> Using original train/val/test splits")
        splits = df["split"].unique()
        print(f"    Found splits: {list(splits)}")
        
        for split in splits:
            split_df = df[df["split"] == split].copy()
            
            # Normalize split name
            out_split = split.lower()
            if out_split == "validation":
                out_split = "val"
            
            _save_hatexplain_split(split_df, out_split)
    else:
        print("\n>>> Creating stratified train/val/test splits")
        # Stratified split
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, temp_idx = next(sss1.split(df, df["label"]))
        
        temp_df = df.iloc[temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
        val_rel_idx, test_rel_idx = next(sss2.split(temp_df, temp_df["label"]))
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[temp_idx[val_rel_idx]]
        test_df = df.iloc[temp_idx[test_rel_idx]]
        
        _save_hatexplain_split(train_df, "train")
        _save_hatexplain_split(val_df, "val")
        _save_hatexplain_split(test_df, "test")
    
    print("\nHateXplain processing complete!")
    return True


def _save_hatexplain_split(df: pd.DataFrame, split_name: str):
    """Save a HateXplain split to CSV and JSONL."""
    df = df.reset_index(drop=True)
    
    print(f"\n>>> Saving {split_name} split: {len(df)} samples, {df['label'].mean():.1%} positive")
    
    # Save CSV
    csv_path = DATA_DIR / f"hatexplain_{split_name}.csv"
    df[["text", "label"]].to_csv(csv_path, index=False)
    print(f"    CSV: {csv_path}")
    
    # Save JSONL (for compatibility with loader)
    jsonl_path = DATA_DIR / f"hatexplain_{split_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {
                "text": row["text"],
                "label": "hatespeech" if row["label"] == 1 else "normal",
                "post_tokens": row["text"].split()
            }
            f.write(json.dumps(obj) + "\n")
    print(f"    JSONL: {jsonl_path}")


# ============================================================================
# JIGSAW PROCESSING
# ============================================================================

def process_jigsaw():
    """Process Jigsaw from raw_data/jigsaw/train.csv to data/"""
    print("\n" + "="*80)
    print("PROCESSING JIGSAW")
    print("="*80)
    
    jigsaw_dir = RAW_DIR / "jigsaw"
    train_path = jigsaw_dir / "train.csv"
    
    if not train_path.exists():
        print(f"WARNING: {train_path} not found")
        print("Jigsaw dataset requires manual download from Kaggle.")
        print("See: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data")
        return False
    
    print(f">>> Loading {train_path}")
    df_raw = pd.read_csv(train_path)
    print(f"    Loaded {len(df_raw)} records")
    
    # Find available identity columns
    available_id_cols = [c for c in IDENTITY_COLS if c in df_raw.columns]
    print(f"    Identity columns found: {len(available_id_cols)}")
    
    # Create processed dataframe
    df = pd.DataFrame({
        "id": df_raw["id"],
        "text": df_raw["comment_text"].apply(clean_text),
        "label": (df_raw["target"] >= TOXICITY_THRESHOLD).astype(int)
    })
    
    # Add identity columns
    for c in available_id_cols:
        df[f"g_{c}"] = (df_raw[c].fillna(0) >= TOXICITY_THRESHOLD).astype(int)
    
    # Remove empty and duplicate texts
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    print(f"    After cleaning: {len(df)} records")
    print(f"    Positive rate: {df['label'].mean():.1%}")
    
    # Stratified split 80/10/10
    print("\n>>> Creating stratified train/val/test splits")
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(df, df["label"]))
    
    temp_df = df.iloc[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_rel_idx, test_rel_idx = next(sss2.split(temp_df, temp_df["label"]))
    
    splits = {
        "train": df.iloc[train_idx],
        "val": df.iloc[temp_idx[val_rel_idx]],
        "test": df.iloc[temp_idx[test_rel_idx]]
    }
    
    for split_name, split_df in splits.items():
        split_df = split_df.reset_index(drop=True)
        
        print(f"\n>>> {split_name}: {len(split_df)} samples, {split_df['label'].mean():.1%} positive")
        
        # Save basic version
        basic_path = DATA_DIR / f"jigsaw_{split_name}.csv"
        split_df[["text", "label"]].to_csv(basic_path, index=False)
        print(f"    Basic: {basic_path}")
        
        # Save full version
        full_cols = ["id", "text", "label"] + [f"g_{c}" for c in available_id_cols]
        full_path = DATA_DIR / f"jigsaw_{split_name}_full.csv"
        split_df[full_cols].to_csv(full_path, index=False)
        print(f"    Full: {full_path}")
    
    print("\nJigsaw processing complete!")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process raw datasets from raw_data/ to data/"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["civil", "hatexplain", "jigsaw", "all"],
        default=["all"],
        help="Which datasets to process"
    )
    
    args = parser.parse_args()
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["civil", "hatexplain", "jigsaw"]
    
    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RAW DATA PROCESSOR")
    print("="*80)
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Output directory: {DATA_DIR}")
    print(f"Datasets to process: {datasets}")
    
    results = {}
    
    for ds in datasets:
        if ds == "civil":
            results["civil"] = process_civil_comments()
        elif ds == "hatexplain":
            results["hatexplain"] = process_hatexplain()
        elif ds == "jigsaw":
            results["jigsaw"] = process_jigsaw()
    
    # Save protocols.json
    protocols = {
        "datasets": [ds for ds, success in results.items() if success],
        "splits": ["train", "val", "test"],
        "processed_from_raw": True,
        "toxicity_threshold": TOXICITY_THRESHOLD,
        "seed": SEED
    }
    protocols_path = DATA_DIR / "protocols.json"
    with open(protocols_path, "w") as f:
        json.dump(protocols, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print("\nResults:")
    for ds, success in results.items():
        status = "✓ Success" if success else "✗ Failed/Skipped"
        print(f"  {ds}: {status}")
    
    print("\nGenerated files:")
    for f in sorted(DATA_DIR.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name:40} {size:>12,} bytes")


if __name__ == "__main__":
    main()
