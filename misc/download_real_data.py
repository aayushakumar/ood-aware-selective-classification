#!/usr/bin/env python3
"""
Download Real Datasets for Toxicity Classification
Downloads and preprocesses Jigsaw, Civil Comments, and HateXplain datasets.

Requirements:
- pip install kaggle datasets pandas scikit-learn

For Kaggle datasets, you need:
1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account -> Create New API Token
3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional

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

# Identity columns
IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "physical_disability", "intellectual_or_learning_disability", 
    "psychiatric_or_mental_illness", "other_disability"
]


# ============================================================================
# TEXT CLEANING
# ============================================================================

URL_RE = re.compile(r"http\S+")
AT_RE = re.compile(r"@\w+")

def clean_text(s: str) -> str:
    """Clean text by normalizing URLs, mentions, and whitespace."""
    s = str(s) if pd.notna(s) else ""
    s = URL_RE.sub(" URL ", s)
    s = AT_RE.sub("@USER", s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================================
# STRATIFIED SPLIT
# ============================================================================

def stratified_split(df: pd.DataFrame, label_col: str = "label", 
                     train_ratio: float = 0.8, val_ratio: float = 0.1,
                     seed: int = SEED):
    """
    Perform stratified 80/10/10 train/val/test split.
    """
    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-train_ratio), random_state=seed)
    train_idx, temp_idx = next(sss1.split(df, df[label_col]))
    
    # Second split: val vs test (50/50 of remaining)
    temp = df.iloc[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel_idx, test_rel_idx = next(sss2.split(temp, temp[label_col]))
    
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


# ============================================================================
# JIGSAW DATASET
# ============================================================================

def download_jigsaw():
    """Download Jigsaw Toxic Comment dataset from Kaggle."""
    print("\n" + "="*80)
    print("DOWNLOADING JIGSAW DATASET")
    print("="*80)
    
    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        return False
    
    jigsaw_dir = RAW_DIR / "jigsaw"
    jigsaw_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading from Kaggle (this may take a while)...")
    try:
        kaggle.api.competition_download_files(
            "jigsaw-unintended-bias-in-toxicity-classification",
            path=str(jigsaw_dir),
            quiet=False
        )
        
        # Unzip if needed
        zip_file = jigsaw_dir / "jigsaw-unintended-bias-in-toxicity-classification.zip"
        if zip_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(jigsaw_dir)
            zip_file.unlink()
        
        print(f"Downloaded to: {jigsaw_dir}")
        return True
    except Exception as e:
        print(f"ERROR downloading Jigsaw: {e}")
        print("\nTo download manually:")
        print("1. Go to: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data")
        print("2. Download train.csv")
        print(f"3. Place it in: {jigsaw_dir}/")
        return False


def process_jigsaw():
    """Process Jigsaw dataset into standard format."""
    print("\n>>> Processing Jigsaw dataset...")
    
    # Find train.csv
    jigsaw_dir = RAW_DIR / "jigsaw"
    train_path = jigsaw_dir / "train.csv"
    
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run with --download first.")
        return False
    
    print(f"Loading {train_path}...")
    df_raw = pd.read_csv(train_path)
    print(f"Loaded {len(df_raw)} records")
    
    # Find available identity columns
    use_id_cols = [c for c in IDENTITY_COLS if c in df_raw.columns]
    print(f"Found {len(use_id_cols)} identity columns")
    
    # Create cleaned dataset
    df = pd.DataFrame({
        "id": df_raw["id"],
        "text": df_raw["comment_text"].map(clean_text),
        "label": (df_raw["target"] >= 0.5).astype(int)
    })
    
    # Add identity columns
    for c in use_id_cols:
        df[f"g_{c}"] = (df_raw[c].fillna(0) >= 0.5).astype(int)
    
    # Deduplicate
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"After deduplication: {len(df)} records")
    
    # Split
    train_df, val_df, test_df = stratified_split(df)
    
    # Save
    id_cols = ["id"] + [f"g_{c}" for c in use_id_cols]
    
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Basic version
        basic_path = DATA_DIR / f"jigsaw_{split_name}.csv"
        split_df[["text", "label"]].to_csv(basic_path, index=False)
        
        # Full version
        full_path = DATA_DIR / f"jigsaw_{split_name}_full.csv"
        split_df[["id", "text", "label"] + [f"g_{c}" for c in use_id_cols]].to_csv(full_path, index=False)
        
        pos_rate = split_df["label"].mean()
        print(f"  {split_name}: {len(split_df)} samples, {pos_rate:.1%} positive")
    
    print("Jigsaw processing complete!")
    return True


# ============================================================================
# CIVIL COMMENTS DATASET
# ============================================================================

def download_civil():
    """Download Civil Comments dataset."""
    print("\n" + "="*80)
    print("DOWNLOADING CIVIL COMMENTS DATASET")
    print("="*80)
    
    civil_dir = RAW_DIR / "civil"
    civil_dir.mkdir(parents=True, exist_ok=True)
    
    # Try Hugging Face datasets first
    try:
        from datasets import load_dataset
        print("Downloading from Hugging Face...")
        
        ds = load_dataset("google/civil_comments", trust_remote_code=True)
        
        # Save as CSV
        for split in ["train", "validation", "test"]:
            if split in ds:
                df = ds[split].to_pandas()
                save_path = civil_dir / f"civil_{split}.csv"
                df.to_csv(save_path, index=False)
                print(f"Saved {split}: {len(df)} records")
        
        return True
    except Exception as e:
        print(f"Hugging Face download failed: {e}")
    
    # Try Kaggle
    try:
        import kaggle
        print("Trying Kaggle...")
        kaggle.api.dataset_download_files(
            "stefanoleone992/jigsaw-toxic-comment-classification-challenge",
            path=str(civil_dir),
            unzip=True
        )
        return True
    except Exception as e:
        print(f"Kaggle download failed: {e}")
    
    print("\nTo download manually:")
    print("Option 1: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('google/civil_comments')\"")
    print("Option 2: Download from https://www.tensorflow.org/datasets/catalog/civil_comments")
    print(f"Place files in: {civil_dir}/")
    return False


def process_civil():
    """Process Civil Comments dataset into standard format."""
    print("\n>>> Processing Civil Comments dataset...")
    
    civil_dir = RAW_DIR / "civil"
    
    # Find CSV files
    csv_files = list(civil_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {civil_dir}. Run with --download first.")
        return False
    
    # Try to load train/val/test splits if they exist
    splits_exist = all((civil_dir / f"civil_{s}.csv").exists() for s in ["train", "validation", "test"])
    
    if splits_exist:
        print("Found pre-split files, loading...")
        dfs = {}
        for split, filename in [("train", "civil_train.csv"), ("val", "civil_validation.csv"), ("test", "civil_test.csv")]:
            filepath = civil_dir / filename
            if filepath.exists():
                dfs[split] = pd.read_csv(filepath)
                print(f"  Loaded {split}: {len(dfs[split])} records")
    else:
        # Load single file and split
        main_file = csv_files[0]
        print(f"Loading {main_file}...")
        df_raw = pd.read_csv(main_file)
        print(f"Loaded {len(df_raw)} records")
        
        # Process and split
        use_id_cols = [c for c in IDENTITY_COLS if c in df_raw.columns]
        
        # Determine text column
        text_col = "text" if "text" in df_raw.columns else "comment_text"
        toxicity_col = "toxicity" if "toxicity" in df_raw.columns else "target"
        
        df = pd.DataFrame({
            "id": range(len(df_raw)),
            "text": df_raw[text_col].map(clean_text),
            "toxicity": df_raw[toxicity_col],
            "label": (df_raw[toxicity_col] >= 0.5).astype(int)
        })
        
        for c in use_id_cols:
            df[f"g_{c}"] = (df_raw[c].fillna(0) >= 0.5).astype(int)
        
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        
        train_df, val_df, test_df = stratified_split(df)
        dfs = {"train": train_df, "val": val_df, "test": test_df}
    
    # Process each split
    for split_name, df_raw in dfs.items():
        # Clean and prepare
        use_id_cols = [c for c in IDENTITY_COLS if c in df_raw.columns or f"g_{c}" in df_raw.columns]
        
        text_col = "text" if "text" in df_raw.columns else "comment_text"
        
        if "label" not in df_raw.columns:
            toxicity_col = "toxicity" if "toxicity" in df_raw.columns else "target"
            df_raw["label"] = (df_raw[toxicity_col] >= 0.5).astype(int)
        
        if "id" not in df_raw.columns:
            df_raw["id"] = range(len(df_raw))
        
        df = pd.DataFrame({
            "id": df_raw["id"],
            "text": df_raw[text_col].map(clean_text) if text_col in df_raw.columns else df_raw["text"],
            "label": df_raw["label"]
        })
        
        if "toxicity" in df_raw.columns:
            df["toxicity"] = df_raw["toxicity"]
        
        for c in use_id_cols:
            src_col = f"g_{c}" if f"g_{c}" in df_raw.columns else c
            if src_col in df_raw.columns:
                df[f"g_{c}"] = (df_raw[src_col].fillna(0) >= 0.5).astype(int)
        
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        
        # Save basic version
        basic_cols = ["text", "label"]
        if "toxicity" in df.columns:
            basic_cols.append("toxicity")
        basic_path = DATA_DIR / f"civil_{split_name}.csv"
        df[basic_cols].to_csv(basic_path, index=False)
        
        # Save full version
        full_path = DATA_DIR / f"civil_{split_name}_full.csv"
        df.to_csv(full_path, index=False)
        
        pos_rate = df["label"].mean()
        print(f"  {split_name}: {len(df)} samples, {pos_rate:.1%} positive")
    
    print("Civil Comments processing complete!")
    return True


# ============================================================================
# HATEXPLAIN DATASET
# ============================================================================

def download_hatexplain():
    """Download HateXplain dataset from Hugging Face."""
    print("\n" + "="*80)
    print("DOWNLOADING HATEXPLAIN DATASET")
    print("="*80)
    
    hatex_dir = RAW_DIR / "hatexplain"
    hatex_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("Downloading from Hugging Face...")
        
        ds = load_dataset("hatexplain", trust_remote_code=True)
        
        for split in ds.keys():
            df = ds[split].to_pandas()
            save_path = hatex_dir / f"hatexplain_{split}.parquet"
            df.to_parquet(save_path)
            print(f"Saved {split}: {len(df)} records")
        
        return True
    except Exception as e:
        print(f"Hugging Face download failed: {e}")
    
    # Try GitHub direct download
    print("Trying GitHub download...")
    try:
        import urllib.request
        
        url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
        save_path = hatex_dir / "dataset.json"
        
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded to: {save_path}")
        return True
    except Exception as e:
        print(f"GitHub download failed: {e}")
    
    print("\nTo download manually:")
    print("1. Go to: https://github.com/hate-alert/HateXplain")
    print("2. Download Data/dataset.json")
    print(f"3. Place it in: {hatex_dir}/")
    return False


def process_hatexplain():
    """Process HateXplain dataset into standard format."""
    print("\n>>> Processing HateXplain dataset...")
    
    hatex_dir = RAW_DIR / "hatexplain"
    
    # Try loading from different formats
    df = None
    
    # Try parquet files (from Hugging Face)
    parquet_files = list(hatex_dir.glob("*.parquet"))
    if parquet_files:
        print("Loading from parquet files...")
        dfs = {}
        for pf in parquet_files:
            split = pf.stem.replace("hatexplain_", "")
            dfs[split] = pd.read_parquet(pf)
            print(f"  Loaded {split}: {len(dfs[split])} records")
        
        # Process Hugging Face format
        for split_name, df_raw in dfs.items():
            # Map split names
            out_split = "val" if split_name == "validation" else split_name
            
            # Extract text and labels
            rows = []
            for _, row in df_raw.iterrows():
                # HuggingFace format has 'post_tokens' as list
                if "post_tokens" in row and isinstance(row["post_tokens"], list):
                    text = " ".join(row["post_tokens"])
                elif "text" in row:
                    text = row["text"]
                else:
                    continue
                
                # Get label (0=normal, 1=hatespeech, 2=offensive in HF format)
                label_val = row.get("label", 0)
                if isinstance(label_val, int):
                    label = 1 if label_val in [1, 2] else 0  # hatespeech or offensive -> toxic
                else:
                    label = 1 if str(label_val).lower() in ["hatespeech", "offensive", "hate"] else 0
                
                rows.append({"text": clean_text(text), "label": label})
            
            df = pd.DataFrame(rows)
            df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
            
            # Save CSV
            csv_path = DATA_DIR / f"hatexplain_{out_split}.csv"
            df.to_csv(csv_path, index=False)
            
            # Save JSONL
            jsonl_path = DATA_DIR / f"hatexplain_{out_split}.jsonl"
            with open(jsonl_path, "w") as f:
                for _, row in df.iterrows():
                    obj = {
                        "text": row["text"],
                        "label": "hatespeech" if row["label"] == 1 else "normal",
                        "post_tokens": row["text"].split()
                    }
                    f.write(json.dumps(obj) + "\n")
            
            pos_rate = df["label"].mean()
            print(f"  {out_split}: {len(df)} samples, {pos_rate:.1%} positive")
        
        print("HateXplain processing complete!")
        return True
    
    # Try JSON file
    json_files = list(hatex_dir.glob("*.json"))
    if json_files:
        json_file = json_files[0]
        print(f"Loading from {json_file}...")
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # HateXplain JSON format: {id: {post_tokens: [...], annotators: [...], ...}}
        rows = []
        for post_id, record in data.items():
            # Get text
            tokens = record.get("post_tokens", [])
            text = " ".join(tokens)
            
            # Get majority label from annotators
            labels = [a.get("label", "normal") for a in record.get("annotators", [])]
            if labels:
                from collections import Counter
                majority_label = Counter(labels).most_common(1)[0][0]
            else:
                majority_label = "normal"
            
            label = 1 if majority_label.lower() in ["hatespeech", "offensive", "hate"] else 0
            
            # Get original split if available
            split = record.get("split", "train")
            
            rows.append({
                "id": post_id,
                "text": clean_text(text),
                "label": label,
                "split": split
            })
        
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        
        # Split by original split field or create new split
        if "split" in df.columns and df["split"].nunique() > 1:
            for split_name in df["split"].unique():
                split_df = df[df["split"] == split_name].copy()
                out_split = "val" if split_name == "validation" else split_name
                
                # Save
                csv_path = DATA_DIR / f"hatexplain_{out_split}.csv"
                split_df[["text", "label"]].to_csv(csv_path, index=False)
                
                jsonl_path = DATA_DIR / f"hatexplain_{out_split}.jsonl"
                with open(jsonl_path, "w") as f:
                    for _, row in split_df.iterrows():
                        obj = {
                            "text": row["text"],
                            "label": "hatespeech" if row["label"] == 1 else "normal",
                            "post_tokens": row["text"].split()
                        }
                        f.write(json.dumps(obj) + "\n")
                
                pos_rate = split_df["label"].mean()
                print(f"  {out_split}: {len(split_df)} samples, {pos_rate:.1%} positive")
        else:
            # Create our own split
            train_df, val_df, test_df = stratified_split(df)
            
            for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                csv_path = DATA_DIR / f"hatexplain_{split_name}.csv"
                split_df[["text", "label"]].to_csv(csv_path, index=False)
                
                jsonl_path = DATA_DIR / f"hatexplain_{split_name}.jsonl"
                with open(jsonl_path, "w") as f:
                    for _, row in split_df.iterrows():
                        obj = {
                            "text": row["text"],
                            "label": "hatespeech" if row["label"] == 1 else "normal",
                            "post_tokens": row["text"].split()
                        }
                        f.write(json.dumps(obj) + "\n")
                
                pos_rate = split_df["label"].mean()
                print(f"  {split_name}: {len(split_df)} samples, {pos_rate:.1%} positive")
        
        print("HateXplain processing complete!")
        return True
    
    print(f"ERROR: No data files found in {hatex_dir}. Run with --download first.")
    return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and process real toxicity datasets"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets (requires internet and API keys for some)"
    )
    parser.add_argument(
        "--process",
        action="store_true", 
        help="Process downloaded datasets into standard format"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["jigsaw", "civil", "hatexplain", "all"],
        default=["all"],
        help="Which datasets to download/process"
    )
    
    args = parser.parse_args()
    
    if not args.download and not args.process:
        args.download = True
        args.process = True
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["jigsaw", "civil", "hatexplain"]
    
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("REAL DATA DOWNLOAD AND PROCESSING")
    print("="*80)
    print(f"Data directory: {DATA_DIR}")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Datasets: {datasets}")
    print(f"Download: {args.download}, Process: {args.process}")
    
    # Download
    if args.download:
        for ds in datasets:
            if ds == "jigsaw":
                download_jigsaw()
            elif ds == "civil":
                download_civil()
            elif ds == "hatexplain":
                download_hatexplain()
    
    # Process
    if args.process:
        for ds in datasets:
            if ds == "jigsaw":
                process_jigsaw()
            elif ds == "civil":
                process_civil()
            elif ds == "hatexplain":
                process_hatexplain()
    
    # Generate protocols.json
    protocols = {
        "datasets": datasets,
        "splits": ["train", "val", "test"],
        "generated": False,
        "real_data": True,
    }
    protocols_path = DATA_DIR / "protocols.json"
    with open(protocols_path, "w") as f:
        json.dump(protocols, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    for f in sorted(DATA_DIR.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name:40} {size:>12,} bytes")


if __name__ == "__main__":
    main()
