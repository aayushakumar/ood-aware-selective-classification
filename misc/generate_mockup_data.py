#!/usr/bin/env python3
"""
Generate Mockup Data for Toxicity Classification Experiments
Creates synthetic datasets for testing the pipeline without real data.

Generates train/val/test splits for:
- Jigsaw (toxic comment classification)
- Civil Comments (toxicity with identity attributes)
- HateXplain (hate speech detection)
"""

import os
import random
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42

# Sample sizes for mockup data
TRAIN_SIZE = 500
VAL_SIZE = 100
TEST_SIZE = 100

# Approximate class balance (proportion of toxic/positive class)
POSITIVE_RATE = 0.3

# Identity groups for Jigsaw and Civil Comments
IDENTITY_GROUPS = [
    "male", "female", "transgender", "other_gender",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "homosexual_gay_or_lesbian", "bisexual", "heterosexual", "other_sexual_orientation",
    "psychiatric_or_mental_illness", "physical_disability", "intellectual_or_learning_disability",
]


# ============================================================================
# TEXT TEMPLATES
# ============================================================================

# Non-toxic comment templates
NON_TOXIC_TEMPLATES = [
    "I really appreciate your perspective on this topic.",
    "This is a well-written article with some interesting points.",
    "I disagree with the author but respect their viewpoint.",
    "Thanks for sharing this information, it was very helpful.",
    "I think we should consider multiple perspectives here.",
    "This is a complex issue that deserves more discussion.",
    "I appreciate the thoughtful analysis presented here.",
    "Could you explain more about this point?",
    "I found this article to be very informative.",
    "Let's keep the discussion civil and productive.",
    "That's an interesting point of view.",
    "I learned something new from this post.",
    "Well said, I agree with most of your points.",
    "This needs more context to understand fully.",
    "I think there are valid arguments on both sides.",
    "Thanks for bringing attention to this issue.",
    "I would like to see more evidence for this claim.",
    "This is a fair assessment of the situation.",
    "I appreciate the balanced approach here.",
    "Could you provide sources for this information?",
    "This comment section has been very educational.",
    "I respect your opinion even though I see it differently.",
    "Great discussion, everyone has made good points.",
    "I think we can find common ground on this.",
    "This article raises important questions.",
    "I found this to be a thoughtful piece.",
    "The comments here have been surprisingly constructive.",
    "I'd like to add some context to this discussion.",
    "This is exactly what I was looking for, thank you.",
    "I think we should focus on the facts here.",
]

# Toxic comment templates (synthetic, clearly labeled as mockup)
TOXIC_TEMPLATES = [
    "[MOCKUP TOXIC] This comment contains simulated offensive language.",
    "[MOCKUP TOXIC] Simulated hostile comment for testing purposes.",
    "[MOCKUP TOXIC] This is a synthetic negative comment.",
    "[MOCKUP TOXIC] Mockup aggressive language placeholder.",
    "[MOCKUP TOXIC] Simulated derogatory statement for testing.",
    "[MOCKUP TOXIC] Synthetic toxic content for pipeline validation.",
    "[MOCKUP TOXIC] This represents offensive content in testing.",
    "[MOCKUP TOXIC] Placeholder for hostile speech detection testing.",
    "[MOCKUP TOXIC] Mockup hateful language for classifier training.",
    "[MOCKUP TOXIC] Simulated harassment content placeholder.",
    "[MOCKUP TOXIC] Synthetic threat-like statement for testing.",
    "[MOCKUP TOXIC] Mockup insulting comment for validation.",
    "[MOCKUP TOXIC] Simulated discriminatory language placeholder.",
    "[MOCKUP TOXIC] This is a toxic content placeholder for testing.",
    "[MOCKUP TOXIC] Synthetic abusive language for pipeline testing.",
]

# HateXplain-specific templates
HATEXPLAIN_NON_TOXIC = [
    "I love how diverse our community is becoming.",
    "Everyone deserves to be treated with respect.",
    "We should celebrate our differences, not fear them.",
    "Inclusion makes our society stronger.",
    "I appreciate learning about different cultures.",
    "Let's work together to build a better world.",
    "Diversity is our strength as a nation.",
    "Respect and understanding go a long way.",
    "We're all part of the same human family.",
    "Kindness costs nothing but means everything.",
]

HATEXPLAIN_TOXIC = [
    "[MOCKUP HATE] Simulated hate speech content for testing.",
    "[MOCKUP OFFENSIVE] Synthetic offensive language placeholder.",
    "[MOCKUP HATE] Mockup discriminatory statement for testing.",
    "[MOCKUP OFFENSIVE] Simulated slur placeholder for testing.",
    "[MOCKUP HATE] This represents hateful content in testing data.",
]


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_texts(n_samples: int, positive_rate: float, 
                   toxic_templates: List[str], 
                   non_toxic_templates: List[str]) -> pd.DataFrame:
    """
    Generate synthetic text data with labels.
    
    Args:
        n_samples: Number of samples to generate
        positive_rate: Proportion of positive (toxic) samples
        toxic_templates: List of toxic comment templates
        non_toxic_templates: List of non-toxic comment templates
    
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    n_positive = int(n_samples * positive_rate)
    n_negative = n_samples - n_positive
    
    # Generate positive (toxic) samples
    positive_texts = [
        random.choice(toxic_templates) + f" Sample ID: {i}" 
        for i in range(n_positive)
    ]
    
    # Generate negative (non-toxic) samples  
    negative_texts = [
        random.choice(non_toxic_templates) + f" Sample ID: {i + n_positive}"
        for i in range(n_negative)
    ]
    
    # Combine and shuffle
    texts = positive_texts + negative_texts
    labels = [1] * n_positive + [0] * n_negative
    
    # Create DataFrame and shuffle
    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    return df


def add_identity_columns(df: pd.DataFrame, 
                         identity_groups: List[str]) -> pd.DataFrame:
    """
    Add random identity group indicators to a DataFrame.
    
    Args:
        df: DataFrame with text and label columns
        identity_groups: List of identity group names
    
    Returns:
        DataFrame with added g_* columns
    """
    df = df.copy()
    df["id"] = range(len(df))
    
    for group in identity_groups:
        # Randomly assign identity indicators (about 10% positive for each group)
        df[f"g_{group}"] = np.random.binomial(1, 0.1, len(df))
    
    # Ensure at least one group is marked for some samples
    # to make fairness analysis meaningful
    for i in range(0, len(df), 10):
        group = random.choice(identity_groups)
        df.loc[i, f"g_{group}"] = 1
    
    return df


def generate_jigsaw_data():
    """Generate Jigsaw dataset mockup files."""
    print("Generating Jigsaw dataset mockup...")
    
    splits = {
        "train": TRAIN_SIZE,
        "val": VAL_SIZE,
        "test": TEST_SIZE,
    }
    
    for split_name, size in splits.items():
        # Basic version (text, label only)
        df = generate_texts(size, POSITIVE_RATE, TOXIC_TEMPLATES, NON_TOXIC_TEMPLATES)
        basic_path = DATA_DIR / f"jigsaw_{split_name}.csv"
        df[["text", "label"]].to_csv(basic_path, index=False)
        print(f"  Created: {basic_path} ({len(df)} samples)")
        
        # Full version (with identity groups)
        df_full = add_identity_columns(df, IDENTITY_GROUPS)
        full_path = DATA_DIR / f"jigsaw_{split_name}_full.csv"
        df_full.to_csv(full_path, index=False)
        print(f"  Created: {full_path} ({len(df_full)} samples, {len(IDENTITY_GROUPS)} groups)")


def generate_civil_data():
    """Generate Civil Comments dataset mockup files."""
    print("\nGenerating Civil Comments dataset mockup...")
    
    splits = {
        "train": TRAIN_SIZE,
        "val": VAL_SIZE,
        "test": TEST_SIZE,
    }
    
    for split_name, size in splits.items():
        # Generate base data
        df = generate_texts(size, POSITIVE_RATE, TOXIC_TEMPLATES, NON_TOXIC_TEMPLATES)
        
        # Add toxicity score (continuous version of label)
        # Toxic samples get high toxicity, non-toxic get low
        df["toxicity"] = df["label"].apply(
            lambda x: np.random.uniform(0.5, 1.0) if x == 1 
                      else np.random.uniform(0.0, 0.5)
        )
        
        # Basic version (text, label only)
        basic_path = DATA_DIR / f"civil_{split_name}.csv"
        df[["text", "label", "toxicity"]].to_csv(basic_path, index=False)
        print(f"  Created: {basic_path} ({len(df)} samples)")
        
        # Full version (with identity groups)
        df_full = add_identity_columns(df, IDENTITY_GROUPS)
        full_path = DATA_DIR / f"civil_{split_name}_full.csv"
        df_full.to_csv(full_path, index=False)
        print(f"  Created: {full_path} ({len(df_full)} samples, {len(IDENTITY_GROUPS)} groups)")


def generate_hatexplain_data():
    """Generate HateXplain dataset mockup files."""
    print("\nGenerating HateXplain dataset mockup...")
    
    splits = {
        "train": TRAIN_SIZE,
        "val": VAL_SIZE,
        "test": TEST_SIZE,
    }
    
    label_map = {0: "normal", 1: "hatespeech"}
    
    for split_name, size in splits.items():
        # Generate data with HateXplain-specific templates
        df = generate_texts(
            size, POSITIVE_RATE, 
            HATEXPLAIN_TOXIC, 
            HATEXPLAIN_NON_TOXIC
        )
        
        # CSV format (text, label) - matches what the loader expects
        csv_path = DATA_DIR / f"hatexplain_{split_name}.csv"
        df[["text", "label"]].to_csv(csv_path, index=False)
        print(f"  Created: {csv_path} ({len(df)} samples)")
        
        # JSONL format (alternative format)
        jsonl_path = DATA_DIR / f"hatexplain_{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                obj = {
                    "text": row["text"],
                    "label": label_map[row["label"]],
                    "post_tokens": row["text"].split(),
                }
                f.write(json.dumps(obj) + "\n")
        print(f"  Created: {jsonl_path} ({len(df)} samples)")


def generate_protocols_json():
    """Generate protocols.json metadata file."""
    print("\nGenerating protocols.json metadata...")
    
    protocols = {
        "datasets": ["jigsaw", "civil", "hatexplain"],
        "splits": ["train", "val", "test"],
        "generated": True,
        "mockup_data": True,
        "sizes": {
            "train": TRAIN_SIZE,
            "val": VAL_SIZE,
            "test": TEST_SIZE,
        },
        "positive_rate": POSITIVE_RATE,
    }
    
    protocols_path = DATA_DIR / "protocols.json"
    with open(protocols_path, "w", encoding="utf-8") as f:
        json.dump(protocols, f, indent=2)
    print(f"  Created: {protocols_path}")


def main():
    """Generate all mockup datasets."""
    print("=" * 80)
    print("MOCKUP DATA GENERATOR")
    print("=" * 80)
    print(f"\nOutput directory: {DATA_DIR}")
    print(f"Random seed: {SEED}")
    print(f"Train size: {TRAIN_SIZE}, Val size: {VAL_SIZE}, Test size: {TEST_SIZE}")
    print(f"Positive (toxic) rate: {POSITIVE_RATE}")
    print()
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    set_seed(SEED)
    
    # Generate all datasets
    generate_jigsaw_data()
    generate_civil_data()
    generate_hatexplain_data()
    generate_protocols_json()
    
    print("\n" + "=" * 80)
    print("MOCKUP DATA GENERATION COMPLETE")
    print("=" * 80)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(DATA_DIR.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name:40} {size:>10,} bytes")
    
    print("\nYou can now run:")
    print("  python scripts/run_tfidf_baselines.py --source jigsaw --targets civil hatexplain")
    print("  python scripts/run_roberta.py --source jigsaw --targets civil hatexplain")


if __name__ == "__main__":
    main()
