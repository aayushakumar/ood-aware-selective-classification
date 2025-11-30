#!/usr/bin/env python3
"""
Process large Jigsaw dataset using chunked reading to save memory.
"""

import pandas as pd
from pathlib import Path
import re
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import gc

print('Processing Jigsaw dataset (memory-efficient chunked approach)...', flush=True)

RAW_PATH = Path('raw_data/jigsaw/train.csv')
DATA_DIR = Path('data')
SEED = 42
TOXICITY_THRESHOLD = 0.5
CHUNK_SIZE = 100000  # Process 100k rows at a time

IDENTITY_COLS = [
    'male', 'female', 'transgender', 'other_gender',
    'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
    'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion',
    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
    'physical_disability', 'intellectual_or_learning_disability', 
    'psychiatric_or_mental_illness', 'other_disability'
]

# Text cleaning functions
URL_RE = re.compile(r'http\S+')
AT_RE = re.compile(r'@\w+')
WHITESPACE_RE = re.compile(r'\s+')

def clean_text(s):
    if pd.isna(s):
        return ''
    s = str(s)
    s = URL_RE.sub(' URL ', s)
    s = AT_RE.sub('@USER', s)
    s = s.replace('\n', ' ').replace('\t', ' ')
    s = WHITESPACE_RE.sub(' ', s).strip()
    return s

# Read columns to determine available identity columns
print('Reading header...', flush=True)
header = pd.read_csv(RAW_PATH, nrows=0)
available_id_cols = [c for c in IDENTITY_COLS if c in header.columns]
use_cols = ['id', 'target', 'comment_text'] + available_id_cols
print(f'Using columns: id, target, comment_text + {len(available_id_cols)} identity columns', flush=True)

# Process in chunks
print(f'Processing in chunks of {CHUNK_SIZE}...', flush=True)
processed_chunks = []
chunk_num = 0

for chunk in pd.read_csv(RAW_PATH, usecols=use_cols, chunksize=CHUNK_SIZE):
    chunk_num += 1
    
    # Clean text
    chunk['text'] = chunk['comment_text'].apply(clean_text)
    
    # Create label
    chunk['label'] = (chunk['target'] >= TOXICITY_THRESHOLD).astype(int)
    
    # Create identity group columns
    for c in available_id_cols:
        chunk[f'g_{c}'] = (chunk[c].fillna(0) >= TOXICITY_THRESHOLD).astype(int)
    
    # Keep only needed columns
    keep_cols = ['id', 'text', 'label'] + [f'g_{c}' for c in available_id_cols]
    chunk = chunk[keep_cols]
    
    # Remove empty texts
    chunk = chunk[chunk['text'].str.len() > 0]
    
    processed_chunks.append(chunk)
    print(f'  Chunk {chunk_num}: {len(chunk)} rows', flush=True)
    
    gc.collect()

# Combine all chunks
print('Combining chunks...', flush=True)
df = pd.concat(processed_chunks, ignore_index=True)
del processed_chunks
gc.collect()

print(f'Total before dedup: {len(df)}', flush=True)

# Remove duplicates
print('Removing duplicates...', flush=True)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f'After dedup: {len(df)}', flush=True)
print(f'Positive rate: {df["label"].mean():.1%}', flush=True)

# Stratified split 80/10/10
print('Creating stratified splits...', flush=True)
np.random.seed(SEED)

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, temp_idx = next(sss1.split(df, df['label']))

temp_df = df.iloc[temp_idx]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_rel_idx, test_rel_idx = next(sss2.split(temp_df, temp_df['label']))

splits = {
    'train': df.iloc[train_idx],
    'val': df.iloc[temp_idx[val_rel_idx]],
    'test': df.iloc[temp_idx[test_rel_idx]]
}

# Save splits
DATA_DIR.mkdir(parents=True, exist_ok=True)
full_cols = ['id', 'text', 'label'] + [f'g_{c}' for c in available_id_cols]

for split_name, split_df in splits.items():
    split_df = split_df.reset_index(drop=True)
    
    # Basic version
    basic_path = DATA_DIR / f'jigsaw_{split_name}.csv'
    split_df[['text', 'label']].to_csv(basic_path, index=False)
    
    # Full version
    full_path = DATA_DIR / f'jigsaw_{split_name}_full.csv'
    split_df[full_cols].to_csv(full_path, index=False)
    
    print(f'{split_name}: {len(split_df)} samples, {split_df["label"].mean():.1%} positive', flush=True)
    print(f'  Saved: {basic_path}', flush=True)
    print(f'  Saved: {full_path}', flush=True)

print('\nJigsaw processing complete!', flush=True)
