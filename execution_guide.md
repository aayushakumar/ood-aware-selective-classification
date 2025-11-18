# Complete Execution Guide

## Prerequisites

### On Kaggle:
1. **Enable GPU**: Settings → Accelerator → GPU
2. **Add Datasets**:
   - Jigsaw Unintended Bias in Toxicity Classification
   - Civil Comments (if available)
   - HateXplain dataset

### On Local Machine:
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn
pip install peft  # Optional for LoRA
```

---

## Step-by-Step Execution

### PHASE 1: Data Preprocessing (Run Notebooks in Order)

#### 1.1 Preprocess Jigsaw Dataset
**File:** `cs483_data.ipynb`

```python
# Run all cells in sequence
# Expected outputs:
# - data/jigsaw_train.csv
# - data/jigsaw_val.csv
# - data/jigsaw_test.csv
# - data/jigsaw_train_full.csv (with identity groups)
# - data/jigsaw_val_full.csv
# - data/jigsaw_test_full.csv
# - data/protocols.json
```

**Verification:**
```python
import pandas as pd
train = pd.read_csv('data/jigsaw_train.csv')
print(f"Jigsaw train: {len(train)} samples")
print(f"Columns: {list(train.columns)}")
print(f"Positive rate: {train['label'].mean():.3f}")
```

#### 1.2 Preprocess Civil Comments
**File:** `civildata.ipynb`

Run all cells to generate:
- `data/civil_train.csv`
- `data/civil_val.csv`
- `data/civil_test.csv`
- `data/civil_*_full.csv` files

#### 1.3 Preprocess HateXplain
**File:** `hatexplaindata.ipynb`

Run all cells to generate:
- `data/hatexplain_train.csv`
- `data/hatexplain_val.csv`
- `data/hatexplain_test.csv`

**⚠️ IMPORTANT:** After running all preprocessing notebooks, verify data:

```python
import os
from pathlib import Path

required_files = [
    'jigsaw_train.csv', 'jigsaw_val.csv', 'jigsaw_test.csv',
    'civil_train.csv', 'civil_val.csv', 'civil_test.csv',
    'hatexplain_train.csv', 'hatexplain_val.csv', 'hatexplain_test.csv'
]

data_dir = Path('data')
for fname in required_files:
    fpath = data_dir / fname
    if fpath.exists():
        size_mb = fpath.stat().st_size / (1024*1024)
        print(f"✓ {fname} ({size_mb:.1f} MB)")
    else:
        print(f"✗ {fname} - MISSING!")
```

---

### PHASE 2: Train Models

#### 2.1 Train TF-IDF Baseline

**On Kaggle/Notebook:**
```python
!python scripts/run_tfidf_baselines.py \
    --source_dataset jigsaw \
    --target_datasets civil hatexplain \
    --model logreg \
    --seed 42 \
    --data_dir /kaggle/working/data \
    --save_preds
```

**Local/Terminal:**
```bash
cd scripts
python run_tfidf_baselines.py \
    --source_dataset jigsaw \
    --target_datasets civil hatexplain \
    --model logreg \
    --seed 42 \
    --save_preds
```

**Expected outputs:**
- `experiments/summary_tfidf_jigsaw_logreg.csv`
- `experiments/preds_tfidf_logreg_jigsaw_test.csv`
- `experiments/preds_tfidf_logreg_jigsaw_to_civil.csv`

#### 2.2 Train Basic RoBERTa

```bash
python scripts/run_roberta.py \
    --source_dataset jigsaw \
    --target_datasets civil hatexplain \
    --model_name roberta-base \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5 \
    --seed 42 \
    --calibration isotonic \
    --early_stop \
    --tune_threshold \
    --save_preds
```

**Time estimate:** 30-60 minutes on GPU

**Expected outputs:**
- `experiments/summary_jigsaw.csv`
- `experiments/preds_jigsaw_test.csv`
- `experiments/preds_jigsaw_to_civil.csv`
- `experiments/jigsaw_val_reliability.csv`

#### 2.3 Train RoBERTa with LoRA (Optional)

```bash
python scripts/run_roberta.py \
    --source_dataset jigsaw \
    --target_datasets civil hatexplain \
    --peft lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 5 \
    --calibration temperature \
    --seed 42 \
    --save_preds
```

#### 2.4 Train with CORAL Domain Adaptation (Optional)

```bash
python scripts/run_roberta.py \
    --source_dataset jigsaw \
    --target_datasets civil \
    --coral_target civil \
    --coral_lambda 0.1 \
    --epochs 5 \
    --seed 42 \
    --save_preds
```

---

### PHASE 3: Fairness Analysis

#### 3.1 Compute Fairness Metrics

```bash
python scripts/fairness_metrics.py \
    --dataset civil \
    --split test \
    --pred_file experiments/preds_jigsaw_to_civil.csv \
    --full_data_file data/civil_test_full.csv \
    --out_prefix experiments/fairness_jigsaw_to_civil
```

**Expected outputs:**
- `experiments/fairness_jigsaw_to_civil_summary.csv`
- `experiments/fairness_jigsaw_to_civil_per_group.csv`

---

### PHASE 4: Generate Visualizations

**File:** `analysis_plots.ipynb`

Run all cells to generate:
- Reliability diagrams
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Cross-domain comparisons
- Fairness visualizations

All plots saved to: `experiments/plots/`

---

## Alternative: Run Everything at Once

**File:** `run_all_experiments.ipynb`

This master notebook orchestrates all steps, but requires:
1. Data preprocessing completed first
2. Scripts in proper location
3. Sufficient GPU time

---

## Troubleshooting Common Issues

### Issue: "Module not found" errors
**Solution:**
```python
import sys
sys.path.append('/kaggle/working/scripts')
```

### Issue: "Data file not found"
**Solution:** Verify data preprocessing completed:
```python
!ls -lh /kaggle/working/data/
```

### Issue: Out of memory during training
**Solution:** Reduce batch size:
```bash
--batch_size 8  # or even 4
```

### Issue: CUDA out of memory
**Solution:** Enable mixed precision:
```bash
--amp  # Add this flag
```

---

## Quick Start (Minimum Viable Experiment)

If you want to run a quick experiment to test everything:

```bash
# 1. Preprocess Jigsaw only (run cs483_data.ipynb)

# 2. Train small model
python scripts/run_roberta.py \
    --source_dataset jigsaw \
    --epochs 1 \
    --batch_size 16 \
    --seed 42

# 3. Check results
ls -lh experiments/
```

---

## Expected Timeline

| Phase | Time (GPU) | Time (CPU) |
|-------|-----------|-----------|
| Data Preprocessing | 5-10 min | 5-10 min |
| TF-IDF Baseline | 2-5 min | 5-10 min |
| RoBERTa (3 epochs) | 30-60 min | 8-12 hours |
| Fairness Analysis | 1-2 min | 1-2 min |
| Visualization | 5-10 min | 5-10 min |
| **Total** | **~1-1.5 hours** | **~9-13 hours** |

---

## Verification Checklist

Before generating your report, verify:

- [ ] All data files exist in `data/`
- [ ] At least one model trained (check `experiments/`)
- [ ] Summary CSV files generated
- [ ] Predictions saved (for fairness analysis)
- [ ] Fairness metrics computed
- [ ] Plots generated in `experiments/plots/`

```python
# Run this verification script
from pathlib import Path

checks = {
    'Data files': list(Path('data').glob('*.csv')),
    'Experiment summaries': list(Path('experiments').glob('summary_*.csv')),
    'Predictions': list(Path('experiments').glob('preds_*.csv')),
    'Fairness results': list(Path('experiments').glob('fairness_*.csv')),
    'Plots': list(Path('experiments/plots').glob('*.png'))
}

for category, files in checks.items():
    print(f"{category}: {len(files)} files")
    if not files:
        print(f"  ⚠️  No files found!")
```