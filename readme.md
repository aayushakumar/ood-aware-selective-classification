# Out-of-Distribution Evaluation of Toxicity Classifiers

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **CS 483 – Final Project**  
> Cross-domain robustness, calibration, fairness, and OOD detection of toxic comment classifiers across **Jigsaw**, **Civil Comments**, and **HateXplain**.

## 🎯 Quick Start: Reproduce All Results

**The fastest way to reproduce all experiments is to run the self-contained notebook:**

```bash
# Open in Jupyter/VS Code/Colab:
executable_script/final_experiments_smallScale.ipynb
```

This single notebook runs the complete pipeline: data loading → TF-IDF baselines → RoBERTa training → OOD detection → calibration → fairness analysis → all visualizations.

---

## 📋 Overview

This repository provides a complete pipeline to:

- Train and evaluate **TF–IDF baselines** (Logistic Regression / SVM) and **RoBERTa** models
- Perform **cross-domain transfer** evaluation (e.g., train on Jigsaw → test on Civil & HateXplain)
- Analyze **probability calibration** (ECE, reliability diagrams, temperature scaling, isotonic regression)
- Compute **group fairness metrics** (Demographic Parity, Equal Opportunity, Equalized Odds)
- Benchmark **OOD detection** methods (MaxSoftmax, ODIN, Energy)
- Generate publication-ready plots and tables

---

## 1. Repository Structure

```text
ood-eval-toxic-classifiers/
├── executable_script/
│   └── final_experiments_smallScale.ipynb  # ⭐ MAIN: Complete reproducible pipeline
├── scripts/
│   ├── run_roberta.py              # RoBERTa training & evaluation
│   ├── run_tfidf_baselines.py      # TF-IDF baseline models
│   ├── fairness_metrics.py         # Group fairness computation
│   ├── ood_algorithms.py           # OOD detection methods (MaxSoftmax, ODIN, Energy)
│   └── test_pipeline.py            # Sanity checks
├── notebooks/
│   ├── analysis_plots.ipynb        # Additional visualizations
│   ├── civildata.ipynb             # Civil Comments preprocessing
│   ├── cs483_data.ipynb            # Jigsaw preprocessing
│   └── hatexplaindata.ipynb        # HateXplain preprocessing
├── data/                           # Preprocessed CSV files (see Section 3)
├── output/                         # Generated results, models, figures
├── final_report.tex                # LaTeX report
├── requirements.txt                # Python dependencies
└── execution_guide.md              # Detailed execution instructions
```

---

## 2. How to Reproduce (Step-by-Step)

### Option A: Run the All-in-One Notebook (Recommended)

The notebook `executable_script/final_experiments_smallScale.ipynb` is self-contained and includes:

| Cell | Section | Description |
|------|---------|-------------|
| 1 | Environment Setup | Auto-detects Colab/Kaggle/Local, clones repo if needed |
| 2 | Imports | Loads project modules from `scripts/` |
| 3 | Configuration | Sets hyperparameters, seeds, device |
| 4 | Data Loading | Loads all three datasets |
| 5 | Results Tracker | Initializes storage for metrics |
| 6 | TF-IDF Baselines | Trains LogReg/SVM on all sources, cross-domain eval |
| 7 | RoBERTa Training | Fine-tunes RoBERTa on each dataset (100K samples) |
| 8 | OOD Detection | MaxSoftmax, ODIN, Energy scoring |
| 9 | Calibration | Temperature scaling, isotonic regression, ECE |
| 10 | Fairness | Demographic parity, equalized odds gaps |
| 11-21 | Visualizations | Heatmaps, ROC/PR curves, reliability diagrams |
| 22+ | Advanced Analysis | Selective prediction, OOD-weighted ensembles |

**To run:**

```bash
# 1. Install dependencies
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm

# 2. Ensure data/ folder has the preprocessed CSVs (see Section 3)

# 3. Open and run all cells
jupyter notebook executable_script/final_experiments_smallScale.ipynb
```

**On Google Colab:**
1. Upload the notebook to Colab
2. The first cell will automatically clone the repo and install dependencies
3. Run all cells sequentially

**On Kaggle:**
1. Create a new notebook and attach the datasets
2. Upload the notebook or copy cells
3. Run all cells

### Option B: Run Individual Scripts (CLI)

```bash
# TF-IDF Baselines
python scripts/run_tfidf_baselines.py \
    --source_dataset jigsaw \
    --target_datasets civil hatexplain \
    --model logreg \
    --data_dir data \
    --save_preds

# RoBERTa Training
python scripts/run_roberta.py \
    --source_dataset jigsaw \
    --target_datasets civil hatexplain \
    --model_name roberta-base \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_len 128 \
    --data_dir data \
    --calibration isotonic \
    --early_stop \
    --tune_threshold \
    --save_preds

# Fairness Metrics
python scripts/fairness_metrics.py \
    --dataset civil \
    --split test \
    --pred_file experiments/preds_jigsaw_to_civil.csv \
    --full_data_file data/civil_test_full.csv \
    --group_prefix g_ \
    --out_prefix experiments/fairness_jigsaw_to_civil
```

---

## 3. Datasets

The `data/` folder should contain preprocessed CSVs. If missing, run the preprocessing notebooks first.

| Dataset | Train | Val | Test | Positive Rate | Source |
|---------|-------|-----|------|---------------|--------|
| Jigsaw | 1,420,932 | 177,616 | 89,780 | 8.0% | Kaggle |
| Civil Comments | 1,776,165 | 96,717 | 96,702 | 8.0% | Kaggle |
| HateXplain | 16,087 | 2,011 | 2,011 | 59.6% | GitHub |

**Required files in `data/`:**
```
jigsaw_train.csv, jigsaw_val.csv, jigsaw_test.csv
civil_train.csv, civil_val.csv, civil_test.csv
hatexplain_train.csv, hatexplain_val.csv, hatexplain_test.csv
*_full.csv variants (include identity group columns for fairness)
```

### Preprocessing

Run these notebooks to generate the data files (only needed once):

1. **Jigsaw**: `notebooks/cs483_data.ipynb`
   - Binarizes toxicity at threshold 0.5
   - Creates identity group indicators (`g_male`, `g_female`, etc.)

2. **Civil Comments**: `notebooks/civildata.ipynb`
   - Same preprocessing as Jigsaw

3. **HateXplain**: `notebooks/hatexplaindata.ipynb`
   - Maps {hatespeech, offensive} → 1, {normal} → 0

---

## 4. Methods & Metrics

### Models
- **TF-IDF + LogReg/SVM**: Bag-of-words baseline with unigrams/bigrams
- **RoBERTa-base**: Fine-tuned transformer (100K training samples for speed)

### Evaluation Metrics
| Category | Metrics |
|----------|---------|
| Classification | F1, AUROC, Accuracy |
| Calibration | ECE (Expected Calibration Error) |
| OOD Detection | AUROC for ID vs OOD separation |
| Fairness | Demographic Parity gap, Equalized Odds gap |

### OOD Detection Methods
- **MaxSoftmax**: max class probability
- **ODIN**: temperature-scaled softmax with perturbation
- **Energy**: negative log-sum-exp of logits

---

## 5. Key Results

### Cross-Domain F1 (RoBERTa)
| Source → Target | Jigsaw | Civil | HateXplain |
|-----------------|--------|-------|------------|
| Jigsaw | **0.67** | 0.65 | 0.72 |
| Civil | 0.68 | **0.67** | 0.69 |
| HateXplain | 0.32 | 0.33 | **0.84** |

### Key Findings
1. **Transfer is asymmetric**: Models trained on HateXplain struggle on Jigsaw/Civil
2. **OOD detection works for cross-platform shift**: AUROC ~0.75 for Jigsaw↔HateXplain
3. **OOD detection fails within-platform**: AUROC ~0.50 for Jigsaw↔Civil
4. **Calibration degrades under shift**: ECE increases 2-4x on OOD data

---

## 6. Output Files

After running the notebook, outputs are saved to `output/`:

```
output/
├── results/          # JSON metrics summaries
├── figures/          # All plots (PNG + PDF)
│   ├── classification_heatmaps.png
│   ├── roc_pr_curves.png
│   ├── calibration_reliability.png
│   ├── ood_fairness.png
│   ├── ood_score_distributions.png
│   └── ood_summary_publication.png
└── models/           # Saved model weights
    ├── roberta_jigsaw.pt
    ├── roberta_civil.pt
    └── roberta_hatexplain.pt
```

---

## 7. Configuration

Key parameters in the notebook (Cell 3):

```python
CONFIG = {
    'seed': 42,
    'datasets': ['jigsaw', 'civil', 'hatexplain'],
    
    # RoBERTa
    'model_name': 'roberta-base',
    'max_length': 128,
    'batch_size': 64,  # Reduce to 16-32 for smaller GPUs
    'epochs': 5,
    'learning_rate': 2e-5,
    
    # TF-IDF
    'tfidf_max_features': 50000,
    'tfidf_ngram_range': (1, 2),
}

# Speed settings
MAX_TRAIN_SAMPLES = 100000  # Subsample for faster training
USE_FP16 = True             # Mixed precision
```

---

## 8. Dependencies

```bash
pip install torch>=2.0 transformers>=4.30 scikit-learn pandas numpy matplotlib seaborn tqdm
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

**Hardware**: GPU recommended (tested on A100, works on T4/V100 with smaller batch size)

---

## 10. Acknowledgements

Datasets:
- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) (Kaggle)
- [Civil Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle)
- [HateXplain](https://github.com/hate-alert/HateXplain) (GitHub)

Libraries: PyTorch, Hugging Face Transformers, scikit-learn
