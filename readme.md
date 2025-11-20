# BiasBreakers: Out-of-Distribution Evaluation of Toxicity Classifiers

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-experimental-yellow.svg)]
[![Requirements](https://img.shields.io/badge/requirements-ready-lightgrey.svg)]


> **CS 483 – Final Project**  
> Cross-domain robustness, calibration, and fairness of toxic comment classifiers across **Jigsaw**, **Civil Comments**, and **HateXplain**.

This repository contains a full end-to-end pipeline to:

- Preprocess three toxicity datasets into a **unified format** (with identity group annotations where available).
- Train and evaluate **classical baselines** (TF–IDF + Logistic Regression / SVM) and **RoBERTa-based** neural models.
- Perform **cross-domain evaluation** (e.g., train on Jigsaw → test on Civil & HateXplain).
- Analyze **calibration** (ECE, reliability diagrams) and perform **post-hoc calibration** (temperature scaling, isotonic regression).
- Compute **group fairness metrics** (Demographic Parity, Equal Opportunity, Equalized Odds) across identity attributes.
- Generate plots and tables for the **final report / slides**.

---

## 1. Repository Structure

```text
ood-eval-toxic-classifiers
├── notebooks
│   ├── analysis_plots.ipynb         # All figures for calibration, ROC/PR, fairness, cross-domain comparison
│   ├── civildata.ipynb              # Preprocessing for Civil Comments
│   ├── cs483_data.ipynb             # Preprocessing for Jigsaw train.csv
│   ├── hatexplaindata.ipynb         # Preprocessing for HateXplain JSON/JSONL
│   └── run_all_experiments.ipynb    # One-click orchestration for experiments and summary artifacts
├── scripts
│   ├── fairness_metrics.py          # Group fairness computation (DP, EO, EOdds) and CLI
│   ├── run_roberta.py               # RoBERTa training + calibration + cross-domain evaluation
│   └── run_tfidf_baselines.py       # TF–IDF + Logistic Regression / SVM baselines
└── execution_guide.md               # Step-by-step how to run everything (Kaggle & local)
````

Runtime-generated directories:

```text
data/         # Preprocessed CSVs (jigsaw_*, civil_*, hatexplain_*)
experiments/  # Predictions, metrics, summaries, and plots
experiments/plots/   # All visualizations (.png)
working/      # (Kaggle/Colab-local, created by config cells)
```

All notebooks start with an **environment-aware configuration cell** that auto-detects:

* **Kaggle** (`/kaggle/input`, `/kaggle/working`)
* **Google Colab** (`/content/input`, `/content/working`)
* **Local** (uses `<repo_root>/input` and `<repo_root>/working`)

and then creates the standard directories:

* `OUT_DIR` → `.../working/data`
* `EXPERIMENTS_DIR` → `.../working/experiments`
* `SCRIPTS_DIR` → `.../working/scripts`

---

## 2. Datasets & Preprocessing

### 2.1 Jigsaw Unintended Bias in Toxicity Classification

* **Raw source**: `train.csv` from the Kaggle Jigsaw dataset.

* **Notebook**: `notebooks/cs483_data.ipynb`

* **Key steps**:

  * Load `train.csv` from `jigsaw-unintended-bias-in-toxicity-classification`.
  * Text cleaning:

    * Replace URLs with `URL`.
    * Replace `@user` handles with `@USER`.
    * Normalize whitespace and strip newlines/tabs.
  * **Binarize** toxicity:

    * `label = 1` if `target >= 0.5`, else `0`.
  * Add **identity group indicators**:

    * Original columns like `male`, `female`, `black`, `white`, `christian`, `muslim`, etc.
    * Converted to binary columns `g_male`, `g_female`, `g_black`, `g_white`, etc. using threshold `>= 0.5`.
  * Deduplicate by `text`.
  * **Stratified 8/1/1 split** (train/val/test) on `label` using `StratifiedShuffleSplit`.

* **Outputs** (in `data/`):

  * Standard (for training):

    * `jigsaw_train.csv`, `jigsaw_val.csv`, `jigsaw_test.csv`
      → columns: `text`, `label`
  * Full (for fairness analysis):

    * `jigsaw_train_full.csv`, `jigsaw_val_full.csv`, `jigsaw_test_full.csv`
      → columns: `id`, `text`, `label`, `g_*` (identity groups)
  * Protocol metadata:

    * `protocols.json` (sizes + positive rates per split)

---

### 2.2 Civil Comments

* **Notebook**: `notebooks/civildata.ipynb`

* **Raw detection**:

  * Automatically searches Kaggle inputs for any directory containing `"civil"` and `"comment"` in its name.

* **Key steps**:

  * Pick first Civil Comments CSV found in that directory.
  * `text` column:

    * Use `comment_text` if present, else `text`.
    * Same cleaning pipeline as Jigsaw (URLs → `URL`, mentions → `@USER`, whitespace normalization).
  * `label`:

    * Use `toxicity` or `target` column; `label = 1` if toxicity ≥ 0.5.
  * Identity columns:

    * Predefined list (e.g., `male`, `female`, `muslim`, `black`, etc.).
    * For each present in the raw CSV, produce `g_<col>` binary indicator (`>= 0.5`).
  * Deduplicate by `text`.
  * Stratified 8/1/1 split (train/val/test) on `label`.

* **Outputs**:

  * Standard:

    * `civil_train.csv`, `civil_val.csv`, `civil_test.csv` (`text`, `label`)
  * Full + groups:

    * `civil_train_full.csv`, `civil_val_full.csv`, `civil_test_full.csv` (`id`, `text`, `label`, `g_*`)
  * Protocols:

    * `civil_protocols.json` (sizes and positive rates)

---

### 2.3 HateXplain

* **Notebook**: `notebooks/hatexplaindata.ipynb`

* **Raw detection**:

  * Looks for a directory in `/kaggle/input` with `"hatexplain"` or `"hate_explain"` in its name.
  * Uses first JSON / JSONL file found in that directory.

* **Key steps**:

  * Loading:

    * Support both **JSONL** (one JSON per line) and **JSON** (list or dict).
  * Text extraction:

    * If `"text"` present, use directly.
    * Else, if `"post_tokens"` present, join tokens into a string.
  * Label mapping:

    * Original: `"hatespeech"`, `"offensive"`, `"normal"` (and variants).
    * Binary mapping:

      * `label = 1` for `{hatespeech, offensive, offensive_language, hate}`
      * `label = 0` otherwise.
  * Deduplicate by `text`.
  * Stratified 8/1/1 split on `label`.

* **Outputs**:

  * `hatexplain_train.csv`, `hatexplain_val.csv`, `hatexplain_test.csv` (columns: `text`, `label`)
  * `hatexplain_protocols.json` (sizes + positive rates)

---

## 3. Models & Methods

### 3.1 TF–IDF Baselines (Classical)

**Script**: `scripts/run_tfidf_baselines.py`

* Features:

  * `TfidfVectorizer` with configurable:

    * `ngram_range=(1, ngram_max)` (default 1–2)
    * `min_df` (default 5)
    * optional `max_features`.
* Models:

  * **Logistic Regression** (`logreg`):

    * `solver="lbfgs"`, `max_iter=1000`, `n_jobs=-1`.
    * Outputs probabilistic scores (`predict_proba`) → AUROC, PR–AUC, NLL, Brier.
  * **Linear SVM** (`svm`):

    * `LinearSVC(max_iter=5000, dual=False)`.
    * Uses `decision_function` for AUROC/PR–AUC; no calibrated probs.
* Evaluation:

  * In-domain:

    * Train on `source_train`
    * Validate on `source_val` (for sanity).
    * Test on `source_test`.
  * Cross-domain:

    * Apply TF–IDF mapping and classifier to each target dataset’s test split:

      * e.g., Jigsaw → (Civil, HateXplain).
* Metrics:

  * Accuracy, F1 (binary), AUROC, PR–AUC, NLL (logreg only), Brier (logreg only).

Saves:

* `experiments/summary_tfidf_<source>_<model>.csv`

  * Rows: `split` ∈ `{in_domain_val, in_domain_test, cross_<target>}`.
  * Columns: `accuracy`, `f1`, `auroc`, `pr_auc`, `nll`, `brier`.
* (If `--save_preds`)

  * `experiments/preds_tfidf_<model>_<source>_test.csv`
  * `experiments/preds_tfidf_<model>_<source>_to_<target>.csv`
  * Columns: `id`, `text`, `label`, `pred`, `score`.

---

### 3.2 RoBERTa-based Models

**Script**: `scripts/run_roberta.py`

High-level:

* Hugging Face `AutoTokenizer` + `AutoModelForSequenceClassification`.
* Optional **LoRA/PEFT** to reduce trainable parameter count.
* Optional **CORAL** domain alignment for better cross-domain robustness.
* Optional **AMP (mixed precision)** for faster training.
* Optional **class-imbalance weighting** via `WeightedRandomSampler`.
* **Early stopping** on validation (F1 or loss).
* **Post-hoc calibration** (temperature or isotonic).
* **Threshold tuning** on validation to maximize F1.

#### 3.2.1 Data Loading

* Unified `load_dataset(name, split, data_dir)` dispatcher:

  * `jigsaw` → from `jigsaw_<split>.csv`
  * `civil` → from `civil_<split>.csv` (uses `toxicity` or `label` if present)
  * `hatexplain` → CSV or JSONL as described above.

#### 3.2.2 PyTorch Dataset

Class: `ToxicityDataset`

* Wraps text + label into:

  * `input_ids`
  * `attention_mask`
  * `labels`
* Uses max sequence length `max_len` (default 128).

#### 3.2.3 Training Options

Key functions:

* `build_model(...)`:

  * Loads base model (e.g., `roberta-base`) with 2 labels.
  * Optionally wraps with LoRA (`peft_method="lora"`) using `peft` library.

* `train_one_epoch(...)`:

  * Standard cross-entropy loss.
  * Optional **CORAL loss**:

    * Pulls an unlabeled target batch (`coral_target`) and aligns **feature covariances** using:

      [
      \mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} | C_s - C_t |_F^2
      ]

      where (C_s, C_t) are covariance matrices of CLS representations.
  * Optional **AMP** (torch.cuda.amp) with gradient scaling.

* Early stopping:

  * Monitors either **F1** or **NLL** on validation.
  * Restores best-performing checkpoint if enabled.

#### 3.2.4 Calibration & ECE

* `expected_calibration_error(y_true, pos_prob, n_bins=15)`:

  * Computes ECE using equal-width bins in [0, 1].
  * Also returns bin-level stats (count, accuracy, confidence, gap).

* **Temperature scaling**:

  * `fit_temperature(logits, labels)`:

    * Optimizes a single temperature scalar `T` using L-BFGS to minimize validation NLL.

* **Isotonic regression**:

  * `fit_isotonic(probs, labels)`:

    * Fits `sklearn.isotonic.IsotonicRegression` on positive class probabilities.

* **Inference with calibration**:

  * `apply_calibration_from_logits(logits, temperature, iso_model)`

    * Applies temperature scaling (on logits) and isotonic regression (on probabilities) if provided.

#### 3.2.5 Threshold Tuning

* `tune_threshold_for_f1(y_true, pos_prob)`:

  * Scans thresholds in [0, 1] in steps of 0.001.
  * Chooses the threshold maximizing F1 on validation.
  * Used as decision threshold on `pos_prob` during test and cross-domain evaluation.

#### 3.2.6 Cross-Domain Evaluation

* `evaluate_cross_domain(...)`:

  * For each target dataset:

    * Loads `target_test` split.
    * Evaluates with calibration + threshold.
    * Optionally dumps `preds_<source>_to_<target>.csv` with `text`, `label`, `pos_prob`, `pred`.

#### 3.2.7 Outputs

From a typical run (e.g., Jigsaw → Civil, HateXplain):

* Models:

  * `experiments/jigsaw_roberta-base_seed42.pt`
  * `experiments/jigsaw_roberta-base_seed42_best.pt` (if early stopping)
* Calibration artifacts:

  * `experiments/jigsaw_val_reliability.csv` (validation bins)
  * `experiments/jigsaw_temp_42.txt` or `jigsaw_iso_42.pkl`
  * `experiments/jigsaw_thr_42.txt` (tuned threshold)
* Predictions:

  * `experiments/preds_jigsaw_test.csv` (in-domain)
  * `experiments/preds_jigsaw_to_civil.csv`
  * `experiments/preds_jigsaw_to_hatexplain.csv`
* Summary metrics:

  * `experiments/summary_jigsaw.csv`

    * `split` ∈ {`in_domain_test`, `cross_civil`, `cross_hatexplain`}
    * metrics: `accuracy`, `f1`, `auroc`, `pr_auc`, `nll`, `brier`, `ece`, `tn`, `fp`, `fn`, `tp`.

---

## 4. Fairness Metrics

**Script**: `scripts/fairness_metrics.py`

You can either:

* Import and call `compute_group_fairness(...)` from Python, or
* Use the CLI to compute fairness from prediction and full-data CSVs.

### 4.1 Metrics per Group

For each identity group column `g_*` and group value `0` / `1` (not in group / in group), the script computes:

* `support`: number of instances
* `pos_rate`: ( P(\hat{Y}=1 \mid g=\text{val}) )
  → **Demographic Parity** component.
* `tpr`: ( P(\hat{Y}=1 \mid Y=1, g=\text{val}) )
  → **Equal Opportunity** component.
* `fpr`: ( P(\hat{Y}=1 \mid Y=0, g=\text{val}) )
  → **Equalized Odds** component.

### 4.2 Aggregate Gaps per Group Column

For each `group_col` (e.g., `g_male`):

* `dp_diff`:

  * Max difference in `pos_rate` across values (typically between group 0 vs 1).
* `eop_diff`:

  * Max difference in `tpr` across values.
* `eo_diff`:

  * Max of `tpr` gap and `fpr` gap → **Equalized Odds** gap.

### 4.3 CLI Usage

Example: fairness for **Jigsaw → Civil** test set.

Assuming:

* Predictions: `experiments/preds_jigsaw_to_civil.csv` (must contain `id`, `label`, `pred`, optionally `pos_prob`/`score`).
* Full data with groups: `data/civil_test_full.csv` (`id`, `label`, `g_*`).

```bash
python scripts/fairness_metrics.py \
  --dataset civil \
  --split test \
  --pred_file experiments/preds_jigsaw_to_civil.csv \
  --full_data_file data/civil_test_full.csv \
  --group_prefix g_ \
  --out_prefix experiments/fairness_jigsaw_to_civil
```

This will create:

* `experiments/fairness_jigsaw_to_civil_summary.csv`
* `experiments/fairness_jigsaw_to_civil_per_group.csv`

and print the **Top 10** groups by each fairness gap.

---

## 5. Analysis & Visualization

**Notebook**: `notebooks/analysis_plots.ipynb`

Once experiments are run and CSVs are populated in `experiments/`, this notebook generates all figures needed for the report.

### 5.1 Reliability Diagrams

* From `*_val_reliability.csv` or other reliability CSVs.
* Function: `plot_reliability_diagram(csv_path, title, save_path)`
* Filters out empty bins; point size scaled by bin count; overlays perfect calibration line.
* Saves to: `experiments/plots/reliability_*.png`

### 5.2 ROC Curves

* Input prediction CSV: must have `label` & `pos_prob` or `score`.
* Function: `plot_roc_curve(pred_csv_path, title, save_path)`
* Computes AUC and plots ROC vs random baseline.
* Saves: `experiments/plots/roc_*.png`

### 5.3 Precision–Recall Curves

* Function: `plot_pr_curve(pred_csv_path, title, save_path)`
* Computes Average Precision (AP) and plots PR curve.
* Saves: `experiments/plots/pr_*.png`

### 5.4 Confusion Matrices

* Requires prediction CSV with `label` and `pred`.
* Function: `plot_confusion_matrix(pred_csv_path, title, save_path)`
* Uses `sns.heatmap` with labels {Non-Toxic, Toxic}.
* Saves: `experiments/plots/cm_*.png`

### 5.5 Cross-Domain Performance Bar Plots

* Function: `plot_cross_domain_comparison(summaries_dict, metric)`

  * Accepts dict: `{model_name: summary_df}` where each summary_df is a `summary_*.csv`.
  * Bar plots metrics (e.g., `f1`) across splits for multiple models (e.g., RoBERTa vs TF–IDF).
* Saves: `experiments/plots/comparison_*.png`

### 5.6 Fairness Visualizations

* Function: `plot_fairness_summary(fairness_csv_path, title, top_k)`

  * For a fairness summary CSV (e.g., `fairness_jigsaw_test_summary.csv`):
  * Plots top-K groups by:

    * Demographic Parity gap (`dp_diff`)
    * Equal Opportunity gap (`eop_diff`)
    * Equalized Odds gap (`eo_diff`)
* Function: `plot_per_group_metrics(fairness_per_group_csv, groups_to_plot)`

  * Bar plots for:

    * Positive rate
    * TPR
    * FPR
  * For specific identity groups (e.g., `g_male`, `g_female`, `g_black`, `g_white`, `g_lgbtq`, `g_muslim`).

### 5.7 Calibration Comparison (Before / After)

* Function: `plot_calibration_comparison(bins_before_csv, bins_after_csv)`

  * Two-panel figure showing reliability diagrams pre/post calibration.
  * Annotates ECE in each sub-plot.

### 5.8 Metrics Table

* Function: `create_metrics_table(summary_dfs_dict, save_path)`

  * Aggregates metrics across models and splits into **one CSV**:

    * `Model`, `Split`, `Accuracy`, `F1`, `AUROC`, `PR-AUC`, `ECE`.
  * Saves as `experiments/metrics_table.csv` for convenient table import into LaTeX.

### 5.9 Final Summary Print

At the end, the notebook:

* Lists all plots generated in `experiments/plots/`.
* Highlights which plots to use for:

  * Reliability
  * ROC/PR
  * Confusion matrices
  * Fairness
  * Cross-domain comparisons

---

## 6. How to Run

### 6.1 Kaggle (Recommended)

1. **Create a Kaggle Notebook** and attach:

   * Jigsaw Unintended Bias in Toxicity Classification dataset.
   * Civil Comments dataset (or equivalent).
   * HateXplain dataset.

2. **Upload / mount this repo** into the notebook environment or copy the `notebooks/` and `scripts/` folders.

3. **Run preprocessing notebooks in order**:

   * `cs483_data.ipynb`  → Jigsaw CSVs
   * `civildata.ipynb`   → Civil Comments CSVs
   * `hatexplaindata.ipynb` → HateXplain CSVs

   Check that `data/` now contains (at least):

   ```text
   jigsaw_train.csv, jigsaw_val.csv, jigsaw_test.csv
   civil_train.csv, civil_val.csv, civil_test.csv
   hatexplain_train.csv, hatexplain_val.csv, hatexplain_test.csv
   ```

4. **Run all experiments in one shot**:

   * Open `run_all_experiments.ipynb` and run all cells.

   This will:

   * Verify data files.
   * Train **TF–IDF + Logistic Regression** baseline (Jigsaw → Civil, HateXplain).
   * Train **RoBERTa** with isotonic calibration, early stopping, threshold tuning.
   * Evaluate cross-domain performance.
   * Compute fairness metrics for Jigsaw → Civil.
   * Summarize metrics to `model_comparison.csv`.
   * Generate a quick comparison plot.

5. **Generate full analysis plots**:

   * Open `analysis_plots.ipynb` and run all cells.
   * Collect figures from `experiments/plots/` for your report and presentation.

---

### 6.2 Local Execution

1. **Install dependencies**:

   ```bash
   pip install torch transformers scikit-learn pandas numpy matplotlib seaborn peft
   ```

2. **Prepare raw datasets**:

   * Either mirror the Kaggle directory layout under `input/` (as expected by notebooks), or
   * Edit the `INPUT_ROOT` paths in the preprocessing notebooks to point to your local data.

3. **Run preprocessing**:

   * Execute `cs483_data.ipynb`, `civildata.ipynb`, `hatexplaindata.ipynb` in a local Jupyter / VSCode environment.
   * Confirm that `data/` contains all required CSVs.

4. **Run CLI scripts** (from repo root):

   * TF–IDF logistic regression baseline:

     ```bash
     python scripts/run_tfidf_baselines.py \
       --source_dataset jigsaw \
       --target_datasets civil hatexplain \
       --model logreg \
       --data_dir data \
       --save_preds
     ```

   * RoBERTa with isotonic calibration + threshold tuning:

     ```bash
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
       --patience 2 \
       --tune_threshold \
       --save_preds
     ```

   * Fairness for Jigsaw → Civil:

     ```bash
     python scripts/fairness_metrics.py \
       --dataset civil \
       --split test \
       --pred_file experiments/preds_jigsaw_to_civil.csv \
       --full_data_file data/civil_test_full.csv \
       --group_prefix g_ \
       --out_prefix experiments/fairness_jigsaw_to_civil
     ```

5. **Run analysis_plots.ipynb** to generate final visualizations.

---

## 7. Reproducibility

* All training scripts include `set_seed(seed)`:

  * Seeds Python, NumPy, and PyTorch.
* Mixed precision, LoRA, CORAL can be toggled via CLI flags:

  * `--amp`, `--peft lora`, `--coral_target civil --coral_lambda 0.1`, etc.
* All key experimental configuration is logged via:

  * `summary_*.csv`
  * `model_comparison.csv`
  * Temperature / threshold / isotonic files in `experiments/`.

---

## 8. Limitations & TODOs

Some potential extensions / remaining work items:

* [ ] **More seeds**: Run multi-seed evaluations for RoBERTa (`--seeds 42 43 44`) and aggregate performance.
* [ ] **More models**: Add a DistilBERT / DeBERTa baseline for efficiency comparison.
* [ ] **Richer fairness analysis**:

  * Conditional fairness metrics stratified by toxicity level.
  * Intersectional group analysis (e.g., `g_black & g_female`).
* [ ] **Hyperparameter sweeps**:

  * TF–IDF: vary `ngram_max`, `min_df`, `max_features`.
  * RoBERTa: tune `lr`, `batch_size`, and `max_len`.
* [ ] **Better handling of label noise** in Civil/HateXplain via robust loss or label smoothing.
* [ ] **More calibration methods**:

  * Platt scaling, histogram binning for comparison to isotonic/temperature.
* [ ] **Documentation polish**:

  * Add example result tables / plots into this README.
  * Add badges, environment.yml / requirements.txt for one-click setup.


## 9. Acknowledgements

This project reuses public datasets from:

* **Jigsaw Unintended Bias in Toxicity Classification** (Kaggle)
* **Civil Comments** (Kaggle / Jigsaw)
* **HateXplain** (hate speech dataset with explanations)

and builds on standard libraries:

* PyTorch, Hugging Face Transformers, scikit-learn, matplotlib, seaborn, and peft.
