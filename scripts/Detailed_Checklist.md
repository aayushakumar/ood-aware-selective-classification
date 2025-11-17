Detailed Checklist

A. Data & Splits
(Completed) Multiple datasets with unified schema (`jigsaw`, `civil`, `hatexplain`) via `load_*`.
(Completed) Reproducibility controls (`set_seed`) and multi-seed runner.
(Partly) Partial: No explicit cleaning (dedup, empty/very short text removal, URL/handle normalization).
(Incompleted) Missing: Descriptive data reports (class balance, length stats) written to disk.

B. Training Framework
(Completed) Optimizer/scheduler: AdamW + linear warmup/decay.
(Completed) AMP mixed precision (`--amp`).
(Completed) Early stopping (`--early_stop`, patience/metric).
(Completed) Class imbalance handling via `WeightedRandomSampler`.
(Completed) LoRA (PEFT) switch (`--peft lora`).
(Completed) CORAL domain alignment (CLS features).
(Partly) Partial: Only CORAL among DA/DG tricks; no mixup, R-Drop, adversarial training, or GRL.
(Incompleted) Missing: Gradient clipping; gradient accumulation; resume-from-checkpoint; “freeze encoder” switch.

C. Evaluation & Cross-Domain
(Completed) In-domain val/test + cross-domain evaluation (source → targets).
(Completed) Metrics: Accuracy, F1 (binary), AUROC, NLL, Brier, Confusion Matrix, ECE (+ val reliability bins CSV).
(Completed) Calibration: Temperature scaling / Isotonic regression.
(Completed) Threshold tuning for F1 on validation, reused at test/cross.
(Completed) CSV dumps of predictions; summary CSV of metrics.
(Partly) Partial: No macro-averaged metrics or class-wise P/R/F1.
(Incompleted) Missing: PR-AUC; automatic plots (ROC, PR, reliability, confusion matrix heatmap).

D. Baselines & Ablations
(Completed) Ablation switches (LoRA / CORAL / calibration / threshold).
(Incompleted) Missing: Classical text baseline (TF-IDF + Logistic Regression / Linear SVM).
(Incompleted) Missing: Automated grid/random search to produce a consolidated ablation table.

E. Reproducibility & Logging
(Completed) Saves model weights, validation logits/probs, predictions, summary.
(Partly) Partial: No args/config/environment snapshot beyond filenames; no best/last metadata bundle.
(Incompleted) Missing: External experiment tracking (W&B/MLflow); standalone evaluation mode for saved checkpoints; batch inference mode.

F. Fairness & Sliced Analysis (if in scope)
(Incompleted) Missing: Slice metrics by identity groups or heuristics; DP/EO style fairness reporting.
(Incompleted) Missing: Error analysis slices (length bins, profanity intensity, URLs, domain origin).

G. Data Pipeline & Efficiency
(Completed) Custom `Dataset` + `DataLoader`.
(Partly) Partial: No dynamic padding (`DataCollatorWithPadding`), no HF Datasets for caching/streaming.
(Incompleted) Missing: Tokenization cache to disk; tuned `num_workers` / `worker_init_fn` for speed.

H. Reporting & Visualization
(Incompleted) Missing: Auto-generated figures (ROC/PR/reliability/confusion-matrix) saved to `experiments/`.
(Incompleted) Missing: Top-K FP/FN examples and qualitative error summaries.
