
# Toxicity-OOD-Fairness

**Toxicity-OOD-Fairness** is a study of how toxicity / hate-speech classifiers behave when they are **trained on one dataset** (e.g. Jigsaw) but **evaluated on another** (e.g. Civil Comments, HateXplain). Our focus is not only on accuracy but also on **calibration** and **group fairness** under domain shift.

## Project Summary

Modern toxicity classifiers are usually evaluated in-domain, on the same distribution they were trained on. In practice, however, models are often deployed on platforms or time periods that look different from the original training data. This **domain shift** can:

- reduce task performance (F1, Accuracy),
- worsen probability quality (higher ECE / NLL),
- and amplify fairness gaps across identity-related subgroups.

This project builds a **reproducible, low-compute** setup to **measure** that degradation and to test **practical remedies** such as post-hoc calibration and lightweight domain adaptation.

## Objectives

1. **Quantify OOD impact**  
   Train on a source dataset and evaluate on other toxicity datasets to measure the drop in F1/Accuracy and the increase in calibration error.

2. **Examine fairness under shift**  
   When identity attributes are available (e.g. Civil Comments), compute group-based metrics (TPR/FPR gaps, equalized-odds gap) to see whether shift disproportionately hurts certain groups.

3. **Test low-cost fixes**  
   Evaluate whether simple, commonly used techniques — post-hoc calibration (temperature scaling, isotonic regression), CORAL-style feature alignment, and small adapter tuning — can partially restore both performance and fairness without retraining large models from scratch.

4. **Keep it class/project friendly**  
   Everything is designed to run on typical university hardware and to be used by multiple teammates working on data, baselines, fairness, and visualization.

## Datasets (planned)

- **Jigsaw Toxic Comment Classification (Kaggle)**  
  Main supervised toxicity dataset. We map to: `text`, `label`.

- **Civil Comments**  
  Contains toxicity score and identity attributes. We binarize toxicity and use identity **only for evaluation**.

- **HateXplain**  
  Multi-label hate/offense/neutral dataset. We harmonize to binary toxicity for cross-dataset experiments.

## Methods (planned)

- **Models / Baselines:** TF–IDF + Logistic Regression, Linear SVM, RoBERTa-base fine-tuning
- **Calibration:** temperature scaling, isotonic regression, reliability diagrams, ECE/NLL/Brier
- **Domain Adaptation:** CORAL (source–target covariance alignment), adapter tuning (PEFT) on small labeled target subsets
- **Evaluation:** F1, Accuracy, AUROC, ECE, NLL, Brier, group fairness metrics

## Project Timeline (high level)

- **Week 9:** Data ingestion, label harmonization, basic fairness metric scripts
- **Week 10:** Train/eval baselines in-domain, first cross-domain results
- **Week 11:** Add post-hoc calibration, compare pre/post on OOD
- **Week 12+:** Lightweight domain adaptation (CORAL, adapters), ablations, robustness, conformal prediction (if time)

## Why this matters

Toxicity models are increasingly used in moderation, safety, and research settings where **explainability, stability, and fairness** matter as much as raw accuracy. A model that looks good in-domain but silently becomes unfair when the data changes is not deployable. This project provides a concrete, dataset-backed way to **surface that risk** and **test realistic mitigations**.
