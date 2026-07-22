# A Robustness-First Cross-Disease Framework Identifies a Universal Microbial Redox Axis and Disease-Specific Metabolic Divergence in Colorectal Cancer, Crohn's Disease, and Liver Cirrhosis

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn 1.7.2](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the complete analysis pipeline for:

> **"A robustness-first analytical framework identifies conserved microbial signatures and disease-specific metabolic responses across three gut-associated diseases: a cross-sectional multi-omics computational study"**

The pipeline systematically benchmarks **1,152 preprocessing–model configurations** across six binary classification tasks (three diseases × two omics platforms), selects a unified pipeline using a **maximin robustness criterion** (maximising worst-case AUC across all tasks), and applies **SHAP-based feature attribution** and **KEGG pathway over-representation analysis** to identify conserved and disease-specific biological signals.

**Diseases:** Colorectal Cancer (CRC) · Crohn's Disease (CD) · Liver Cirrhosis (LC)  
**Omics:** Gut metagenomics (species-level abundance) · Serum metabolomics (LC-MS)

## Repository Structure

```
robustness-first-XAI-Gut-Liver/
├── LICENSE
├── README.md
├── requirements.txt
│
├── Data/
│   ├── Processed/
│   │   ├── Metabolomics/
│   │   │   ├── CD/
│   │   │   │   ├── CD_HILIC NEGATIVE ION MODE.xlsx
│   │   │   │   ├── CD_HILIC POSITIVE ION MODE.xlsx
│   │   │   │   ├── CD_Reversed phase NEGATIVE ION MODE.xlsx
│   │   │   │   ├── CD_Reversed phase POSITIVE ION MODE.xlsx
│   │   │   │   └── CD_label.xlsx
│   │   │   ├── CRC/
│   │   │   │   ├── CRC_Metabolomics_Data.xlsx
│   │   │   │   └── CRC_Metabolomics_label.xlsx
│   │   │   └── LC/
│   │   │       └── LC_Metabolomics.xlsx
│   │   └── Metagenomics/
│   │       ├── CD_external/
│   │       │   └── phylum_abundance_PRJEB2054.xlsx
│   │       ├── CD_train/
│   │       ├── CRC_external/
│   │       │   └── phylum_abundance_PRJEB6070.xlsx
│   │       ├── CRC_train/
│   │       │   └── phylum_abundance_PRJEB10878.xlsx
│   │       ├── LC_external/
│   │       │   └── phylum_abundance_PRJEB38481.xlsx
│   │       ├── LC_train/
│   │       │   ├── phylum_abundance_PRJEB15371.xlsx
│   │       │   └── phylum_abundance_PRJEB6337.xlsx
│   │       ├── abbreviation_mapping.xlsx
│   │       ├── msp_kegg_pathways.xlsx
│   │       └── mspmap.xlsx
│   └── raw/
│       └── DOWNLOAD_INSTRUCTIONS.md
│
├── Figures/
│   ├── Main/
│   │   ├── figure1_Prosedure.tif
│   │   ├── figure2_Benchmarking.tif
│   │   ├── Figure3_Feature.tif
│   │   ├── figure4_Pathway.tif
│   │   └── Figure5_Inference.tif
│   └── Supplimentary/
│       ├── S_figure1_flowdiagram.png
        ├── S_figure2_feature_selection_snr.png
│       ├── S_figure3_best_config_per_task.png
│       ├── S_figure4_univariate_auc_distribution.png
│       └── S_figure5_upset_plot.jpg
│
├── Results/
│   ├── 1_Benchmarking/
│   │   ├── Best-worst rank of configs/
│   │   │   └── all_configs_ranked_by_robustness.csv
│   │   ├── Signature_validation/
│   │   │   └── ALL_RESULTS.csv
│   │   ├── Univariate_results/
│   │   │   └── all_feature_auc_results.csv
│   │   ├── metabolomics_CD/
│   │   │   └── all_results.csv
│   │   ├── metabolomics_CRC/
│   │   │   └── all_results.csv
│   │   ├── metabolomics_LC/
│   │   │   └── all_results.csv
│   │   ├── microbiome_CD/
│   │   │   └── results.csv
│   │   ├── microbiome_CRC/
│   │   │   └── results.csv
│   │   └── microbiome_LC/
│   │       └── results.csv
│   ├── 2_SHAP_results/
│   │   ├── shap_CD_metabolomics/
│   │   │   └── ALL_SHAP_VALUES.csv
│   │   ├── shap_CD_microbiome/
│   │   │   └── ALL_SHAP_VALUES.csv
│   │   ├── shap_CRC_metabolomics/
│   │   │   └── ALL_SHAP_VALUES.csv
│   │   ├── shap_CRC_microbiome/
│   │   │   └── ALL_SHAP_VALUES.csv
│   │   ├── shap_LC_metabolomics/
│   │   │   └── ALL_SHAP_VALUES.csv
│   │   └── shap_LC_microbiome/
│   │       └── ALL_SHAP_VALUES.csv
│   ├── 3_Cross_disease_feature/
│   │   └── triplecommon_features.xlsx
│   ├── 4_pathway_results/
│   │   ├── CD_metabolomics/
│   │   │   └── p0.05.xlsx
│   │   ├── CD_microbiome/
│   │   │   └── PATHWAY_ORA_SIGNIFICANT_FDR0.05.xlsx
│   │   ├── CRC_metabolomics/
│   │   │   └── p0.05.xlsx
│   │   ├── CRC_microbiome/
│   │   │   └── PATHWAY_ORA_SIGNIFICANT_FDR0.05.xlsx
│   │   ├── LC_metabolomics/
│   │   │   └── p0.05.xlsx
│   │   └── LC_microbiome/
│   │       └── PATHWAY_ORA_SIGNIFICANT_FDR0.05.xlsx
│   └── SNR_analysis/
│       └── meta_component_effects.csv
│
└── scripts/
    ├── 01_Benchmarking/
    │   ├── microbe_ml.py                  # ML benchmarking — microbiome (all 3 diseases)
    │   ├── LC_metabolomics_ml.py          # ML benchmarking — LC metabolomics
    │   ├── CD_metabolomics_ml.py          # ML benchmarking — CD metabolomics
    │   ├── CRC_metabolomics_ml.py         # ML benchmarking — CRC metabolomics
    │   ├── univariate_ml.py               # Single-feature AUC (dysbiosis architecture)
    │   ├── signature_validation_ml.py     # 19-species signature validation
    │   └── best_worst.py                  # Configuration robustness ranking
    ├── 02_SHAP_pipelines/
    │   ├── shap_Microbe.py                # SHAP attribution — microbiome
    │   ├── shap_LC_Metabolomics.py        # SHAP attribution — LC metabolomics
    │   ├── shap_CD_Metabolomics.py        # SHAP attribution — CD metabolomics
    │   └── Shap_CRC_Metabolomics.py       # SHAP attribution — CRC metabolomics
    ├── 03_Pathway_ora_pipelines/
    │   ├── metagenomics_ora.py            # KEGG pathway ORA — microbiome
    │   └── metabolomics_ora.py            # KEGG pathway ORA — metabolomics
    ├── 04_Other_analysis_pipelines/
    │   ├── cross_disease_feature_signature.py  # Cross-disease feature overlap
    │   └── snr_analysis.py                     # Preprocessing SNR analysis
    └── 05_Figure_generation_pipelines/
    |    └── raincloud_plot.py              # Half-violin raincloud plots
    |__ Reviewer's_analysis
```

---

## Methods Summary

### Configuration Space — 1,152 per task

| Component | Options | Count |
|-----------|---------|-------|
| **Scaling** | None · Standard ((x−μ)/σ) · Robust ((x−med)/IQR) · Log (log1p) | 4 |
| **Feature selection** | Variance threshold · ANOVA F-test · Mutual information · RF importance · Mann-Whitney U (BH) · None — each at 10 / 30 / 50 % retention (except None) | 16 |
| **Class balancing** | None · SMOTE · Random undersampling | 3 |
| **Classifier** | Random Forest · XGBoost · LightGBM · Logistic Regression · SVM-RBF · MLP | 6 |

**Total: 4 × 16 × 3 × 6 = 1,152 configurations per task**

### Maximin Optimisation

The selected configuration maximises the *minimum* AUC across all six tasks:

```
c* = argmax_c  min_{t ∈ {1,...,6}}  AUC(c, t)
```

**Optimal pipeline:** XGBoost + Standard scaling + Random Forest feature selection + No balancing  
**Minimum cross-task AUC:** 0.882 (permutation *p* < 0.01, 1,000 iterations)

### Evaluation

- **Nested cross-validation:** outer 5-fold stratified / inner 3-fold for hyperparameter tuning
- All preprocessing applied strictly within folds (no data leakage)
- External validation on independent European cohorts (CRC only; CD and LC severely limited)
- Seed 42 throughout for reproducibility

---

## Datasets

### Microbiome (metagenomics)

| Disease | Training cohort | n (case / control) | External cohort 
|---------|----------------|-------------------|----------------|---|
| CRC | PRJEB10878 (China) | — | PRJEB6070 (France) 
| CD | PRJEB15371 (China) | — | PRJEB2054 (Spain) 
| LC | PRJEB6337 (China) | — | PRJEB38481 (UK) 

Species-level abundance profiles (1,990 metagenomic species pan-genomes) from the [Human Gut Microbiome Atlas](https://www.microbiomeatlas.org/).

### Metabolomics (LC-MS)

| Disease | Source | n | Notes |
|---------|--------|---|-------|
| CRC | Metabolomics Workbench **ST000284**  Single data + label file |
| CD | Metabolomics Workbench **ST000899**  Four ion-mode files + label file |
| LC | Hoyles *et al.* 2021 (supplementary) 

See `data/raw_data_download/DOWNLOAD_INSTRUCTIONS.md` for step-by-step download guidance.

---

## Installation

### Requirements

- Python **3.10** (tested with 3.10.11)
- pip

### Setup

```bash
# Clone
git clone https://github.com/sahukartika/robustness-first-XAI-Gut-Liver.git
cd robustness-first-XAI-Gut-Liver

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dependencies

| Package | Version |
|---------|---------|
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scipy | 1.15.3 |
| statsmodels | 0.14.5 |
| scikit-learn | 1.7.2 |
| imbalanced-learn | 0.14.0 |
| xgboost | 3.0.5 |
| lightgbm | 4.6.0 |
| shap | 0.49.1 |
| joblib | 1.5.2 |
| requests | 2.32.5 |
| matplotlib-venn | 1.1.2 |
| upsetplot | 0.9.0 |
| openpyxl | 3.1.5 |

---

## Usage

All scripts are self-contained. Edit the configuration block inside `if __name__ == "__main__"` at the bottom of each script, then run it directly. Scripts are designed to be executed from the repository root.

---

### Recommended Execution Order

```
Stage 1 — ML Benchmarking
  1a.  microbe_ml.py
  1b.  LC_metabolomics_ml.py
  1c.  CD_metabolomics_ml.py
  1d.  CRC_metabolomics_ml.py
  1.e  raincloud_plot.py
  1.f  best_worst.py

Stage 2 — Preprocessing Effect Analysis
  2a.  snr_analysis.py
  2b.  univariate_ml.py

Stage 3 — Univariate Analysis
  3.  univariate_ml.py

Stage 4 — SHAP Feature Attribution
  4a.  shap_Microbe.py          (run once per disease: CRC, CD, LC)
  4b.  shap_LC_Metabolomics.py
  4c.  shap_CD_Metabolomics.py
  4d.  Shap_CRC_Metabolomics.py

Stage 5 — Cross-Disease Signature
  5a.  cross_disease_feature_signature.py
  5b.  signature_validation_ml.py

Stage 6 — Pathway Analysis
  6a.  metagenomics_ora.py      (run once per disease)
  6b.  metabolomics_ora.py      (run once per disease)
```


