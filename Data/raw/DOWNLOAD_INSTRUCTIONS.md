# Raw Data Download Instructions

This file describes how to obtain all raw datasets used in the study. Processed versions ready for direct use with the pipeline are available in `../processed/`.

---

## 1. Gut Microbiome Data — Human Gut Microbiome Atlas

All microbiome datasets were obtained from the **Human Gut Microbiome Atlas**:

> **URL:** https://www.microbiomeatlas.org/downloads.php

### Files to download

From the Downloads page, obtain the following two files:

| File | Description |
|------|-------------|
| `vect_atlas.csv.gz` | Species abundance matrix (1,990 metagenomic species pan-genomes × all samples) |
| `sampleID.csv` | Sample metadata table (sample IDs, study accessions, disease labels, cohort info) |

### Steps

1. Go to https://www.microbiomeatlas.org/downloads.php
2. Download **Species abundance matrix** (`vect_atlas.csv.gz`) 
3. Download **Sample metadata table** (`sampleID.csv`)
4. Decompress: `vect_atlas.csv.gz`
5. Use `sampleID.csv` to subset samples by BioProject accession for each cohort:

| Disease | Training cohort | BioProject | Country |
|---------|----------------|-----------|---------|
| CRC | Training | PRJEB10878 | China |
| CD | Training | PRJEB15371 | China |
| LC | Training | PRJEB6337 | China |
| CRC | External validation | PRJEB6070 | France |
| CD | External validation | PRJEB2054 | Spain |
| LC | External validation | PRJEB38481 | UK |

6. Filter `vect_atlas.csv` rows to retain only samples from the target BioProject using the `sampleID.csv` metadata.
7. The resulting matrices (species × samples) form the input Excel files for `microbe_ml.py`.

> **Note:** Missing values (undetected species) are imputed with zero, consistent with the assumption that undetected species are absent or below the detection limit.

---

## 2. Metabolomics Data — Colorectal Cancer (CRC)

**Source:** Metabolomics Workbench, Study **ST000284**

### Steps

1. Go to https://www.metabolomicsworkbench.org/
2. Search for study **ST000284** (or browse directly: https://www.metabolomicsworkbench.org/data/DRCCStudySummary.php?Mode=SetupSearch&ResultType=1&StudyID=ST000284)
3. Download the processed data file (peak intensity table) and associated metadata/label file
4. The downloaded files correspond to:
   - `CRC_Metabolomics_Data.xlsx` — feature × sample intensity matrix
   - `CRC_Metabolomics_label.xlsx` — sample group labels

These are the direct inputs to `CRC_metabolomics_ml.py` and `Shap_CRC_Metabolomics.py`.

---

## 3. Metabolomics Data — Crohn's Disease (CD)

**Source:** Metabolomics Workbench, Study **ST000899**

### Steps

1. Go to https://www.metabolomicsworkbench.org/
2. Search for study **ST000899** (or browse directly: https://www.metabolomicsworkbench.org/data/DRCCStudySummary.php?Mode=SetupSearch&ResultType=1&StudyID=ST000899)
3. Download the four ion-mode data files and the label file. The study provides data split by ionisation mode:

| File | Ion mode |
|------|----------|
| `CD_HILIC POSITIVE ION MODE.xlsx` | HILIC positive |
| `CD_HILIC NEGATIVE ION MODE.xlsx` | HILIC negative |
| `CD_Reversed phase POSITIVE ION MODE.xlsx` | Reversed-phase positive |
| `CD_Reversed phase NEGATIVE ION MODE.xlsx` | Reversed-phase negative |
| `CD_label.xlsx` | Sample group labels |

These are the direct inputs to `CD_metabolomics_ml.py` and `shap_CD_Metabolomics.py`.

> **Note:** The CD metabolomics cohort is small (n = 40; 20 cases vs 20 controls). Results should be interpreted with caution.

---

## 4. Metabolomics Data — Liver Cirrhosis (LC)

**Source:** Supplementary material of the following published paper:

>Moreau R, Clària J, Aguilar F, et al. Blood metabolomics uncovers inflammation-associated mitochondrial dysfunction as a potential mechanism underlying ACLF. J Hepatol. 2020;72:688–701. doi: 10.1016/j.jhep.2019.11.009

> **Note:** This paper originally concerned hepatic steatosis, but the serum metabolomics data from the LC cohort referenced in the current study was obtained from the supplementary data of:


### Steps

1. Access the paper via the DOI above 
2. Download the supplementary data file containing the serum metabolomics matrix
3. Format as:
   - First column: metabolite names
   - Remaining columns: samples prefixed with class identifier (e.g., `ACLF_001` for cirrhosis, `HS_001` for healthy)
4. Save as `LC_Metabolomics.xlsx` — this is the direct input to `LC_metabolomics_ml.py` and `shap_LC_Metabolomics.py`

> **Note:** The LC metabolomics cohort has severe class imbalance (874 disease vs 29 healthy; ratio ~30:1), which may affect performance estimation despite balancing strategies applied in the pipeline.

---


