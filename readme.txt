## Project Overview

This repository contains code for analyzing stress-specific volatile organic compound (VOC) fingerprints in Beech and Oak trees under different environmental stress conditions.

## Objectives

- Identify **stress-specific VOC fingerprints** in Beech and Oak under heat, ozone, and combined stresses.
- Use **PMF** to resolve latent VOC emission factors with biological and temporal meaning.
- Train **Random Forest models** to classify stress type from VOC profiles and interpret key features using **SHAP** (SHapley Additive exPlanations).
- Quantify prediction **uncertainty using entropy** and visualize variability across stress classes.

## Methodology

### PMF (Positive Matrix Factorization)

- Decomposes VOC time series data into latent **factors** representing shared compound emission patterns.
- Factor profiles and time series are validated **compound-wise** by comparing reconstructed signals against raw measurements.
- A **summary Excel file** is generated, showing the dominant factor and unexplained fraction per compound.
- Helpful for identifying biologically relevant emission groups (e.g., heat-induced sesquiterpenes or ozone-linked ketones).

> See: 'PMF_Validation.py'

---

### Random Forest Classifier 
- Input: Preprocessed VOC matrix labeled by stress type (`Pre-stress`, `Heat`, `Ozone`, `Ozone+Heat`).
- Splitting strategy ensures temporal separation of test/train data per class.
- GridSearchCV is used for RF hyperparameter tuning.
- Performance evaluated via:
  - **Confusion Matrix**
  - **ROC-AUC curves**
  - **Bootstrapped Accuracy & F1**
- **SHAP values** explain feature contribution globally and for each stress class.
- Prediction **entropy** is analyzed to assess confidence and model uncertainty.

> See: `Random_forest_SHAP.py`


## Requirements

- Python 3.8+
- Required packages:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `shap`, `openpyxl`, `upsetplot`, `matplotlib-venn`, `tqdm`
- PMF data must be in Excel format with:
  - Time series (`*_TS`)
  - Factor profile (`*_Prof`)
  - VOC ion matrix (`datawave`)

---


This project was developed within the Biogenic Emissions and Air Quality Impacts group, led by Dr. Eva Pfannerstill at Forschungszentrum Jülich. The group focuses on understanding the role of biogenic volatile organic compounds (BVOCs) in plant–atmosphere interactions and their implications for air quality and climate.

https://www.fz-juelich.de/en/research/research-fields/young-scientists/young-investigator-groups/biogenic-emissions-and-air-quality-impacts


---

## Contact

For questions, suggestions, or collaborations, please contact:

[biplobforestry@gmail.com]
[b.dey@fz-juelich.de]


