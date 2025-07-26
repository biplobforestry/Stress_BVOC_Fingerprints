# README

## Project Overview

This repository contains code for analyzing stress-specific volatile organic compound (VOC) fingerprints in Beech and Oak trees under different environmental stress conditions. The work integrates Positive Matrix Factorization (PMF) and Random Forest (RF) machine learning techniques to detect and interpret plant stress responses based on VOC emissions.

## Objectives

* Identify stress-specific VOC fingerprints in Beech and Oak under heat, ozone, and combined ozone + heat stresses.
* Use Positive Matrix Factorization (PMF) to resolve latent VOC emission factors with biological and temporal meaning.
* Train Random Forest models to classify stress types from VOC profiles and interpret key features using SHAP (SHapley Additive exPlanations).
* Quantify prediction uncertainty using entropy and visualize variability across stress classes.

## Methodology

### Positive Matrix Factorization (PMF)

* Decomposes VOC time series data into latent factors representing shared compound emission patterns.
* Factor profiles and time series are validated compound-wise by comparing reconstructed signals against raw measurements.
* A summary Excel file is generated, showing the dominant factor and unexplained fraction per compound.
* This approach is helpful for identifying biologically relevant emission groups, such as heat-induced sesquiterpenes or ozone-linked ketones.

Relevant script: `PMF_Validation.py`

### Random Forest Classifier

* Input: Preprocessed VOC matrix labeled by stress type (Pre-stress, Heat, Ozone, Ozone+Heat).
* The dataset is split with temporal separation to avoid information leakage across training and testing sets.
* GridSearchCV is used for Random Forest hyperparameter tuning.
* Performance is evaluated using:

  * Confusion Matrix
  * ROC-AUC curves
  * Bootstrapped Accuracy and F1 scores
* SHAP values are used to explain the contribution of individual features globally and for each stress class.
* Prediction entropy is analyzed to assess classification confidence and model uncertainty.

Relevant script: `Random_forest_SHAP.py`

## Requirements

* Python 3.8+
* Required packages:

  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn
  * shap
  * openpyxl
  * upsetplot
  * matplotlib-venn
  * tqdm

**Note**: PMF data must be in Excel format and contain:

* Time series data (sheet named `*_TS`)
* Factor profiles (sheet named `*_Prof`)
* VOC ion matrix (sheet named `datawave`)

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to use, modify, and distribute this code with appropriate credit to the original author.

## Citation

If you use this code in your research, please cite:

Dey et al. (2025), *Biogeosciences* (in review)

## Acknowledgements

This work was developed within the Biogenic Emissions and Air Quality Impacts group, led by Dr. Eva Pfannerstill at Forschungszentrum Jülich. The group focuses on understanding the role of biogenic volatile organic compounds (BVOCs) in plant-atmosphere interactions and their implications for air quality and climate.

More information: [https://www.fz-juelich.de/en/research/research-fields/young-scientists/young-investigator-groups/biogenic-emissions-and-air-quality-impacts](https://www.fz-juelich.de/en/research/research-fields/young-scientists/young-investigator-groups/biogenic-emissions-and-air-quality-impacts)

## Contact

For questions, suggestions, or collaborations, please contact:

* [biplobforestry@gmail.com](mailto:biplobforestry@gmail.com)
* [b.dey@fz-juelich.de](mailto:b.dey@fz-juelich.de)
