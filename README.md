# FAABE: Firefly Algorithm-Tuned Analogy-Based Estimation

Replication package and code for the paper:
****

## Overview
This repository contains the source code, datasets, and experimental results for FAABE and baseline analogy-based software effort estimation models (ABE, GA-ABE, PSO-ABE, DE-ABE, GWO-ABE, and WOA-ABE), as well as the ablation study.

## Requirements
- Python 3.8+
- PyTorch (>= 2.0.0)
- scikit-learn (>= 1.2.0)
- scipy (>= 1.10.0)
- pandas (>= 2.0.0)
- numpy (>= 1.24.0)

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate faabe
```

## Datasets
The `Datasets/` directory includes the 10 benchmark software effort estimation datasets used in the study:
- Albrecht
- China
- COCOMO81
- Desharnais
- Kemerer
- Kitchenham
- Maxwell
- Miyazaki94
- Subbiah
- Valdes-Souto

Summary statistics for each dataset (projects, features, effort ranges) are provided in `stats.txt`.

## Running the Experiments

### 1. Baseline Comparisons
To run 10-fold cross-validation across all 10 datasets for ABE, GA-ABE, PSO-ABE, DE-ABE, GWO-ABE, WOA-ABE, and FAABE:
```bash
python FAABE.py
```
GPU acceleration is used automatically if CUDA is available; otherwise it runs on CPU. Results and logs are saved to `results_ALL_BASELINES_CUDA/`.

### 2. Ablation Study
To run the ablation experiments:
```bash
python FAABE_Ablation.py
```
Results and logs are saved to `results_FAABE_ABLATION_ALL_10_DATASETS/`.

## Results and Outputs
- `results_ALL_BASELINES_CUDA/`: Contains overall evaluation metrics (MMRE, MAE, MSE, RMSE), fold-level results, out-of-fold (OOF) predictions, optimized feature weights, optimization history, and pairwise Wilcoxon signed-rank and Cliff's Delta tests with Holm-Bonferroni correction.
- `results_FAABE_ABLATION_ALL_10_DATASETS/`: Contains fold-level results, overall metrics, and pairwise statistical comparisons for the ablation variants (`ABE_BASE`, `FAABE_NO_PEARSON`, `FAABE_NO_FIREFLY`, `FAABE_NO_IWM`, `FAABE_FULL`).

## License
Apache License 2.0. See `LICENSE` for details.
