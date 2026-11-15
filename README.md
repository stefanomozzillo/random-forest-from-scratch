# California Housing — Random Forest from Scratch

A regression project on the California Housing dataset comparing custom Decision Tree and Random Forest implementations (written in Python and NumPy) against simple baselines and `scikit-learn`.

## Overview

The goal is to predict median house values across California districts while checking whether a custom tree ensemble behaves consistently with standard library implementations in terms of accuracy, overfitting, and runtime.

The repository is organized into three main parts:
- **Baselines:** Median predictor and standardized Linear Regression to set a reference error floor.
- **From-Scratch Models:** Custom `DecisionTree` and `ScratchForest` implementations using variance reduction (MSE) for node splitting.
- **Validation & Comparison:** Side-by-side benchmarks comparing custom models, `scikit-learn`, and baselines on identical data splits.

## Dataset & Experimental Setup

- **Dataset:** California Housing (`20,640` samples, `8` numerical features).
- **Train / Test Split:** `80%` training (`16,512` samples) and `20%` hold-out test set (`4,128` samples), kept fixed across all experiments (`random_state=42`).
- **Subsampling (`N = 2,000` and `N = 5,000`):** Pure-Python tree training is slower than Cython-optimized libraries. Training on fixed subsets of `2,000` and `5,000` rows keeps runtimes practical while still capturing spatial and income patterns across the dataset. All models—including baselines—are evaluated on the exact same subsets for a fair comparison.
- **Metrics:** Models are evaluated using **Mean Absolute Error (MAE)** on both training and test sets to track generalization and overfitting (`Gap = Test MAE - Train MAE`).

## Main Results

- **Tree Equivalence:** At equal depth (`max_depth=6`), `DecisionTree` produces nearly identical predictions to `scikit-learn`'s `DecisionTreeRegressor`, confirming that the splitting logic matches the reference behavior.
- **Improvement Over Baselines:** On `N = 5,000`, the Random Forest models reduce Test MAE to around **`0.39`**, compared to **`~0.88`** for the median baseline and **`~0.53`** for linear regression.
- **Feature Subsampling:** Restricting candidate features at each split (`max_features=3`) lowers test error compared to using all features (`max_features=1.0`), as it reduces correlation between individual trees.
- **Sample Size Effect:** Increasing training samples from `2,000` to `5,000` consistently lowers Test MAE and narrows the train-test gap across both custom and `scikit-learn` forests.

## Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
