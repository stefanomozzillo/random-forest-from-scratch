# Random Forest from Scratch — First-Principles Implementation & Real-World Validation

An educational machine learning project implementing **CART Decision Trees** (`ScratchTree`) and **Random Forest Regressors** (`ScratchForest`) from scratch in Python and NumPy, progressively validated against `scikit-learn` across both a controlled benchmark dataset and noisy, real-world real estate records.

## Project Objective

Rather than treating ensemble methods as black-box APIs, this repository was built to explore the mechanics of tree-based regression from first principles—covering variance-reduction (MSE) node splitting, bootstrap aggregation (bagging), and random feature subsampling.

The project follows a structured, progressive experimental workflow:
1. **Algorithmic Verification:** Building `ScratchTree` and `ScratchForest` in pure Python/NumPy and verifying behavioral parity against `scikit-learn` on a clean benchmark (**California Housing**).
2. **Real-World Generalization:** Scaling the workflow end-to-end to raw, uncleaned **Italian Real Estate Market (OMI)** records—handling data sanitization, target leakage detection, hyperparameter tuning, and error diagnostics.

---

## Repository Structure

```text
random-forest-from-scratch/
├── rf/
│   ├── tree.py                  # Custom CART regression tree (variance reduction splitting)
│   └── forest.py                # Bootstrap aggregation & random feature subsampling
├── notebooks/
│   ├── 01_...ipynb              # Single tree implementation, baselines & split equivalence
│   ├── 02_...ipynb              # Ensemble mechanics, bias-variance & subsampling on California Housing
│   └── 03_milan_omi.ipynb       # End-to-end real-world pipeline on Italian OMI real estate data
└── README.md
