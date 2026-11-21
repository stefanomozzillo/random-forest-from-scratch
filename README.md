# Random Forest from Scratch — First-Principles Implementation & Real-World Validation

An educational machine learning project implementing **CART Decision Trees** (`ScratchTree`) and **Random Forest Regressors** (`ScratchForest`) from scratch in Python and NumPy, progressively validated against `scikit-learn` across both a controlled benchmark dataset and noisy, real-world real estate records.

## Project Objective

Rather than treating ensemble methods as black-box APIs, this repository was built to explore the mechanics of tree-based regression from first principles—covering variance-reduction (MSE) node splitting, bootstrap aggregation (bagging), and random feature subsampling.

The project follows a structured, three-stage workflow:
1. **From-Scratch Core Implementation (`rf/`):** Designing and coding `ScratchTree` and `ScratchForest` in pure Python and NumPy with zero reliance on external tree libraries.
2. **Algorithmic Verification (`Notebooks 01–02`):** Benchmarking the custom models against simple baselines and `scikit-learn` on a controlled dataset (**California Housing**) to verify split equivalence, bagging variance reduction, and feature subsampling behavior.
3. **Real-World Generalization (`Notebook 03`):** Scaling the workflow end-to-end to raw, uncleaned **Italian Real Estate Market (OMI)** records—handling data sanitization, target leakage detection, hyperparameter tuning, and error diagnostics.



---

## Repository Structure

```text
random-forest-from-scratch/
├── rf/
│   ├── tree.py                  # Custom CART regression tree (variance reduction splitting)
│   └── forest.py                # Bootstrap aggregation & random feature subsampling
├── notebooks/
│   ├── 01_decision_tree.ipynb   # Single tree validation, baselines & split equivalence
│   ├── 02_random_forest.ipynb   # Ensemble mechanics, bias-variance & subsampling on California Housing
│   └── 03_milan_omi.ipynb       # End-to-end real-world pipeline on Italian OMI real estate data
└── README.md
```

---

## Development & Experimental Progression

### Stage 1: Core From-Scratch Implementation (`rf/tree.py`, `rf/forest.py`)
- **`ScratchTree`:** Recursive binary CART regressor selecting optimal feature-threshold splits via variance (MSE) reduction, controlled by stopping criteria (`max_depth`, `min_samples_split`).
- **`ScratchForest`:** Ensemble wrapper implementing bootstrap sampling with replacement (bagging), per-split random feature subsampling (`max_features`), and prediction averaging across trees.

### Stage 2: Controlled Benchmark — California Housing (`Notebooks 01–02`)
- **Setup:** `20,640` samples, `8` numerical features (`80/20` train/test split, `random_state=42`). Subsets of `N = 2,000` and `N = 5,000` were used to analyze scaling behavior while keeping pure-Python runtimes practical.
- **Baselines:** Median predictor (`Test MAE ~0.88`) and standardized Linear Regression (`Test MAE ~0.53`) established non-trivial error floors.
- **Key Takeaways:**
  - **Split Equivalence:** At equal depth (`max_depth=6`), `ScratchTree` produces nearly identical predictions to `DecisionTreeRegressor`, confirming exact mathematical alignment in split selection.
  - **Decorrelation via `max_features`:** Restricting candidate features per split (`max_features=3` vs. `all`) reduces inter-tree correlation and lowers Test MAE to **`~0.39`**.
  - **Sample Size & Overfitting:** Scaling from `2,000` to `5,000` training samples consistently reduces generalization error and narrows the `Test - Train MAE` gap.

### Stage 3: Real-World Validation — Italian OMI Real Estate Data (`Notebook 03`)
- **Setup:** Predicting property sale prices (`€/m²`) from raw Osservatorio del Mercato Immobiliare (OMI) records, requiring type sanitization, missing-value handling, and encoding of high-cardinality geographic and structural features.
- **Key Takeaways:**
  - **Target Leakage Detection:** Exploratory correlation analysis revealed that rental rates (`Loc_min`, `Loc_max`) correlated strongly with sale prices (Pearson = 0.914$, Spearman = 0.912$). Excluding rental columns prevented trivial target leakage and forced the ensemble to learn genuine spatial and structural relationships.
  - **Depth & Bias–Variance Trade-off:** Increasing `max_depth` from `10` to `20` reduced Test MAE by over **60%** (from **`251.28 €/m²`** down to **`92.74 €/m²`**), demonstrating that deep splits are essential to capture fine-grained municipal and micro-zone price hierarchies.
  - **Parity with `scikit-learn`:** With `n_trees=25` and `max_depth=20`, `ScratchForest` achieved a **Test MAE of `92.74 €/m²`** compared to `scikit-learn`'s **`92.66 €/m²`** (`< 0.1%` difference), tracking both the bimodal peaks and the upper luxury tail (`~20,000 €/m²`) of the true price distribution.
  - **Compute vs. Accuracy Trade-off:** Subsampling features (`max_features=80`) reduced training runtime by over `20%` with a modest accuracy trade-off (`98.52 €/m²`), while contrasting pure-Python execution against `scikit-learn` highlighted the systems impact of Cython compilation and multi-core parallelism (`n_jobs=-1`).
