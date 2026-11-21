import numpy as np
from .tree import ScratchTree


class ScratchForest:
    """Ensemble of CART regression trees using bagging and feature subsampling."""

    def __init__(
        self,
        n_trees=10,       # number of trees in the forest
        max_depth=None,        # max depth passed to each individual tree
        min_samples_split=2,   # min samples to split passed to each tree
        max_features=None,     # number of features considered at each split
        random_state=None,     # seeds the forest RNG for reproducible bootstraps and trees
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """Train an ensemble of trees on bootstrap samples of (X, y)."""
        X_arr = np.asarray(X, dtype=float) # ensure inputs are float NumPy arrays (handles lists, DataFrames, bools, etc.)
        y_arr = np.asarray(y, dtype=float)

        self._rng = np.random.default_rng(self.random_state)
        self.trees = [] # empty tree list

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_samples(X_arr, y_arr)

            # Generate a distinct, reproducible seed for each tree's feature subsampling
            tree_seed = int(self._rng.integers(0, 2**32))

            tree = ScratchTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """Predict target values by averaging predictions across all trees."""
        if not self.trees:
            raise RuntimeError("Call fit() before calling predict().")

        X_arr = np.asarray(X, dtype=float)
        # Matrix of shape (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X_arr) for tree in self.trees])
        # Average across axis 0 (across trees) to get one prediction per sample
        return np.mean(tree_preds, axis=0)

    def _bootstrap_samples(self, X, y):
        """Draw a random sample of size n_samples with replacement."""
        n_samples = X.shape[0]
        idxs = self._rng.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]