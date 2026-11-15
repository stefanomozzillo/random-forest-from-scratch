import numpy as np


class Node:
    """Single node in a regression tree (either an internal split or a leaf)."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """CART-style regression tree built from scratch using variance reduction."""

    def __init__(
        self,
        max_depth=None,        # Grows unpruned trees by default
        min_samples_split=2,   # Allows splitting down to single-sample leaves
        max_features=None,     # Evaluates all features per split for a standard tree
        random_state=None,     # Seeds the RNG for reproducible feature subsampling
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        """Build the regression tree recursively from training data (X, y)."""
        X_arr = np.asarray(X, dtype=float) # ensure inputs are float NumPy arrays (handles lists, DataFrames, bools, etc.)
        y_arr = np.asarray(y, dtype=float)


        self._rng = np.random.default_rng(self.random_state)
        self.n_features_ = X_arr.shape[1]
        self.max_features_ = (
            self.n_features_
            if self.max_features is None
            else min(self.n_features_, self.max_features)
        )
        self.root = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def predict(self, X):
        """Predict target values by routing each sample from root to a leaf."""
        if self.root is None:
            raise RuntimeError("Call fit() before calling predict().")  

        X_arr = np.asarray(X, dtype=float) # ensure inputs are float NumPy arrays (handles lists, DataFrames, bools, etc.)
        return np.array([self._traverse_tree(x, self.root) for x in X_arr])

    def _build_tree(self, X, y, depth=0):
        """Recursively partition data until a stopping criterion is met."""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria: max depth reached, too few samples, or pure target
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_labels == 1
        ):
            return Node(value=float(np.mean(y)))  # leaf value prediction

        feat_idxs = self._rng.choice(n_feats, size=self.max_features_, replace=False)

        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        # If no valid split improves variance, turn current node into a leaf
        if best_feature is None:
            return Node(value=float(np.mean(y)))  # no valid split found, fall back to leaf mean

        left_idxs = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_idxs = np.argwhere(X[:, best_feature] > best_threshold).flatten()

        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)


        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _best_split(self, X, y, feat_idxs):
        """Find the best (feature, threshold) split maximizing gain."""
        best_gain = 0.0  # A split is useful only if it strictly improves over the parent (gain > 0)
        best_feature = None
        best_threshold = None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]  # X column turned into a monodimensional array
            thresholds = np.unique(X_col) #remove duplicate to find possible cutting points

            for threshold in thresholds:
                left_idxs = np.argwhere(X_col <= threshold).flatten()    
                right_idxs = np.argwhere(X_col > threshold).flatten()
                gain = self._information_gain(y, y[left_idxs], y[right_idxs])

                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(feat_idx)
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _information_gain(self, y, y_left, y_right):
        """Compute gain as parent MSE minus weighted average of children MSE."""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0.0  # no data division means 0 gain

        parent_mse = self._mse(y) #parent entropy

        # average children entropy
        weighted_children_mse = (    
            (n_left / n) * self._mse(y_left)
            + (n_right / n) * self._mse(y_right)
        )
        # return information gain 
        return parent_mse - weighted_children_mse

    def _mse(self, y):
        """Compute Mean Squared Error of target values."""
        return float(np.mean((y - np.mean(y)) ** 2))

    def _traverse_tree(self, x, node):
        """Recursively navigate down the tree for a single sample x until reaching a leaf."""
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
