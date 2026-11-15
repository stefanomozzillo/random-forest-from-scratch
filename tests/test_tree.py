

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from rf.tree import DecisionTree

# Toy data: area (m^2) -> price (thousand EUR). Best first split is area <= 75.
X_TOY = np.array([[40.0], [50.0], [60.0], [90.0], [100.0], [120.0]])
y_TOY = np.array([100.0, 120.0, 130.0, 250.0, 270.0, 300.0])


def make_data(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 3 * X[:, 0] + np.sin(2 * X[:, 1]) + rng.normal(scale=0.3, size=n)
    return X, y


def test_single_split_matches_hand_computation():
    """A depth-1 tree must split at area <= 75 and predict each side's mean.

    Catches: wrong split criterion, off-by-one in thresholds, wrong leaf value.
    """
    tree = DecisionTree(max_depth=1)
    tree.fit(X_TOY, y_TOY)
    pred = tree.predict(X_TOY)
    assert np.allclose(pred[:3], 350 / 3)
    assert np.allclose(pred[3:], 820 / 3)


def test_unbounded_tree_memorizes_training_set():
    """Without limits, every leaf holds one sample, so training targets are reproduced.

    Catches: premature stopping, broken recursion, wrong leaf values on larger data.
    """
    X, y = make_data(n=100)
    tree = DecisionTree(max_depth=None, min_samples_split=2)
    tree.fit(X, y)
    assert np.allclose(tree.predict(X), y)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_max_depth_limits_the_number_of_leaves(depth):
    """A tree of depth k has at most 2**k leaves, hence at most 2**k distinct predictions.

    Catches: depth counted off by one (the most common bug with this parameter).
    """
    X, y = make_data()
    tree = DecisionTree(max_depth=depth)
    tree.fit(X, y)
    assert len(np.unique(tree.predict(X))) <= 2 ** depth


def test_min_samples_split_boundary():
    """A node with n samples may split iff n >= min_samples_split (scikit-learn semantics).

    Catches: '>' instead of '>=' when deciding whether to stop.
    """
    X, y = make_data(n=50)

    no_split = DecisionTree(min_samples_split=len(y) + 1)
    no_split.fit(X, y)
    assert np.allclose(no_split.predict(X), y.mean())

    one_split = DecisionTree(min_samples_split=len(y), max_depth=1)
    one_split.fit(X, y)
    assert len(np.unique(one_split.predict(X))) == 2


def test_degenerate_inputs_do_not_crash_or_recurse_forever():
    """Constant targets or identical rows leave no valid split: the node must become a leaf.

    Catches: infinite recursion, empty child nodes, crashes when the best gain is zero.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))

    constant_y = DecisionTree()
    constant_y.fit(X, np.full(30, 7.0))
    assert np.allclose(constant_y.predict(X), 7.0)

    y = rng.normal(size=30)
    identical_rows = DecisionTree()
    identical_rows.fit(np.ones((30, 3)), y)
    assert np.allclose(identical_rows.predict(np.ones((30, 3))), y.mean())


def test_random_state_only_matters_when_subsampling_features():
    """Without feature subsampling the tree is deterministic; with it, the seed decides.

    Catches: hidden randomness, ignored random_state, ignored max_features.
    """
    X, y = make_data(n=300, d=6)

    a = DecisionTree(max_depth=6, random_state=0)
    b = DecisionTree(max_depth=6, random_state=1)
    a.fit(X, y)
    b.fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))

    s0 = DecisionTree(max_depth=6, max_features=2, random_state=0)
    s0_again = DecisionTree(max_depth=6, max_features=2, random_state=0)
    s1 = DecisionTree(max_depth=6, max_features=2, random_state=1)
    for tree in (s0, s0_again, s1):
        tree.fit(X, y)
    assert np.array_equal(s0.predict(X), s0_again.predict(X))
    assert not np.array_equal(s0.predict(X), s1.predict(X))


def test_predictions_are_means_of_training_targets():
    """Every leaf is an average of training targets, so predictions stay inside their range.

    Catches: wrong leaf values, wrong output shape (including single-row input).
    """
    X, y = make_data()
    tree = DecisionTree(max_depth=5)
    tree.fit(X, y)
    X_new, _ = make_data(n=50, seed=1)
    pred = tree.predict(X_new)
    assert pred.shape == (50,)
    assert pred.min() >= y.min() and pred.max() <= y.max()
    assert tree.predict(X_new[:1]).shape == (1,)


def test_matches_scikit_learn_decision_tree():
    """With continuous features there are no ties, so both trees must pick the same splits.

    Catches: any error in the split search, which the toy tests cannot expose.
    """
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=0)
    X_tr, y_tr, X_te, y_te = X[:300], y[:300], X[300:], y[300:]

    mine = DecisionTree(max_depth=5)
    mine.fit(X_tr, y_tr)
    ref = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X_tr, y_tr)

    assert np.allclose(mine.predict(X_tr), ref.predict(X_tr))
    mae_mine = np.mean(np.abs(mine.predict(X_te) - y_te))
    mae_ref = np.mean(np.abs(ref.predict(X_te) - y_te))
    assert abs(mae_mine - mae_ref) / mae_ref < 0.05