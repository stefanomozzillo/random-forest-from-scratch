
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor as SklearnForest

from rf.forest import ScratchForest
from rf.tree import DecisionTree


def make_data(n=300, seed=0):
    return make_friedman1(n_samples=n, n_features=5, noise=1.0, random_state=seed)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def test_builds_the_requested_number_of_trees_and_refit_resets():
    """Catches: wrong tree count, trees accumulating across fit() calls."""
    X, y = make_data(n=100)
    forest = ScratchForest(n_trees=7, max_depth=3, random_state=0)
    forest.fit(X, y)
    assert len(forest.trees) == 7
    forest.fit(X, y)
    assert len(forest.trees) == 7


def test_prediction_is_the_mean_of_the_trees():
    """The forest output must equal the average of its trees' outputs.

    Catches: averaging along the wrong axis, using only some of the trees.
    """
    X, y = make_data(n=150)
    forest = ScratchForest(n_trees=6, max_depth=4, random_state=0)
    forest.fit(X, y)
    expected = np.mean([tree.predict(X) for tree in forest.trees], axis=0)
    pred = forest.predict(X)
    assert pred.shape == (len(X),)
    assert np.allclose(pred, expected)


def test_bootstrap_keeps_features_and_targets_aligned():
    """Row i of the sample must still be paired with its own target.

    Catches: indexing X and y with different random draws, a silent bug that ruins accuracy.
    """
    n = 500
    X = np.column_stack([np.arange(n), np.random.default_rng(0).normal(size=n)]).astype(float)
    y = 10.0 * np.arange(n)
    forest = ScratchForest(n_trees=1, max_depth=2, random_state=0)
    forest.fit(X, y)
    Xb, yb = forest._bootstrap_samples(X, y)
    assert Xb.shape == X.shape and yb.shape == y.shape
    assert np.array_equal(yb, 10.0 * Xb[:, 0])


def test_bootstrap_draws_with_replacement():
    """About 63.2% of rows appear in a bootstrap sample (1 - 1/e), with duplicates.

    Catches: sampling without replacement, which would use every row exactly once.
    """
    n = 2000
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.arange(n, dtype=float)
    forest = ScratchForest(n_trees=1, max_depth=1, random_state=0)
    forest.fit(X, y)
    Xb, _ = forest._bootstrap_samples(X, y)
    unique_fraction = len(np.unique(Xb[:, 0])) / n
    assert 0.60 < unique_fraction < 0.665


def test_same_seed_same_forest_different_seed_different_forest():
    """Catches: ignored random_state (irreproducible experiments) or seed reuse across runs."""
    X, y = make_data(n=150)
    a = ScratchForest(n_trees=5, max_depth=5, random_state=0)
    b = ScratchForest(n_trees=5, max_depth=5, random_state=0)
    c = ScratchForest(n_trees=5, max_depth=5, random_state=1)
    for forest in (a, b, c):
        forest.fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert not np.allclose(a.predict(X), c.predict(X))


def test_trees_in_the_forest_are_different():
    """Each tree sees its own bootstrap sample, so the trees must not all be identical.

    Catches: bootstrap not applied, or the same data/seed given to every tree.
    """
    X, y = make_data(n=200)
    forest = ScratchForest(n_trees=5, max_depth=6, random_state=0)
    forest.fit(X, y)
    preds = [tree.predict(X) for tree in forest.trees]
    assert any(not np.allclose(preds[0], p) for p in preds[1:])


def test_hyperparameters_are_forwarded_to_the_trees():
    """Catches: max_depth or max_features silently ignored by the forest."""
    X, y = make_data(n=200)

    stump = ScratchForest(n_trees=1, max_depth=1, random_state=0)
    stump.fit(X, y)
    assert len(np.unique(stump.predict(X))) <= 2

    all_features = ScratchForest(n_trees=3, max_depth=5, max_features=None, random_state=0)
    one_feature = ScratchForest(n_trees=3, max_depth=5, max_features=1, random_state=0)
    all_features.fit(X, y)
    one_feature.fit(X, y)
    assert not np.allclose(all_features.predict(X), one_feature.predict(X))


def test_input_types_are_equivalent():
    """DataFrames, Series and lists must give the same result as NumPy arrays.

    Catches: type handling that breaks when the notebooks pass pandas objects.
    """
    X, y = make_data(n=100)
    reference = ScratchForest(n_trees=4, max_depth=4, random_state=0)
    reference.fit(X, y)
    expected = reference.predict(X)

    from_pandas = ScratchForest(n_trees=4, max_depth=4, random_state=0)
    from_pandas.fit(pd.DataFrame(X, columns=list("abcde")), pd.Series(y))
    assert np.array_equal(from_pandas.predict(pd.DataFrame(X, columns=list("abcde"))), expected)

    from_lists = ScratchForest(n_trees=4, max_depth=4, random_state=0)
    from_lists.fit(X.tolist(), y.tolist())
    assert np.array_equal(from_lists.predict(X.tolist()), expected)


def test_predict_before_fit_raises():
    """Catches: silent garbage (or a confusing error) when the model was never trained."""
    with pytest.raises(RuntimeError):
        ScratchForest().predict(np.zeros((3, 2)))


def test_forest_beats_a_single_tree_on_noisy_data():
    """Averaging bootstrapped trees must reduce variance relative to one deep tree.

    This is the whole point of the algorithm. Catches: bagging that does not actually help.
    """
    X, y = make_data(n=500)
    X_tr, y_tr, X_te, y_te = X[:300], y[:300], X[300:], y[300:]

    tree = DecisionTree(max_depth=8)
    tree.fit(X_tr, y_tr)
    forest = ScratchForest(n_trees=20, max_depth=8, random_state=0)
    forest.fit(X_tr, y_tr)

    assert mae(y_te, forest.predict(X_te)) < mae(y_te, tree.predict(X_te))


def test_close_to_scikit_learn_forest():
    """With equal settings the two forests should reach a similar error (within 20%).

    Catches: subtle defects that the unit tests above cannot see.
    """
    X, y = make_data(n=500)
    X_tr, y_tr, X_te, y_te = X[:300], y[:300], X[300:], y[300:]

    mine = ScratchForest(n_trees=20, max_depth=8, max_features=None, random_state=0)
    mine.fit(X_tr, y_tr)
    ref = SklearnForest(n_estimators=20, max_depth=8, max_features=1.0, random_state=0)
    ref.fit(X_tr, y_tr)

    mae_mine = mae(y_te, mine.predict(X_te))
    mae_ref = mae(y_te, ref.predict(X_te))
    assert mae_mine < 1.2 * mae_ref