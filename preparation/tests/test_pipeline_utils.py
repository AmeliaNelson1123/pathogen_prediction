import numpy as np
import pandas as pd
import preparation.pipeline_utils as pu


def test_constants_present():
    assert pu.RANDOM_STATE == 42
    assert pu.TEST_SIZE == 0.22
    assert pu.Y_COL == "binary_listeria_presense"
    assert "log of index" in pu.LEAK_COLS
    assert "cluster_kmeans" in pu.LEAK_COLS


def test_data_path_exists(real_data_path):
    assert pu.data_path() == real_data_path
    assert pu.data_path().exists()


def test_set_seeds_is_deterministic():
    pu.set_seeds(42)
    a = np.random.rand(5)
    pu.set_seeds(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_load_and_prep_target_is_binary(real_data_path):
    df = pu.load_and_prep()
    assert pu.Y_COL in df.columns
    assert set(df[pu.Y_COL].unique()) <= {0, 1}
    # dataset is balanced 50/50 (311/311)
    assert df[pu.Y_COL].sum() == 311
    assert len(df) == 622


def test_load_and_prep_drops_leak_and_raw_columns():
    df = pu.load_and_prep()
    for c in pu.LEAK_COLS + [pu.RAW_COUNT_COL]:
        assert c not in df.columns, f"{c} should have been dropped"


def test_load_and_prep_features_are_numeric_and_keep_nans():
    df = pu.load_and_prep()
    X = df.drop(columns=[pu.Y_COL])
    # every feature column numeric
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    # NaNs are preserved (NOT replaced with +/-99999 sentinels)
    assert (X.abs() == 99999).sum().sum() == 0


def test_split_xy_separates_target():
    df = pu.load_and_prep()
    X, y = pu.split_xy(df)
    assert pu.Y_COL not in X.columns
    assert y.name == pu.Y_COL
    assert len(X) == len(y) == 622


def test_make_train_test_is_stratified_and_deterministic():
    df = pu.load_and_prep()
    Xtr1, Xte1, ytr1, yte1 = pu.make_train_test(df)
    Xtr2, Xte2, ytr2, yte2 = pu.make_train_test(df)
    # deterministic
    assert list(ytr1.index) == list(ytr2.index)
    # test size ~22%
    assert abs(len(Xte1) / 622 - 0.22) < 0.01
    # stratified: train/test prevalence within 3pp of overall 0.5
    assert abs(ytr1.mean() - 0.5) < 0.03
    assert abs(yte1.mean() - 0.5) < 0.03


def test_cluster_adder_appends_one_column():
    X = np.random.RandomState(0).rand(50, 4)
    adder = pu.ClusterFeatureAdder(n_clusters=3)
    Xt = adder.fit_transform(X)
    assert Xt.shape == (50, 5)
    labels = Xt[:, -1]
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_cluster_adder_is_fit_on_training_only():
    # centroids depend ONLY on fit data, not on later transform inputs
    rng = np.random.RandomState(1)
    Xtrain = rng.rand(60, 4)
    Xother = rng.rand(10, 4) + 100.0  # far away block
    adder = pu.ClusterFeatureAdder(n_clusters=3).fit(Xtrain)
    centroids_before = adder.kmeans_.cluster_centers_.copy()
    _ = adder.transform(Xother)  # transform must NOT refit
    assert np.allclose(centroids_before, adder.kmeans_.cluster_centers_)


def test_cluster_adder_deterministic():
    X = np.random.RandomState(2).rand(40, 3)
    a = pu.ClusterFeatureAdder().fit_transform(X)
    b = pu.ClusterFeatureAdder().fit_transform(X)
    assert np.allclose(a, b)


def test_preprocessor_removes_nans_and_adds_cluster_col():
    df = pu.load_and_prep()
    X, y = pu.split_xy(df)
    pre = pu.make_preprocessor(add_clusters=True)
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any()
    # one extra column for the cluster id
    assert Xt.shape[1] == X.shape[1] + 1


def test_preprocessor_without_clusters_matches_feature_count():
    df = pu.load_and_prep()
    X, _ = pu.split_xy(df)
    pre = pu.make_preprocessor(add_clusters=False)
    Xt = pre.fit_transform(X)
    assert Xt.shape[1] == X.shape[1]
    assert not np.isnan(Xt).any()
