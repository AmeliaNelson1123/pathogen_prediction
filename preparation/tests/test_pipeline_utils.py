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


def test_inf_converter_converts_inf_to_nan():
    """Test that _InfToNanConverter correctly converts inf/-inf to NaN."""
    X = np.array([[np.inf, 1.0], [-np.inf, 2.0]])
    converter = pu._InfToNanConverter()
    Xt = converter.fit_transform(X)
    # Inf values should be NaN
    assert np.isnan(Xt[0, 0])
    assert np.isnan(Xt[1, 0])
    # Finite values should be unchanged
    assert Xt[0, 1] == 1.0
    assert Xt[1, 1] == 2.0


def test_preprocessor_does_not_mutate_input():
    """Test that make_preprocessor does not mutate the caller's input DataFrame."""
    df = pu.load_and_prep()
    X, _ = pu.split_xy(df)
    # Record the number of inf values in the original data
    before = np.isinf(X.to_numpy(dtype=float)).sum()
    # Run the preprocessor
    pre = pu.make_preprocessor(add_clusters=True)
    _ = pre.fit_transform(X)
    # Verify the caller's X still has the same number of inf values (was NOT mutated)
    after = np.isinf(X.to_numpy(dtype=float)).sum()
    assert after == before, f"Input was mutated: {before} infs before, {after} infs after"


def test_search_spaces_have_all_models():
    spaces = pu.sklearn_search_spaces()
    assert set(spaces) == {
        "logistic_regression", "knn", "decision_tree",
        "random_forest", "svm", "gbm",
    }
    # LogisticRegression must be configured so l1_ratio is valid (saga+elasticnet)
    lr_pipe, lr_grid = spaces["logistic_regression"]
    clf = lr_pipe.named_steps["clf"]
    assert clf.get_params()["solver"] == "saga"
    assert clf.get_params()["penalty"] == "elasticnet"


def test_logistic_regression_fits_without_api_error():
    # the exact call that crashed on modern sklearn must now work
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    lr_pipe, lr_grid = pu.sklearn_search_spaces()["logistic_regression"]
    lr_pipe.set_params(clf__l1_ratio=1.0, clf__C=1.0).fit(Xtr, ytr)  # must not raise


def test_run_sklearn_selection_returns_fitted_best(monkeypatch):
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    # keep it fast: restrict to two cheap models via a small monkeypatched space
    small = {k: pu.sklearn_search_spaces()[k] for k in ["decision_tree", "gbm"]}
    monkeypatch.setattr(pu, "sklearn_search_spaces", lambda: small)
    out = pu.run_sklearn_selection(Xtr, ytr)
    assert set(out) == {"decision_tree", "gbm"}
    for name, res in out.items():
        assert 0.5 <= res["cv_best"] <= 1.0
        # fitted: can predict
        assert res["estimator"].predict(Xte).shape[0] == len(yte)


def test_build_nn_shapes():
    pu.set_seeds()
    model = pu.build_nn(input_dim=10, n_layers=2, n_neurons=16)
    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 1)


def test_nn_grid_is_small():
    grid = pu.nn_grid()
    assert 1 <= len(grid) <= 12  # kept small on purpose
    assert all({"n_layers", "n_neurons", "epochs", "batch_size"} <= set(g) for g in grid)


def test_run_nn_selection_returns_params():
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    out = pu.run_nn_selection(Xtr, ytr)
    assert 0.5 <= out["cv_best"] <= 1.0
    assert {"n_layers", "n_neurons", "epochs", "batch_size"} <= set(out["params"])


def test_evaluate_perfect_and_keys():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]
    m = pu.evaluate(y_true, y_proba)
    assert set(m) == {"accuracy", "precision", "recall", "f1",
                      "roc_auc", "pr_auc", "confusion_matrix"}
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_predict_proba_any_sklearn():
    df = pu.load_and_prep()
    Xtr, Xte, ytr, yte = pu.make_train_test(df)
    _, (pipe, _) = "gbm", (pu.sklearn_search_spaces()["gbm"])
    pipe.fit(Xtr, ytr)
    proba = pu.predict_proba_any(pipe, Xte)
    assert proba.shape[0] == len(yte)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_cv_distribution_shape():
    df = pu.load_and_prep()
    Xtr, _, ytr, _ = pu.make_train_test(df)
    pipe, _ = pu.sklearn_search_spaces()["decision_tree"]
    dist = pu.cv_accuracy_distribution(pipe, Xtr, ytr)
    assert set(dist) == {"mean", "std", "scores"}
    assert len(dist["scores"]) == 5
    assert 0.5 <= dist["mean"] <= 1.0
