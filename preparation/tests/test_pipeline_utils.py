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
