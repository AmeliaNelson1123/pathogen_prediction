import numpy as np
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
