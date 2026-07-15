from pathlib import Path
import pytest

@pytest.fixture
def real_data_path():
    return Path(__file__).resolve().parents[2] / "data" / "ListeriaSoil_clean_log.csv"
