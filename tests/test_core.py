"""core 层单测。作者：晨星"""

import numpy as np
import pytest

from anomalyforge.core.config import Config
from anomalyforge.core.errors import (
    ConfigError,
    DataError,
    DetectorError,
)
from anomalyforge.core.types import Dataset, DetectorName


def test_dataset_validation():
    X = np.random.default_rng(0).normal(size=(100, 5))
    y = np.zeros(100, dtype=int)
    ds = Dataset(X=X, y=y)
    assert ds.n_samples == 100
    assert ds.n_features == 5
    assert ds.n_anomalies == 0


def test_dataset_shape_mismatch_raises():
    X = np.zeros((10, 3))
    y = np.zeros(9, dtype=int)
    with pytest.raises(ValueError):
        Dataset(X=X, y=y)


def test_dataset_2d_required():
    X = np.zeros(10)
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        Dataset(X=X, y=y)


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ANOMALYFORGE_DETECTOR", "lof")
    monkeypatch.setenv("ANOMALYFORGE_CONTAMINATION", "0.2")
    c = Config(env_override=True)
    assert c.detector == "lof"
    assert abs(c.contamination - 0.2) < 1e-9


def test_config_bad_contamination():
    with pytest.raises(ConfigError):
        Config(contamination=1.5)


def test_config_bad_env_contamination(monkeypatch):
    monkeypatch.setenv("ANOMALYFORGE_CONTAMINATION", "not-a-float")
    with pytest.raises(ConfigError):
        Config(env_override=True)


def test_errors_have_codes():
    assert DataError("x").code == "E100"
    assert DetectorError("x").code == "E200"
    assert ConfigError("x").code == "E500"


def test_detector_name_enum():
    assert DetectorName.IForest.value == "iforest"
    assert DetectorName.ZScore.value == "zscore"
