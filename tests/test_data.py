"""data 层单测。作者：晨星"""

import os
import tempfile

import pytest

import numpy as np

from anomalyforge.core.errors import DataError
from anomalyforge.core.types import Dataset
from anomalyforge.data.synthetic import (
    load_csv,
    make_synthetic,
    split_train_test,
)


def test_make_synthetic_shape_and_contamination():
    ds = make_synthetic(n_samples=1000, n_features=8, contamination=0.1, random_state=1)
    assert ds.X.shape == (1000, 8)
    assert ds.y.shape[0] == 1000
    assert abs(ds.n_anomalies / 1000 - 0.1) < 0.02


def test_make_synthetic_reproducible():
    a = make_synthetic(random_state=7)
    b = make_synthetic(random_state=7)
    assert np.allclose(a.X, b.X)
    assert np.array_equal(a.y, b.y)


def test_make_synthetic_bad_contamination():
    with pytest.raises(DataError):
        make_synthetic(contamination=0.0)


def test_load_csv_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = (rng.normal(size=50) > 0).astype(int)
    rows = np.hstack([X, y.reshape(-1, 1)])

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a,b,c,label\n")
            for r in rows:
                f.write(",".join(f"{v:.6f}" for v in r) + "\n")
        ds = load_csv(path, label_column="label")
    assert isinstance(ds, Dataset)
    assert ds.X.shape[1] == 3
    assert ds.n_anomalies == int(y.sum())


def test_load_csv_missing_file():
    with pytest.raises(DataError):
        load_csv("__no_such_file__.csv")


def test_split_train_test():
    ds = make_synthetic(n_samples=500, random_state=3)
    train, test = split_train_test(ds, test_size=0.3, random_state=3)
    assert train.n_samples + test.n_samples == 500
    assert test.n_samples == 150
