"""eval 层单测。作者：晨星"""

import numpy as np
import pytest

from anomalyforge.core.errors import EvalError
from anomalyforge.core.types import DetectorResult
from anomalyforge.data.synthetic import make_synthetic
from anomalyforge.eval.metrics import (
    Evaluator,
    average_precision_score,
    precision_recall_at_k,
    roc_auc_score,
)

DS = make_synthetic(n_samples=800, n_features=6, contamination=0.1, random_state=5)


def test_roc_auc_perfect():
    # 异常分数远超正常
    scores = np.where(DS.y == 1, 100.0, 0.0)
    assert abs(roc_auc_score(DS.y, scores) - 1.0) < 1e-6


def test_roc_auc_random_above_chance():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=DS.n_samples)
    auc = roc_auc_score(DS.y, scores)
    assert 0.3 < auc < 0.7  # 随机分数接近 0.5


def test_roc_auc_requires_both_classes():
    y = np.zeros(10, dtype=int)
    with pytest.raises(EvalError):
        roc_auc_score(y, np.arange(10))


def test_precision_recall_at_k():
    # 完美排序：异常分数最高
    scores = np.where(DS.y == 1, 1.0, 0.0)
    p, r, f1 = precision_recall_at_k(DS.y, scores, k=int(np.sum(DS.y == 1)))
    assert abs(p - 1.0) < 1e-6
    assert abs(r - 1.0) < 1e-6
    assert abs(f1 - 1.0) < 1e-6


def test_average_precision_perfect():
    scores = np.where(DS.y == 1, 1.0, 0.0)
    assert abs(average_precision_score(DS.y, scores) - 1.0) < 1e-6


def test_evaluator_on_detector_result():
    from anomalyforge.detectors.factory import build_detector

    det = build_detector("iforest", contamination=0.1, random_state=2)
    det.fit(DS.X)
    res = det.predict(DS.X)
    ev = Evaluator()
    m = ev.evaluate(res, DS)
    assert 0.0 <= m.roc_auc <= 1.0
    assert m.name == "iforest"
    assert set(m.as_dict().keys()) >= {"roc_auc", "precision_at_k", "f1_at_k"}
