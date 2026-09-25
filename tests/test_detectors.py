"""detectors 层单测。作者：晨星"""

import numpy as np
import pytest

from anomalyforge.core.errors import DetectorError
from anomalyforge.core.types import Dataset
from anomalyforge.data.synthetic import make_synthetic
from anomalyforge.detectors.factory import (
    ALL_DETECTORS,
    PYOD_DETECTORS,
    STATISTICAL_DETECTORS,
    available_detectors,
    build_detector,
    build_default_ensemble,
)
from anomalyforge.detectors.statistical import (
    IQRDetector,
    MADDetector,
    ZScoreDetector,
)

rng = np.random.default_rng(0)
DS = make_synthetic(n_samples=600, n_features=6, contamination=0.1, random_state=0)


def test_statistical_detectors():
    for name, cls in [("zscore", ZScoreDetector), ("iqr", IQRDetector), ("mad", MADDetector)]:
        det = cls(contamination=0.1)
        det.fit(DS.X)
        res = det.predict(DS.X)
        assert res.scores.shape[0] == DS.n_samples
        assert int(np.sum(res.labels)) == max(1, round(0.1 * DS.n_samples))
        assert res.name == name


def test_pyod_detectors_run():
    # auto_encoder 需要可选依赖 torch，缺失时跳过
    skip = set()
    try:
        import torch  # noqa: F401
    except ImportError:
        skip.add("auto_encoder")
    for name in PYOD_DETECTORS:
        if name in skip:
            pytest.skip(f"{name} 需要 torch（可选依赖，未安装）")
            continue
        det = build_detector(name, contamination=0.1, random_state=1)
        det.fit(DS.X)
        res = det.predict(DS.X)
        assert res.scores.shape[0] == DS.n_samples
        # 分数应具区分度（异常样本均值 > 正常样本均值）
        score_anom = res.scores[DS.y == 1].mean()
        score_norm = res.scores[DS.y == 0].mean()
        assert score_anom > score_norm


def test_build_detector_unknown():
    with pytest.raises(DetectorError):
        build_detector("does_not_exist")


def test_available_detectors_includes_both():
    avail = available_detectors()
    for s in STATISTICAL_DETECTORS:
        assert s in avail
    for p in PYOD_DETECTORS:
        assert p in avail


def test_build_default_ensemble_nonempty():
    ens = build_default_ensemble(contamination=0.1, random_state=2)
    assert len(ens) >= 1
    assert all(hasattr(d, "name") for d in ens)


def test_labels_count_matches_contamination():
    det = build_detector("iforest", contamination=0.15, random_state=3)
    det.fit(DS.X)
    res = det.predict(DS.X)
    assert int(np.sum(res.labels)) == max(1, round(0.15 * DS.n_samples))
