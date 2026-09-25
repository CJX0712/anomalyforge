"""pipeline 层单测。作者：晨星"""

import pytest

from anomalyforge.core.config import Config
from anomalyforge.core.types import Dataset
from anomalyforge.data.synthetic import make_synthetic
from anomalyforge.detectors.factory import build_detector
from anomalyforge.pipeline.anomaly_pipeline import AnomalyPipeline, benchmark


def test_pipeline_run_returns_metrics():
    cfg = Config(detector="iforest", random_state=11)
    pipe = AnomalyPipeline(cfg)
    m = pipe.run()
    assert 0.0 <= m.roc_auc <= 1.0
    assert m.name == "iforest"


def test_pipeline_run_split():
    cfg = Config(detector="knn", random_state=12)
    pipe = AnomalyPipeline(cfg)
    m = pipe.run_split(test_size=0.3)
    assert 0.0 <= m.roc_auc <= 1.0


def test_benchmark_returns_sorted():
    cfg = Config(random_state=13, n_samples=800, n_features=6)
    names = ["iforest", "copod", "zscore", "iqr", "mad"]
    metrics = benchmark(names, config=cfg)
    assert len(metrics) >= 2
    rocs = [m.roc_auc for m in metrics]
    assert rocs == sorted(rocs, reverse=True)


def test_benchmark_statistical_only():
    # 即使 PyOD 缺失，统计后端也应产出结果
    cfg = Config(random_state=14)
    metrics = benchmark(["zscore", "iqr", "mad"], config=cfg)
    assert len(metrics) == 3
    assert {m.name for m in metrics} == {"zscore", "iqr", "mad"}


def test_pipeline_with_external_dataset():
    ds = make_synthetic(random_state=15)
    cfg = Config(detector="lof", random_state=15)
    pipe = AnomalyPipeline(cfg)
    m = pipe.run(dataset=ds)
    assert m.roc_auc > 0.5
