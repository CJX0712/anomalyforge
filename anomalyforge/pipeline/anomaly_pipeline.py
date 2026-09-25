"""端到端异常检测流水线。

AnomalyPipeline：config -> (可选)生成/载入数据 -> 预处理 -> 训练 -> 推理 -> 评测。
benchmark：跨多个检测器统一评测，输出可比对的指标表。
作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import Config
from ..core.types import Dataset, DetectorResult, EvalMetrics
from ..data.synthetic import load_csv, make_synthetic, split_train_test
from ..detectors.factory import build_detector
from ..eval.metrics import Evaluator
from ..preprocess.standardize import Standardizer
from ..training.fit import Trainer


class AnomalyPipeline:
    """单一检测器端到端流水线。"""

    def __init__(
        self,
        config: Config,
        detector: Optional[Any] = None,
        preprocess: Optional[Standardizer] = None,
    ) -> None:
        self.config = config
        self.detector = detector or build_detector(
            config.detector,
            contamination=config.contamination,
            random_state=config.random_state,
        )
        self.preprocess = preprocess
        self.evaluator = Evaluator()

    def _ensure_dataset(self, dataset: Optional[Dataset]) -> Dataset:
        if dataset is not None:
            return dataset
        return make_synthetic(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )

    def run(self, dataset: Optional[Dataset] = None) -> EvalMetrics:
        """运行流水线并返回评测指标。"""
        ds = self._ensure_dataset(dataset)
        trainer = Trainer(preprocess=self.preprocess)
        trainer.fit(self.detector, ds)
        X = trainer.transform_if_fitted(ds.X)
        result = self.detector.predict(X)
        return self.evaluator.evaluate(result, ds)

    def run_split(
        self, dataset: Optional[Dataset] = None, test_size: float = 0.3
    ) -> EvalMetrics:
        """训练/测试分离模式：训练集拟合，测试集评测。"""
        ds = self._ensure_dataset(dataset)
        train, test = split_train_test(
            ds, test_size=test_size, random_state=self.config.random_state
        )
        trainer = Trainer(preprocess=self.preprocess)
        trainer.fit(self.detector, train)
        X_test = trainer.transform_if_fitted(test.X)
        result = self.detector.predict(X_test)
        return self.evaluator.evaluate(result, test)


def benchmark(
    detector_names: List[str],
    config: Optional[Config] = None,
    dataset: Optional[Dataset] = None,
    random_state: int = 42,
    use_preprocess: bool = True,
) -> List[EvalMetrics]:
    """跨多个检测器统一评测，返回 EvalMetrics 列表（按 ROC-AUC 降序）。"""
    config = config or Config(random_state=random_state)
    if dataset is None:
        dataset = make_synthetic(
            n_samples=config.n_samples,
            n_features=config.n_features,
            contamination=config.contamination,
            random_state=config.random_state,
        )
    evaluator = Evaluator()
    results: List[EvalMetrics] = []
    for name in detector_names:
        try:
            det = build_detector(
                name,
                contamination=config.contamination,
                random_state=config.random_state,
            )
        except Exception:  # noqa: BLE001
            continue
        preprocess = Standardizer() if use_preprocess else None
        trainer = Trainer(preprocess=preprocess)
        try:
            trainer.fit(det, dataset)
        except Exception:  # noqa: BLE001
            continue
        X = trainer.transform_if_fitted(dataset.X)
        try:
            res: DetectorResult = det.predict(X)
            metrics = evaluator.evaluate(res, dataset)
        except Exception:  # noqa: BLE001
            continue
        results.append(metrics)
    results.sort(key=lambda m: m.roc_auc, reverse=True)
    return results
