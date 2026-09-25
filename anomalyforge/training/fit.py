"""训练编排：将检测器拟合到数据集，并计时。

支持可选的预处理（标准化）步骤。返回 FitResult。
作者：晨星
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ..core.errors import TrainingError
from ..core.types import Dataset, FitResult
from ..preprocess.standardize import Standardizer


class Trainer:
    """检测器训练编排器。

    preprocess: 可选 Standardizer；若为 None 则原样拟合。
    """

    def __init__(self, preprocess: Optional[Standardizer] = None) -> None:
        self.preprocess = preprocess
        self._fitted_preprocessor: Optional[Standardizer] = None

    def fit(self, detector: Any, dataset: Dataset) -> FitResult:
        """拟合检测器，返回 FitResult（含耗时）。"""
        try:
            X = dataset.X
            if self.preprocess is not None:
                self._fitted_preprocessor = self.preprocess.fit(X)
                X = self._fitted_preprocessor.transform(X)

            t0 = time.perf_counter()
            detector.fit(X)
            elapsed = time.perf_counter() - t0

            return FitResult(
                name=getattr(detector, "name", "detector"),
                fitted=True,
                n_train=dataset.n_samples,
                elapsed_sec=elapsed,
                meta={"preprocessed": self.preprocess is not None},
            )
        except Exception as e:  # noqa: BLE001
            raise TrainingError(
                f"训练 {getattr(detector, 'name', '?')} 失败", cause=e
            ) from e

    def transform_if_fitted(self, X: Any) -> Any:
        if self._fitted_preprocessor is not None:
            return self._fitted_preprocessor.transform(X)
        return X
