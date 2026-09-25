"""检测器基类与通用工具。

PyODDetector：对任意 PyOD 估计器的统一封装，输出归一为
(higher score = more anomalous) 的 DetectorResult。
作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import DetectorError
from ..core.types import DetectorResult


def labels_from_contamination(scores: np.ndarray, contamination: float) -> np.ndarray:
    """按分数降序取前 k 个为异常（k = round(contamination*n)）。

    保证判定数量与 contamination 一致，便于跨检测器公平评测。
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.shape[0]
    k = max(1, int(round(contamination * n)))
    k = min(k, n)
    labels = np.zeros(n, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    labels[order[:k]] = 1
    return labels


class PyODDetector:
    """PyOD 估计器统一封装。

    约定：decision_function 越大越异常；predict 返回 DetectorResult。
    若 PyOD 不可用（ImportError），应在 factory 层降级为统计兜底。
    """

    name: str = "pyod"

    def __init__(
        self,
        name: str,
        pyod_cls: Any,
        contamination: float = 0.1,
        random_state: int = 42,
        **model_kwargs: Any,
    ) -> None:
        self.name = name
        self.contamination = float(contamination)
        self.random_state = random_state
        self._estimator = self._build(pyod_cls, model_kwargs)
        self._fitted = False

    def _build(self, pyod_cls: Any, model_kwargs: dict) -> Any:
        kwargs: dict = {"contamination": self.contamination}
        try:
            # 大多数 PyOD 估计器接受 random_state
            estimator = pyod_cls(random_state=self.random_state, **model_kwargs)
        except TypeError:
            estimator = pyod_cls(**model_kwargs)
        return estimator

    def fit(self, X: np.ndarray) -> "PyODDetector":
        try:
            self._estimator.fit(np.asarray(X, dtype=np.float64))
        except Exception as e:  # noqa: BLE001
            raise DetectorError(f"{self.name} 拟合失败", cause=e) from e
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> DetectorResult:
        if not self._fitted:
            raise DetectorError(f"{self.name} 未拟合即调用 predict")
        X = np.asarray(X, dtype=np.float64)
        try:
            # PyOD 约定：decision_function 越大越异常
            raw = np.asarray(self._estimator.decision_function(X), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            raise DetectorError(f"{self.name} 推理失败", cause=e) from e
        scores = raw.ravel()
        labels = labels_from_contamination(scores, self.contamination)
        return DetectorResult(
            name=self.name,
            scores=scores,
            labels=labels,
            contamination=self.contamination,
        )
