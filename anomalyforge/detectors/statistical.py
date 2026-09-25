"""纯 numpy 统计异常检测器（零下载兜底后端）。

当 PyOD / 网络 / 密钥不可用（或显式指定统计后端）时，
自动降级到以下轻量检测器，保证 demo 零下载可跑：
  - ZScore：逐特征标准分，取最大绝对分
  - IQR：四分位距法，超出 [Q1-1.5IQR, Q3+1.5IQR] 越界量
  - MAD：中位数绝对偏差的修正 Z 分数
作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import DetectorResult
from .base import labels_from_contamination


class BaseStatisticalDetector:
    """统计检测器基类：在训练集上拟合参考统计量。"""

    name: str = "statistical"

    def __init__(self, contamination: float = 0.1) -> None:
        self.contamination = float(contamination)
        self._fitted = False
        self._train_scores: np.ndarray | None = None

    def _score(self, X: np.ndarray) -> np.ndarray:
        """子类实现：返回 (n,) 分数，越大越异常。"""
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> "BaseStatisticalDetector":
        X = np.asarray(X, dtype=np.float64)
        self._train_scores = self._score(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> DetectorResult:
        if not self._fitted:
            raise RuntimeError(f"{self.name} 未拟合即调用 predict")
        X = np.asarray(X, dtype=np.float64)
        scores = self._score(X)
        labels = labels_from_contamination(scores, self.contamination)
        return DetectorResult(
            name=self.name,
            scores=scores,
            labels=labels,
            contamination=self.contamination,
        )


class ZScoreDetector(BaseStatisticalDetector):
    """逐特征标准分，异常分数 = 各特征 |z| 的最大值。"""

    name = "zscore"

    def __init__(self, contamination: float = 0.1) -> None:
        super().__init__(contamination)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _score(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-9
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=1)


class IQRDetector(BaseStatisticalDetector):
    """四分位距法，分数 = 各特征超出围栏的越界量之和。"""

    name = "iqr"

    def __init__(self, contamination: float = 0.1, k: float = 1.5) -> None:
        super().__init__(contamination)
        self.k = k

    def _score(self, X: np.ndarray) -> np.ndarray:
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = (q3 - q1) + 1e-9
        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr
        below = np.clip(lower - X, 0, None)
        above = np.clip(X - upper, 0, None)
        return (below + above).sum(axis=1)


class MADDetector(BaseStatisticalDetector):
    """中位数绝对偏差（MAD）修正 Z 分数。"""

    name = "mad"

    def __init__(self, contamination: float = 0.1) -> None:
        super().__init__(contamination)
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def _score(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            self.median_ = np.median(X, axis=0)
            self.mad_ = np.median(np.abs(X - self.median_), axis=0) + 1e-9
        modified_z = 0.6745 * (X - self.median_) / self.mad_
        return np.abs(modified_z).max(axis=1)
