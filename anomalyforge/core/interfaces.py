"""模块接口契约（Protocol）。

每个模块对外暴露的接口用 runtime_checkable Protocol 描述，
保证单向无环调用、可独立替换与验证。
作者：晨星
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .types import Dataset, DetectorResult, EvalMetrics, FitResult


@runtime_checkable
class IDataLoader(Protocol):
    """数据加载/生成接口。"""

    def load(self, **kwargs: Any) -> Dataset:
        ...


@runtime_checkable
class IDetector(Protocol):
    """检测器接口。

    任何检测器（PyOD 封装或纯 numpy 兜底）都应实现：
      name -> str
      fit(X) -> 自身
      predict(X) -> DetectorResult
    """

    name: str

    def fit(self, X: np.ndarray) -> "IDetector":
        ...

    def predict(self, X: np.ndarray) -> DetectorResult:
        ...


@runtime_checkable
class IPreprocessor(Protocol):
    """预处理接口（如标准化）。"""

    def fit(self, X: np.ndarray) -> "IPreprocessor":
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class ITrainer(Protocol):
    """训练编排接口。"""

    def fit(self, detector: Any, dataset: Dataset) -> FitResult:
        ...


@runtime_checkable
class IEvaluator(Protocol):
    """评测接口。"""

    def evaluate(self, result: DetectorResult, dataset: Dataset) -> EvalMetrics:
        ...


@runtime_checkable
class IPipeline(Protocol):
    """端到端流水线接口。"""

    def run(self, dataset: Optional[Dataset] = None) -> EvalMetrics:
        ...
