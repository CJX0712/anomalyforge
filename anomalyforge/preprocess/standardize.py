"""特征标准化预处理。

优先使用 scikit-learn StandardScaler（SOTA 后端），
不可用时降级为纯 numpy 实现（零下载）。
作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DetectorError

try:  # 优先 SOTA 后端
    from sklearn.preprocessing import StandardScaler as _SkStandardScaler

    _SKLEARN_OK = True
except ImportError:  # pragma: no cover - 兜底路径
    _SKLEARN_OK = False


class Standardizer:
    """统一标准化接口。

    fit: 计算均值与标准差（sklearn 或 numpy）
    transform: (X - mean) / std
    """

    def __init__(self, with_std: bool = True, eps: float = 1e-9) -> None:
        self.with_std = with_std
        self.eps = eps
        self._backend = "sklearn" if _SKLEARN_OK else "numpy"
        self._scaler = _SkStandardScaler() if _SKLEARN_OK else None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        if self._backend == "sklearn":
            try:
                self._scaler.fit(X)  # type: ignore[union-attr]
                self.mean_ = np.asarray(self._scaler.mean_)  # type: ignore[union-attr]
                self.scale_ = np.asarray(self._scaler.scale_)  # type: ignore[union-attr]
                return self
            except Exception as e:  # noqa: BLE001
                raise DetectorError("sklearn StandardScaler 拟合失败", cause=e) from e
        # numpy 兜底
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        if not self.with_std:
            std = np.ones_like(std)
        self.scale_ = std + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._backend == "sklearn" and self._scaler is not None:
            return np.asarray(self._scaler.transform(X))
        if self.mean_ is None or self.scale_ is None:
            raise DetectorError("Standardizer 未拟合即 transform")
        return (X - self.mean_) / self.scale_


def build_standardizer(**kwargs) -> Standardizer:
    """构造标准化器。"""
    return Standardizer(**kwargs)
