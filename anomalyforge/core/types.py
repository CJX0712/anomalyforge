"""核心数据类型定义。

所有跨模块传递的数据载体统一用 dataclass 描述，字段带类型注解，
保证模块间接口稳定、可独立验证。
作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class DetectorName(str, Enum):
    """受支持的检测器标识。

    PyOD SOTA 后端 + 纯 numpy 统计兜底后端。
    """

    # --- PyOD SOTA 后端 ---
    IForest = "iforest"
    KNN = "knn"
    LOF = "lof"
    OCSVM = "ocsvm"
    AutoEncoder = "auto_encoder"
    HBOS = "hbos"
    COPOD = "copod"

    # --- 纯 numpy 统计兜底后端（零下载）---
    ZScore = "zscore"
    IQR = "iqr"
    MAD = "mad"


@dataclass
class Dataset:
    """一份异常检测数据集。

    X: 特征矩阵 (n_samples, n_features)
    y: 标签向量 (n_samples,)，1=异常 0=正常；合成数据提供，真实数据可全 0
    feature_names: 可选列名
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.int64).ravel()
        if self.X.ndim != 2:
            raise ValueError("X 必须是 2D 矩阵 (n_samples, n_features)")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X 与 y 样本数不一致")

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_anomalies(self) -> int:
        return int(np.count_nonzero(self.y == 1))


@dataclass
class DetectorResult:
    """单个检测器在数据集上的输出。

    scores: 异常分数（越大越异常），已统一为单调递增语义
    labels: 二值判定 1=异常 0=正常
    contamination: 实际判定为异常的比例
    meta: 任意附加信息（运行耗时、阈值等）
    """

    name: str
    scores: np.ndarray
    labels: np.ndarray
    contamination: float = 0.1
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64).ravel()
        self.labels = np.asarray(self.labels, dtype=np.int64).ravel()


@dataclass
class FitResult:
    """训练/拟合结果。"""

    name: str
    fitted: bool
    n_train: int
    elapsed_sec: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """评测指标集合。"""

    name: str
    roc_auc: float
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    average_precision: float
    contamination: float = 0.1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "roc_auc": round(self.roc_auc, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "f1_at_k": round(self.f1_at_k, 4),
            "average_precision": round(self.average_precision, 4),
            "contamination": round(self.contamination, 4),
        }
