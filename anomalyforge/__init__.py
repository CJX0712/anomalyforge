"""AnomalyForge — 模块化异常检测系统。

顶层包导出核心公共 API，便于 `from anomalyforge import ...` 直接调用。
作者：晨星
"""

from .core.types import (
    Dataset,
    DetectorResult,
    EvalMetrics,
    FitResult,
    DetectorName,
)
from .core.errors import (
    AnomalyForgeError,
    DataError,
    DetectorError,
    TrainingError,
    EvalError,
    ConfigError,
)
from .core.config import Config
from .pipeline.anomaly_pipeline import AnomalyPipeline

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "Dataset",
    "DetectorResult",
    "EvalMetrics",
    "FitResult",
    "DetectorName",
    "AnomalyForgeError",
    "DataError",
    "DetectorError",
    "TrainingError",
    "EvalError",
    "ConfigError",
    "Config",
    "AnomalyPipeline",
    "__version__",
    "__author__",
]
