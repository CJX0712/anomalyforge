"""检测器工厂：按名称构造检测器实例。

PyOD SOTA 后端（iforest/knn/lof/ocsvm/auto_encoder/hbos/copod）通过
延迟 import 构造；纯 numpy 统计后端（zscore/iqr/mad）零依赖直接构造。
作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.errors import DetectorError
from ..core.types import DetectorName

from .base import PyODDetector
from .statistical import IQRDetector, MADDetector, ZScoreDetector

# PyOD 类名 -> (模块路径, 类名)
_PYOD_REGISTRY: Dict[str, tuple] = {
    "iforest": ("pyod.models.iforest", "IForest"),
    "knn": ("pyod.models.knn", "KNN"),
    "lof": ("pyod.models.lof", "LOF"),
    "ocsvm": ("pyod.models.ocsvm", "OCSVM"),
    "auto_encoder": ("pyod.models.auto_encoder", "AutoEncoder"),
    "hbos": ("pyod.models.hbos", "HBOS"),
    "copod": ("pyod.models.copod", "COPOD"),
}

_STATISTICAL = {
    "zscore": ZScoreDetector,
    "iqr": IQRDetector,
    "mad": MADDetector,
}

PYOD_DETECTORS: List[str] = list(_PYOD_REGISTRY.keys())
STATISTICAL_DETECTORS: List[str] = list(_STATISTICAL.keys())
ALL_DETECTORS: List[str] = PYOD_DETECTORS + STATISTICAL_DETECTORS


def available_detectors() -> List[str]:
    """返回全部可用检测器名（PyOD 不可用时动态剔除）。"""
    out = list(STATISTICAL_DETECTORS)
    try:
        import pyod  # noqa: F401

        out += PYOD_DETECTORS
    except ImportError:
        pass
    return out


def build_detector(
    name: str,
    contamination: float = 0.1,
    random_state: int = 42,
    **model_kwargs: Any,
) -> Any:
    """按名称构造检测器。

    统计后端直接构造；PyOD 后端延迟 import，若不可用抛 DetectorError。
    """
    name = name.lower()
    if name in _STATISTICAL:
        return _STATISTICAL[name](contamination=contamination)

    if name in _PYOD_REGISTRY:
        module_path, cls_name = _PYOD_REGISTRY[name]
        try:
            import importlib

            mod = importlib.import_module(module_path)
            pyod_cls = getattr(mod, cls_name)
        except (ImportError, AttributeError) as e:
            raise DetectorError(
                f"PyOD 后端不可用，无法构造 {name}；可改用统计后端 "
                f"{STATISTICAL_DETECTORS}",
                cause=e,
            ) from e
        return PyODDetector(
            name=name,
            pyod_cls=pyod_cls,
            contamination=contamination,
            random_state=random_state,
            **model_kwargs,
        )

    raise DetectorError(
        f"未知检测器: {name}；可选: {available_detectors()}"
    )


def build_default_ensemble(
    contamination: float = 0.1,
    random_state: int = 42,
) -> List[Any]:
    """构造一个跨后端的默认集成（PyOD + 统计兜底混合）。"""
    names = ["iforest", "knn", "copod", "zscore", "iqr"]
    detectors: List[Any] = []
    for n in names:
        try:
            detectors.append(
                build_detector(n, contamination=contamination, random_state=random_state)
            )
        except DetectorError:
            # PyOD 缺失时跳过，仅保留统计后端
            continue
    if not detectors:
        detectors = [
            build_detector(n, contamination=contamination, random_state=random_state)
            for n in STATISTICAL_DETECTORS
        ]
    return detectors
