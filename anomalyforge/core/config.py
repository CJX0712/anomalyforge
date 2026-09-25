"""配置与运行时参数。

支持环境变量覆盖（ANOMALYFORGE_*），便于干净环境一键复现。
作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any

from .errors import ConfigError


@dataclass
class Config:
    """AnomalyForge 全局配置。

    detector: 默认检测器名（见 DetectorName）
    contamination: 异常占比先验
    random_state: 随机种子
    n_features: 合成数据特征维度
    n_samples: 合成数据样本数
    env_override: 是否允许环境变量覆盖（默认 True）
    extra: 任意扩展参数
    """

    detector: str = "iforest"
    contamination: float = 0.1
    random_state: int = 42
    n_features: int = 10
    n_samples: int = 1000
    env_override: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.contamination < 1.0):
            raise ConfigError(
                f"contamination 必须位于 (0,1)，收到 {self.contamination}"
            )
        if self.env_override:
            self._apply_env()

    def _apply_env(self) -> None:
        if v := os.getenv("ANOMALYFORGE_DETECTOR"):
            self.detector = v
        if v := os.getenv("ANOMALYFORGE_CONTAMINATION"):
            try:
                self.contamination = float(v)
            except ValueError as e:
                raise ConfigError(
                    f"无法解析 ANOMALYFORGE_CONTAMINATION={v}", cause=e
                ) from e
        if v := os.getenv("ANOMALYFORGE_RANDOM_STATE"):
            self.random_state = int(v)
        if v := os.getenv("ANOMALYFORGE_N_FEATURES"):
            self.n_features = int(v)
        if v := os.getenv("ANOMALYFORGE_N_SAMPLES"):
            self.n_samples = int(v)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "extra": self.extra,
        }
