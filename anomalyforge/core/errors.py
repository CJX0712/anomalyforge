"""错误码体系。

统一错误基类 + 分域错误，错误码区间：
  E100 数据层 (DataError)
  E200 检测器层 (DetectorError)
  E300 训练层 (TrainingError)
  E400 评测层 (EvalError)
  E500 配置层 (ConfigError)
作者：晨星
"""

from __future__ import annotations


class AnomalyForgeError(Exception):
    """所有 AnomalyForge 错误的基类。

    code: 错误码（如 "E200"）
    message: 人类可读描述
    """

    code: str = "E000"

    def __init__(self, message: str, *, code: str | None = None, cause: Exception | None = None):
        self.code = code or self.__class__.code
        self.message = message
        self.cause = cause
        super().__init__(f"[{self.code}] {message}")
        if cause is not None:
            self.__cause__ = cause


class DataError(AnomalyForgeError):
    """E100 数据层错误：加载、校验、合成失败。"""

    code = "E100"


class DetectorError(AnomalyForgeError):
    """E200 检测器层错误：构造、拟合、推理失败。"""

    code = "E200"


class TrainingError(AnomalyForgeError):
    """E300 训练层错误：训练编排失败。"""

    code = "E300"


class EvalError(AnomalyForgeError):
    """E400 评测层错误：指标计算失败。"""

    code = "E400"


class ConfigError(AnomalyForgeError):
    """E500 配置层错误：配置校验、环境变量解析失败。"""

    code = "E500"


__all__ = [
    "AnomalyForgeError",
    "DataError",
    "DetectorError",
    "TrainingError",
    "EvalError",
    "ConfigError",
]
