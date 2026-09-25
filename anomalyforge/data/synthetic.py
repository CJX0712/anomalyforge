"""合成异常检测数据集生成 + CSV 载入。

零依赖：仅 numpy。生成器构造一个服从多元高斯的正常簇，
再按 contamination 注入异常点（远离簇心的稀疏离群）。
作者：晨星
"""

from __future__ import annotations

import csv
import os
from typing import Optional, Tuple

import numpy as np

from ..core.errors import DataError
from ..core.types import Dataset


def make_synthetic(
    n_samples: int = 1000,
    n_features: int = 10,
    contamination: float = 0.1,
    random_state: int = 42,
    cluster_std: float = 1.0,
    outlier_magnitude: float = 6.0,
) -> Dataset:
    """生成二分类异常检测数据集。

    正常样本：多元高斯 N(0, cluster_std^2 * I)
    异常样本：从 N(mu_out, cluster_std^2 * I) 抽取，mu_out 偏离原点
              outlier_magnitude 个标准差，并带随机方向扰动

    返回 Dataset(X, y)，y=1 为异常。
    """
    if not (0.0 < contamination < 1.0):
        raise DataError(f"contamination 必须位于 (0,1)，收到 {contamination}")
    if n_samples < 2 or n_features < 1:
        raise DataError(f"非法维度 n_samples={n_samples}, n_features={n_features}")

    rng = np.random.default_rng(random_state)
    n_out = max(1, int(round(n_samples * contamination)))
    n_in = n_samples - n_out

    # 正常簇
    X_in = rng.normal(loc=0.0, scale=cluster_std, size=(n_in, n_features))

    # 异常点：随机方向、固定偏移幅度
    direction = rng.normal(size=(n_out, n_features))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    offset = outlier_magnitude * cluster_std
    X_out = direction * offset + rng.normal(
        scale=cluster_std * 0.5, size=(n_out, n_features)
    )

    X = np.vstack([X_in, X_out]).astype(np.float64)
    y = np.concatenate(
        [np.zeros(n_in, dtype=np.int64), np.ones(n_out, dtype=np.int64)]
    )

    # 打乱顺序，避免标签连续分布影响评估
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    feature_names = [f"f{i}" for i in range(n_features)]
    return Dataset(X=X, y=y, feature_names=feature_names)


def load_csv(
    path: str,
    label_column: Optional[str] = None,
    contamination: float = 0.1,
) -> Dataset:
    """从 CSV 载入数据。

    label_column 指定标签列（可选）。未指定时按 contamination 假设全正常。
    返回 Dataset。
    """
    if not os.path.exists(path):
        raise DataError(f"文件不存在: {path}")

    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if r]
    except Exception as e:  # noqa: BLE001
        raise DataError(f"CSV 解析失败: {path}", cause=e) from e

    if not header:
        raise DataError("CSV 缺少表头")

    try:
        data_rows = np.asarray(rows, dtype=np.float64)
    except ValueError as e:
        raise DataError("CSV 含非数值单元格", cause=e) from e

    if data_rows.size == 0:
        raise DataError("CSV 为空")

    if label_column is not None:
        if label_column not in header:
            raise DataError(f"label_column 不存在: {label_column}")
        col = header.index(label_column)
        y = data_rows[:, col].astype(np.int64)
        feat_cols = [i for i in range(len(header)) if i != col]
        X = data_rows[:, feat_cols]
        feature_names = [header[i] for i in feat_cols]
    else:
        X = data_rows
        y = np.zeros(data_rows.shape[0], dtype=np.int64)
        feature_names = list(header)

    return Dataset(X=X, y=y, feature_names=feature_names)


def split_train_test(
    dataset: Dataset, test_size: float = 0.3, random_state: int = 42
) -> Tuple[Dataset, Dataset]:
    """按行随机切分为训练/测试两份 Dataset。"""
    rng = np.random.default_rng(random_state)
    n = dataset.n_samples
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train = Dataset(
        X=dataset.X[train_idx],
        y=dataset.y[train_idx],
        feature_names=dataset.feature_names,
    )
    test = Dataset(
        X=dataset.X[test_idx],
        y=dataset.y[test_idx],
        feature_names=dataset.feature_names,
    )
    return train, test
