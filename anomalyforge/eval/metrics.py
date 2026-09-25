"""异常检测评测指标。

优先使用 scikit-learn 计算 ROC-AUC / Average Precision，
不可用时降级为纯 numpy 的秩和（Mann-Whitney）实现，保证零依赖可跑。
作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import EvalError
from ..core.types import Dataset, DetectorResult, EvalMetrics

try:
    from sklearn.metrics import average_precision_score as _sk_ap
    from sklearn.metrics import roc_auc_score as _sk_roc_auc

    _SKLEARN_OK = True
except ImportError:  # pragma: no cover
    _SKLEARN_OK = False


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC，越大越异常。sklearn 优先，否则 numpy 秩和实现。"""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise EvalError("ROC-AUC 需要正负样本均存在")

    if _SKLEARN_OK:
        # sklearn 约定 scores 越大越接近正类
        return float(_sk_roc_auc(y_true, scores))

    # 秩和法（Mann-Whitney U）
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    # 平均秩处理并列
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_pos = ranks[y_true == 1].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average Precision（PR 曲线下面积）。sklearn 优先，否则 numpy。"""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if _SKLEARN_OK:
        return float(_sk_ap(y_true, scores))  # type: ignore[assignment]

    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    precisions = np.cumsum(y == 1) / (np.arange(len(y)) + 1)
    # 仅在正样本处采样
    mask = y == 1
    return float(np.sum(precisions[mask]) / n_pos)


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(scores)))
    return np.argsort(-scores, kind="mergesort")[:k]


def precision_recall_at_k(
    y_true: np.ndarray, scores: np.ndarray, k: int
) -> tuple:
    """返回 (precision_at_k, recall_at_k, f1_at_k)。

    k 默认取真实异常数 n_anom（异常检测常用设定）。
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_anom = int(np.sum(y_true == 1))
    if n_anom == 0:
        raise EvalError("Precision@k 需要至少一个真实异常")
    k = max(1, min(k, n_anom, len(scores)))
    top = _top_k_indices(scores, k)
    hits = int(np.sum(y_true[top] == 1))
    precision = hits / k
    recall = hits / n_anom
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


class Evaluator:
    """对 DetectorResult 进行端到端评测。"""

    def evaluate(
        self,
        result: DetectorResult,
        dataset: Dataset,
        k: int | None = None,
    ) -> EvalMetrics:
        y = dataset.y
        if k is None:
            k = int(np.sum(y == 1))
        try:
            roc = roc_auc_score(y, result.scores)
            ap = average_precision_score(y, result.scores)
            p, r, f1 = precision_recall_at_k(y, result.scores, k)
        except Exception as e:  # noqa: BLE001
            raise EvalError(f"评测 {result.name} 失败", cause=e) from e
        return EvalMetrics(
            name=result.name,
            roc_auc=roc,
            precision_at_k=p,
            recall_at_k=r,
            f1_at_k=f1,
            average_precision=ap,
            contamination=result.contamination,
        )
