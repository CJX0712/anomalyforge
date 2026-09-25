"""AnomalyForge 端到端演示。

零下载可跑：生成合成异常数据，跨 PyOD SOTA 后端 + 纯 numpy 统计兜底
做横向评测，打印指标对比表，并输出 JSON 基线到 examples/benchmark.json。

运行：
  python -m anomalyforge.examples.run_demo
作者：晨星
"""

from __future__ import annotations

import json
import os

from ..core.config import Config
from ..data.synthetic import make_synthetic
from ..detectors.factory import available_detectors
from ..pipeline.anomaly_pipeline import benchmark


def run_demo() -> None:
    config = Config(
        detector="iforest",
        contamination=0.1,
        random_state=42,
        n_samples=2000,
        n_features=12,
    )
    dataset = make_synthetic(
        n_samples=config.n_samples,
        n_features=config.n_features,
        contamination=config.contamination,
        random_state=config.random_state,
    )
    print(
        f"[demo] 数据集: {dataset.n_samples} 样本 / {dataset.n_features} 特征 / "
        f"{dataset.n_anomalies} 异常 (contam={config.contamination})"
    )

    names = available_detectors()
    print(f"[demo] 参与评测检测器: {names}\n")
    metrics = benchmark(names, config=config, dataset=dataset)

    print("=" * 78)
    print("AnomalyForge 横向评测结果（按 ROC-AUC 降序）")
    print("=" * 78)
    header = ["Detector", "ROC-AUC", "AP", "P@k", "R@k", "F1@k"]
    print(" | ".join(f"{h:<12}" for h in header))
    print("-" * 78)
    for m in metrics:
        print(
            " | ".join(
                [
                    f"{m.name:<12}",
                    f"{m.roc_auc:<12.4f}",
                    f"{m.average_precision:<12.4f}",
                    f"{m.precision_at_k:<12.4f}",
                    f"{m.recall_at_k:<12.4f}",
                    f"{m.f1_at_k:<12.4f}",
                ]
            )
        )
    print("=" * 78)

    # 基线落盘
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.to_dict(),
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "n_anomalies": dataset.n_anomalies,
                "metrics": [m.as_dict() for m in metrics],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[demo] 基线已写入: {out_path}")


if __name__ == "__main__":
    run_demo()
