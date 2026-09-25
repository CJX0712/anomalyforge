"""AnomalyForge 命令行入口。

用法示例：
  python -m anomalyforge.cli --detector iforest
  python -m anomalyforge.cli --benchmark --detector iforest knn copod zscore iqr
  python -m anomalyforge.cli --dataset data.csv --label-column label
作者：晨星
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import Config
from .core.types import Dataset
from .data.synthetic import load_csv, make_synthetic
from .detectors.factory import available_detectors
from .pipeline.anomaly_pipeline import AnomalyPipeline, benchmark
from .preprocess.standardize import Standardizer


def _print_table(metrics_list: list) -> None:
    header = ["Detector", "ROC-AUC", "AP", "P@k", "R@k", "F1@k", "contam."]
    print("\n" + " | ".join(f"{h:<10}" for h in header))
    print("-" * (len(header) * 13))
    for m in metrics_list:
        print(
            " | ".join(
                [
                    f"{m.name:<10}",
                    f"{m.roc_auc:<10.4f}",
                    f"{m.average_precision:<10.4f}",
                    f"{m.precision_at_k:<10.4f}",
                    f"{m.recall_at_k:<10.4f}",
                    f"{m.f1_at_k:<10.4f}",
                    f"{m.contamination:<10.4f}",
                ]
            )
        )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anomalyforge", description="AnomalyForge 异常检测系统"
    )
    parser.add_argument("--detector", default="iforest", help="检测器名称")
    parser.add_argument(
        "--benchmark", action="store_true", help="对多个检测器做横向评测"
    )
    parser.add_argument(
        "--detectors", nargs="*", default=None, help="benchmark 模式下的检测器列表"
    )
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset", default=None, help="CSV 数据路径")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--split", action="store_true", help="训练/测试分离评测")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    config = Config(
        detector=args.detector,
        contamination=args.contamination,
        random_state=args.random_state,
        n_features=args.n_features,
        n_samples=args.n_samples,
    )

    if args.dataset:
        dataset = load_csv(
            args.dataset,
            label_column=args.label_column,
            contamination=args.contamination,
        )
    else:
        dataset = make_synthetic(
            n_samples=args.n_samples,
            n_features=args.n_features,
            contamination=args.contamination,
            random_state=args.random_state,
        )

    preprocess = None if args.no_preprocess else Standardizer()

    if args.benchmark:
        names = args.detectors or available_detectors()
        metrics = benchmark(
            names, config=config, dataset=dataset, use_preprocess=not args.no_preprocess
        )
        if args.json:
            print(json.dumps([m.as_dict() for m in metrics], ensure_ascii=False))
        else:
            _print_table(metrics)
        return 0

    pipe = AnomalyPipeline(config, preprocess=preprocess)
    metrics = pipe.run_split(dataset) if args.split else pipe.run(dataset)
    if args.json:
        print(json.dumps(metrics.as_dict(), ensure_ascii=False))
    else:
        _print_table([metrics])
    return 0


if __name__ == "__main__":
    sys.exit(main())
