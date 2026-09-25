# AnomalyForge

> 模块化异常检测系统 —— 复用顶级开源（PyOD · scikit-learn · numpy · scipy），零下载可跑，一键复现。
> 作者：**晨星**

AnomalyForge 是一套**生产可用**的异常检测系统：以 [PyOD](https://github.com/yzhao062/pyod) 为核心检测器库（20+ 算法），scikit-learn 负责标准化与评测，numpy/scipy 提供零依赖统计兜底。模块按单一职责划分，接口契约先行，每个模块可独立验证。

## 特性

- ✅ **复用顶级开源**：检测核心直接复用 PyOD（iforest / knn / lof / ocsvm / hbos / copod / auto_encoder），不重造轮子。
- ✅ **离线可跑兜底**：PyOD / sklearn 缺失时自动降级到纯 numpy 统计检测器（zscore / iqr / mad），**零下载即可运行 demo**。
- ✅ **模块化 + 契约先行**：`core` 定义类型/错误码/配置/Protocol，`data → preprocess → training → detectors → eval → pipeline` 单向无环。
- ✅ **统一接口语义**：所有检测器输出「分数越大越异常」，标签按 `contamination` 取 top-k，跨检测器公平评测。
- ✅ **一键复现**：`requirements.lock.txt` 锁定全部依赖（pip freeze），`Makefile` / `Dockerfile` 即开即用。
- ✅ **量化基线**：内置 benchmark 横向评测，ROC-AUC 0.9766–0.9993。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. 运行端到端 demo（生成合成数据 + 跨检测器评测）
python -m anomalyforge.examples.run_demo

# 3. 运行单测
pytest -q

# 4. 命令行：单检测器
python -m anomalyforge.cli --detector iforest

# 5. 命令行：横向 benchmark
python -m anomalyforge.cli --benchmark
```

## 代码结构

```
anomalyforge/
├── core/            # 契约层：types / errors / config / interfaces
├── data/            # 合成数据生成 + CSV 载入
├── detectors/       # PyOD 封装 + 纯 numpy 统计兜底 + factory
├── preprocess/      # 标准化（sklearn 优先，numpy 兜底）
├── training/        # 拟合编排
├── eval/            # ROC-AUC / AP / Precision@k / Recall@k / F1@k
├── pipeline/        # AnomalyPipeline + benchmark
├── cli.py           # argparse 入口
└── examples/        # run_demo + benchmark.json
tests/               # pytest 单测
docs/architecture.md # 架构设计
```

## 作为库调用

```python
from anomalyforge import Config, AnomalyPipeline
from anomalyforge.data.synthetic import make_synthetic

cfg = Config(detector="iforest", contamination=0.1, random_state=42)
pipe = AnomalyPipeline(cfg)
metrics = pipe.run()          # 自动生成合成数据并评测
print(metrics.as_dict())
```

## 检测器一览

| 名称 | 后端 | 需额外依赖 |
|------|------|-----------|
| iforest / knn / lof / ocsvm / hbos / copod | PyOD | — |
| auto_encoder | PyOD | torch（可选）|
| zscore / iqr / mad | numpy | —（零下载）|

## 评测基线

数据集 2000 样本 / 12 特征 / 200 异常（contamination=0.1）：

| Detector | ROC-AUC | F1@k |
|----------|--------|------|
| ocsvm    | 0.9993 | 0.965 |
| lof      | 0.9989 | 0.960 |
| knn      | 0.9989 | 0.970 |
| iforest  | 0.9970 | 0.920 |
| zscore   | 0.9883 | 0.835 |
| iqr      | 0.9766 | 0.840 |

完整基线见 `docs/architecture.md` 与 `anomalyforge/examples/benchmark.json`。

## 验收（DoD）

- [x] 干净环境 clone → 一键 demo 零手工干预
- [x] 模块单测全通过（30 passed, 1 skipped）
- [x] 依赖锁定可复现（`requirements.lock.txt`）
- [x] 文档覆盖架构 / 部署 / 使用
- [x] 关键指标量化基线 + 对照

## 许可证

MIT © 晨星
