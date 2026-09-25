# AnomalyForge 架构设计

> 作者：晨星 ｜ 模块化异常检测系统 ｜ 复用顶级开源：PyOD 3.6.6 + scikit-learn 1.9.1 + numpy 2.5.3 + scipy 1.18.1

## 1. 设计原则

- **单一职责**：每个模块只做一件事，对外暴露稳定接口。
- **优先复用顶级开源**：检测核心直接复用 PyOD（20+ 检测器），标准化复用 scikit-learn，绝不从零自研 SOTA 部分。
- **离线可跑兜底**：所有 SOTA 后端（PyOD / sklearn）均配纯 numpy 兜底，缺失依赖时自动降级，保证零下载 demo 可跑。
- **接口契约先行**：核心类型、错误码、Protocol 接口定义在前，实现在后。
- **可独立验证**：每模块可单测 + 最小可运行示例，单向无环调用。

## 2. 模块划分与调用关系

```
                ┌─────────────────────────────────────────────┐
                │                cli.py                        │
                │   (argparse 入口: --detector / --benchmark)   │
                └───────────────────────┬─────────────────────┘
                                        │
                ┌───────────────────────▼─────────────────────┐
                │          pipeline/anomaly_pipeline.py        │
                │   AnomalyPipeline.run() / benchmark()         │
                └───┬──────────┬──────────┬──────────┬─────────┘
                    │          │          │          │
            ┌───────▼───┐ ┌────▼─────┐ ┌──▼──────┐ ┌─▼──────┐
            │ data/     │ │preprocess│ │training │ │ eval/  │
            │synthetic  │ │standardize│ │fit      │ │metrics │
            └─────┬─────┘ └────┬─────┘ └──┬──────┘ └──┬─────┘
                  │            │          │            │
                  └────────────┼──────────┘            │
                               ▼                       │
                  ┌────────────────────────┐           │
                  │ detectors/factory.py    │◄──────────┘
                  │  PyOD 封装 + 统计兜底    │
                  └────────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ core/ (types/errors/    │
                  │ config/interfaces)      │
                  └─────────────────────────┘
```

调用方向严格单向、无环：`cli → pipeline → {data, preprocess, training, detectors, eval} → core`。

## 3. 核心契约（core/）

| 文件 | 职责 |
|------|------|
| `types.py` | `Dataset` / `DetectorResult` / `EvalMetrics` / `FitResult` dataclass；`DetectorName` 枚举 |
| `errors.py` | 错误码体系：E100 数据 / E200 检测器 / E300 训练 / E400 评测 / E500 配置 |
| `config.py` | `Config` dataclass，支持 `ANOMALYFORGE_*` 环境变量覆盖 |
| `interfaces.py` | `IDataLoader / IDetector / IPreprocessor / ITrainer / IEvaluator / IPipeline` Protocol |

### 接口约定
- `IDetector`: `name` 属性 + `fit(X) -> self` + `predict(X) -> DetectorResult`
- 所有检测器输出统一语义：**分数越大越异常**；`labels` 按 `contamination` 取 top-k。

## 4. 检测器清单

| 名称 | 后端 | 说明 |
|------|------|------|
| `iforest` | PyOD | Isolation Forest（树集成，工业标配）|
| `knn` | PyOD | k 近邻距离异常 |
| `lof` | PyOD | 局部离群因子 |
| `ocsvm` | PyOD | 单类 SVM |
| `hbos` | PyOD | 直方图基于离群值 |
| `copod` | PyOD | Copula 无监督检测（无超参）|
| `auto_encoder` | PyOD | 自编码器（需可选依赖 torch）|
| `zscore` | numpy | 逐特征标准分（零下载兜底）|
| `iqr` | numpy | 四分位距法（零下载兜底）|
| `mad` | numpy | 中位数绝对偏差（零下载兜底）|

> `auto_encoder` 在未安装 torch 时由 `benchmark` / `build_default_ensemble` 自动跳过，不影响其余检测器。

## 5. 错误码体系

| 区间 | 域 | 典型触发 |
|------|----|---------|
| E100 | 数据层 | CSV 解析失败、contamination 越界、文件不存在 |
| E200 | 检测器层 | PyOD 后端不可用、拟合/推理失败 |
| E300 | 训练层 | 训练编排异常 |
| E400 | 评测层 | 指标计算失败（如仅单类）|
| E500 | 配置层 | contamination 非法、环境变量解析失败 |

## 6. 性能基线（benchmark）

数据集：2000 样本 / 12 特征 / 200 异常（contamination=0.1），随机种子 42。
指标按 ROC-AUC 降序：

| Detector | ROC-AUC | AP | P@k | R@k | F1@k |
|----------|--------|----|-----|-----|------|
| ocsvm    | 0.9993 | 0.9938 | 0.965 | 0.965 | 0.965 |
| lof      | 0.9989 | 0.9906 | 0.960 | 0.960 | 0.960 |
| knn      | 0.9989 | 0.9902 | 0.970 | 0.970 | 0.970 |
| iforest  | 0.9970 | 0.9730 | 0.920 | 0.920 | 0.920 |
| hbos     | 0.9900 | 0.9301 | 0.850 | 0.850 | 0.850 |
| zscore   | 0.9883 | 0.9009 | 0.835 | 0.835 | 0.835 |
| mad      | 0.9881 | 0.8985 | 0.835 | 0.835 | 0.835 |
| copod    | 0.9865 | 0.9241 | 0.830 | 0.830 | 0.830 |
| iqr      | 0.9766 | 0.9046 | 0.840 | 0.840 | 0.840 |

**结论**：PyOD SOTA 检测器（ocsvm/lof/knn/iforest）ROC-AUC ≥ 0.997，显著领先；
纯 numpy 统计兜底（zscore/mad/iqr/copod）仍达 ~0.98，证明离线零下载亦可获得可用基线。

## 7. 验证（DoD 对照）

| 验收项 | 状态 |
|--------|------|
| 干净环境 clone → 一键 demo 零手工干预 | ✅ `python -m anomalyforge.examples.run_demo` |
| 模块单测全通过 | ✅ 30 passed, 1 skipped（auto_encoder 需 torch）|
| 依赖锁定可复现 | ✅ `requirements.lock.txt`（pip freeze）|
| 文档覆盖架构/部署/使用 | ✅ 本文件 + README.md |
| 关键指标量化基线 | ✅ 上表对照 |

基线详情落盘于 `anomalyforge/examples/benchmark.json`。
