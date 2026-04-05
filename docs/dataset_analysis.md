# PLL 数据集分析（面向 class imbalance / SR-CB 实验）

## 目录结构建议

```
PllReduction/
├── datasets/                          # 数据根目录（默认）
│   ├── lost.mat                       # 标准 MATLAB 单文件
│   ├── MyDataset/                     # 或：目录 bundle（便于 CSV/NPY 导出）
│   │   ├── data.npy
│   │   ├── partial_target.npy
│   │   └── target.npy                 # 可选
│   └── ...
├── pll/
│   └── analysis/
│       ├── imbalance_metrics.py       # 类别不均衡 + 候选集指标
│       ├── loaders.py                 # 发现与加载
│       └── report_builder.py          # 汇总报告
├── experiments/
│   └── analyze_pll_datasets.py       # CLI 入口
└── results/
    └── dataset_analysis/<timestamp>/   # 默认输出
        ├── pll_dataset_analysis.json
        ├── pll_dataset_summary.csv
        └── skipped_paths.json
```

## 使用方法

```bash
# 默认分析 datasets/，输出到 results/dataset_analysis/<时间戳>/
python experiments/analyze_pll_datasets.py

# 指定目录与输出
python experiments/analyze_pll_datasets.py --data-dir datasets --out-dir results/dataset_analysis

# 候选标签二值化阈值（与消歧中「是否算候选」一致时可调）
python experiments/analyze_pll_datasets.py --cand-threshold 0.5

# 少打终端表
python experiments/analyze_pll_datasets.py --quiet
```

依赖：`numpy`、`scipy`（读 `.mat`）、写 CSV/JSON 为标库。若使用 **CSV bundle**，有 `pandas` 时更易兼容带表头的文件（可选）。

## 输出说明

| 文件 | 内容 |
|------|------|
| `pll_dataset_analysis.json` | 每数据集完整结构：真实类分布、Many/Medium/Few、候选数直方图、实验提示 `experiment_hints` |
| `pll_dataset_summary.csv` | 标量列，便于 Excel / pandas 与消融结果 join |
| `skipped_paths.json` | 无法解析的路径及原因（不抛异常） |

### 指标与实验含义（简要）

- **imbalance_ratio**：\(N_{\max}/N_{\min}\)（仅计数 >0 的类），大则 **CB（类重加权）** 更可能敏感。
- **cv_counts**：真实类样本数的变异系数，与实现中 CB 用的 soft-count CV 不同，但同向反映不均衡。
- **mean_cand / histogram**：每样本候选类数；高则 **SR（样本可靠性）** 更可能敏感。
- **many_medium_few**：与 `pll.eval.splits.get_many_medium_few_splits` **相同划分规则**（按类样本数排序三等分），便于对照实验分组指标。

## 新数据集接入方式

### 方式 A：`.mat`（推荐，与训练管线一致）

与 `pll.data.loader.load_dataset` 约定相同：

- 变量名：`data`（`n_samples × n_features`）、`partial_target`（`n_classes × n_samples`）
- 可选：`target`（`n_classes × n_samples` one-hot），无则仅输出候选侧统计并带警告

将文件放在 `datasets/<Name>.mat` 即可被自动发现。

### 方式 B：目录 bundle（NPY）

```
datasets/MyPLL/
  data.npy              # 或 X.npy / x.npy
  partial_target.npy
  target.npy            # 可选
```

形状与 `.mat` 一致。

### 方式 C：目录 bundle（CSV）

```
datasets/MyPLL/
  data.csv
  partial_target.csv
  target.csv            # 可选
```

无表头、纯数值、逗号分隔；`pandas` 可选用于兼容。

### 方式 D：自定义格式

在 `pll/analysis/loaders.py` 中：

1. 新增 `_try_load_xxx(path)`，返回 `(LoadedPLL | None, err | None)`；
2. 在 `try_load_path` 的 `loaders` 列表中注册（顺序越靠前越早尝试）。

`LoadedPLL` 需包含：`name`, `X`, `partial_target`, `target`, `source_path`, `format`。

### 无法识别的文件

根目录下单个 `.csv`/`.txt` 若不能映射为完整 PLL（缺 `data` + `partial_target`），会记入 `skipped_paths.json` 并给出提示，**不中断**整批分析。
