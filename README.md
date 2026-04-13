# PllReduction — 偏标签学习降维实验框架

## 一、项目概述

本项目实现了面向偏标签学习（Partial Label Learning, PLL）的降维方法及其统一实验框架。核心算法包括 SDLPP、DELIN、CENDA 三种降维方法，支持 KNN 和 IPAL 两种下游分类器，采用 **Repeated Holdout**（重复留出法）作为评估协议。

框架特点：
- **层次化 YAML 配置**：`base.yaml` → `datasets/*.yaml` → `methods/*.yaml` → CLI 覆盖
- **统一实验入口**：`experiments/run.py` 支持单次运行、批量 benchmark、参数扫描、消融实验
- **灵活的参数扫描**：`--sweep` 支持任意参数的笛卡尔积组合
- **标准化结果管理**：自动生成 JSON / CSV / LaTeX 格式的结果文件

---

## 二、项目结构

```
PllReduction/
├── configs/                         # 配置文件（层次化 YAML）
│   ├── base.yaml                    #   全局默认配置
│   ├── datasets/                    #   数据集特定配置
│   │   ├── lost.yaml
│   │   ├── MSRCv2.yaml
│   │   ├── FG-NET.yaml
│   │   ├── Mirflickr.yaml           #   含 stratified: false 降级
│   │   ├── Soccer Player.yaml
│   │   ├── Yahoo! News.yaml
│   │   ├── slashdotpl-f{1,2,3}.yaml
│   │   └── cifar10-lt-g{100,200}-r{1,2,3}.yaml  # CIFAR10 长尾变体
│   └── methods/                     #   方法配置
│       ├── sdlpp_baseline.yaml      #   SDLPP 基线（无 SR/CB）
│       ├── sdlpp_sr_cb.yaml         #   SDLPP + SR + CB（含 warmup_by_dataset）
│       ├── delin.yaml
│       └── cenda.yaml
│
├── datasets/                        # 数据集 (.mat 文件)
│   ├── lost.mat, MSRCv2.mat, ...
│   └── cifar10/                     #   CIFAR10 长尾 PLL 数据集
│
├── experiments/                     # 实验入口
│   ├── run.py                       #   ★ 统一实验运行器
│   ├── run_all.sh                   #   ★ 一键全跑脚本（分阶段并行）
│   ├── analyze_datasets.py          #   数据集分析工具
│   ├── generate_cifar10_lt_pll.py   #   CIFAR10 长尾 PLL 数据生成
│   └── generate_cifar10_lt_pll_grid.py  #   批量生成 CIFAR10 数据
│
├── pll/                             # 核心库
│   ├── config.py                    #   配置加载 / 合并 / 工具函数
│   ├── data/
│   │   ├── loader.py                #   .mat 数据加载
│   │   └── preprocessor.py          #   Z-score 预处理
│   ├── reducers/                    #   降维方法
│   │   ├── base.py                  #   BaseReducer 接口
│   │   ├── sdlpp.py                 #   SDLPP 实现
│   │   ├── delin.py                 #   DELIN 实现
│   │   └── cenda.py                 #   CENDA 实现
│   ├── disambig/                    #   消歧策略
│   │   ├── base.py                  #   BaseDisambiguator 接口
│   │   └── knn_propagation.py       #   KNN 标签传播 (SR + CB)
│   ├── classifiers/                 #   下游分类器
│   │   ├── knn.py                   #   KNN (sklearn 封装)
│   │   └── ipal.py                  #   IPAL (NNLS + 标签传播)
│   └── eval/                        #   评估框架
│       ├── splitter.py              #   RepeatedHoldout 分层/降级分割
│       ├── metrics.py               #   指标计算 + 聚合
│       ├── evaluator.py             #   评估流水线编排
│       └── reporter.py              #   表格 / CSV / JSON / LaTeX 输出
│
├── refes/                           #  MATLAB 参考代码（只读保留）
│   ├── sdlpp/                       #   SDLPP MATLAB 实现
│   ├── DELIN/                       #   DELIN MATLAB 实现
│   ├── CENDA/                       #   CENDA MATLAB 实现
│   └── IPAL/                        #   IPAL MATLAB 实现
│
└── results/                         # 实验结果输出（自动生成）
```

---

## 三、环境依赖

```bash
pip install numpy scipy scikit-learn pyyaml
```

Python 版本：3.9+

---

## 四、快速上手

### 4.1 单数据集单方法

```bash
python experiments/run.py --dataset lost --method sdlpp_baseline
```

- 使用默认配置（10 次 repeated holdout，20% 测试集，分层抽样）
- 结果保存到 `results/single/lost/sdlpp_baseline+knn/<timestamp>/`

### 4.2 快速冒烟测试

```bash
python experiments/run.py --dataset lost --method sdlpp_baseline --fast
```

`--fast` 模式：仅 3 次 repeat，迭代轮数 T 降至 10，用于快速验证流程。

### 4.3 多数据集批量实验

```bash
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET Mirflickr "Soccer Player" "Yahoo! News" \
    --methods sdlpp_baseline sdlpp_sr_cb delin cenda \
    --campaign benchmark_v1
```

- 自动构建 `datasets × methods` 笛卡尔积
- 生成 campaign 级汇总文件：`results/benchmark_v1/campaign_summary.csv`、`campaign_summary.tex`、`session_manifest.json`

### 4.4 指定分类器

```bash
python experiments/run.py \
    --datasets lost MSRCv2 \
    --methods sdlpp_baseline \
    --classifiers knn ipal \
    --campaign clf_compare
```

`--classifiers` 覆盖方法配置中的默认分类器，生成 `datasets × methods × classifiers` 的实验网格。

---

## 五、参数扫描与消融实验

### 5.1 SR × CB 消融实验

```bash
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET \
    --method sdlpp_sr_cb \
    --sweep disambig.params.use_sample_reliability=true,false \
    --sweep disambig.params.use_class_balance=true,false \
    --campaign ablation_sr_cb
```

生成 2×2=4 个实验配置的笛卡尔积。结果表中 `SR` 和 `CB` 作为独立列显示。

### 5.2 连续参数扫描

```bash
# r_min 参数扫描
python experiments/run.py \
    --datasets lost MSRCv2 \
    --method sdlpp_sr_cb \
    --sweep disambig.params.r_min=0.0,0.05,0.1,0.2,0.3 \
    --campaign sweep_rmin

# alpha 参数扫描
python experiments/run.py \
    --datasets lost \
    --method sdlpp_sr_cb \
    --sweep disambig.params.alpha=0.1,0.3,0.5,0.7,0.9 \
    --campaign sweep_alpha

# warmup 参数扫描
python experiments/run.py \
    --datasets lost \
    --method sdlpp_sr_cb \
    --sweep disambig.params.warmup=0,2,5,10,15 \
    --campaign sweep_warmup

# target_d 维度扫描
python experiments/run.py \
    --datasets lost \
    --method sdlpp_sr_cb \
    --sweep model.params.target_d=5,8,13,20,30 \
    --campaign sweep_target_d

# miu 参数扫描
python experiments/run.py \
    --datasets lost \
    --method sdlpp_sr_cb \
    --sweep model.params.miu=0.01,0.05,0.1,0.5,1.0 \
    --campaign sweep_miu
```

扫描的参数在结果表中作为**独立列**展示（如 `r_min`, `alpha` 等），而非笼统的 `sweep_params` 字段。

### 5.3 多参数联合扫描

```bash
# r_min × alpha 联合扫描（笛卡尔积）
python experiments/run.py \
    --datasets lost \
    --method sdlpp_sr_cb \
    --sweep disambig.params.r_min=0.0,0.1,0.2 \
    --sweep disambig.params.alpha=0.3,0.5,0.7 \
    --campaign sweep_rmin_alpha
```

产生 3×3=9 种参数组合。

### 5.4 `--sweep` 路径参考

| 参数 | `--sweep` 路径 | 表中列名 |
|---|---|---|
| Sample Reliability 开关 | `disambig.params.use_sample_reliability` | SR |
| Class Balance 开关 | `disambig.params.use_class_balance` | CB |
| r_min | `disambig.params.r_min` | r_min |
| alpha | `disambig.params.alpha` | alpha |
| warmup | `disambig.params.warmup` | warmup |
| cb_cv0 | `disambig.params.cb_cv0` | cv0 |
| cb_cv1 | `disambig.params.cb_cv1` | cv1 |
| target_d | `model.params.target_d` | target_d |
| miu | `model.params.miu` | miu |
| 迭代次数 T | `model.params.T` | T |

---

## 六、CLI 参数完整说明

```
python experiments/run.py [OPTIONS]
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `--dataset NAME` | str | 单数据集名称 |
| `--datasets N1 N2 ...` | str+ | 多数据集（与 `--dataset` 互斥） |
| `--method NAME` | str | 单方法配置名称（对应 `configs/methods/NAME.yaml`） |
| `--methods M1 M2 ...` | str+ | 多方法（与 `--method` 互斥） |
| `--classifiers C1 C2 ...` | str+ | 分类器列表（覆盖方法默认值） |
| `--clf-params k1=v1 ...` | str+ | 分类器参数覆盖（如 `n_neighbors=10`） |
| `--sweep path=v1,v2,...` | str | 参数扫描轴（可重复使用，生成笛卡尔积） |
| `--campaign NAME` | str | campaign 名称（默认 `single`），用于结果分组 |
| `--seed N` | int | 随机种子（覆盖配置文件） |
| `--n-repeats N` | int | Repeated Holdout 重复次数（覆盖配置文件） |
| `--fast` | flag | 冒烟模式：3 次 repeat，T=10 |
| `--verbose` | flag | 详细日志输出 |

---

## 七、配置系统详解

### 7.1 配置合并顺序（后者覆盖前者）

```
configs/base.yaml
  ↓ deep_merge
configs/datasets/{dataset_name}.yaml
  ↓ deep_merge
configs/methods/{method_name}.yaml
  ↓ deep_merge
CLI overrides (--seed, --n-repeats, --sweep 等)
```

### 7.2 base.yaml 默认值

```yaml
seed: 42
data:
  data_dir: datasets
preprocessing:
  method: zscore
eval:
  protocol: repeated_holdout
  n_repeats: 10        # 重复次数
  test_ratio: 0.2      # 测试集比例
  stratified: true      # 分层抽样
output:
  base_dir: results
  formats: [json]
```

### 7.3 数据集特定配置

每个数据集可配置：
- `data.name`：数据集名称（对应 `.mat` 文件名）
- `eval.stratified`：是否启用分层抽样（Mirflickr 等稀疏数据集建议设为 `false`）
- `dataset_note`：数据集备注信息（显示在结果表中）

### 7.4 方法配置

每个方法 YAML 包含：

```yaml
model:
  name: sdlpp          # 降维器名称 (sdlpp / delin / cenda)
  params:
    T: 100             # 迭代轮数
    target_d: 13       # 目标维度
    k: 8               # 近邻数
    miu: 0.1           # 权衡参数
    thr: 0.95          # 特征值阈值

disambig:              # 消歧策略（仅 SDLPP 使用）
  name: knn_propagation
  params:
    use_sample_reliability: true
    use_class_balance: true
    r_min: 0.1
    warmup: 5
    warmup_by_dataset:   # 按数据集覆盖 warmup
      MSRCv2: 2
      Mirflickr: 0
    alpha: 0.5

classifier:
  name: knn
  params:
    n_neighbors: 5

method_tag: sdlpp_sr_cb  # 简短标签，用于输出目录命名
```

### 7.5 warmup_by_dataset 机制

`warmup_by_dataset`、`r_min_by_dataset`、`alpha_by_dataset` 等映射允许按数据集覆盖消歧超参。在配置加载时，`resolve_disambig_params_by_dataset()` 会根据当前数据集名称，将对应值写入 `disambig.params` 中的 `warmup` / `r_min` / `alpha` 等字段。

---

## 八、评估协议

### 8.1 Repeated Holdout

- 默认 10 次重复，每次 80% 训练 / 20% 测试
- 优先使用分层抽样（`StratifiedShuffleSplit`）
- 若分层失败（某类样本数过少），自动降级为 `ShuffleSplit` 并记录日志

### 8.2 指标定义

| 指标 | 计算方式 |
|---|---|
| `overall_acc` | 测试集整体准确率 |
| `balanced_acc` | 仅对测试集**实际出现**的类计算 recall 均值 |
| `many_acc` | 训练集样本数最多的 1/3 类的 balanced accuracy |
| `medium_acc` | 训练集样本数中等的 1/3 类的 balanced accuracy |
| `few_acc` | 训练集样本数最少的 1/3 类的 balanced accuracy |

- `many/medium/few` 分组基于训练集类别频率
- 若某个分组在当前测试集中无样本，该组指标报告为 `NaN`（N/A），不计入均值
- 汇总结果以 `mean ± std` 形式展示，附带 `valid_runs`（有效 split 数）

### 8.3 稀疏数据集处理

- 含少于 5 个样本的类的数据集自动标记警告
- 结果表中以 `*` 后缀标识
- 建议仅作为补充结果，不作为主要 benchmark

---

## 九、结果文件结构

### 9.1 单次实验输出

```
results/{campaign}/{dataset}/{method_tag}+{classifier}/{sweep_suffix}/{timestamp}/
├── config.json       # 完整配置快照
├── splits.csv        # 每次 split 的详细指标
└── summary.json      # 汇总统计 (mean ± std)
```

### 9.2 Campaign 级汇总

```
results/{campaign}/
├── campaign_summary.csv    # 所有实验的 CSV 汇总
├── campaign_summary.tex    # LaTeX booktabs 表格
└── session_manifest.json   # 元信息（实验数、耗时、环境快照等）
```

### 9.3 文件命名规则

- `method_tag`：来自方法 YAML 的 `method_tag` 字段
- `sweep_suffix`：如 `r_min=0.1`、`SR=true_CB=false`
- `timestamp`：格式 `YYYYMMDD_HHMMSS`

---

## 十、CIFAR10 长尾 PLL 数据生成

### 10.1 前提条件

需要 CIFAR-10 MATLAB batch 文件位于 `datasets/cifar-10-batches-mat/`。

### 10.2 生成单个数据集

```bash
python experiments/generate_cifar10_lt_pll.py \
    --gamma 100 --r 2 --mode fast --seed 42 --save-meta
```

输出：`datasets/cifar10/cifar10-lt-g100-r2-fast.mat`

参数说明：
- `--gamma`：不平衡比（头类/尾类样本数比值）
- `--r`：每个样本的额外候选标签数
- `--mode`：`fast`（头类 1000 样本）或 `full`（头类 3000 样本）
- `--save-meta`：生成元信息 JSON

### 10.3 批量生成

```bash
python experiments/generate_cifar10_lt_pll_grid.py \
    --gammas 100,200 --rs 1,2,3 --seeds 42,43,44 --mode fast
```

生成 `2 × 3 × 3 = 18` 个数据集。

### 10.4 在实验中使用 CIFAR10

已为 6 个主要变体 (gamma=100/200 x r=1/2/3) 创建好配置文件：

```
configs/datasets/
├── cifar10-lt-g100-r1.yaml    # gamma=100, r=1, seed=42
├── cifar10-lt-g100-r2.yaml    # gamma=100, r=2, seed=42
├── cifar10-lt-g100-r3.yaml    # gamma=100, r=3, seed=42
├── cifar10-lt-g200-r1.yaml    # gamma=200, r=1, seed=42
├── cifar10-lt-g200-r2.yaml    # gamma=200, r=2, seed=42
└── cifar10-lt-g200-r3.yaml    # gamma=200, r=3, seed=42
```

运行 CIFAR10 实验（建议在服务器上执行，d=3072 计算量大）：

```bash
# 单个变体
python experiments/run.py \
    --dataset cifar10-lt-g100-r2 \
    --method sdlpp_sr_cb \
    --campaign cifar_test

# 全部 6 个变体 SR x CB 消融
python experiments/run.py \
    --datasets cifar10-lt-g100-r1 cifar10-lt-g100-r2 cifar10-lt-g100-r3 \
               cifar10-lt-g200-r1 cifar10-lt-g200-r2 cifar10-lt-g200-r3 \
    --method sdlpp_sr_cb \
    --sweep disambig.params.use_sample_reliability=false,true \
    --sweep disambig.params.use_class_balance=false,true \
    --n-repeats 5 \
    --campaign cifar10_ablation_sr_cb
```

另有 4 个多 seed 变体 (s43/s44) 的配置用于鲁棒性验证。

---

## 十一、一键全跑脚本 `run_all.sh`

`experiments/run_all.sh` 整合了全部实验批次，支持分阶段并行执行：

### 11.1 三个阶段

| Phase | 内容 | 批次 | 说明 |
|-------|------|------|------|
| 1 | 参数调优 | 1a,1b,2,3a,3b,4a,4b | 7 路并行，无依赖 |
| 2 | Benchmark + CIFAR10 | 5a,5b,5c,7a,7b | 5 路并行，需 Phase 1 结果 |
| 3 | 可选精细化 | 6a,6b,7c | 3 路并行 |

### 11.2 基本用法

```bash
# 预览所有命令（不执行）
bash experiments/run_all.sh --dry-run

# 执行 Phase 1（调参阶段）
bash experiments/run_all.sh --phase 1

# 执行 Phase 2（benchmark 阶段）
bash experiments/run_all.sh --phase 2

# 全部顺序执行
bash experiments/run_all.sh --phase all

# 只跑某个批次
bash experiments/run_all.sh --batch 7a

# 指定 Python 解释器
bash experiments/run_all.sh --phase 1 --python python3
```

### 11.3 服务器后台运行

```bash
nohup bash experiments/run_all.sh --phase 1 > run_phase1.log 2>&1 &
```

耗时较长的 Phase 1 / `5c` 等，见 [experiments/REMOTE_EXPERIMENTS.md](experiments/REMOTE_EXPERIMENTS.md)。

### 11.4 推荐工作流

```bash
# Step 1: 在服务器上跑 Phase 1 调参
bash experiments/run_all.sh --phase 1

# Step 2: 分析结果，确定最优参数，更新 configs/methods/sdlpp_sr_cb.yaml

# Step 3: 跑 Phase 2 benchmark + CIFAR10
bash experiments/run_all.sh --phase 2

# Step 4:（可选）Phase 3 精细化
bash experiments/run_all.sh --phase 3
```

### 11.5 日志

每个批次的输出记录在 `logs/<timestamp>/<batch_name>.log`。

### 11.6 批次清单

| ID | 名称 | 实验数 | 说明 |
|----|------|--------|------|
| 1a | SR x CB 核心数据集 | 16 | 4 核心数据集 x 4 SR/CB 组合 |
| 1b | SR x CB 大型数据集 | 8 | Soccer + Yahoo x 4 组合 |
| 2 | warmup 诊断 | 42 | 6 主表数据集 x 7 warmup 值 |
| 3a | r_min 敏感性 | 36 | 6 数据集 x 6 r_min 值 |
| 3b | alpha 敏感性 | 30 | 6 数据集 x 5 alpha 值 |
| 4a | target_d 扫描 (sr_cb) | 36 | 6 数据集 x 6 维度 |
| 4b | miu 扫描 | 30 | 6 数据集 x 5 miu 值 |
| 4c | target_d 扫描 (baseline) | 36 | 6 数据集 x 6 维度（`n_repeats=10`；**单独** `--batch 4c`，未并入 phase 1） |
| 5a | 核心 SDLPP benchmark | 40 | 4 数据集 x 2 方法 x 5 repeats |
| 5b | 大型 SDLPP benchmark | 20 | 2 数据集 x 2 方法 x 5 repeats |
| 5c | 主表分类器对比 | 120 | 6 数据集 x 2 方法 x 2 分类器 x 5 repeats |
| 6a | warmup x r_min 联合 | 18 | 2 数据集 x 3x3 |
| 6b | 自适应 CB 阈值 | 18 | 2 数据集 x 3x3 |
| 7a | CIFAR10 SR x CB | 24 | 6 变体 x 4 SR/CB 组合 |
| 7b | CIFAR10 SDLPP benchmark | 60 | 6 变体 x 2 方法 x 5 repeats |
| 7e | CIFAR ResNet18 + KNN/IPAL | 20 | 1 嵌入集 x 2 方法 x 2 分类器（需先跑嵌入脚本） |
| 7c | CIFAR10 多 seed | 4 | 2 变体 x 2 seed |
| | **合计（随批次变动）** | 见各批 | 长任务见 [experiments/REMOTE_EXPERIMENTS.md](experiments/REMOTE_EXPERIMENTS.md) |

---

## 十二、添加新组件

### 12.1 添加新降维方法

1. 在 `pll/reducers/` 下创建新文件，继承 `BaseReducer`，实现 `fit(X, partial_target, disambiguator=None)` 和 `transform(X)`
2. 在 `pll/eval/evaluator.py` 的 `_register_defaults()` 中注册
3. 创建 `configs/methods/your_method.yaml`

### 12.2 添加新分类器

1. 在 `pll/classifiers/` 下创建新文件，实现 `fit(X, y, ...)` 和 `predict(X)`
2. 若需要 `partial_target`，在 `fit` 签名中添加 `partial_target` 参数（`Evaluator` 会通过 `inspect.signature` 自动检测并传递）
3. 在 `pll/eval/evaluator.py` 的 `_register_defaults()` 和 `pll/classifiers/__init__.py` 中注册

### 12.3 添加新数据集

1. 将 `.mat` 文件放入 `datasets/`（需包含 `data`, `partial_target`, `target` 三个字段）
2. 创建 `configs/datasets/{name}.yaml`
3. 直接通过 `--dataset {name}` 使用

---

## 十三、后续实验规划

### 第一阶段：基线复现与验证（优先级 ★★★）

**目标**：在所有可用数据集上建立完整的基线结果，确认 Python 实现与 MATLAB 对齐。

```bash
# 1. 主表 SDLPP benchmark（六数据集；slashdot 仅 f1，DELIN/CENDA 需单独命令）
python experiments/run.py \
    --datasets lost MSRCv2 Mirflickr slashdotpl-f1 "Soccer Player" "Yahoo! News" \
    --methods sdlpp_baseline sdlpp_sr_cb \
    --n-repeats 10 \
    --campaign baseline_main_v1

# 2. KNN vs IPAL（与 run_all.sh batch 5c 一致）
python experiments/run.py \
    --datasets lost MSRCv2 Mirflickr slashdotpl-f1 "Soccer Player" "Yahoo! News" \
    --methods sdlpp_baseline sdlpp_sr_cb \
    --classifiers knn ipal \
    --campaign benchmark_clf_main_v1
```

**验收标准**：
- 所有实验正常完成，无报错
- SDLPP baseline 在 lost/MSRCv2/FG-NET 上的 balanced_acc 与 MATLAB 参考值偏差 < 2%
- 生成完整的 campaign_summary.csv 和 .tex 文件，可直接用于论文

---

### 第二阶段：SR + CB 机制消融（优先级 ★★★）

**目标**：验证 Sample Reliability 和 Class Balance 各自的贡献。

```bash
# 1. SR × CB 四因子消融
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET "Soccer Player" "Yahoo! News" \
    --method sdlpp_sr_cb \
    --sweep disambig.params.use_sample_reliability=true,false \
    --sweep disambig.params.use_class_balance=true,false \
    --campaign ablation_sr_cb_v1

# 2. warmup 影响分析
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET \
    --method sdlpp_sr_cb \
    --sweep disambig.params.warmup=0,2,5,8,10,15 \
    --campaign ablation_warmup_v1
```

**预期产出**：SR×CB 四格表（论文 Table）+ warmup 曲线图数据

---

### 第三阶段：关键超参数敏感性分析（优先级 ★★☆）

**目标**：系统评估各超参数对性能的影响，确定最优参数范围。

```bash
# 1. r_min 敏感性
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET "Soccer Player" \
    --method sdlpp_sr_cb \
    --sweep disambig.params.r_min=0.0,0.05,0.1,0.15,0.2,0.3,0.5 \
    --campaign sensitivity_rmin_v1

# 2. alpha 敏感性
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET "Soccer Player" \
    --method sdlpp_sr_cb \
    --sweep disambig.params.alpha=0.1,0.2,0.3,0.5,0.7,0.9 \
    --campaign sensitivity_alpha_v1

# 3. target_d 维度影响
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET \
    --method sdlpp_sr_cb \
    --sweep model.params.target_d=3,5,8,13,20,30,50 \
    --campaign sensitivity_target_d_v1

# 4. miu 权衡参数
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET \
    --method sdlpp_sr_cb \
    --sweep model.params.miu=0.01,0.05,0.1,0.5,1.0,5.0 \
    --campaign sensitivity_miu_v1

# 5. r_min × alpha 联合分析（核心参数交互）
python experiments/run.py \
    --datasets lost MSRCv2 \
    --method sdlpp_sr_cb \
    --sweep disambig.params.r_min=0.0,0.1,0.2 \
    --sweep disambig.params.alpha=0.3,0.5,0.7 \
    --campaign joint_rmin_alpha_v1
```

**预期产出**：各参数的 line plot 数据 + 热力图数据（r_min × alpha）

---

### 第四阶段：CIFAR10 长尾实验（优先级 ★★☆）

**目标**：在合成的长尾不平衡数据集上验证方法的鲁棒性。

```bash
# 1. 生成 CIFAR10 长尾数据
python experiments/generate_cifar10_lt_pll_grid.py \
    --gammas 100,200 --rs 1,2,3 --seeds 42,43,44 --mode fast

# 2. 创建数据集配置文件（示例，需为每个组合创建）
# configs/datasets/cifar10-lt-g100-r1-fast-s42.yaml 等

# 3. 在 CIFAR 上运行基线
python experiments/run.py \
    --datasets cifar10-lt-g100-r1-fast-s42 cifar10-lt-g100-r2-fast-s42 \
               cifar10-lt-g100-r3-fast-s42 \
    --methods sdlpp_baseline sdlpp_sr_cb \
    --campaign cifar_baseline_g100

# 4. CIFAR 上的 SR×CB 消融
python experiments/run.py \
    --datasets cifar10-lt-g100-r2-fast-s42 cifar10-lt-g200-r2-fast-s42 \
    --method sdlpp_sr_cb \
    --sweep disambig.params.use_sample_reliability=true,false \
    --sweep disambig.params.use_class_balance=true,false \
    --campaign cifar_ablation_sr_cb
```

**预期产出**：不同 gamma / r 设置下各方法的性能比较表

---

### 第五阶段：自适应 CB 与进阶分析（优先级 ★☆☆）

**目标**：探索自适应 alpha 策略和更细粒度的分析。

```bash
# 1. 自适应 CB alpha 的 cv0/cv1 阈值扫描
python experiments/run.py \
    --datasets lost MSRCv2 FG-NET \
    --method sdlpp_sr_cb \
    --sweep disambig.params.cb_cv0=0.05,0.1,0.2 \
    --sweep disambig.params.cb_cv1=0.3,0.5,0.7 \
    --campaign adaptive_cb_threshold_v1

# 2. 大数据集上的完整评估
python experiments/run.py \
    --datasets "Soccer Player" "Yahoo! News" \
    --methods sdlpp_baseline sdlpp_sr_cb delin cenda \
    --n-repeats 10 \
    --campaign large_dataset_v1
```

---

### 实验规划时间线

| 阶段 | 内容 | 预计实验数 | 预计耗时 |
|---|---|---|---|
| 第一阶段 | 基线复现 | 9×4 + 3×2×2 = 48 | 2-4 小时 |
| 第二阶段 | SR×CB 消融 + warmup | 5×4 + 3×6 = 38 | 2-3 小时 |
| 第三阶段 | 超参数敏感性 | ~100+ | 4-8 小时 |
| 第四阶段 | CIFAR10 长尾 | ~50+ | 视数据规模 |
| 第五阶段 | 自适应 CB + 大数据集 | ~50+ | 视需求 |

> **提示**：可使用 `--fast` 先快速试探，确认参数范围合理后再切换到完整 `--n-repeats 10` 运行。

---

### 后续可扩展方向

1. **新方法集成**：如 PiCO、PRODEN 等最新 PLL 方法，仅需实现 `BaseReducer` 接口
2. **Deep Feature 支持**：利用预训练模型提取特征，替换原始 `.mat` 特征
3. **更多分类器**：如 SVM、Random Forest 等，仅需实现 `fit/predict` 接口
4. **可视化模块**：自动绘制参数敏感性曲线、消融热力图
5. **收敛分析**：追踪 SDLPP 迭代过程中 Y 的变化、投影维度的变化
6. **统计检验**：对 repeated holdout 结果进行 paired t-test 或 Wilcoxon signed-rank test

---

## 十四、常见问题 (FAQ)

### Q1: Mirflickr 精度异常低？
Mirflickr 包含极稀疏类别（<5 样本），已配置 `stratified: false` 降级为随机分割。结果仅供参考（标有 `*` 前缀）。

### Q2: IPAL 运行很慢？
IPAL 对每个训练样本执行 NNLS 求解，计算量远大于 KNN。建议先在小数据集上测试，大数据集可考虑增加 `--clf-params k=5` 减少邻居数。

### Q3: 如何添加新数据集？
1. 准备 `.mat` 文件，包含 `data`（N×D）、`partial_target`（C×N）、`target`（C×N，one-hot）
2. 放入 `datasets/`
3. 创建 `configs/datasets/{name}.yaml`
4. 直接使用 `--dataset {name}`

### Q4: sweep 参数路径怎么写？
使用 YAML 配置中的点分路径。例如配置文件中的：
```yaml
disambig:
  params:
    r_min: 0.1
```
对应 `--sweep disambig.params.r_min=0.0,0.1,0.2`

### Q5: 如何只重新运行失败的实验？
目前需手动指定。可从 `session_manifest.json` 中查看失败实验，然后构造对应命令重跑。
