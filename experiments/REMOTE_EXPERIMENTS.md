# 远程服务器实验清单

建议在 `tmux` / `screen` 会话中执行，`tee` 同时输出到终端和日志文件。

```bash
cd /path/to/PllReduction
export PYTHONPATH=.
mkdir -p logs
```

---

## 实验 1：主基准表

**目的**：论文主表。六数据集 × 两方法 × 三分类器，per-dataset 最优 `target_d` 已写入 yaml。

| 数据集 | target_d | sr_cb 额外配置 |
|--------|---------|---------------|
| lost | 20 | r_min=0.2, alpha=0.3 |
| MSRCv2 | 8 | `use_distance_weight=false`（与 SR+CB 主设定一致） |
| Mirflickr | 13 | — |
| slashdotpl-f1 | 13 | — |
| Soccer Player | 50 | — |
| Yahoo! News | 50 | — |

```bash
python experiments/run.py \
  --datasets lost MSRCv2 Mirflickr slashdotpl-f1 "Soccer Player" "Yahoo! News" \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers knn ipal plsvm \
  --n-repeats 10 \
  --campaign main_table_v1 \
  2>&1 | tee logs/main_table.log
```

关注：`balanced_acc`、`few_acc`；SR+CB 预期在 Soccer/Yahoo/Mirflickr 上优于 baseline。

**补充：MSRCv2 `target_d` 对照**（`campaign=msrcv2_target_d_8_13_v1`，结果 `results/msrcv2_target_d_8_13_v1/`，日志 `logs/msrcv2_d8_d13_sweep.log`）：在 `dist_weight` 关闭、`n_repeats=10` 下，**knn / ipal** 在 d=8 与 d=13 上 SR+CB 的 `balanced_acc` 均高于同 d 的 baseline；**plsvm** 上 SR+CB 仍弱于 baseline。d=8 时 knn 的均衡提升最大（约 +0.031）。主表 yaml 已固定 MSRCv2 为 d=8。

```bash
python experiments/run.py \
  --datasets MSRCv2 \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers knn ipal plsvm \
  --n-repeats 10 \
  --sweep model.params.target_d=8,13 \
  --campaign msrcv2_target_d_8_13_v1 \
  2>&1 | tee logs/msrcv2_d8_d13_sweep.log
```

---

## 实验 2：SURE（核 / 线性分开跑）

SURE 通过 `classifier.params.use_kernel` 显式选择：**不要依赖自动阈值**，否则中小数据集与论文设定不一致。

| 模式 | 适用 | 说明 |
|------|------|------|
| `use_kernel=true` | 中小四集 + **可选** Soccer/Yahoo | 与论文一致；大数据集上 K 为 m×m，内存与时间陡增 |
| `use_kernel=false` | Soccer / Yahoo（默认推荐先跑） | 省内存；线性 SURE 与论文 Table 4 不可直接对比 |

**大数据集要不要试核 SURE？**  
**建议试**：论文 Table 4 在 Soccer/Yahoo 上报告的就是核 SURE；线性版在 Soccer 上曾出现多数类坍塌，核版通常更合理。代价是 **RAM**：训练集约 m≈0.8N，单块稠密核矩阵 K 约 **8m² 字节**（float64）——Soccer 约 **1.5–2 GB** 量级，Yahoo 约 **2.5–3 GB** 仅 K；LU 分解 B 时峰值常再乘 1.5–2×。**建议内存 ≥ 32 GB** 再跑 2c；16 GB 可能 OOM 或极慢。可先 `--n-repeats 3` 试通。

未设置 `use_kernel` 时：**自动**在 `m <= kernel_max_samples`（默认 2500）用核，否则线性。

```bash
# 2a：中小型数据集 — 强制 RBF 核（与论文 Table 4 设定一致）
python experiments/run.py \
  --datasets lost MSRCv2 Mirflickr slashdotpl-f1 \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers sure \
  --clf-params use_kernel=true \
  --n-repeats 5 \
  --campaign sure_kernel_small_v1 \
  2>&1 | tee logs/sure_kernel_small.log

# 2b：大数据集 — 强制线性（省资源；与论文核 SURE 不对齐，作快速对照）
python experiments/run.py \
  --datasets "Soccer Player" "Yahoo! News" \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers sure \
  --clf-params use_kernel=false \
  --n-repeats 5 \
  --campaign sure_linear_large_v1 \
  2>&1 | tee logs/sure_linear_large.log

# 2c（可选）：大数据集 — 强制核 SURE，与论文 Table 4 可比；需大内存、耗时长
python experiments/run.py \
  --datasets "Soccer Player" "Yahoo! News" \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers sure \
  --clf-params use_kernel=true \
  --n-repeats 3 \
  --campaign sure_kernel_large_v1 \
  2>&1 | tee logs/sure_kernel_large.log
```

若某机器内存仍紧张：优先跑 2b；或对 2a 中单集减小 `n_repeats`。

---

## 关于 SR+CB 消融实验

已有结果（`results/ablation_sr_cb_core_v2`、`results/ablation_sr_cb_large_v2`）覆盖四种组合
（SR=F/CB=F、SR=T/CB=F、SR=F/CB=T、SR=T/CB=T）× 6 个数据集，**无需重跑**。

---

## CIFAR：ResNet18 嵌入实验（第二阶段）

1. 生成嵌入：

```bash
python experiments/cifar10_mat_to_resnet18.py \
  --input-mat datasets/cifar10/cifar10-lt-g100-eta0.3-s42.mat \
  --output-mat datasets/cifar10/cifar10-lt-g100-eta0.3-s42-resnet18.mat
```

2. 运行评估：

```bash
python experiments/run.py \
  --datasets cifar10-lt-g100-eta0.3-s42-resnet18 \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers knn ipal \
  --n-repeats 5 \
  --campaign cifar10_resnet18_v1 \
  2>&1 | tee logs/cifar_resnet18.log
```
