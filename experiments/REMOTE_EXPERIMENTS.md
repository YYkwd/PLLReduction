# 远程服务器上运行耗时实验

Phase 1（六数据集 sweep）与 Phase 2 主表（`5c` 六集 × 两方法 × 两分类器）耗时长，建议在 GPU/多核机器上用 `tmux`/`screen` 或 `nohup` 执行。

## 环境

```bash
cd /path/to/PllReduction
export PYTHONPATH=.   # 若需要
```

## 推荐：分批次执行

```bash
# 预览命令
bash experiments/run_all.sh --dry-run --batch 5c

# 单批（示例：主表分类器对比，约 120 个实验单元 × 内部 repeat）
nohup bash experiments/run_all.sh --batch 5c > logs/nohup_5c.log 2>&1 &

# Phase 1 全部并行（7 个子进程，总时长取决于最慢的一批）
nohup bash experiments/run_all.sh --phase 1 > logs/nohup_phase1.log 2>&1 &
```

## SDLPP baseline 在不同降维维度 `target_d` 上

与 batch `4a`（`sdlpp_sr_cb`）的 `target_d` 网格一致，但方法为 **`sdlpp_baseline`**，主表六数据集；结果目录 `results/sweep_baseline_target_d_v1/`。

```bash
nohup bash experiments/run_all.sh --batch 4c > logs/nohup_4c.log 2>&1 &
```

（36 组配置：6 数据集 × 6 个 `target_d`，每组 `n_repeats=10`，耗时长，务必在远程跑。）

## 按数据集单独调参（SR+CB 优于 baseline）

在完整 Phase 1 之外，对**仍落后 baseline 的单个数据集**收窄网格，例如只扫 `warmup`：

```bash
python experiments/run.py \
  --datasets "Yahoo! News" \
  --method sdlpp_sr_cb \
  --sweep disambig.params.warmup=0,3,5,10,20 \
  --n-repeats 10 \
  --campaign tune_warmup_yahoo_v1
```

将最优值写入 [configs/methods/sdlpp_sr_cb.yaml](../configs/methods/sdlpp_sr_cb.yaml) 的 `warmup_by_dataset` / `r_min_by_dataset` / `alpha_by_dataset`（后两者需配合 [pll/config.py](../pll/config.py) 中的 `*_by_dataset` 解析，已支持）。

## CIFAR：生成 ResNet18 嵌入 .mat 再跑 `7e`

1. 确保已有像素版 LT-PLL `.mat`（如 `datasets/cifar10/cifar10-lt-g100-eta0.3-s42.mat`）。
2. 在装有 PyTorch 的机器上：

```bash
python experiments/cifar10_mat_to_resnet18.py \
  --input-mat datasets/cifar10/cifar10-lt-g100-eta0.3-s42.mat \
  --output-mat datasets/cifar10/cifar10-lt-g100-eta0.3-s42-resnet18.mat
```

3. **将输出文件名中的数据集名**与 `run_all.sh` 里 `batch_7e` 的 `--datasets` 参数对齐（默认 `cifar10-lt-g100-eta0.3-s42-resnet18`，即 `.mat` 主文件名不含扩展名）。若你使用别的输出路径，请同步修改 `batch_7e` 或直接用 `run.py`：

```bash
python experiments/run.py \
  --datasets cifar10-lt-g100-eta0.3-s42-resnet18 \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers knn ipal \
  --n-repeats 5 \
  --campaign cifar10_resnet18_clf_v1
```

## DELIN / CENDA

当前 `5a`/`5b`/`7b` 已改为 **仅 SDLPP**。若需对比，请单独写 `run.py` 命令，勿放入默认定时 Phase 2。
