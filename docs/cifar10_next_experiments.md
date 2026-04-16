# CIFAR10：`target_d` 扫描结论与下一步实验

依据日志 `logs/cifar10_target_d_sweep_20260414_212002.log`（campaign `cifar10_target_d_sweep_v1`，**已完成** `cifar10-lt-g100-eta0.3-fast-s42` 上 baseline / `sdlpp_sr_cb` 全 6 档 d；**full** `cifar10-lt-g100-eta0.3-s42` 在 d=13 处日志截断，若需全表请重跑或查 `results/cifar10_target_d_sweep_v1/` 是否已落盘）。

## 1. 日志中的关键现象

### 1.1 迭代维数 ≠ 配置的 `target_d`

- `target_d=13` 时多次出现 `SDLPP fit: 100/100 iters, stop=max_iter, dim 3072->29`：迭代内部维数卡在 **~29**，与最终投影维 13 是两套机制（见 `pll/reducers/sdlpp.py` 中 `thr=0.95` 与最终 `solve_projection(..., target_d)`）。
- `target_d≥30` 时多为 `stop=min_dim`，中间维逐步压到 `target_d−1` 附近（如 49、98、191），说明**瓶颈不只在「最后一步取多少维」**，还在 **SDLPP 迭代 + 消歧** 是否能把 Y 学到对长尾友好。

### 1.2 `balanced_acc`（fast 集，DONE 汇总）

**`sdlpp_baseline` + knn**（越高越好）：  
d=13 → 0.1374；30 → 0.1344；50 → 0.1303；100 → 0.1286；150 → 0.1267；200 → **0.1232**  
→ **增大 d 无收益，略变差**。

**`sdlpp_baseline` + plsvm**（相对最稳）：约 **0.153～0.154**，随 d 变化很小。

**`sdlpp_sr_cb` + knn**：在 0.128～0.138 间波动，**未随 d 单调上升**；d=13 与中间档差异不大。  
**`sdlpp_sr_cb` + ipal**：d=13 时 balanced 最高（~0.147），d 增大后降至 ~0.12～0.13——与「只靠加维救 IPAL」假设不符。  
**`sdlpp_sr_cb` + plsvm**：d=50 时 balanced 最高（~0.162），再高略降，说明存在**局部最优 d**，但不是「越大越好」。

### 1.3 与旧 CIFAR 消融的一致性

此前 `results/cifar10_ablation_sr_cb/` 上 SR/CB 几乎拉不开差距；本日志进一步说明：**在像素 + SDLPP 这一路线上，主要矛盾不是「少了几维」**，而是 **表示与算法与长尾 PLL 的匹配度**。

---

## 2. 结论（给汇报用）

1. **单纯提高 `target_d`（13→200）不能作为 CIFAR10 像素 PLL 的主改进方向**；`balanced_acc` 整体无持续增益，部分设置更差。  
2. **若保留 SDLPP 框架**：可固定 **较小 d（如 13～50）** 以省算力，把精力放到 **表征或其它方法**。  
3. **日志未跑完 full 集全部 d** 时，不宜仅凭 fast 外推 full；至少补全 **full 上 d∈{13,50}** 对照即可验证趋势是否一致。

---

## 3. 下一步实验规划（按优先级）

### 阶段 A：表征升级（首推）

| 实验 | 目的 | 命令要点 |
|------|------|------------|
| **ResNet18 嵌入** | 将 3072 像素换为低维语义特征，再跑 SDLPP baseline/sr_cb × knn/ipal | 见 [`experiments/REMOTE_EXPERIMENTS.md`](../experiments/REMOTE_EXPERIMENTS.md)：`cifar10_mat_to_resnet18.py` 生成 `.mat` 后 `run.py`；**需重新扫 d**（嵌入维通常 512 量级，与像素最优 d 不可沿用）。 |

成功标准：`few_acc` 非零或 `balanced_acc` 相对像素基线 **+0.05 以上** 再考虑写进主表。

### 阶段 B：换降维/消歧管线（与 d 解耦）

| 实验 | 目的 |
|------|------|
| **DELIN / CENDA**（仓库已有 [`configs/methods/delin.yaml`](../configs/methods/delin.yaml)、[`cenda.yaml`](../configs/methods/cenda.yaml)） | 验证「非 SDLPP」是否在 CIFAR PLL 上更稳。 |
| **SDLPP 调参** | 固定 `target_d=13` 或 50，扫 `T`、`thr`、`k`、`miu`；或单独为 CIFAR 在 `sdlpp_sr_cb.yaml` 增加 `warmup_by_dataset` / `r_min`（当前 CIFAR 无专项）。 |

### 阶段 C：强基线（证明「不是评估 bug」）

| 实验 | 目的 |
|------|------|
| **原始特征 + kNN**（不降维、或 PCA→d） | 给出「无 PLL 结构」的上界/对照。 |
| **全监督 / 非 PLL 线性基线**（若数据允许） | 可选，用于论文 Related 与讨论。 |

### 阶段 D：补全本次 sweep 的缺口（低成本）

```bash
# 仅补 full 集、两档 d、与日志一致的方法与分类器
python experiments/run.py \
  --datasets cifar10-lt-g100-eta0.3-s42 \
  --methods sdlpp_baseline sdlpp_sr_cb \
  --classifiers knn ipal plsvm \
  --sweep model.params.target_d=13,50,100 \
  --n-repeats 5 \
  --campaign cifar10_target_d_full_verify_v1 \
  2>&1 | tee logs/cifar10_target_d_full_verify.log
```

若 full 上趋势与 fast 一致，可**正式停止**大规模 d 扫描，转入阶段 A/B。

---

## 4. 不建议继续做的

- **在像素上继续细扫 d>200** 或密集网格：收益概率低、单次 `run_multi_classifier` 时间已 ~3–4 分钟/档（fast），full 更慢。  
- **期望仅靠 SR/CB + 提维** 扭转 CIFAR 长尾：日志与旧消融均不支持。

---

## 5. 可选：将 CIFAR 默认 `target_d` 写入 yaml

若论文叙事为「像素 CIFAR 仅作探索性附录」，可固定 **d=13 或 50**（与 plsvm/sr_cb 在 fast 上的局部峰对齐），在 [`configs/methods/sdlpp_baseline.yaml`](../configs/methods/sdlpp_baseline.yaml) / [`sdlpp_sr_cb.yaml`](../configs/methods/sdlpp_sr_cb.yaml) 的 `target_d_by_dataset` 中增加 `cifar10-lt-g100-eta0.3-s42` 等键，避免与 tabular 数据集共用默认 13 的歧义。
