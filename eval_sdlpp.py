"""
SDLPP 参数消融实验: 对比 use_sample_reliability 与 use_class_balance 的四种组合
评估降维后 KNN 分类: Overall_Acc, Many/Medium/Few_Acc, Balanced_Acc
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sdlpp import sdlpp


def load_or_generate_data(data_path):
    """加载 .mat 或生成合成数据"""
    path = Path(data_path)
    if path.exists():
        mat = sio.loadmat(str(path))
        data = mat['data']
        partial_target = mat['partial_target']
        if hasattr(partial_target, 'toarray'):
            partial_target = partial_target.toarray()
        target = mat.get('target')
        if target is not None and hasattr(target, 'toarray'):
            target = target.toarray()
        elif target is not None:
            target = np.asarray(target)
        return data, partial_target, target

    # 合成数据: 5 类, 长尾分布, 部分标签
    np.random.seed(42)
    n_classes = 5
    n_per_class = [200, 150, 100, 80, 50]  # Many -> Few
    n_samples = sum(n_per_class)
    d = 50
    data_list, target_list = [], []
    centers = np.random.randn(n_classes, d) * 2
    for c in range(n_classes):
        X_c = centers[c] + np.random.randn(n_per_class[c], d) * 0.8
        data_list.append(X_c)
        target_list.append(np.full(n_per_class[c], c))
    data = np.vstack(data_list)
    target = np.hstack(target_list)
    # partial_target: 每样本 r=2 个候选标签
    r = 2
    partial_target = np.zeros((n_classes, n_samples))
    for i in range(n_samples):
        true_c = target[i]
        cands = [true_c]
        others = [j for j in range(n_classes) if j != true_c]
        cands.append(np.random.choice(others))
        for c in cands:
            partial_target[c, i] = 1
    return data, partial_target, target


def get_many_medium_few_splits(y_train, n_classes):
    """按类别样本数分为 Many/Medium/Few (各约 1/3 类)"""
    counts = np.bincount(y_train, minlength=n_classes)
    sorted_idx = np.argsort(-counts)
    n = len(sorted_idx)
    m, f = (n + 2) // 3, (2 * n + 2) // 3
    many_classes = set(sorted_idx[:m])
    medium_classes = set(sorted_idx[m:f])
    few_classes = set(sorted_idx[f:])
    return many_classes, medium_classes, few_classes


def compute_metrics(y_true, y_pred, many_set, medium_set, few_set, n_classes):
    """Overall_Acc, Many/Medium/Few_Acc, Balanced_Acc"""
    overall = np.mean(y_true == y_pred)

    def acc_for_set(classes):
        mask = np.isin(y_true, list(classes))
        if mask.sum() == 0:
            return 0.0
        return np.mean(y_pred[mask] == y_true[mask])

    many_acc = acc_for_set(many_set)
    medium_acc = acc_for_set(medium_set)
    few_acc = acc_for_set(few_set)

    per_class_acc = []
    for c in range(n_classes):
        m = y_true == c
        if m.sum() > 0:
            per_class_acc.append(np.mean(y_pred[m] == y_true[m]))
        else:
            per_class_acc.append(0.0)
    balanced_acc = np.mean(per_class_acc)

    return {
        'Overall_Acc': overall,
        'Many_Acc': many_acc,
        'Medium_Acc': medium_acc,
        'Few_Acc': few_acc,
        'Balanced_Acc': balanced_acc,
    }


def run_sdlpp_and_eval(data, partial_target, target, use_sr, use_cb, k_folds=5, knn_k=5):
    """在给定开关下运行 SDLPP + KNN 评估"""
    n_classes = partial_target.shape[0]
    X = data.copy()
    y = np.argmax(target, axis=0) if target.ndim > 1 else target

    para = {
        'T': 50,
        'target_d': 13,
        'k': 8,
        'miu': 0.1,
        'thr': 0.95,
        'use_sample_reliability': use_sr,
        'use_class_balance': use_cb,
        'imbalance_alpha': 0.5,
        'imbalance_eps': 1e-8,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        pt_train = partial_target[:, train_idx]

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0, ddof=1)
        std[std == 0] = 1
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        lower_train, P, _ = sdlpp(X_train, pt_train, para)
        lower_test = X_test @ P

        many_set, medium_set, few_set = get_many_medium_few_splits(y_train, n_classes)

        knn = KNeighborsClassifier(n_neighbors=knn_k)
        knn.fit(lower_train, y_train)
        y_pred = knn.predict(lower_test)

        m = compute_metrics(y_test, y_pred, many_set, medium_set, few_set, n_classes)
        all_metrics.append(m)

    avg = {}
    for k in all_metrics[0]:
        avg[k] = np.mean([x[k] for x in all_metrics])
    return avg


def main():
    base = Path(__file__).parent
    data_path = base / 'datasets' / 'lost.mat'
    data, partial_target, target = load_or_generate_data(data_path)

    if target is None:
        print("数据集中无 target，无法评估分类。使用合成数据。")
        data, partial_target, target = load_or_generate_data('')
    if target.ndim > 1:
        target = np.argmax(target, axis=0)

    n_samples, n_features = data.shape
    n_classes = partial_target.shape[0]
    print(f"数据: {n_samples} 样本, {n_features} 维, {n_classes} 类")
    print("=" * 70)

    configs = [
        (False, False, "两者都关"),
        (True, False, "仅 sample reliability"),
        (False, True, "仅 class balance"),
        (True, True, "两者都开"),
    ]

    results = []
    for use_sr, use_cb, name in configs:
        print(f"运行: {name} ...")
        m = run_sdlpp_and_eval(data, partial_target, target, use_sr, use_cb)
        results.append((name, m))
        print(f"  Overall={m['Overall_Acc']:.4f} Many={m['Many_Acc']:.4f} "
              f"Medium={m['Medium_Acc']:.4f} Few={m['Few_Acc']:.4f} Balanced={m['Balanced_Acc']:.4f}")

    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    headers = ["配置", "Overall_Acc", "Many_Acc", "Medium_Acc", "Few_Acc", "Balanced_Acc"]
    row_fmt = "{:20s} {:12.4f} {:10.4f} {:10.4f} {:10.4f} {:12.4f}"
    print(f"{'配置':<20} {'Overall_Acc':>12} {'Many_Acc':>10} {'Medium_Acc':>10} {'Few_Acc':>10} {'Balanced_Acc':>12}")
    print("-" * 70)
    for name, m in results:
        print(row_fmt.format(name, m['Overall_Acc'], m['Many_Acc'], m['Medium_Acc'], m['Few_Acc'], m['Balanced_Acc']))

    print("\n" + "=" * 70)
    print("各优化贡献分析")
    print("=" * 70)
    base_m = results[0][1]
    sr_only = results[1][1]
    cb_only = results[2][1]
    both_m = results[3][1]

    print("\n1. Sample Reliability 贡献 (单独开启 vs 两者都关):")
    for k in base_m:
        delta = sr_only[k] - base_m[k]
        print(f"   {k}: {delta:+.4f} ({base_m[k]:.4f} -> {sr_only[k]:.4f})")

    print("\n2. Class Balance 贡献 (单独开启 vs 两者都关):")
    for k in base_m:
        delta = cb_only[k] - base_m[k]
        print(f"   {k}: {delta:+.4f} ({base_m[k]:.4f} -> {cb_only[k]:.4f})")

    print("\n3. 两者协同 (两者都开 vs 基线):")
    for k in base_m:
        delta = both_m[k] - base_m[k]
        print(f"   {k}: {delta:+.4f} ({base_m[k]:.4f} -> {both_m[k]:.4f})")

    print("\n4. 结论:")
    if both_m['Balanced_Acc'] > base_m['Balanced_Acc']:
        print("   两者都开对 Balanced_Acc 有提升，有利于长尾/少样本类。")
    if both_m['Few_Acc'] > base_m['Few_Acc']:
        print("   两者都开使 Few 类准确率显著提升。")
    if sr_only['Few_Acc'] < base_m['Few_Acc'] and both_m['Few_Acc'] > base_m['Few_Acc']:
        print("   sample reliability 单独会损害 Few 类，但与 class balance 协同后转为正向。")
    if cb_only['Few_Acc'] > base_m['Few_Acc']:
        print("   class balance 单独即可提升 Few 类表现，是少样本类的核心优化。")


if __name__ == '__main__':
    main()
