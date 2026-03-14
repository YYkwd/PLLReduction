#!/usr/bin/env python
"""
数据集查看工具
用于查看 MATLAB 数据集的详细信息
"""

import scipy.io as sio
import numpy as np
import argparse
from pathlib import Path


def view_dataset(mat_file):
    """查看 MATLAB 数据集的详细信息"""
    
    # 加载数据集
    print(f"\n加载数据集: {mat_file}")
    mat_data = sio.loadmat(mat_file)
    
    print("=" * 70)
    print(f"数据集: {Path(mat_file).name}")
    print("=" * 70)
    
    # 显示所有字段
    print("\n包含的字段:")
    fields = [key for key in mat_data.keys() if not key.startswith('__')]
    for key in fields:
        print(f"  - {key}")
    
    print("\n" + "=" * 70)
    
    # 详细信息
    for key in fields:
        data = mat_data[key]
        print(f"\n{'='*70}")
        print(f"字段: {key}")
        print(f"{'='*70}")
        print(f"类型: {type(data).__name__}")
        print(f"形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        
        # 处理稀疏矩阵
        if hasattr(data, 'toarray'):
            print(f"稀疏矩阵: 是")
            print(f"非零元素: {data.nnz}")
            print(f"稀疏度: {(1 - data.nnz / (data.shape[0] * data.shape[1])) * 100:.2f}%")
            dense_data = data.toarray()
            
            # 如果是标签矩阵
            if 'target' in key.lower():
                print(f"\n标签信息:")
                print(f"  类别数: {data.shape[0]}")
                print(f"  样本数: {data.shape[1]}")
                
                # 每个样本的标签数
                labels_per_sample = np.sum(dense_data, axis=0)
                print(f"\n  每个样本的标签数:")
                print(f"    最少: {int(labels_per_sample.min())}")
                print(f"    最多: {int(labels_per_sample.max())}")
                print(f"    平均: {labels_per_sample.mean():.2f}")
                
                # 显示前几个样本的标签
                print(f"\n  前 5 个样本的标签:")
                for i in range(min(5, data.shape[1])):
                    labels = np.where(dense_data[:, i] > 0)[0]
                    print(f"    样本 {i}: 标签 {labels.tolist()}")
                
                # 如果是真实标签，统计类别分布
                if key == 'target' and labels_per_sample.max() == 1:
                    true_labels = np.argmax(dense_data, axis=0)
                    unique_labels, counts = np.unique(true_labels, return_counts=True)
                    print(f"\n  类别分布:")
                    for label, count in sorted(zip(unique_labels, counts)):
                        percentage = count / len(true_labels) * 100
                        print(f"    类别 {label:2d}: {count:4d} 样本 ({percentage:5.2f}%)")
        
        # 处理普通数组（特征数据）
        else:
            print(f"\n数据统计:")
            print(f"  最小值: {data.min():.4f}")
            print(f"  最大值: {data.max():.4f}")
            print(f"  均值:   {data.mean():.4f}")
            print(f"  标准差: {data.std():.4f}")
            print(f"  中位数: {np.median(data):.4f}")
            
            # 显示数据的一小部分
            print(f"\n  数据预览 (前 3 个样本, 前 5 个特征):")
            preview = data[:3, :5] if data.ndim == 2 else data[:5]
            for i, row in enumerate(preview):
                if data.ndim == 2:
                    values = ', '.join([f"{v:8.2f}" for v in row])
                    print(f"    样本 {i}: [{values}, ...]")
                else:
                    print(f"    {row}")
    
    print("\n" + "=" * 70)
    print("数据集查看完成")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='查看 MATLAB 数据集的详细信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python view_dataset.py datasets/lost.mat
  python view_dataset.py datasets/lost.mat --simple
        """
    )
    
    parser.add_argument('mat_file', type=str, help='MATLAB 数据文件路径 (.mat)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.mat_file).exists():
        print(f"错误: 文件不存在: {args.mat_file}")
        return 1
    
    # 查看数据集
    view_dataset(args.mat_file)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
