#!/usr/bin/env python3
"""
数据集划分工具

功能：将完整的配置文件按比例划分为训练集和验证集
- 对每个 pipeline 的图像对按指定比例随机划分
- 生成 train_config.json 和 val_config.json
- 使用固定随机种子确保可复现
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def split_config(
    input_config_path,
    output_dir,
    train_ratio=0.9,
    seed=42,
    train_name="train_config.json",
    val_name="val_config.json"
):
    """
    将配置文件按比例划分为训练集和验证集

    Args:
        input_config_path: 输入配置文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例（默认 0.9）
        seed: 随机种子（默认 42）
        train_name: 训练配置文件名
        val_name: 验证配置文件名
    """
    # 设置随机种子
    random.seed(seed)

    # 读取原始配置
    print(f"读取配置文件: {input_config_path}")
    with open(input_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 创建训练集和验证集配置
    train_config = {
        "description": f"训练集配置 ({train_ratio*100:.0f}% 划分)",
        "total_pipelines": 0,
        "pipelines": []
    }

    val_config = {
        "description": f"验证集配置 ({(1-train_ratio)*100:.0f}% 划分)",
        "total_pipelines": 0,
        "pipelines": []
    }

    # 统计信息
    stats = {
        'total_pipelines': 0,
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'pipeline_stats': []
    }

    print(f"\n开始划分数据集（训练/验证 = {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%）...")
    print("="*60)

    # 遍历每个 pipeline
    for pipeline_config in config['pipelines']:
        pipeline_id = pipeline_config['id']
        pipeline = pipeline_config['pipeline']
        data = pipeline_config['data']

        # 统计
        stats['total_pipelines'] += 1
        stats['total_images'] += len(data)

        # 随机打乱数据
        data_shuffled = data.copy()
        random.shuffle(data_shuffled)

        # 计算划分点
        split_idx = int(len(data_shuffled) * train_ratio)

        # 划分数据
        train_data = data_shuffled[:split_idx]
        val_data = data_shuffled[split_idx:]

        # 统计
        stats['train_images'] += len(train_data)
        stats['val_images'] += len(val_data)

        # 记录每个 pipeline 的统计信息
        pipeline_name = '+'.join(pipeline)
        stats['pipeline_stats'].append({
            'id': pipeline_id,
            'pipeline': pipeline_name,
            'total': len(data),
            'train': len(train_data),
            'val': len(val_data)
        })

        # 添加到训练集配置
        if train_data:
            train_config['pipelines'].append({
                'id': pipeline_id,
                'pipeline': pipeline,
                'data': train_data
            })

        # 添加到验证集配置
        if val_data:
            val_config['pipelines'].append({
                'id': pipeline_id,
                'pipeline': pipeline,
                'data': val_data
            })

        print(f"Pipeline {pipeline_id:3d} | {pipeline_name:50s} | "
              f"总计: {len(data):4d} | 训练: {len(train_data):4d} | 验证: {len(val_data):3d}")

    # 更新 pipeline 计数
    train_config['total_pipelines'] = len(train_config['pipelines'])
    val_config['total_pipelines'] = len(val_config['pipelines'])

    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置文件
    train_config_path = output_dir / train_name
    val_config_path = output_dir / val_name

    with open(train_config_path, 'w', encoding='utf-8') as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    with open(val_config_path, 'w', encoding='utf-8') as f:
        json.dump(val_config, f, ensure_ascii=False, indent=2)

    # 保存统计信息
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 打印统计摘要
    print("="*60)
    print("\n📊 划分统计摘要:")
    print(f"  总 Pipelines: {stats['total_pipelines']}")
    print(f"  总图像数: {stats['total_images']}")
    print(f"  训练集图像: {stats['train_images']} ({stats['train_images']/stats['total_images']*100:.1f}%)")
    print(f"  验证集图像: {stats['val_images']} ({stats['val_images']/stats['total_images']*100:.1f}%)")
    print(f"\n✓ 配置文件已保存:")
    print(f"  训练集: {train_config_path}")
    print(f"  验证集: {val_config_path}")
    print(f"  统计信息: {stats_path}")

    return train_config_path, val_config_path, stats_path


def main():
    """主函数"""
    # 动态路径计算
    script_dir = Path(__file__).parent.absolute()
    restormer_root = script_dir.parent
    chain_cuda_root = restormer_root.parent
    data_dir = chain_cuda_root / "data"
    config_dir = data_dir / "Comb_Config"

    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('--input', type=str,
                       default=str(config_dir / "1_3_6_config.json"),
                       help='输入配置文件路径')
    parser.add_argument('--output-dir', type=str,
                       default=str(config_dir),
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='训练集比例 (默认 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认 42)')
    parser.add_argument('--train-name', type=str, default="train_config.json",
                       help='训练配置文件名')
    parser.add_argument('--val-name', type=str, default="val_config.json",
                       help='验证配置文件名')

    args = parser.parse_args()

    print("="*60)
    print("数据集划分工具")
    print("="*60)
    print(f"输入: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练/验证比例: {args.train_ratio*100:.0f}% / {(1-args.train_ratio)*100:.0f}%")
    print(f"随机种子: {args.seed}")

    split_config(
        input_config_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        train_name=args.train_name,
        val_name=args.val_name
    )


if __name__ == "__main__":
    main()
