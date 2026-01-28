#!/usr/bin/env python3
"""
划分训练集和验证集
从 unified_config.json 中按比例随机划分数据集
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

# 动态路径计算
SCRIPT_DIR = Path(__file__).parent.absolute()
CHAIN_CUDA_ROOT = SCRIPT_DIR.parent
DATA_DIR = CHAIN_CUDA_ROOT / "data"
CONFIG_DIR = DATA_DIR / "Comb_Config"


def split_train_val(
    input_config_path=None,
    output_val_path=None,
    output_train_path=None,
    val_ratio=0.1,
    random_seed=42
):
    """
    从训练配置中划分验证集

    Args:
        input_config_path: 输入的完整配置文件路径（默认使用动态路径）
        output_val_path: 输出的验证配置文件路径（默认使用动态路径）
        output_train_path: 输出的训练配置文件路径（默认使用动态路径）
        val_ratio: 验证集比例 (0.0-1.0)，默认 0.1 (10%)
        random_seed: 随机种子，保证可复现
    """
    # 使用动态默认路径
    if input_config_path is None:
        input_config_path = str(CONFIG_DIR / "unified_config.json")
    if output_val_path is None:
        output_val_path = str(CONFIG_DIR / "val_config.json")
    if output_train_path is None:
        output_train_path = str(CONFIG_DIR / "train_config.json")

    print(f"读取配置文件: {input_config_path}")

    # 读取原始配置
    with open(input_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"原始配置:")
    print(f"  - Total pipelines: {config['total_pipelines']}")

    # 统计总数据量
    total_data_count = sum(len(p['data']) for p in config['pipelines'])
    print(f"  - Total data entries: {total_data_count}")

    # 计算验证集样本数
    num_val_samples = int(total_data_count * val_ratio)
    print(f"\n验证集比例: {val_ratio:.1%}")
    print(f"验证集样本数: {num_val_samples}")

    if num_val_samples == 0:
        print(f"警告: 验证集样本数为 0，请增大数据量或提高 val_ratio")
        return

    # 设置随机种子
    random.seed(random_seed)

    # 收集所有数据项（带 pipeline 信息）
    all_data_items = []
    for pipeline_idx, pipeline in enumerate(config['pipelines']):
        for data_item in pipeline['data']:
            all_data_items.append({
                'pipeline_idx': pipeline_idx,
                'pipeline': pipeline['pipeline'],
                'data': data_item
            })

    print(f"\n随机划分数据集...")

    # 随机打乱并选择
    random.shuffle(all_data_items)
    val_items = all_data_items[:num_val_samples]
    train_items = all_data_items[num_val_samples:]

    print(f"  - 验证集样本: {len(val_items)} ({len(val_items)/total_data_count:.1%})")
    print(f"  - 训练集样本: {len(train_items)} ({len(train_items)/total_data_count:.1%})")

    # 重新组织验证数据
    val_pipelines_dict = defaultdict(list)
    for item in val_items:
        pipeline_key = tuple(item['pipeline'])
        val_pipelines_dict[pipeline_key].append(item['data'])

    # 构建验证配置
    val_pipelines = []
    for idx, (pipeline_key, data_list) in enumerate(sorted(val_pipelines_dict.items()), start=1):
        val_pipelines.append({
            "id": idx,
            "pipeline": list(pipeline_key),
            "data": data_list
        })

    val_config = {
        "description": "验证集配置文件（从训练数据中随机抽取）",
        "total_pipelines": len(val_pipelines),
        "pipelines": val_pipelines
    }

    # 重新组织训练数据
    train_pipelines_dict = defaultdict(list)
    for item in train_items:
        pipeline_key = tuple(item['pipeline'])
        train_pipelines_dict[pipeline_key].append(item['data'])

    # 构建训练配置
    train_pipelines = []
    for idx, (pipeline_key, data_list) in enumerate(sorted(train_pipelines_dict.items()), start=1):
        train_pipelines.append({
            "id": idx,
            "pipeline": list(pipeline_key),
            "data": data_list
        })

    train_config = {
        "description": config.get('description', '训练配置文件') + "（已划分出验证集）",
        "total_pipelines": len(train_pipelines),
        "pipelines": train_pipelines
    }

    # 保存验证配置
    print(f"\n保存验证配置到: {output_val_path}")
    with open(output_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_config, f, ensure_ascii=False, indent=2)

    print(f"验证配置统计:")
    print(f"  - Total pipelines: {val_config['total_pipelines']}")
    print(f"  - Total data entries: {sum(len(p['data']) for p in val_pipelines)}")

    # 保存训练配置
    print(f"\n保存训练配置到: {output_train_path}")
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    print(f"训练配置统计:")
    print(f"  - Total pipelines: {train_config['total_pipelines']}")
    print(f"  - Total data entries: {sum(len(p['data']) for p in train_pipelines)}")

    # 打印一些示例
    print(f"\n验证配置中的前 3 个 pipeline:")
    for i, pipeline in enumerate(val_pipelines[:3], 1):
        print(f"  Pipeline {i}:")
        print(f"    - Tools: {pipeline['pipeline']}")
        print(f"    - Data count: {len(pipeline['data'])}")
        if pipeline['data']:
            print(f"    - 第一个样本: {pipeline['data'][0]['lq']}")

    print("\n数据划分完成！")


def main():
    parser = argparse.ArgumentParser(
        description="划分训练集和验证集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python split_train_val.py --val-ratio 0.1    # 10% 验证集 (默认)
  python split_train_val.py --val-ratio 0.2    # 20% 验证集
  python split_train_val.py --seed 123         # 使用不同的随机种子
        """
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='验证集比例 (0.0-1.0)，默认 0.1 (10%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，默认 42'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='输入配置文件路径，默认 unified_config.json'
    )
    parser.add_argument(
        '--output-train',
        type=str,
        default=None,
        help='输出训练配置文件路径，默认 train_config.json'
    )
    parser.add_argument(
        '--output-val',
        type=str,
        default=None,
        help='输出验证配置文件路径，默认 val_config.json'
    )

    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("val-ratio 必须在 0.0 和 1.0 之间")

    split_train_val(
        input_config_path=args.input,
        output_val_path=args.output_val,
        output_train_path=args.output_train,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
