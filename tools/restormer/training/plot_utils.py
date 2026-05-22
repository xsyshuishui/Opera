#!/usr/bin/env python3
"""
训练曲线绘图工具

功能：为训练过程生成 4 个清晰的曲线图
1. train_loss_components.png - 训练 Loss 分量
2. val_loss_components.png - 验证 Loss 分量
3. total_loss_comparison.png - 训练 vs 验证总 Loss
4. val_metrics.png - 验证指标 (2x2 子图)
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 统一配色方案
COLORS = {
    'pixel': '#FF6B35',       # 橙色
    'perceptual': '#9B59B6',  # 紫色
    'lpips': '#00B4D8',       # 青色
    'train': '#E74C3C',       # 红色
    'val': '#3498DB',         # 蓝色
    'psnr': '#27AE60',        # 绿色
    'ssim': '#2980B9',        # 深蓝
    'lpips_metric': '#8E44AD', # 深紫
    'clipiqa': '#16A085',     # 青绿
    'musiq': '#D35400',       # 深橙
}


def plot_loss_components_combined(metrics_history, save_path, prefix='train'):
    """
    绘制 3 种 Loss 分量的曲线图

    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
        prefix: 'train' 或 'val'，决定使用哪组 loss 数据
    """
    if prefix == 'train':
        keys = ['pixel_loss', 'perceptual_loss', 'lpips_loss', 'musiq_loss', 'clipiqa_loss']
        title = 'Training Loss Components'
    else:
        keys = ['val_pixel_loss', 'val_perceptual_loss', 'val_lpips_loss', 'val_musiq_loss', 'val_clipiqa_loss']
        title = 'Validation Loss Components'

    # 检查数据是否存在
    has_data = any(key in metrics_history and metrics_history[key] for key in keys)
    if not has_data:
        print(f"⚠ 跳过 {title}: 没有数据")
        return

    plt.figure(figsize=(10, 6))

    labels = {
        'pixel_loss': 'Pixel (L1)',
        'perceptual_loss': 'Perceptual (VGG)',
        'lpips_loss': 'LPIPS',
        'musiq_loss': 'MUSIQ',
        'clipiqa_loss': 'CLIPIQA',
        'val_pixel_loss': 'Pixel (L1)',
        'val_perceptual_loss': 'Perceptual (VGG)',
        'val_lpips_loss': 'LPIPS',
        'val_musiq_loss': 'MUSIQ',
        'val_clipiqa_loss': 'CLIPIQA',
    }
    colors = {
        'pixel_loss': COLORS['pixel'],
        'perceptual_loss': COLORS['perceptual'],
        'lpips_loss': COLORS['lpips'],
        'musiq_loss': COLORS['musiq'],
        'clipiqa_loss': COLORS['clipiqa'],
        'val_pixel_loss': COLORS['pixel'],
        'val_perceptual_loss': COLORS['perceptual'],
        'val_lpips_loss': COLORS['lpips'],
        'val_musiq_loss': COLORS['musiq'],
        'val_clipiqa_loss': COLORS['clipiqa'],
    }

    for key in keys:
        if key in metrics_history and metrics_history[key]:
            raw_values = metrics_history[key]
            # 过滤掉 None 值，保持 epoch 对应关系
            valid_data = [(i + 1, v) for i, v in enumerate(raw_values) if v is not None]
            if not valid_data:
                continue
            epochs, values = zip(*valid_data)
            epochs, values = list(epochs), list(values)

            plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2,
                     markersize=4, color=colors[key], alpha=0.8, label=labels[key])

            # 标记最小值
            best_idx = np.argmin(values)
            plt.plot(epochs[best_idx], values[best_idx], '*', markersize=12,
                     color=colors[key], markeredgecolor='black', markeredgewidth=0.5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 已保存: {save_path}")


def plot_total_loss_comparison(metrics_history, save_path):
    """
    绘制训练总 Loss 和验证总 Loss 的对比图

    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
    """
    train_loss_raw = metrics_history.get('train_loss', [])
    val_loss_raw = metrics_history.get('val_loss', [])

    if not train_loss_raw and not val_loss_raw:
        print("⚠ 跳过总 Loss 对比图: 没有数据")
        return

    plt.figure(figsize=(10, 6))

    # 过滤 None 值
    if train_loss_raw:
        valid_data = [(i + 1, v) for i, v in enumerate(train_loss_raw) if v is not None]
        if valid_data:
            epochs, train_loss = zip(*valid_data)
            plt.plot(epochs, train_loss, marker='o', linestyle='-', linewidth=2,
                     markersize=4, color=COLORS['train'], alpha=0.8, label='Train Loss')

    if val_loss_raw:
        valid_data = [(i + 1, v) for i, v in enumerate(val_loss_raw) if v is not None]
        if valid_data:
            epochs, val_loss = zip(*valid_data)
            epochs, val_loss = list(epochs), list(val_loss)
            plt.plot(epochs, val_loss, marker='s', linestyle='-', linewidth=2,
                     markersize=4, color=COLORS['val'], alpha=0.8, label='Validation Loss')

            # 标记验证 loss 最小值
            best_idx = np.argmin(val_loss)
            plt.plot(epochs[best_idx], val_loss[best_idx], '*', markersize=15,
                     color=COLORS['val'], markeredgecolor='black', markeredgewidth=1,
                     label=f'Best Val: {val_loss[best_idx]:.4f} @ Epoch {epochs[best_idx]}')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 已保存: {save_path}")


def plot_val_metrics_subplots(metrics_history, save_path):
    """
    绘制验证指标的 2x2 子图

    布局:
    - 左上: PSNR (higher is better)
    - 右上: SSIM (higher is better)
    - 左下: LPIPS (lower is better)
    - 右下: CLIPIQA + MUSIQ (双 Y 轴)

    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
    """
    # 检查是否有验证指标数据
    metric_keys = ['val_psnr', 'val_ssim', 'val_lpips', 'val_clipiqa', 'val_musiq']
    has_data = any(key in metrics_history and metrics_history[key] for key in metric_keys)
    if not has_data:
        print("⚠ 跳过验证指标图: 没有数据")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Validation Metrics', fontsize=16, fontweight='bold', y=0.98)

    def plot_single_subplot(ax, key, ylabel, title, color, higher_is_better=True):
        """绘制单个子图"""
        if key not in metrics_history or not metrics_history[key]:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            return

        raw_values = metrics_history[key]
        # 过滤 None 值
        valid_data = [(i + 1, v) for i, v in enumerate(raw_values) if v is not None]
        if not valid_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            return

        epochs, values = zip(*valid_data)
        epochs, values = list(epochs), list(values)

        ax.plot(epochs, values, marker='o', linestyle='-', linewidth=2,
                markersize=4, color=color, alpha=0.8)

        # 标记最佳值
        if higher_is_better:
            best_idx = np.argmax(values)
            best_label = 'Best (max)'
        else:
            best_idx = np.argmin(values)
            best_label = 'Best (min)'

        ax.plot(epochs[best_idx], values[best_idx], '*', markersize=12,
                color=color, markeredgecolor='black', markeredgewidth=0.5)
        ax.annotate(f'{best_label}: {values[best_idx]:.4f}\n@ Epoch {epochs[best_idx]}',
                    xy=(epochs[best_idx], values[best_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=9, color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 左上: PSNR
    plot_single_subplot(axes[0, 0], 'val_psnr', 'PSNR (dB)', 'PSNR',
                        COLORS['psnr'], higher_is_better=True)

    # 右上: SSIM
    plot_single_subplot(axes[0, 1], 'val_ssim', 'SSIM', 'SSIM',
                        COLORS['ssim'], higher_is_better=True)

    # 左下: LPIPS
    plot_single_subplot(axes[1, 0], 'val_lpips', 'LPIPS', 'LPIPS (lower is better)',
                        COLORS['lpips_metric'], higher_is_better=False)

    # 右下: CLIPIQA + MUSIQ (双 Y 轴)
    ax4 = axes[1, 1]

    # 过滤 None 值
    clipiqa_data = None
    musiq_data = None
    if 'val_clipiqa' in metrics_history and metrics_history['val_clipiqa']:
        valid = [(i + 1, v) for i, v in enumerate(metrics_history['val_clipiqa']) if v is not None]
        if valid:
            clipiqa_data = list(zip(*valid))
    if 'val_musiq' in metrics_history and metrics_history['val_musiq']:
        valid = [(i + 1, v) for i, v in enumerate(metrics_history['val_musiq']) if v is not None]
        if valid:
            musiq_data = list(zip(*valid))

    if not clipiqa_data and not musiq_data:
        ax4.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='gray')
        ax4.set_title('CLIPIQA & MUSIQ', fontsize=12, fontweight='bold')
    else:
        lines = []
        labels = []

        if clipiqa_data:
            epochs, values = clipiqa_data
            epochs, values = list(epochs), list(values)
            line1, = ax4.plot(epochs, values, marker='o', linestyle='-', linewidth=2,
                              markersize=6, color=COLORS['clipiqa'], alpha=0.8, label='CLIPIQA')
            lines.append(line1)
            labels.append('CLIPIQA')
            ax4.set_ylabel('CLIPIQA Score', fontsize=10, color=COLORS['clipiqa'])
            ax4.tick_params(axis='y', labelcolor=COLORS['clipiqa'])

            # 标记最佳值 (五角星, zorder=10 确保在最上层)
            best_idx = np.argmax(values)
            ax4.plot(epochs[best_idx], values[best_idx], marker='*', markersize=15,
                     color=COLORS['clipiqa'], markeredgecolor='black', markeredgewidth=0.8,
                     linestyle='None', zorder=10)

        if musiq_data:
            if clipiqa_data:
                ax4_twin = ax4.twinx()
            else:
                ax4_twin = ax4

            epochs, values = musiq_data
            epochs, values = list(epochs), list(values)
            line2, = ax4_twin.plot(epochs, values, marker='s', linestyle='--', linewidth=2,
                                   markersize=6, color=COLORS['musiq'], alpha=0.8, label='MUSIQ')
            lines.append(line2)
            labels.append('MUSIQ')
            ax4_twin.set_ylabel('MUSIQ Score', fontsize=10, color=COLORS['musiq'])
            ax4_twin.tick_params(axis='y', labelcolor=COLORS['musiq'])

            # 标记最佳值 (菱形, 避免与 CLIPIQA 的星号重叠)
            best_idx = np.argmax(values)
            ax4_twin.plot(epochs[best_idx], values[best_idx], marker='D', markersize=10,
                          color=COLORS['musiq'], markeredgecolor='black', markeredgewidth=0.8,
                          linestyle='None', zorder=9)

        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_title('CLIPIQA & MUSIQ', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax4.legend(lines, labels, loc='best', fontsize=9)

    plt.tight_layout()
    fig.subplots_adjust(top=0.93)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 已保存: {save_path}")


def plot_all_curves(metrics_history, save_dir):
    """
    主入口函数：生成 4 个 PNG 文件

    Args:
        metrics_history: 指标历史字典
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 训练 Loss 分量图
    plot_loss_components_combined(
        metrics_history,
        save_dir / 'train_loss_components.png',
        prefix='train'
    )

    # 2. 验证 Loss 分量图
    plot_loss_components_combined(
        metrics_history,
        save_dir / 'val_loss_components.png',
        prefix='val'
    )

    # 3. 总 Loss 对比图
    plot_total_loss_comparison(
        metrics_history,
        save_dir / 'total_loss_comparison.png'
    )

    # 4. 验证指标 2x2 子图
    plot_val_metrics_subplots(
        metrics_history,
        save_dir / 'val_metrics.png'
    )


# ============================================================
# 保留的工具函数 (用于保存指标数据)
# ============================================================

def save_metrics_to_csv(metrics_history, save_path):
    """
    将指标历史保存为 CSV 文件

    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
    """
    import csv

    if not metrics_history:
        print("⚠ 没有指标数据可保存")
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有指标名称
    metric_names = list(metrics_history.keys())

    # 确定最大长度（以防不同指标长度不一致）
    max_length = max(len(values) for values in metrics_history.values())

    # 写入 CSV
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(['epoch'] + metric_names)

        # 写入数据
        for i in range(max_length):
            row = [i + 1]  # epoch 从 1 开始
            for metric_name in metric_names:
                values = metrics_history[metric_name]
                if i < len(values):
                    value = values[i]
                    # 格式化数值
                    if value is not None:
                        row.append(f"{value:.6f}")
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            writer.writerow(row)

    print(f"✓ 已保存指标 CSV: {save_path}")


def save_metrics_to_json(metrics_history, save_path):
    """
    将指标历史保存为 JSON 文件

    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
    """
    import json

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=2)

    print(f"✓ 已保存指标 JSON: {save_path}")


# ============================================================
# 保留旧函数 (兼容性，可单独调用)
# ============================================================

def plot_single_metric(
    epochs,
    values,
    metric_name,
    save_path,
    ylabel=None,
    title=None,
    color='blue',
    higher_is_better=True
):
    """
    绘制单个指标的曲线图 (保留供单独使用)

    Args:
        epochs: epoch 列表
        values: 指标值列表
        metric_name: 指标名称
        save_path: 保存路径
        ylabel: Y 轴标签（默认使用 metric_name）
        title: 图表标题（默认使用 metric_name）
        color: 曲线颜色
        higher_is_better: 指标值越高越好（用于显示最优值标记）
    """
    if not values:
        print(f"⚠ 跳过绘图 {metric_name}: 没有数据")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2,
             markersize=4, color=color, alpha=0.8)

    # 标记最佳值
    if higher_is_better:
        best_idx = np.argmax(values)
        best_label = 'Best (max)'
    else:
        best_idx = np.argmin(values)
        best_label = 'Best (min)'

    plt.plot(epochs[best_idx], values[best_idx], 'r*', markersize=15,
             label=f'{best_label}: {values[best_idx]:.4f} @ Epoch {epochs[best_idx]}')

    # 设置标签和标题
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel or metric_name, fontsize=12)
    plt.title(title or f'{metric_name} vs Epoch', fontsize=14, fontweight='bold')

    # 网格和图例
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)

    # 设置 X 轴为整数
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 已保存图表: {save_path}")


# 保留旧函数名以保持向后兼容
def plot_training_curves(metrics_history, save_dir, current_epoch=None):
    """
    旧函数，现在调用 plot_all_curves
    """
    plot_all_curves(metrics_history, save_dir)


def plot_loss_comparison(train_loss, val_loss, save_path):
    """
    旧函数，保留以兼容旧代码调用
    """
    metrics_history = {'train_loss': train_loss, 'val_loss': val_loss}
    plot_total_loss_comparison(metrics_history, save_path)


def plot_loss_components(metrics_history, save_path):
    """
    旧函数，保留以兼容旧代码调用
    """
    plot_loss_components_combined(metrics_history, save_path, prefix='train')
