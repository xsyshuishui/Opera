#!/usr/bin/env python3
"""
生成单一退化图片批处理脚本

功能:
- 为5种退化类型生成单一退化图片
- 每种退化生成500张图片
- 同时生成配套的配置文件 single_config.json

输入:
- HQ图片: data/Train_HQ (前500张)
- 深度图: data/Train_Depth (haze退化需要)

输出:
- LQ图片: data/Single_LQ/
- 配置文件: data/Comb_Config/single_config.json
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# 动态路径计算
SCRIPT_DIR = Path(__file__).parent.absolute()
CHAIN_CUDA_ROOT = SCRIPT_DIR.parent
DATA_DIR = CHAIN_CUDA_ROOT / "data"

# 导入退化函数
from add_single_degradation import (
    add_noise,
    add_rain,
    add_haze,
    add_motion_blur,
    add_defocus_blur
)


# 退化配置字典
DEGRADATIONS = {
    # Noise - 3个固定sigma值
    "noise+sigma15": {
        "func": lambda img, idx: add_noise(img, "Gaussian", 15),
        "pipeline": "restormer.denoise.color-sigma15.v1"
    },
    "noise+sigma25": {
        "func": lambda img, idx: add_noise(img, "Gaussian", 25),
        "pipeline": "restormer.denoise.color-sigma25.v1"
    },
    "noise+sigma50": {
        "func": lambda img, idx: add_noise(img, "Gaussian", 50),
        "pipeline": "restormer.denoise.color-sigma50.v1"
    },

    # Rain - 1组图片，2个模型
    "rain": {
        "func": lambda img, idx: add_rain(img),  # 使用随机参数
        "pipelines": [
            "xrestormer.derain.rain.v1",
            "restormer.derain.rain.v1"
        ]
    },

    # Haze - 使用深度图
    "haze": {
        "func": lambda img, idx: add_haze(
            img,
            idx,
            depth_dir=DATA_DIR / "Train_Depth"
        ),
        "pipeline": "xrestormer.dehaze.haze.v1"
    },

    # Motion Blur - 随机severity
    "motionblur": {
        "func": lambda img, idx: add_motion_blur(img),
        "pipeline": "restormer.deblur.motion.v1"
    },

    # Defocus Blur - 随机severity
    "defocusblur": {
        "func": lambda img, idx: add_defocus_blur(img),
        "pipeline": "restormer.deblur.defocus-single.v1"
    }
}


def generate_degraded_images():
    """生成退化图片"""
    print("=" * 60)
    print("开始生成单一退化图片")
    print("=" * 60)

    # 输入输出目录（使用动态路径）
    hq_dir = DATA_DIR / "Train_HQ"
    output_dir = DATA_DIR / "Single_LQ"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择前500张图片
    image_list = sorted(hq_dir.glob("*.png"))[:500]
    print(f"\n找到 {len(image_list)} 张HQ图片")
    print(f"输出目录: {output_dir}")
    print(f"退化类型: {len(DEGRADATIONS)} 种")
    print(f"预计生成: {len(image_list) * len(DEGRADATIONS)} 张图片\n")

    # 生成退化图片
    success_count = 0
    error_count = 0

    for img_path in tqdm(image_list, desc="处理图片"):
        img_id = img_path.stem  # "000001"

        # 读取HQ图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n警告: 无法读取图片 {img_path}")
            error_count += 1
            continue

        # 应用每种退化
        for deg_name, deg_config in DEGRADATIONS.items():
            try:
                # 应用退化函数
                lq_img = deg_config["func"](img, img_id)

                # 保存退化图片
                output_path = output_dir / f"{img_id}+{deg_name}.png"
                cv2.imwrite(str(output_path), lq_img)
                success_count += 1

            except Exception as e:
                print(f"\n错误: 处理 {img_id}+{deg_name} 失败: {e}")
                error_count += 1

    print(f"\n生成完成!")
    print(f"成功: {success_count} 张")
    print(f"失败: {error_count} 张")
    print(f"保存位置: {output_dir}")

    return success_count, error_count


def generate_config_file():
    """生成配置文件"""
    print("\n" + "=" * 60)
    print("开始生成配置文件")
    print("=" * 60)

    config = {
        "description": "单一退化数据配置文件 - 5种退化类型，共9个模型pipeline，每个500张",
        "total_pipelines": 9,
        "pipelines": []
    }

    pipeline_id = 1

    # 遍历所有退化配置
    for deg_name, deg_config in DEGRADATIONS.items():
        # 确定该退化对应的pipeline列表
        if "pipeline" in deg_config:
            pipelines = [deg_config["pipeline"]]
        else:  # rain有多个pipeline
            pipelines = deg_config["pipelines"]

        # 为每个pipeline创建配置
        for pipeline_name in pipelines:
            data_list = []

            # 生成500对LQ-GT路径（使用相对路径）
            for i in range(1, 501):
                img_id = f"{i:06d}"
                data_list.append({
                    "lq": str(DATA_DIR / "Single_LQ" / f"{img_id}+{deg_name}.png"),
                    "gt": str(DATA_DIR / "Train_HQ" / f"{img_id}.png")
                })

            config["pipelines"].append({
                "id": pipeline_id,
                "pipeline": [pipeline_name],
                "data": data_list
            })

            print(f"Pipeline {pipeline_id}: {pipeline_name} - {len(data_list)} 条数据")
            pipeline_id += 1

    # 保存配置文件（使用动态路径）
    output_path = DATA_DIR / "Single_LQ" / "single_config.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n配置文件已保存: {output_path}")
    print(f"总pipelines: {len(config['pipelines'])}")
    print(f"总数据对: {sum(len(p['data']) for p in config['pipelines'])}")

    return output_path


def verify_output():
    """验证输出结果"""
    print("\n" + "=" * 60)
    print("验证输出结果")
    print("=" * 60)

    # 使用动态路径
    output_dir = DATA_DIR / "Single_LQ"
    config_path = output_dir / "single_config.json"

    # 检查目录
    if not output_dir.exists():
        print("❌ 输出目录不存在")
        return False

    # 检查图片数量
    png_files = list(output_dir.glob("*.png"))
    expected_images = 500 * len(DEGRADATIONS)
    print(f"\n图片数量: {len(png_files)} / {expected_images}")
    if len(png_files) != expected_images:
        print(f"⚠️  警告: 图片数量不符合预期")
    else:
        print("✅ 图片数量正确")

    # 检查配置文件
    if not config_path.exists():
        print("❌ 配置文件不存在")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"\n配置文件验证:")
        print(f"  Pipelines数量: {len(config['pipelines'])} / 9")
        print(f"  总数据对: {sum(len(p['data']) for p in config['pipelines'])} / 4500")

        # 检查每个pipeline
        all_valid = True
        for pipeline in config['pipelines']:
            if len(pipeline['data']) != 500:
                print(f"  ⚠️  Pipeline {pipeline['id']} 数据量异常: {len(pipeline['data'])}")
                all_valid = False

        if all_valid:
            print("✅ 配置文件格式正确")

    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False

    # 抽样检查文件命名
    print(f"\n文件命名示例:")
    for i, png_file in enumerate(sorted(png_files)[:5]):
        print(f"  {png_file.name}")

    print("\n验证完成!")
    return True


def main():
    """主函数"""
    try:
        # 步骤1: 生成退化图片
        success_count, error_count = generate_degraded_images()

        if error_count > 0:
            print(f"\n⚠️  警告: 有 {error_count} 张图片生成失败")

        # 步骤2: 生成配置文件
        config_path = generate_config_file()

        # 步骤3: 验证输出
        verify_output()

        print("\n" + "=" * 60)
        print("所有任务完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
