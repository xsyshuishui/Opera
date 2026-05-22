#!/usr/bin/env python3
"""
配置文件生成器
从 Comb_Plan 的 plan.json 生成训练所需的统一配置文件
删除了 generation_info、degradation_types、data_count 等字段
"""

import json
import re
import os
from collections import defaultdict
from pathlib import Path

# 动态路径计算
SCRIPT_DIR = Path(__file__).parent.absolute()
CHAIN_CUDA_ROOT = SCRIPT_DIR.parent
DATA_DIR = CHAIN_CUDA_ROOT / "data"


def extract_answer_from_output(output_text):
    """从输出文本中提取 <answer></answer> 标签中的工具序列"""
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        try:
            # 使用 eval 解析 Python 列表字符串
            tools = eval(answer_match.group(1).strip())
            if isinstance(tools, list):
                return tools
        except Exception as e:
            print(f"警告：解析 answer 标签失败: {e}")
            return None
    return None


def extract_image_id(rel_path):
    """从相对路径中提取图片 ID（如 folder/000001+rain.png → 000001）

    支持递归目录结构，从文件名部分提取数字 ID
    """
    # 获取文件名部分（去掉目录路径）
    filename = os.path.basename(rel_path)
    # 匹配数字 ID（文件名开头的数字部分）
    match = re.match(r'^(\d+)', filename)
    if match:
        return match.group(1)
    return None


def get_hq_path_from_lq(rel_path, lq_dir, hq_dir, flat_hq=False):
    """根据 LQ 的相对路径，推导出 HQ 的路径

    Args:
        rel_path: LQ 图片的相对路径
        lq_dir: LQ 目录根路径
        hq_dir: HQ 目录根路径
        flat_hq: 如果为 True，HQ 目录是扁平的（只按图片 ID 查找，忽略子目录结构）
                 适用于多对一映射场景：
                 LQ: Group A/dark+noise/001.png → HQ: 001.png

    Returns:
        (hq_path, image_id) 或 (None, None) 如果无法提取 ID
    """
    filename = os.path.basename(rel_path)
    image_id = extract_image_id(filename)
    if image_id is None:
        return None, None

    if flat_hq:
        # 扁平 HQ 目录：直接用 image_id.png（多对一映射）
        hq_path = os.path.join(hq_dir, f"{image_id}.png")
    else:
        # 保持目录结构（原有逻辑，一对一映射）
        dir_part = os.path.dirname(rel_path)
        if dir_part:
            hq_path = os.path.join(hq_dir, dir_part, f"{image_id}.png")
        else:
            hq_path = os.path.join(hq_dir, f"{image_id}.png")

    return hq_path, image_id


def map_tool_name(tool_name):
    """将简短的工具名称映射到完整的模型名称"""
    # 工具名称映射表：简短名称 → 完整模型名称
    tool_mapping = {
        'restormer.gaussian_denoise_15': 'restormer.denoise.color-sigma15.v1',
        'restormer.gaussian_denoise_25': 'restormer.denoise.color-sigma25.v1',
        'restormer.gaussian_denoise_50': 'restormer.denoise.color-sigma50.v1',
        'restormer.real_denoise': 'restormer.denoise.real.v1',
        'restormer.derain': 'restormer.derain.rain.v1',
        'restormer.motion_deblur': 'restormer.deblur.motion.v1',
        'restormer.defocus_deblur': 'restormer.deblur.defocus-single.v1',
        'xrestormer.denoise_50': 'xrestormer.denoise.gaussian.v1',
        'xrestormer.derain': 'xrestormer.derain.rain.v1',
        'xrestormer.dehaze': 'xrestormer.dehaze.haze.v1',
        'xrestormer.deblur': 'xrestormer.deblur.motion.v1',
        'xrestormer.super_resolution': 'xrestormer.sr.real.v1',
    }

    # 返回映射后的名称，如果没有映射则返回原名称
    return tool_mapping.get(tool_name, tool_name)


def generate_config():
    """主函数：生成配置文件"""

    # 输入输出路径（使用动态路径）
    plan_json_path = str(DATA_DIR / "Comb_Plan" / "Agent3_Inference_plan.json")
    lq_dir = str(DATA_DIR / "Train_LQ")
    hq_dir = str(DATA_DIR / "Train_HQ")
    output_path = str(DATA_DIR / "Comb_Config" / "Agent3_Inference_config.json")

    # HQ 目录是否为扁平结构（多对一映射：多个 LQ 对应同一 HQ）
    flat_hq = True

    print(f"读取 plan.json: {plan_json_path}")

    # 读取 plan.json
    with open(plan_json_path, 'r', encoding='utf-8') as f:
        plan_data = json.load(f)

    print(f"总共读取了 {len(plan_data)} 个条目")

    # 按 pipeline 分组数据
    # key: tuple(pipeline), value: list of data entries
    pipeline_groups = defaultdict(list)

    skipped_count = 0
    processed_count = 0

    for filename, entry in plan_data.items():
        # 提取工具调用序列
        output_text = entry.get('output', '')
        tools = extract_answer_from_output(output_text)

        if tools is None or len(tools) == 0:
            print(f"跳过（无法提取工具序列）: {filename}")
            skipped_count += 1
            continue

        # 跳过包含空字符串的工具列表
        if any(not tool or tool.strip() == '' for tool in tools):
            print(f"跳过（工具列表包含空字符串）: {filename} - tools: {tools}")
            skipped_count += 1
            continue

        # filename 是相对路径（支持递归目录结构，如 "folder/000001+rain.png"）
        # 也可以通过 entry.get("image") 获取完整路径，但这里用相对路径更通用
        rel_path = filename

        # 构建 LQ 路径
        lq_path = os.path.join(lq_dir, rel_path)

        # 验证 LQ 文件是否存在
        if not os.path.exists(lq_path):
            print(f"警告：LQ 文件不存在: {lq_path}")
            skipped_count += 1
            continue

        # 根据 LQ 路径推导 HQ 路径（支持递归目录结构和扁平结构）
        hq_path, image_id = get_hq_path_from_lq(rel_path, lq_dir, hq_dir, flat_hq=flat_hq)

        if image_id is None:
            print(f"跳过（无法提取 ID）: {rel_path}")
            skipped_count += 1
            continue

        # 验证 HQ 文件是否存在（要求 lq_dir 和 hq_dir 目录结构完全对应）
        if not os.path.exists(hq_path):
            print(f"警告：HQ 文件不存在: {hq_path}")
            skipped_count += 1
            continue

        # 将数据添加到对应的 pipeline 分组
        pipeline_key = tuple(tools)
        pipeline_groups[pipeline_key].append({
            "lq": lq_path,
            "gt": hq_path
        })

        processed_count += 1

    print(f"\n处理完成:")
    print(f"  - 成功处理: {processed_count} 个图片")
    print(f"  - 跳过: {skipped_count} 个图片")
    print(f"  - 生成的 pipeline 组数: {len(pipeline_groups)}")

    # 构建最终的配置文件
    pipelines = []
    for idx, (pipeline_key, data_list) in enumerate(sorted(pipeline_groups.items()), start=1):
        # 将工具名称映射为完整的模型名称
        mapped_pipeline = [map_tool_name(tool) for tool in pipeline_key]

        pipeline_config = {
            "id": idx,
            "pipeline": mapped_pipeline,
            "data": data_list
        }
        pipelines.append(pipeline_config)

    config = {
        "description": "基于 Comb 数据生成的配置文件（已删除 generation_info 和 data_count 字段）",
        "total_pipelines": len(pipelines),
        "pipelines": pipelines
    }

    # 保存配置文件
    print(f"\n保存配置文件到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n配置文件生成成功!")
    print(f"  - 文件路径: {output_path}")
    print(f"  - Total pipelines: {config['total_pipelines']}")
    print(f"  - Total data entries: {sum(len(p['data']) for p in pipelines)}")

    # 打印前3个 pipeline 示例
    print(f"\n前 3 个 pipeline 示例:")
    for i, pipeline in enumerate(pipelines[:3], 1):
        print(f"  Pipeline {i}:")
        print(f"    - Tools: {pipeline['pipeline']}")
        print(f"    - Data count: {len(pipeline['data'])}")
        if pipeline['data']:
            print(f"    - 第一个数据: LQ={pipeline['data'][0]['lq']}")


if __name__ == "__main__":
    generate_config()
