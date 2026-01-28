from openai import OpenAI
import time
import re
import ast
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import json
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os, json, traceback
import random
import uuid
MAX_WORKERS = 1  # 并发线程数，可根据 API 限速调整


client_agent = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model_name_agent = "Qwen2.5-VL-7B-Instruct"

sys_prompt_agent = """\
You are a professional image restoration assistant.

You will be given an image as input. Your task is to:
1. Visually analyze the image and identify what degradations it contains 
2. Design an optimal sequence of restoration tool calls to enhance the image quality.

# Possible Degradations:
- noise
- rain
- haze
- defocus_blur
- motion_blur
- low_resolution
- jpeg

# Tools from Restormer
- restormer.gaussian_denoise_15
- restormer.gaussian_denoise_25
- restormer.gaussian_denoise_50
- restormer.derain
- restormer.defocus_deblur
- restormer.motion_deblur

# Tools from X-Restormer
- xrestormer.denoise_50
- xrestormer.derain
- xrestormer.dehaze
- xrestormer.deblur
- xrestormer.super_resolution

# Tools from SWIN-IR
- swinir.super_resolution
- swinir.gaussian_denoise_15
- swinir.gaussian_denoise_25
- swinir.gaussian_denoise_50
- swinir.dejpeg

First, think visually in <think> </think> by describing the quality of the image and how you plan to restore it. Then, output a Python list of the detected degradations in <degradation> </degradation>`. Finally output a Python list of the restoration plan in the order in <answer> </answer>. 
e.g.: <think> thinking progress here </think> <degradation>['rain', 'noise']</degradation> <answer>['restormer.gaussian_denoise_25', 'xrestormer.derain']</answer>.

Note that the **order** of tools matters, and tools from different repos may behave differently. Select carefully."""

import base64
def encode_image_png(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"

def ask_agent(image_path):
    content = [
        {"type": "image_url", "image_url": {"url": encode_image_png(image_path)}},
        {
            "type": "text",
            "text":  "How to restore this image? Think first then answer. Note that the image contains low resolution! Do not forget to call super resolution model." + hint,
        },
    ]
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": sys_prompt_agent,
                }
            ],
        },
        {"role": "user", "content": content},
    ]

    # === 调用 vLLM server ===

    response = client_agent.chat.completions.create(
        model=model_name_agent,
        messages=messages,
        temperature=1,
        max_tokens=2048,
    )

    output_text = response.choices[0].message.content
    return output_text


def parse_plan(predict_str):
    available_tools = [
        'restormer.gaussian_denoise_15',
        'restormer.gaussian_denoise_25',
        'restormer.gaussian_denoise_50',
        'restormer.real_denoise',
        'restormer.derain',
        'restormer.defocus_deblur',
        'restormer.motion_deblur',
        'xrestormer.denoise_50',
        'xrestormer.derain',
        'xrestormer.dehaze',
        'xrestormer.deblur',
        'xrestormer.super_resolution',
        'swinir.super_resolution',
        'swinir.gaussian_denoise_15',
        'swinir.gaussian_denoise_25',
        'swinir.gaussian_denoise_50',
        'swinir.dejpeg'
        ]
    def extract_model_order_from_answer(answer_text: str):
        try:
            import ast
            # 解析 JSON 数组
            tools = ast.literal_eval(answer_text)
            for tool in tools:
                if tool not in available_tools:
                    print(f"Unavailable tool: {tool}")
                    return []
            return tools
        except Exception as e:
            print(f"Error decoding answer_text or invalid task: {e}")
            return []  # 如果解析失败，返回空列表


    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    model_order = extract_model_order_from_answer(answer_text)
    return model_order


import requests
def get_image(image_path, plan, output_path):
    """
curl -X POST http://localhost:6001/restore \\
        -H "Content-Type: application/json" \\
        -d '{"pipeline": ["restormer.derain", "xrestormer.dehaze"], "input_path": "/path/to/input.png", "output_path": "/path/to/output.png"}'
    """
    # 构建请求数据
    data = {
        "input_path": image_path,
        "pipeline": plan,
        "output_path": output_path
    }
    # 调用REST API进行图像复原
    response = requests.post(f"http://localhost:6001/restore", json=data, timeout=180)

    #
    return output_path


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image.")
    args = parser.parse_args()

    image_path = args.image_path
    output_path = args.output_path

    print("="*50)
    print(f"INPUT: {image_path}")
    predict_str = ask_agent(image_path)

    plan = parse_plan(predict_str)

    output_image = get_image(image_path, plan, output_path)
    print(f"OUTPUT: {output_image}")