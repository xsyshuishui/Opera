from openai import OpenAI
import json
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import base64

# 动态路径计算
SCRIPT_DIR = Path(__file__).parent.absolute()
CHAIN_CUDA_ROOT = SCRIPT_DIR.parent
DATA_DIR = CHAIN_CUDA_ROOT / "data"

MAX_WORKERS = 4  # 并发线程数，可根据 API 限速调整


client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model_name = "Qwen2.5-VL-7B-Instruct"

LQ_BASE = str(DATA_DIR / "Train_LQ")
OUTPUT_JSON = str(DATA_DIR / "Comb_Plan" / "plan_1_3_6.json")   # 输出文件

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}


sys_prompt = """\
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

# Tools from Restormer
- restormer.gaussian_denoise_15
- restormer.gaussian_denoise_25
- restormer.gaussian_denoise_50
- restormer.real_denoise
- restormer.derain
- restormer.defocus_deblur
- restormer.motion_deblur

# Tools from X-Restormer
- xrestormer.denoise_50
- xrestormer.derain
- xrestormer.dehaze
- xrestormer.deblur
- xrestormer.super_resolution

First, think visually in <think> </think> by describing the quality of the image and how you plan to restore it. Then, output a Python list of the detected degradations in <degradation> </degradation>`. Finally output a Python list of the restoration plan in the order in <answer> </answer>. 
e.g.: <think> thinking progress here </think> <degradation>['rain', 'noise']</degradation> <answer>['restormer.gaussian_denoise_25', 'xrestormer.derain']</answer>.

Note that the **order** of tools matters, and tools from different repos may behave differently. Select carefully."""


def encode_image_png(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"

def eval_model_row(image_path):
    content = [
        {"type": "image_url", "image_url": {"url": encode_image_png(image_path)}},
        {
            "type": "text",
            "text":  "How to restore this image? Think first then answer.",
        },
    ]

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": sys_prompt,
                }
            ],
        },
        {"role": "user", "content": content},
    ]

    # === 调用 vLLM server ===
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.01,
        top_p=0.001,
        max_tokens=2048,
    )

    output_text = response.choices[0].message.content
    # print("Raw output:", output_text)

    return output_text
    

# 获取所有待处理图片（扁平目录结构）
print(f"Processing images from: {LQ_BASE}")

# 找到所有图片文件
image_files = [
    fname for fname in sorted(os.listdir(LQ_BASE))
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"Found {len(image_files)} image files")

# 过滤掉已有结果的
pending_files = []
for fname in image_files:
    if fname not in results:
        image_path = os.path.join(LQ_BASE, fname)
        pending_files.append((fname, image_path))

print(f"Pending to process: {len(pending_files)} images")

if not pending_files:
    print("All images already processed!")
else:
    # === 并发执行 ===
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(eval_model_row, image_path): (fname, image_path)
            for fname, image_path in pending_files
        }

        for future in tqdm(as_completed(future_to_file), total=len(future_to_file),
                        desc="Processing images"):
            fname, image_path = future_to_file[future]
            try:
                text_output = future.result()
                results[fname] = {
                    "image": image_path,
                    "output": text_output
                }
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                traceback.print_exc()
                results[fname] = {"error": str(e)}

            # 实时保存（防止中断丢失）
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nProcessing complete! Results saved to: {OUTPUT_JSON}")
print(f"Total processed: {len(results)} images")
