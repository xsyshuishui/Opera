import os
import json
import pandas as pd
from glob import glob
import tqdm

sys_prompt = """\
You are part of a image restoration planning agent. You will be given an image as input. Your task is to visually analyze the image, and provide necessary information to another pure language model, who will decide the final restoration pipeline. Your output should contain enough information to facilitate the language model.

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

Think visually by describing the quality of the image. Give all information of the image, and analysis of tool choices. DO BOT explicitly give your final decision, since the language model will do this.

Note that the **order** of tools matters, and tools from different repos may behave differently. Select carefully."""


import argparse as _ap
_parser = _ap.ArgumentParser(description="Format LQ images into verl parquet")
_parser.add_argument("--image-dir",  required=True, help="Directory of LQ images")
_parser.add_argument("--output-dir", required=True, help="Directory to save output .parquet files")
_args, _ = _parser.parse_known_args()
image_base = _args.image_dir
os.makedirs(_args.output_dir, exist_ok=True)

data_source = "agent_pangu"
# 获取所有图片
image_paths = glob(os.path.join(image_base, "*"))
# image_paths = image_paths[0:5]
processed_data = []

for img_path in tqdm.tqdm(image_paths):
    # print(img_path)
    fname = os.path.basename(img_path)
    name, _ = os.path.splitext(fname)
    # print(name)
    # 读取图片
    with open(img_path, "rb") as f:
        image_bytes = f.read()


    processed_item = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": "<image> How to restore this image? Think first then answer."
            }
        ],
        "images": [{"bytes": image_bytes}],
        "reward_model": {"ground_truth":"notuseful"}, 
        "extra_info": {
            "image_name": fname,
        },
    }
    # print(processed_item['extra_info'])
    processed_data.append(processed_item)

# 分块存储 parquet
interval=800
for i in range((len(processed_data)+interval-1)//interval):
    l = i*interval
    r = min(l+interval, len(processed_data))
    print(f"debug: {l}, {r}")
    df = pd.DataFrame(processed_data[l:r])
    df.to_parquet(os.path.join(_args.output_dir, f"split{i}.parquet"))

print("Data successfully saved")
