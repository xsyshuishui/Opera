import os
import json
import pandas as pd
from glob import glob
import tqdm
from PIL import Image
from io import BytesIO

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--image-dir", required=True, help="Directory of LQ images to check")
_args = _parser.parse_args()
image_base = _args.image_dir

image_paths = glob(os.path.join(image_base, "*"))

error_images = []
for img_path in tqdm.tqdm(image_paths):
    # print(img_path)
    try:
        fname = os.path.basename(img_path)
        name, _ = os.path.splitext(fname)
        # print(name)
        # 读取图片
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        print(f"Error reading image: {img_path}")
        print(e)
        error_images.append(img_path)
        

print(error_images)