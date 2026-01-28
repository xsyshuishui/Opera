import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat, savemat
import argparse

# Dynamic path computation - no hardcoded paths needed
SCRIPT_DIR = Path(__file__).parent.absolute()
CHAIN_CUDA_ROOT = SCRIPT_DIR.parent
DATA_DIR = CHAIN_CUDA_ROOT / "data"


def center_crop_to_limit(img, max_pixels=1_000_000):
    """中心裁剪图像，使总像素 <= max_pixels，且宽高均为16的倍数"""
    w, h = img.size
    total = w * h
    if total <= max_pixels and w % 16 == 0 and h % 16 == 0:
        return img, (0, 0, w, h)  # 已经够小且满足条件，返回裁剪框

    # 目标边长比例
    aspect = w / h
    new_h = int((max_pixels / aspect) ** 0.5)
    new_w = int(new_h * aspect)

    # 向下取到16的倍数（避免超过max_pixels）
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16

    # 防止尺寸过小或为0
    new_w = max(16, min(new_w, w))
    new_h = max(16, min(new_h, h))

    # 中心裁剪
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    left = (left // 4) * 4
    top = (top // 4) * 4
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def process_folder(input_dir, depth_dir, output_dir, output_depth_dir, max_pixels=1_000_000):
    """
    同步处理图像和深度矩阵
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    exts = ('.png')

    for name in tqdm(os.listdir(input_dir)):
        if not name.lower().endswith(exts):
            continue

        stem = Path(name).stem
        in_path = os.path.join(input_dir, name)
        depth_path = Path(depth_dir) / f"{str(stem).split('.')[0]}.mat"

        out_path = os.path.join(output_dir, name)
        out_depth_path = Path(output_depth_dir) / f"{str(stem).split('.')[0]}.mat"


        if not depth_path.exists():
            print(f"[跳过] 缺少深度文件: {depth_path}")
            continue

        try:
            # ---- 处理图像 ----
            # print(111)
            with Image.open(in_path) as img:
                img = img.convert("RGB")
                cropped_img, (left, top, right, bottom) = center_crop_to_limit(img, max_pixels=max_pixels)
                assert left % 4==0 and top % 4 == 0 and right % 4 == 0 and bottom % 4 == 0
                assert (top - bottom) % 16 == 0
                assert (right - left) % 16 == 0
                # print(f"img {img.size} ---> {bottom-top}, {right-left}")
                cropped_img.save(out_path, lossless=True, compress_level=0)

            # ---- 同步裁剪深度矩阵 ----
            mat = loadmat(depth_path)
            d = mat["data_obj"]  # shape: (H, W)
            h, w = d.shape
            # print(f"mat: {h}, {w} --> {(bottom//4-top//4, right//4-left//4)}")
            # 注意：PIL 图像是 (W, H)，矩阵是 (H, W)
            cropped_d = d[top//4:bottom//4, left//4:right//4]

            savemat(out_depth_path, {"data_obj": cropped_d})

        except Exception as e:
            print(f"[跳过] {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images and depth maps")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input HQ image directory")
    parser.add_argument("--depth_dir", type=str, required=True,
                        help="Input depth map directory")
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR / "Train_HQ"),
                        help="Output HQ image directory")
    parser.add_argument("--output_depth_dir", type=str, default=str(DATA_DIR / "Train_Depth"),
                        help="Output depth map directory")
    parser.add_argument("--max_pixels", type=int, default=1_000_000,
                        help="Maximum pixels per image")

    args = parser.parse_args()

    process_folder(args.input_dir, args.depth_dir, args.output_dir,
                   args.output_depth_dir, max_pixels=args.max_pixels)
