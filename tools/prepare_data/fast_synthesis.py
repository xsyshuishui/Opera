import multiprocessing
from pathlib import Path
import cv2
from tqdm import tqdm
import os
import json
from add_single_degradation import *
import random
import itertools
import argparse

# Dynamic path computation - no hardcoded paths needed
SCRIPT_DIR = Path(__file__).parent.absolute()
CHAIN_CUDA_ROOT = SCRIPT_DIR.parent
DATA_DIR = CHAIN_CUDA_ROOT / "data"

# Global variable for depth directory (set by main)
_DEPTH_DIR = None


def degrade(img, degradation, idx):
    global _DEPTH_DIR
    router = {
        "lr": lr,
        "dark": darken,
        "noise": add_noise,
        "jpeg": add_jpeg_comp_artifacts,
        "haze": add_haze,
        "motionblur": add_motion_blur,
        "defocusblur": add_defocus_blur,
        "rain": add_rain,
    }
    if degradation == "haze":
        return add_haze(img, idx=idx, depth_dir=_DEPTH_DIR)
    return router[degradation](img)


def process_image(idx_args):
    idx, hq_path, comb, lq_dir = idx_args
    # print(str(hq_path))
    try:
        img = cv2.imread(str(hq_path))
        for degra in comb:
            img = degrade(img, degra, idx=hq_path.stem)
        ext = hq_path.suffix
        target_name = f"{hq_path.stem}+" + "+".join(comb) + ext
        target = lq_dir / target_name
        cv2.imwrite(target, img)
        return True
    except Exception as e:
        print(f"Error processing {hq_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def init_worker(depth_dir):
    """Initialize worker with depth directory."""
    global _DEPTH_DIR
    _DEPTH_DIR = Path(depth_dir)


def main():
    global _DEPTH_DIR

    parser = argparse.ArgumentParser(description="Fast image degradation synthesis")
    parser.add_argument("--hq_dir", type=str, default=str(DATA_DIR / "Train_HQ"),
                        help="Input HQ image directory")
    parser.add_argument("--lq_dir", type=str, default=str(DATA_DIR / "Train_LQ"),
                        help="Output LQ image directory")
    parser.add_argument("--depth_dir", type=str, default=str(DATA_DIR / "Train_Depth"),
                        help="Depth map directory (for haze degradation)")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of worker processes")
    parser.add_argument("--samples_per_image", type=int, default=5,
                        help="Number of degraded samples per HQ image")

    args = parser.parse_args()

    hq_dir = Path(args.hq_dir)
    lq_dir = Path(args.lq_dir)
    _DEPTH_DIR = Path(args.depth_dir)
    lq_dir.mkdir(exist_ok=True, parents=True)

    choose_1 = ['rain', 'haze']
    choose_2 = ['motionblur', 'defocusblur']
    choose_3 = ['noise', 'lr']

    def init():
        # 所有候选
        all_choices = [choose_1, choose_2, choose_3]
        all_items = sum(all_choices, [])

        valid_sequences = {1: [], 2: [], 3: []}  # 按长度分类存放

        for L in range(1, 4):  # 长度 1~3
            for seq in itertools.permutations(all_items, L):
                # 检查顺序约束
                idx_1 = [i for i, x in enumerate(seq) if x in choose_1]
                idx_2 = [i for i, x in enumerate(seq) if x in choose_2]
                idx_3 = [i for i, x in enumerate(seq) if x in choose_3]

                # choose_1 在 choose_2 和 choose_3 之前
                if (not idx_1 or not idx_2 or max(idx_1) < min(idx_2)) and \
                (not idx_1 or not idx_3 or max(idx_1) < min(idx_3)) and \
                (not idx_2 or not idx_3 or max(idx_2) < min(idx_3)):
                    valid_sequences[L].append(seq)
        return valid_sequences


    def random_sequence(valid_sequences):
        # 等概率选择退化数量 (1, 2, 3)
        L = random.choices([1, 2, 3], weights=[1, 3, 6], k=1)[0]
        # 在对应类别中随机选择一个合法序列
        return random.choice(valid_sequences[L])

    valid_sequences = init()
    print(valid_sequences)
    hq_paths = sorted(list(hq_dir.glob("*")))
    print(f"Found {len(hq_paths)} HQ images")

    # 设置进程池，使用 init_worker 传递 depth_dir 到每个进程
    pool = multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(str(_DEPTH_DIR),)
    )

    # 准备任务参数
    idx_args = []
    for idx, hq_path in enumerate(hq_paths):
        for _ in range(args.samples_per_image):
            idx_arg = [idx, hq_path, random_sequence(valid_sequences), lq_dir]
            idx_args.append(idx_arg)

    # idx_args = [[idx, hq_path, comb, lq_dir] for idx, hq_path in enumerate(hq_paths) for comb in combs]

    # 进度条和任务处理
    save_json = []
    with tqdm(total=len(idx_args), desc="Processing images") as pbar:
        for result in pool.imap(process_image, idx_args):
            if result:
                save_json.append(result)
            pbar.update(1)

    # 关闭进程池
    pool.close()
    pool.join()



if __name__ == "__main__":
    main()
