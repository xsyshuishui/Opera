from torch.utils import data as data
from torch.utils.data import Sampler, DataLoader
from torchvision.transforms.functional import normalize
import random
import math
import numpy as np
import torch
import cv2
from pathlib import Path
from training.img_util import crop_border, imfrombytes, img2tensor, imwrite, tensor2img, padding, padding_DP, imfrombytesDP
from training.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
import json
from collections import defaultdict

# 动态路径计算
_SCRIPT_DIR = Path(__file__).parent.absolute()
_RESTORMER_ROOT = _SCRIPT_DIR.parent
_CHAIN_CUDA_ROOT = _RESTORMER_ROOT.parent
_DATA_DIR = _CHAIN_CUDA_ROOT / "data"

def mixed_image_collator(batch):
    """
    batch 是一个 list，每个元素是 dataset[i] 的输出
    同一batch内的样本有相同的pipeline和scale
    """
    pipelines = [item['pipeline'] for item in batch]
    scales = [item['scale'] for item in batch]

    lq = torch.stack([item['lq'] for item in batch], dim=0)
    gt = torch.stack([item['gt'] for item in batch], dim=0)
    lq_path = [item['lq_path'] for item in batch]
    gt_path = [item['gt_path'] for item in batch]

    return {
        'lq': lq,
        'gt': gt,
        'lq_path': lq_path,
        'gt_path': gt_path,
        'pipeline': pipelines[0],
        'scale': scales[0]  # 同一batch的scale相同
    }

class MixedBatchSampler(Sampler):
    """
    根据 pipeline 和 scale 分 batch
    确保同一batch内的样本有相同的pipeline和scale，以便正确处理SR任务
    dataset: 你的 Dataset_PairedImage
    batch_size: batch 大小
    shuffle: 是否打乱样本
    """
    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 按照 pipeline + scale 分组
        self.group_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            pipeline = '+'.join(dataset.get_pipeline(idx))
            scale = dataset.get_scale(idx)
            group_key = f"{pipeline}|scale={scale}"
            self.group_to_indices[group_key].append(idx)

        # 构建最终 batch 索引列表
        self.batches = []
        for group_key, indices in self.group_to_indices.items():
            if shuffle:
                random.shuffle(indices)
            # 切成 batch
            num = len(indices) // batch_size * batch_size
            for i in range(0, num, batch_size):
                self.batches.append(indices[i:i+batch_size])

        if shuffle:
            random.shuffle(self.batches)  # 打乱 batch 顺序

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class DistributedMixedBatchSampler(Sampler):
    """
    分布式混合批采样器 (DDP 安全版本)

    在分布式训练中，确保:
    1. 同一 batch 内的样本有相同的 pipeline 和 scale
    2. **关键**: 所有 rank 在同一迭代处理相同 pipeline 的数据
       (避免 DDP 死锁 - 不同 rank 调用不同模型会导致梯度同步死锁)
    3. 不同 rank 处理同一 pipeline group 内的不同样本
    4. 支持 shuffle 和 epoch 种子控制

    DDP 死锁问题说明:
        当不同 rank 调用不同的模型时，DDP 在 backward 阶段会等待所有 rank
        同步同一模型的梯度。如果 rank 0 调用了模型 A 而 rank 1 没有调用，
        rank 0 会永远等待 rank 1 的梯度，导致死锁。

    解决方案:
        确保所有 rank 在同一迭代调用相同的模型序列。方法是让所有 rank
        处理相同 pipeline 的 batch，只是处理的样本不同。

    Args:
        dataset: Dataset_PairedImage 数据集
        batch_size: 每个 GPU 的 batch size
        num_replicas: 总进程数 (world_size)，None 表示自动获取
        rank: 当前进程排名，None 表示自动获取
        shuffle: 是否打乱
        seed: 随机种子
        drop_last: 是否丢弃不完整的 batch
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 4,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        from training.dist_util import get_dist_info

        if num_replicas is None or rank is None:
            _rank, _world_size = get_dist_info()
            num_replicas = num_replicas if num_replicas is not None else _world_size
            rank = rank if rank is not None else _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # 按照 pipeline + scale 分组
        self.group_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            pipeline = '+'.join(dataset.get_pipeline(idx))
            scale = dataset.get_scale(idx)
            group_key = f"{pipeline}|scale={scale}"
            self.group_to_indices[group_key].append(idx)

        self._rebuild_batches()

    def _rebuild_batches(self):
        """
        重建 batch 列表 (DDP 安全版本)

        关键改变: 所有 rank 处理相同的 batch 顺序（相同 pipeline），
        但每个 rank 处理该 batch 内的不同样本子集。

        这确保了所有 rank 在每次迭代调用相同的模型序列，避免 DDP 死锁。
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []

        for group_key, indices in self.group_to_indices.items():
            indices = indices.copy()

            if self.shuffle:
                # 使用确定性 shuffle (所有 rank 使用相同种子，得到相同结果)
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            # === DDP 安全的 batch 创建 ===
            # 每个 "super batch" 包含 num_replicas * batch_size 个样本
            # 这样每个 rank 可以从中取 batch_size 个不重叠的样本
            samples_per_super_batch = self.num_replicas * self.batch_size
            num_samples = len(indices)

            # 计算需要多少个 super batch
            num_super_batches = num_samples // samples_per_super_batch

            if self.drop_last:
                # 丢弃不完整的 super batch
                indices = indices[:num_super_batches * samples_per_super_batch]
            else:
                # 填充到完整的 super batch
                total_needed = math.ceil(num_samples / samples_per_super_batch) * samples_per_super_batch
                padding_size = total_needed - num_samples
                if padding_size > 0:
                    # 循环复用样本进行填充
                    indices.extend(indices[:padding_size])

            # 为每个 super batch 创建当前 rank 的 batch
            for i in range(0, len(indices), samples_per_super_batch):
                super_batch = indices[i:i + samples_per_super_batch]
                if len(super_batch) == samples_per_super_batch:
                    # 当前 rank 取其中的一部分
                    # rank 0 取 [0:batch_size], rank 1 取 [batch_size:2*batch_size], ...
                    start = self.rank * self.batch_size
                    end = start + self.batch_size
                    batch = super_batch[start:end]
                    all_batches.append(batch)

        # Shuffle batch 顺序 (所有 rank 使用相同种子，得到相同顺序)
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        self.batches = all_batches

    def __iter__(self):
        self._rebuild_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch: int):
        """
        设置 epoch (用于确保每个 epoch 的 shuffle 不同)

        在每个 epoch 开始时调用此方法
        """
        self.epoch = epoch


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = None,
):
    """
    创建 DataLoader 的工厂函数 (自动选择普通或分布式采样器)

    Args:
        dataset: Dataset_PairedImage 数据集
        batch_size: 每个 GPU 的 batch size
        num_workers: DataLoader workers
        shuffle: 是否打乱
        pin_memory: 是否使用 pinned memory
        drop_last: 是否丢弃不完整 batch
        distributed: 是否使用分布式采样器，None 表示自动检测

    Returns:
        tuple: (dataloader, sampler)
    """
    from training.dist_util import is_dist_initialized

    if distributed is None:
        distributed = is_dist_initialized()

    if distributed:
        sampler = DistributedMixedBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        sampler = MixedBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=mixed_image_collator,
        pin_memory=pin_memory,
    )

    return dataloader, sampler


class Dataset_PairedImage(data.Dataset):
    opt = {
        'in_ch': 3,    ## RGB image
        'geometric_augs': True,
        'filename_tmpl': '{}',

        # data loader
        'use_shuffle': True,
        'num_worker_per_gpu': 8,
        'batch_size_per_gpu': 8,

        ### -------------Progressive training--------------------------
        'mini_batch_sizes': [8,5,4,2,1,1],             # Batch size per gpu
        'iters': [92000,64000,48000,36000,36000,24000],
        'gt_size': 128,   # Max patch size for progressive training
        'gt_sizes': [128,160,192,256,320,384],  # Patch sizes for progressive training.
        ### ------------------------------------------------------------

        'dataset_enlarge_ratio': 1,
    }

    def __init__(self, train_data):
        import os
        self.data = []
        self.pipeline = []
        # 使用动态路径
        self.base_dir = str(_DATA_DIR / "synthesized") + "/"

        for group in train_data:
            self.pipeline.append(group['pipeline'])
            pipeline_idx = len(self.pipeline) - 1
            for item in group['data']:
                # 添加基础目录到路径（仅对相对路径）
                if not os.path.isabs(item['lq']):
                    item['lq'] = self.base_dir + item['lq']
                if not os.path.isabs(item['gt']):
                    item['gt'] = self.base_dir + item['gt']
                item['pipeline_idx'] = pipeline_idx
                self.data.append(item)

    def get_pipeline(self, index):
        """Get pipeline without loading images (for efficient batch sampler initialization)"""
        index = index % len(self.data)
        return self.pipeline[self.data[index]['pipeline_idx']]

    def get_scale(self, index):
        """Get scale factor based on LQ filename (without loading images).

        Files containing '+lr+' or ending with '+lr.png' are 4x downsampled.
        Returns: 4 for SR samples, 1 for others.
        """
        index = index % len(self.data)
        lq_path = self.data[index]['lq']
        if '+lr+' in lq_path or lq_path.endswith('+lr.png'):
            return 4
        return 1

    def __getitem__(self, index):
        index = index % len(self.data)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.data[index]['gt']
        with open(gt_path, 'rb') as f:
            img_bytes = f.read()
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.data[index]['lq']
        with open(lq_path, 'rb') as f:
            img_bytes = f.read()
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        gt_size = self.opt['gt_size']
        # 动态计算scale（根据实际图像尺寸）
        h_lq, w_lq = img_lq.shape[:2]
        h_gt, w_gt = img_gt.shape[:2]
        scale_h = h_gt // h_lq if h_lq > 0 else 1
        scale_w = w_gt // w_lq if w_lq > 0 else 1
        # 验证scale一致性，取较小值作为scale
        scale = min(scale_h, scale_w)
        if scale < 1:
            scale = 1
        # 如果scale不是整数倍，调整为最接近的整数
        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            # 尝试常见的scale值: 1, 2, 4
            for s in [1, 2, 4]:
                if h_gt == h_lq * s and w_gt == w_lq * s:
                    scale = s
                    break
            else:
                # 如果都不匹配，默认scale=1，后续会报错
                scale = 1
        lq_size = gt_size // scale  # LQ patch size = GT patch size / scale
        # padding (注意: padding函数期望参数顺序为 img_lq, img_gt)
        img_lq, img_gt = padding(img_lq, img_gt, gt_size, scale)

        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, lq_size, scale,
                                                gt_path)

        # flip, rotation augmentations
        img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'pipeline': self.pipeline[self.data[index]['pipeline_idx']],
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'scale': scale
        }

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # 使用动态路径
    test_config_path = _DATA_DIR / "Comb_Config" / "train_config.json"
    with open(test_config_path) as f:
        train_data = json.load(f)
    dataset = Dataset_PairedImage(train_data['pipelines'])
    print(len(dataset))
    print(dataset[0])