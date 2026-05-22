# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py
"""
分布式训练工具模块

支持:
- NPU (Ascend) 多卡训练 (hccl 后端)
- CUDA (NVIDIA) 多卡训练 (nccl 后端)
- CPU 分布式训练 (gloo 后端)

启动方式:
- torchrun --nproc_per_node=N training/train_combined.py --distributed
"""

import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging

logger = logging.getLogger(__name__)


def get_device_type() -> str:
    """
    自动检测可用的设备类型

    Returns:
        str: 'npu', 'cuda', or 'cpu'
    """
    try:
        import torch_npu
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass

    if torch.cuda.is_available():
        return 'cuda'

    return 'cpu'


def get_backend_for_device(device_type: str = None) -> str:
    """
    根据设备类型返回合适的分布式后端

    Args:
        device_type: 'npu', 'cuda', or 'cpu'。None 表示自动检测

    Returns:
        str: 分布式后端名称 ('hccl', 'nccl', or 'gloo')
    """
    if device_type is None:
        device_type = get_device_type()

    if device_type == 'npu':
        return 'hccl'
    elif device_type == 'cuda':
        return 'nccl'
    else:
        return 'gloo'


def init_dist(launcher='pytorch', backend=None, **kwargs):
    """
    初始化分布式训练环境

    Args:
        launcher: 'pytorch' 或 'slurm'
        backend: 分布式后端 ('nccl', 'hccl', 'gloo')，None 表示自动选择
        **kwargs: 传递给 dist.init_process_group 的额外参数
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # 自动选择后端
    if backend is None:
        device_type = get_device_type()
        backend = get_backend_for_device(device_type)
        logger.info(f"Auto-selected backend: {backend} for device type: {device_type}")

    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    """
    PyTorch 原生分布式初始化 (支持 torchrun 启动)

    环境变量要求 (由 torchrun 自动设置):
        - RANK: 全局进程排名
        - LOCAL_RANK: 本地进程排名
        - WORLD_SIZE: 总进程数
        - MASTER_ADDR: 主节点地址
        - MASTER_PORT: 主节点端口
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 根据后端选择设备设置方式
    if backend == 'hccl':
        # NPU 设备
        try:
            import torch_npu
        except ImportError:
            raise RuntimeError("hccl backend requires torch_npu, but it's not installed")
        num_devices = torch.npu.device_count()
        device_id = local_rank % num_devices
        torch.npu.set_device(device_id)
        logger.info(f"[Rank {rank}] Set NPU device: npu:{device_id}")
    elif backend == 'nccl':
        # CUDA 设备
        num_devices = torch.cuda.device_count()
        device_id = local_rank % num_devices
        torch.cuda.set_device(device_id)
        logger.info(f"[Rank {rank}] Set CUDA device: cuda:{device_id}")
    else:
        # CPU (gloo)
        device_id = 0
        logger.info(f"[Rank {rank}] Using CPU with gloo backend")

    dist.init_process_group(backend=backend, **kwargs)
    logger.info(f"[Rank {rank}] Distributed initialized: world_size={world_size}, backend={backend}")


def _init_dist_slurm(backend, port=None):
    """
    SLURM 集群分布式初始化

    Args:
        backend: 分布式后端
        port: Master port，None 表示使用环境变量或默认值 29500
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']

    # 根据后端选择设备
    if backend == 'hccl':
        import torch_npu
        num_devices = torch.npu.device_count()
        torch.npu.set_device(proc_id % num_devices)
    elif backend == 'nccl':
        num_devices = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_devices)
    else:
        num_devices = 1

    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')

    # 设置端口
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_devices)
    os.environ['RANK'] = str(proc_id)

    dist.init_process_group(backend=backend)


def get_dist_info():
    """
    获取分布式信息

    Returns:
        tuple: (rank, world_size)
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_local_rank() -> int:
    """
    获取本地 rank (用于设备选择)

    Returns:
        int: 本地进程排名
    """
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """
    判断当前进程是否为主进程 (rank == 0)

    Returns:
        bool: 是否为主进程
    """
    rank, _ = get_dist_info()
    return rank == 0


def is_dist_initialized() -> bool:
    """
    判断分布式训练是否已初始化

    Returns:
        bool: 是否已初始化
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    获取 world size

    Returns:
        int: 总进程数
    """
    _, world_size = get_dist_info()
    return world_size


def barrier():
    """
    进程同步屏障 (仅在分布式模式下有效)
    """
    if is_dist_initialized():
        dist.barrier()


def reduce_tensor(tensor, op=dist.ReduceOp.SUM, dst=0):
    """
    Reduce 张量到指定进程

    Args:
        tensor: 输入张量
        op: 归约操作 (SUM, AVG, MAX, MIN)
        dst: 目标进程 rank

    Returns:
        归约后的张量 (仅在 dst 进程有效)
    """
    if not is_dist_initialized():
        return tensor

    dist.reduce(tensor, dst=dst, op=op)
    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """
    All-reduce 张量到所有进程

    Args:
        tensor: 输入张量
        op: 归约操作

    Returns:
        归约后的张量
    """
    if not is_dist_initialized():
        return tensor

    dist.all_reduce(tensor, op=op)
    return tensor


def all_reduce_mean(tensor):
    """
    All-reduce 并计算平均值

    Args:
        tensor: 输入张量

    Returns:
        所有进程的平均值
    """
    if not is_dist_initialized():
        return tensor

    world_size = dist.get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)
    return tensor


def broadcast_tensor(tensor, src=0):
    """
    从 src 进程广播张量到所有进程

    Args:
        tensor: 输入张量
        src: 源进程 rank

    Returns:
        广播后的张量
    """
    if not is_dist_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def master_only(func):
    """
    装饰器: 仅在主进程 (rank=0) 执行函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
        return None
    return wrapper


def get_device_for_rank() -> str:
    """
    获取当前进程对应的设备字符串

    Returns:
        str: 设备字符串 (如 'npu:0', 'cuda:1', 'cpu')
    """
    device_type = get_device_type()
    local_rank = get_local_rank()

    if device_type == 'npu':
        return f'npu:{local_rank}'
    elif device_type == 'cuda':
        return f'cuda:{local_rank}'
    else:
        return 'cpu'


def destroy_process_group():
    """
    销毁分布式进程组

    在训练结束时调用以清理分布式环境
    """
    if is_dist_initialized():
        dist.destroy_process_group()
