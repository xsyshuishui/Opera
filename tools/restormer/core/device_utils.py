"""
设备工具模块 - 自动设备选择和管理

提供自动检测和选择最佳计算设备的功能，支持:
- Ascend NPU (torch_npu)
- NVIDIA CUDA
- CPU

支持分布式训练中的设备分配
"""

import torch
import logging
import os

logger = logging.getLogger(__name__)


def _is_npu_available() -> bool:
    """检查 NPU 是否可用"""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def get_device_type() -> str:
    """
    自动检测可用的设备类型

    Returns:
        str: 'npu', 'cuda', or 'cpu'
    """
    if _is_npu_available():
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_available_device(preferred: str = None) -> str:
    """
    获取可用的计算设备

    在分布式训练中，会根据 LOCAL_RANK 自动选择对应的设备

    优先级: NPU > CUDA > CPU

    Args:
        preferred: 用户指定的设备 (如 'npu:0', 'cuda:0', 'cpu')
                   如果为 None，则自动选择最佳设备

    Returns:
        str: 设备字符串 (如 'npu:0', 'cuda:0', 'cpu')
    """
    # 检查是否在分布式环境中
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    if local_rank >= 0:
        # 分布式模式: 根据 local_rank 选择设备
        return _get_distributed_device(local_rank, preferred)

    # 非分布式模式
    if preferred is not None:
        return _validate_and_normalize_device(preferred)

    # 自动选择设备
    # 1. 优先尝试 NPU
    if _is_npu_available():
        device = 'npu:0'
        logger.info(f"自动选择设备: {device} (Ascend NPU)")
        return device

    # 2. 尝试 CUDA
    if torch.cuda.is_available():
        device = 'cuda:0'
        logger.info(f"自动选择设备: {device} (NVIDIA CUDA)")
        return device

    # 3. 回退到 CPU
    device = 'cpu'
    logger.info(f"自动选择设备: {device}")
    return device


def _get_distributed_device(local_rank: int, preferred: str = None) -> str:
    """
    获取分布式训练中当前进程的设备

    Args:
        local_rank: 本地进程排名
        preferred: 用户偏好的设备类型 (如 'npu', 'cuda')

    Returns:
        str: 设备字符串 (如 'npu:0', 'cuda:1')
    """
    # 确定设备类型
    if preferred:
        device_type = preferred.split(':')[0].lower()
    else:
        # 自动检测
        device_type = get_device_type()

    if device_type == 'npu':
        num_devices = torch.npu.device_count()
        device_id = local_rank % num_devices
        device = f'npu:{device_id}'
        logger.info(f"分布式模式: 选择设备 {device} (local_rank={local_rank})")
        return device
    elif device_type == 'cuda':
        num_devices = torch.cuda.device_count()
        device_id = local_rank % num_devices
        device = f'cuda:{device_id}'
        logger.info(f"分布式模式: 选择设备 {device} (local_rank={local_rank})")
        return device
    else:
        return 'cpu'


def _validate_and_normalize_device(device: str) -> str:
    """
    验证并规范化设备字符串

    Args:
        device: 用户输入的设备字符串

    Returns:
        str: 规范化的设备字符串

    Raises:
        ValueError: 设备不可用或格式无效
    """
    device = device.strip().lower()

    # 处理 NPU 设备
    if device.startswith('npu'):
        if not _is_npu_available():
            raise ValueError(f"NPU 设备不可用，请检查 torch_npu 是否安装且 NPU 硬件正常")

        # 如果没有指定设备号，默认使用 npu:0
        if device == 'npu':
            return 'npu:0'

        # 验证设备号格式
        if ':' in device:
            try:
                device_id = int(device.split(':')[1])
                device_count = torch.npu.device_count()
                if device_id >= device_count:
                    raise ValueError(f"NPU 设备号 {device_id} 超出范围，可用设备数: {device_count}")
            except (ValueError, IndexError) as e:
                if 'NPU 设备号' in str(e):
                    raise
                raise ValueError(f"无效的设备格式: {device}")
        return device

    # 处理 CUDA 设备
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA 设备不可用，请检查 CUDA 是否安装且 GPU 硬件正常")

        # 如果没有指定设备号，默认使用 cuda:0
        if device == 'cuda':
            return 'cuda:0'

        # 验证设备号格式
        if ':' in device:
            try:
                device_id = int(device.split(':')[1])
                device_count = torch.cuda.device_count()
                if device_id >= device_count:
                    raise ValueError(f"CUDA 设备号 {device_id} 超出范围，可用设备数: {device_count}")
            except (ValueError, IndexError) as e:
                if 'CUDA 设备号' in str(e):
                    raise
                raise ValueError(f"无效的设备格式: {device}")
        return device

    # CPU
    if device == 'cpu':
        return 'cpu'

    raise ValueError(f"不支持的设备类型: {device}，支持的类型: npu, cuda, cpu")


def set_device(device: str):
    """
    设置当前计算设备

    在分布式训练中，这会设置当前进程使用的 GPU/NPU

    Args:
        device: 设备字符串 (如 'npu:0', 'cuda:0')
    """
    if device.startswith('npu'):
        try:
            import torch_npu
            torch.npu.set_device(device)
            logger.info(f"已设置当前 NPU 设备: {device}")
        except Exception as e:
            logger.warning(f"设置 NPU 设备失败: {e}")
    elif device.startswith('cuda'):
        try:
            torch.cuda.set_device(device)
            logger.info(f"已设置当前 CUDA 设备: {device}")
        except Exception as e:
            logger.warning(f"设置 CUDA 设备失败: {e}")


def get_device_info(device: str) -> dict:
    """
    获取设备信息

    Args:
        device: 设备字符串

    Returns:
        dict: 设备信息字典
    """
    info = {'device': device, 'type': 'unknown'}

    if device.startswith('npu'):
        info['type'] = 'npu'
        try:
            import torch_npu
            device_id = int(device.split(':')[1]) if ':' in device else 0
            info['name'] = torch.npu.get_device_name(device_id)
            info['memory_total'] = torch.npu.get_device_properties(device_id).total_memory
        except Exception:
            pass
    elif device.startswith('cuda'):
        info['type'] = 'cuda'
        try:
            device_id = int(device.split(':')[1]) if ':' in device else 0
            info['name'] = torch.cuda.get_device_name(device_id)
            info['memory_total'] = torch.cuda.get_device_properties(device_id).total_memory
        except Exception:
            pass
    elif device == 'cpu':
        info['type'] = 'cpu'

    return info


def empty_cache(device: str = None):
    """
    清空设备缓存

    Args:
        device: 设备字符串，None 表示清空所有可用设备的缓存
    """
    if device is None:
        # 清空所有可用设备
        if _is_npu_available():
            torch.npu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif device.startswith('npu'):
        torch.npu.empty_cache()
    elif device.startswith('cuda'):
        torch.cuda.empty_cache()


def synchronize(device: str = None):
    """
    同步设备上的计算

    Args:
        device: 设备字符串，None 表示同步所有可用设备
    """
    if device is None:
        device_type = get_device_type()
    else:
        device_type = device.split(':')[0]

    if device_type == 'npu':
        torch.npu.synchronize()
    elif device_type == 'cuda':
        torch.cuda.synchronize()


def get_device_count() -> int:
    """
    获取可用设备数量

    Returns:
        int: 可用设备数量
    """
    device_type = get_device_type()
    if device_type == 'npu':
        return torch.npu.device_count()
    elif device_type == 'cuda':
        return torch.cuda.device_count()
    return 1  # CPU


def get_memory_info(device: str) -> dict:
    """
    获取设备内存使用信息

    Args:
        device: 设备字符串

    Returns:
        dict: 内存信息 {'allocated': bytes, 'reserved': bytes, 'total': bytes}
    """
    info = {'allocated': 0, 'reserved': 0, 'total': 0}

    if device.startswith('npu'):
        try:
            info['allocated'] = torch.npu.memory_allocated(device)
            info['reserved'] = torch.npu.memory_reserved(device)
            device_id = int(device.split(':')[1]) if ':' in device else 0
            info['total'] = torch.npu.get_device_properties(device_id).total_memory
        except Exception:
            pass
    elif device.startswith('cuda'):
        try:
            info['allocated'] = torch.cuda.memory_allocated(device)
            info['reserved'] = torch.cuda.memory_reserved(device)
            device_id = int(device.split(':')[1]) if ':' in device else 0
            info['total'] = torch.cuda.get_device_properties(device_id).total_memory
        except Exception:
            pass

    return info
