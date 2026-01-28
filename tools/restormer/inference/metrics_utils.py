"""
图像质量评估指标计算工具

支持的指标:
- PSNR (Peak Signal-to-Noise Ratio) on RGB channels - using pyiqa
- PSNR_Y (Peak Signal-to-Noise Ratio) on Y channel - using pyiqa
- SSIM (Structural Similarity Index) on Y channel - using pyiqa
- LPIPS (Learned Perceptual Image Patch Similarity) - using pyiqa
- MANIQA (可选，如果库可用)
- CLIPIQA (可选，如果库可用)
- MUSIQ (可选，如果库可用)
"""

import torch
import numpy as np

# 尝试导入 pyiqa
try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False
    print("⚠ pyiqa 库不可用，所有指标将被禁用")

# 设备管理
def get_default_device():
    """自动检测可用的计算设备"""
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu:0"
    except:
        pass

    if torch.cuda.is_available():
        return "cuda:0"

    return "cpu"

# 全局默认设备（可以被外部修改）
DEFAULT_DEVICE = None

def set_default_device(device):
    """设置默认设备"""
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device

# 全局指标模型列表 (类似 score_webapi.py 的设计)
_metrics_list = []
_metrics_dict = {}
_metrics_initialized = False


def init_all_metrics(device=None):
    """
    初始化所有指标模型 (类似 score_webapi.py 的设计)

    Args:
        device: 计算设备
    """
    global _metrics_list, _metrics_dict, _metrics_initialized

    if _metrics_initialized:
        print("✓ 指标模型已初始化，跳过重复初始化")
        return _metrics_dict

    if device is None:
        device = DEFAULT_DEVICE if DEFAULT_DEVICE else get_default_device()

    if not PYIQA_AVAILABLE:
        print("✗ pyiqa 不可用，无法初始化指标模型")
        return {}

    print(f"\n{'='*60}")
    print("初始化图像质量评估指标模型...")
    print(f"{'='*60}")

    try:
        # 按照 score_webapi.py 的顺序和配置初始化指标
        _metrics_list = [
            pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device),
            pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device=device),
            pyiqa.create_metric('lpips', device=device),
            pyiqa.create_metric('clipiqa', device=device),
            pyiqa.create_metric('musiq', device=device),
            # 额外的指标（不在 score_webapi.py 参考中）
            pyiqa.create_metric('psnr', device=device),  # PSNR-RGB (额外)
            pyiqa.create_metric('maniqa', device=device),  # MANIQA (额外)
        ]

        # 创建指标名称到模型的映射
        _metrics_dict = {
            'psnr_y': _metrics_list[0],     # PSNR on Y channel
            'ssim': _metrics_list[1],        # SSIM on Y channel
            'lpips': _metrics_list[2],       # LPIPS
            'clipiqa': _metrics_list[3],     # CLIPIQA
            'musiq': _metrics_list[4],       # MUSIQ
            'psnr': _metrics_list[5],        # PSNR on RGB (额外)
            'maniqa': _metrics_list[6],      # MANIQA (额外)
        }

        _metrics_initialized = True

        print(f"✓ 已初始化 {len(_metrics_list)} 个指标模型:")
        print(f"  - PSNR-Y (Y channel, YCbCr color space)")
        print(f"  - SSIM (Y channel, YCbCr color space)")
        print(f"  - LPIPS")
        print(f"  - CLIPIQA")
        print(f"  - MUSIQ")
        print(f"  - PSNR-RGB (额外指标)")
        print(f"  - MANIQA (额外指标)")
        print(f"{'='*60}\n")

        return _metrics_dict

    except Exception as e:
        print(f"✗ 初始化指标模型失败: {e}")
        _metrics_initialized = False
        return {}


def get_metric_model(metric_name):
    """
    获取指定名称的指标模型

    Args:
        metric_name: 指标名称 ('psnr', 'psnr_y', 'ssim', 'lpips', 'maniqa', 'clipiqa', 'musiq')

    Returns:
        指标模型或 None
    """
    global _metrics_dict
    return _metrics_dict.get(metric_name, None)


# 保留旧的初始化函数以保持向后兼容
def init_psnr_rgb_model(device=None):
    """初始化 PSNR-RGB 模型 (使用 pyiqa)"""
    if not _metrics_initialized:
        init_all_metrics(device)
    return get_metric_model('psnr')


def init_psnr_y_model(device=None):
    """初始化 PSNR-Y 模型 (使用 pyiqa)"""
    if not _metrics_initialized:
        init_all_metrics(device)
    return get_metric_model('psnr_y')


def init_ssim_y_model(device=None):
    """初始化 SSIM-Y 模型 (使用 pyiqa)"""
    if not _metrics_initialized:
        init_all_metrics(device)
    return get_metric_model('ssim')


def init_lpips_model(device=None):
    """初始化 LPIPS 模型 (使用 pyiqa)"""
    if not _metrics_initialized:
        init_all_metrics(device)
    return get_metric_model('lpips')


def init_maniqa_model(device=None):
    """初始化 MANIQA 模型"""
    if not _metrics_initialized:
        init_all_metrics(device)
    return get_metric_model('maniqa')


def init_clipiqa_model(device=None):
    """初始化 CLIPIQA 模型"""
    if not _metrics_initialized:
        init_all_metrics(device)
    return get_metric_model('clipiqa')


def init_musiq_model(device=None):
    """初始化 MUSIQ 模型"""
    if not _metrics_initialized:
        init_all_metrics(device)
    return get_metric_model('musiq')


def calculate_psnr_rgb(pred_img, gt_img, device=None):
    """
    计算 RGB 通道的 PSNR (使用 pyiqa)

    Args:
        pred_img: 预测图像, numpy array or torch tensor, shape (H, W, 3), range [0, 1], RGB order
        gt_img: 真实图像（参考图像）, numpy array or torch tensor, shape (H, W, 3), range [0, 1], RGB order
        device: str, device to run model

    Returns:
        float: PSNR value in dB
    """
    if not _metrics_initialized:
        init_all_metrics(device)

    psnr_model = get_metric_model('psnr')
    if psnr_model is None:
        return None  # 模型不可用

    try:
        # 转换为 torch tensor
        if isinstance(pred_img, np.ndarray):
            pred_img = torch.from_numpy(pred_img).float()
        if isinstance(gt_img, np.ndarray):
            gt_img = torch.from_numpy(gt_img).float()

        # 确保值域在 [0, 1]
        pred_img = torch.clamp(pred_img, 0, 1)
        gt_img = torch.clamp(gt_img, 0, 1)

        # pyiqa 需要输入格式: (1, C, H, W)
        if pred_img.ndim == 3:  # (H, W, C)
            pred_img = pred_img.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
        if gt_img.ndim == 3:
            gt_img = gt_img.permute(2, 0, 1).unsqueeze(0)

        # 计算 PSNR (pyiqa: test_img, ref_img)
        with torch.no_grad():
            psnr = psnr_model(pred_img, gt_img)

        return psnr.item()

    except Exception as e:
        print(f"PSNR-RGB 计算失败: {e}")
        return None


def calculate_psnr_y(pred_img, gt_img, device=None):
    """
    计算 Y 通道的 PSNR (使用 pyiqa)

    Args:
        pred_img: 预测图像, numpy array or torch tensor, shape (H, W, 3), range [0, 1], RGB order
        gt_img: 真实图像（参考图像）, numpy array or torch tensor, shape (H, W, 3), range [0, 1], RGB order
        device: str, device to run model

    Returns:
        float: PSNR value in dB
    """
    if not _metrics_initialized:
        init_all_metrics(device)

    psnr_y_model = get_metric_model('psnr_y')
    if psnr_y_model is None:
        return None  # 模型不可用

    try:
        # 转换为 torch tensor
        if isinstance(pred_img, np.ndarray):
            pred_img = torch.from_numpy(pred_img).float()
        if isinstance(gt_img, np.ndarray):
            gt_img = torch.from_numpy(gt_img).float()

        # 确保值域在 [0, 1]
        pred_img = torch.clamp(pred_img, 0, 1)
        gt_img = torch.clamp(gt_img, 0, 1)

        # pyiqa 需要输入格式: (1, C, H, W)
        if pred_img.ndim == 3:  # (H, W, C)
            pred_img = pred_img.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
        if gt_img.ndim == 3:
            gt_img = gt_img.permute(2, 0, 1).unsqueeze(0)

        # 计算 PSNR-Y (pyiqa: test_img, ref_img)
        with torch.no_grad():
            psnr = psnr_y_model(pred_img, gt_img)

        return psnr.item()

    except Exception as e:
        print(f"PSNR-Y 计算失败: {e}")
        return None


def calculate_ssim(pred_img, gt_img, device=None):
    """
    计算 SSIM on Y channel (使用 pyiqa)

    Args:
        pred_img: 预测图像, numpy array or torch tensor, shape (H, W, 3), range [0, 1]
        gt_img: 真实图像（参考图像）, numpy array or torch tensor, shape (H, W, 3), range [0, 1]
        device: str, device to run model

    Returns:
        float: SSIM value
    """
    if not _metrics_initialized:
        init_all_metrics(device)

    ssim_model = get_metric_model('ssim')
    if ssim_model is None:
        return None  # 模型不可用

    try:
        # 转换为 torch tensor
        if isinstance(pred_img, np.ndarray):
            pred_img = torch.from_numpy(pred_img).float()
        if isinstance(gt_img, np.ndarray):
            gt_img = torch.from_numpy(gt_img).float()

        # 确保值域在 [0, 1]
        pred_img = torch.clamp(pred_img, 0, 1)
        gt_img = torch.clamp(gt_img, 0, 1)

        # pyiqa 需要输入格式: (1, C, H, W)
        if pred_img.ndim == 3:  # (H, W, C)
            pred_img = pred_img.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
        if gt_img.ndim == 3:
            gt_img = gt_img.permute(2, 0, 1).unsqueeze(0)

        # 计算 SSIM-Y (pyiqa: test_img, ref_img)
        with torch.no_grad():
            ssim = ssim_model(pred_img, gt_img)

        return ssim.item()

    except Exception as e:
        print(f"SSIM-Y 计算失败: {e}")
        return None


def calculate_lpips(pred_img, gt_img, device=None):
    """
    计算 LPIPS (使用 pyiqa)

    Args:
        pred_img: 预测图像, numpy array or torch tensor, shape (H, W, 3), range [0, 1]
        gt_img: 真实图像（参考图像）, numpy array or torch tensor, shape (H, W, 3), range [0, 1]
        device: str, device to run model

    Returns:
        float: LPIPS value (lower is better)
    """
    if not _metrics_initialized:
        init_all_metrics(device)

    lpips_model = get_metric_model('lpips')
    if lpips_model is None:
        return None  # 模型不可用

    try:
        # 转换为 torch tensor
        if isinstance(pred_img, np.ndarray):
            pred_img = torch.from_numpy(pred_img).float()
        if isinstance(gt_img, np.ndarray):
            gt_img = torch.from_numpy(gt_img).float()

        # 确保值域在 [0, 1]
        pred_img = torch.clamp(pred_img, 0, 1)
        gt_img = torch.clamp(gt_img, 0, 1)

        # pyiqa 需要输入格式: (1, C, H, W)
        if pred_img.ndim == 3:  # (H, W, C)
            pred_img = pred_img.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
        if gt_img.ndim == 3:
            gt_img = gt_img.permute(2, 0, 1).unsqueeze(0)

        # 计算 LPIPS (pyiqa: test_img, ref_img)
        with torch.no_grad():
            lpips_value = lpips_model(pred_img, gt_img)

        return lpips_value.item()

    except Exception as e:
        print(f"LPIPS 计算失败: {e}")
        return None


def calculate_maniqa(img, device=None):
    """
    计算 MANIQA (无参考图像质量评估)

    Args:
        img: numpy array or torch tensor, shape (H, W, 3), range [0, 1], RGB order
        device: str, device to run model

    Returns:
        float: MANIQA score (0-1, 越高越好) 或 None 如果模型不可用
    """
    if not _metrics_initialized:
        init_all_metrics(device)

    maniqa_model = get_metric_model('maniqa')
    if maniqa_model is None:
        return None  # 模型不可用

    try:
        # 转换为 torch tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).float()
        else:
            img_tensor = img.clone().float()

        # 确保值域在 [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # MANIQA 需要输入格式: (1, C, H, W)
        if img_tensor.ndim == 3:  # (H, W, C)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)

        # 计算 MANIQA
        with torch.no_grad():
            score = maniqa_model(img_tensor)

        return score.item()

    except Exception as e:
        print(f"MANIQA 计算失败: {e}")
        return None


def calculate_clipiqa(img, device=None):
    """
    计算 CLIPIQA (无参考图像质量评估)

    Args:
        img: numpy array or torch tensor, shape (H, W, 3), range [0, 1], RGB order
        device: str, device to run model

    Returns:
        float: CLIPIQA score (0-1, 越高越好) 或 None 如果模型不可用
    """
    if not _metrics_initialized:
        init_all_metrics(device)

    clipiqa_model = get_metric_model('clipiqa')
    if clipiqa_model is None:
        return None  # 模型不可用

    try:
        # 转换为 torch tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).float()
        else:
            img_tensor = img.clone().float()

        # 确保值域在 [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # CLIPIQA 需要输入格式: (1, C, H, W)
        if img_tensor.ndim == 3:  # (H, W, C)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)

        # 计算 CLIPIQA
        with torch.no_grad():
            score = clipiqa_model(img_tensor)

        return score.item()

    except Exception as e:
        print(f"CLIPIQA 计算失败: {e}")
        return None


def calculate_musiq(img, device=None):
    """
    计算 MUSIQ (无参考图像质量评估)

    Args:
        img: numpy array or torch tensor, shape (H, W, 3), range [0, 1], RGB order
        device: str, device to run model (实际使用CPU避免版本冲突)

    Returns:
        float: MUSIQ score (0-100, 越高越好) 或 None 如果模型不可用
    """
    if not _metrics_initialized:
        init_all_metrics(device)

    musiq_model = get_metric_model('musiq')
    if musiq_model is None:
        return None  # 模型不可用

    try:
        # 转换为 torch tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).float()
        else:
            img_tensor = img.clone().float()

        # 确保值域在 [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # MUSIQ 需要输入格式: (1, C, H, W)
        if img_tensor.ndim == 3:  # (H, W, C)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)

        # 计算 MUSIQ
        with torch.no_grad():
            score = musiq_model(img_tensor)

        return score.item()

    except Exception as e:
        print(f"MUSIQ 计算失败: {e}")
        return None


def calculate_all_metrics(pred_img, gt_img, device=None):
    """
    计算所有可用的图像质量指标

    Args:
        pred_img: 预测图像, numpy array or torch tensor, shape (H, W, 3), range [0, 1]
        gt_img: 真实图像, numpy array or torch tensor, shape (H, W, 3), range [0, 1]
        device: str, device to run model

    Returns:
        dict: 包含所有指标的字典
    """
    if device is None:
        device = DEFAULT_DEVICE if DEFAULT_DEVICE else get_default_device()

    metrics = {}

    # PSNR (RGB)
    try:
        psnr = calculate_psnr_rgb(pred_img, gt_img)
        metrics['psnr'] = psnr
    except Exception as e:
        print(f"PSNR (RGB) 计算失败: {e}")
        metrics['psnr'] = None

    # PSNR (Y channel)
    try:
        psnr_y = calculate_psnr_y(pred_img, gt_img)
        metrics['psnr_y'] = psnr_y
    except Exception as e:
        print(f"PSNR (Y) 计算失败: {e}")
        metrics['psnr_y'] = None

    # SSIM
    try:
        ssim = calculate_ssim(pred_img, gt_img)
        metrics['ssim'] = ssim
    except Exception as e:
        print(f"SSIM 计算失败: {e}")
        metrics['ssim'] = None

    # LPIPS
    try:
        lpips_val = calculate_lpips(pred_img, gt_img, device)
        metrics['lpips'] = lpips_val
    except Exception as e:
        print(f"LPIPS 计算失败: {e}")
        metrics['lpips'] = None

    # MANIQA
    try:
        maniqa_val = calculate_maniqa(pred_img, device)
        metrics['maniqa'] = maniqa_val
    except Exception as e:
        print(f"MANIQA 计算失败: {e}")
        metrics['maniqa'] = None

    # CLIPIQA
    try:
        clipiqa_val = calculate_clipiqa(pred_img, device)
        metrics['clipiqa'] = clipiqa_val
    except Exception as e:
        print(f"CLIPIQA 计算失败: {e}")
        metrics['clipiqa'] = None

    # MUSIQ
    try:
        musiq_val = calculate_musiq(pred_img, device)
        metrics['musiq'] = musiq_val
    except Exception as e:
        print(f"MUSIQ 计算失败: {e}")
        metrics['musiq'] = None

    return metrics


if __name__ == "__main__":
    # 测试代码
    print("测试图像质量指标计算...")

    # 创建测试图像
    img1 = np.random.rand(128, 128, 3).astype(np.float32)
    img2 = img1 + np.random.rand(128, 128, 3).astype(np.float32) * 0.1
    img2 = np.clip(img2, 0, 1)

    # 测试 PSNR
    psnr = calculate_psnr_y(img1, img2)
    print(f"PSNR (Y): {psnr:.2f} dB")

    # 测试 SSIM
    ssim = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")

    # 测试 LPIPS
    lpips_val = calculate_lpips(img1, img2, device='cpu')
    if lpips_val is not None:
        print(f"LPIPS: {lpips_val:.4f}")

    print("✓ 测试完成")
