"""
感知损失模块 - VGG + LPIPS + MUSIQ + CLIPIQA

支持的损失类型:
- VGGPerceptualLoss: 使用预训练 VGG19 提取多层特征计算感知损失
- LPIPSLoss: 使用 LPIPS 计算感知损失
- MUSIQLoss: 使用 MUSIQ 无参考质量评估作为损失 (最大化质量分数)
- CLIPIQALoss: 使用 CLIPIQA 无参考质量评估作为损失 (最大化质量分数)
- CombinedLoss: 组合 L1 + VGG + LPIPS + MUSIQ + CLIPIQA 损失，支持渐进训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging

logger = logging.getLogger(__name__)


class VGGPerceptualLoss(nn.Module):
    """
    VGG19 特征感知损失

    使用预训练的 VGG19 提取多层特征，计算预测图像和目标图像在特征空间的 L1 距离。
    这种损失能够捕捉图像的高级语义信息和纹理细节。
    """

    # VGG19 各层对应的索引
    VGG_LAYERS = {
        'relu1_1': 1, 'relu1_2': 3,
        'relu2_1': 6, 'relu2_2': 8,
        'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
        'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
        'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
    }

    def __init__(self, device, layers=None, weights=None):
        """
        Args:
            device: 计算设备
            layers: 要提取的 VGG 层名称列表，默认 ['relu2_2', 'relu3_4', 'relu4_4']
            weights: 各层的权重，默认均等
        """
        super().__init__()

        self.device = device

        # 默认使用的层
        if layers is None:
            layers = ['relu2_2', 'relu3_4', 'relu4_4']
        self.layer_names = layers

        # 各层权重
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        self.weights = weights

        # 加载预训练 VGG19
        vgg = models.vgg19(pretrained=True).features

        # 获取最大需要的层索引
        max_idx = max(self.VGG_LAYERS[name] for name in layers) + 1

        # 只保留需要的层
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max_idx]).to(device)

        # 冻结 VGG 参数
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # 设置为评估模式
        self.vgg_layers.eval()

        # ImageNet 归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

        logger.info(f"VGGPerceptualLoss initialized with layers: {layers}")

    def normalize(self, x):
        """ImageNet 归一化"""
        return (x - self.mean) / self.std

    def extract_features(self, x):
        """提取指定层的特征"""
        x = self.normalize(x)
        features = {}

        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            idx = int(name)
            # 检查当前层是否是我们需要的
            for layer_name in self.layer_names:
                if self.VGG_LAYERS[layer_name] == idx:
                    features[layer_name] = x

        return features

    def forward(self, pred, target):
        """
        计算感知损失

        Args:
            pred: 预测图像 (B, C, H, W)，范围 [0, 1]
            target: 目标图像 (B, C, H, W)，范围 [0, 1]

        Returns:
            感知损失值
        """
        # 确保在 [0, 1] 范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 提取特征
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        # 计算各层损失的加权和
        loss = 0.0
        for i, layer_name in enumerate(self.layer_names):
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            loss += self.weights[i] * F.l1_loss(pred_feat, target_feat)

        return loss


class LPIPSLoss(nn.Module):
    """
    LPIPS 感知损失

    使用 pyiqa 库的 LPIPS 实现，基于 AlexNet/VGG 的学习感知相似度。
    LPIPS 在感知质量评估上比传统的 PSNR/SSIM 更接近人眼感知。
    """

    def __init__(self, device):
        """
        Args:
            device: 计算设备
        """
        super().__init__()

        self.device = device

        try:
            import pyiqa
            # 重要: as_loss=True 启用梯度流，使 loss 能够回传到生成网络
            self.lpips = pyiqa.create_metric('lpips', device=device, as_loss=True)

            # 冻结 LPIPS 模型参数 (只冻结 metric 模型本身的参数，不影响梯度流过)
            for param in self.lpips.parameters():
                param.requires_grad = False

            self.lpips.eval()
            logger.info("LPIPSLoss initialized (as_loss=True, gradients enabled)")

        except ImportError:
            logger.error("pyiqa not available, LPIPSLoss will not work")
            self.lpips = None

    def forward(self, pred, target):
        """
        计算 LPIPS 损失

        Args:
            pred: 预测图像 (B, C, H, W)，范围 [0, 1]
            target: 目标图像 (B, C, H, W)，范围 [0, 1]

        Returns:
            LPIPS 损失值 (标量)
        """
        if self.lpips is None:
            return torch.tensor(0.0, device=self.device)

        # 确保在 [0, 1] 范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 计算 LPIPS (as_loss=True 已启用梯度流)
        lpips_value = self.lpips(pred, target)

        return lpips_value.mean()


class MUSIQLoss(nn.Module):
    """
    MUSIQ 无参考图像质量损失

    使用 MUSIQ (Multi-scale Image Quality Transformer) 评估生成图像的质量。
    由于 MUSIQ 分数越高越好，我们使用负分数作为损失来最大化质量。

    注意: MUSIQ 是无参考指标，只需要输入图像，不需要 GT。
    """

    def __init__(self, device, normalize=True):
        """
        Args:
            device: 计算设备
            normalize: 是否将分数归一化到 [0, 1] 范围 (原始范围约 0-100)
        """
        super().__init__()

        self.device = device
        self.normalize = normalize

        try:
            import pyiqa
            # 重要: as_loss=True 启用梯度流，使 loss 能够回传到生成网络
            self.musiq = pyiqa.create_metric('musiq', device=device, as_loss=True)

            # 冻结 MUSIQ 模型参数 (只冻结 metric 模型本身的参数，不影响梯度流过)
            for param in self.musiq.parameters():
                param.requires_grad = False

            self.musiq.eval()
            logger.info("MUSIQLoss initialized (as_loss=True, gradients enabled)")

        except ImportError:
            logger.error("pyiqa not available, MUSIQLoss will not work")
            self.musiq = None

    def forward(self, pred, target=None):
        """
        计算 MUSIQ 损失 (最大化质量分数)

        Args:
            pred: 预测图像 (B, C, H, W)，范围 [0, 1]
            target: 未使用，保持接口一致性

        Returns:
            MUSIQ 损失值 (负分数，用于最小化)
        """
        if self.musiq is None:
            return torch.tensor(0.0, device=self.device)

        # 确保在 [0, 1] 范围内
        pred = torch.clamp(pred, 0, 1)

        # 计算 MUSIQ 分数 (越高越好)
        musiq_score = self.musiq(pred)

        # 归一化到 [0, 1] (原始范围约 0-100)
        if self.normalize:
            musiq_score = musiq_score / 100.0

        # 返回负分数作为损失 (最大化质量 = 最小化负分数)
        # 使用 (1 - score) 使损失在 [0, 1] 范围，更易于权重调节
        loss = 1.0 - musiq_score.mean()

        return loss


class CLIPIQALoss(nn.Module):
    """
    CLIPIQA 无参考图像质量损失

    使用 CLIP-IQA 评估生成图像的质量。
    由于 CLIPIQA 分数越高越好，我们使用 (1 - score) 作为损失来最大化质量。

    注意: CLIPIQA 是无参考指标，只需要输入图像，不需要 GT。
    """

    def __init__(self, device):
        """
        Args:
            device: 计算设备
        """
        super().__init__()

        self.device = device

        try:
            import pyiqa
            # 重要: as_loss=True 启用梯度流，使 loss 能够回传到生成网络
            self.clipiqa = pyiqa.create_metric('clipiqa', device=device, as_loss=True)

            # 冻结 CLIPIQA 模型参数 (只冻结 metric 模型本身的参数，不影响梯度流过)
            for param in self.clipiqa.parameters():
                param.requires_grad = False

            self.clipiqa.eval()
            logger.info("CLIPIQALoss initialized (as_loss=True, gradients enabled)")

        except ImportError:
            logger.error("pyiqa not available, CLIPIQALoss will not work")
            self.clipiqa = None

    def forward(self, pred, target=None):
        """
        计算 CLIPIQA 损失 (最大化质量分数)

        Args:
            pred: 预测图像 (B, C, H, W)，范围 [0, 1]
            target: 未使用，保持接口一致性

        Returns:
            CLIPIQA 损失值 (1 - score，用于最小化)
        """
        if self.clipiqa is None:
            return torch.tensor(0.0, device=self.device)

        # 确保在 [0, 1] 范围内
        pred = torch.clamp(pred, 0, 1)

        # 计算 CLIPIQA 分数 (范围 0-1，越高越好)
        clipiqa_score = self.clipiqa(pred)

        # 返回 (1 - score) 作为损失 (最大化质量 = 最小化损失)
        loss = 1.0 - clipiqa_score.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    组合损失 - L1 + VGG + LPIPS + MUSIQ + CLIPIQA

    支持渐进训练：可以在训练过程中动态调整各损失分量的权重。

    总损失 = pixel_weight * L1_Loss
           + perceptual_weight * VGG_Loss
           + lpips_weight * LPIPS_Loss
           + musiq_weight * MUSIQ_Loss
           + clipiqa_weight * CLIPIQA_Loss
    """

    def __init__(
        self,
        device,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        lpips_weight: float = 0.0,
        musiq_weight: float = 0.0,
        clipiqa_weight: float = 0.0,
        vgg_layers=None,
    ):
        """
        Args:
            device: 计算设备
            pixel_weight: L1 损失权重
            perceptual_weight: VGG 感知损失权重
            lpips_weight: LPIPS 损失权重
            musiq_weight: MUSIQ 无参考质量损失权重
            clipiqa_weight: CLIPIQA 无参考质量损失权重
            vgg_layers: VGG 特征层列表
        """
        super().__init__()

        self.device = device
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_weight = lpips_weight
        self.musiq_weight = musiq_weight
        self.clipiqa_weight = clipiqa_weight

        # 惰性初始化感知损失模块（在权重 > 0 时才初始化）
        self.vgg_loss = None
        self.lpips_loss = None
        self.musiq_loss = None
        self.clipiqa_loss = None
        self.vgg_layers = vgg_layers

        # 如果初始权重 > 0，立即初始化
        if perceptual_weight > 0:
            self._init_vgg_loss()
        if lpips_weight > 0:
            self._init_lpips_loss()
        if musiq_weight > 0:
            self._init_musiq_loss()
        if clipiqa_weight > 0:
            self._init_clipiqa_loss()

        logger.info(f"CombinedLoss initialized: pixel={pixel_weight}, perceptual={perceptual_weight}, "
                   f"lpips={lpips_weight}, musiq={musiq_weight}, clipiqa={clipiqa_weight}")

    def _init_vgg_loss(self):
        """初始化 VGG 感知损失"""
        if self.vgg_loss is None:
            self.vgg_loss = VGGPerceptualLoss(self.device, layers=self.vgg_layers)
            logger.info("VGG perceptual loss module initialized")

    def _init_lpips_loss(self):
        """初始化 LPIPS 损失"""
        if self.lpips_loss is None:
            self.lpips_loss = LPIPSLoss(self.device)
            logger.info("LPIPS loss module initialized")

    def _init_musiq_loss(self):
        """初始化 MUSIQ 损失"""
        if self.musiq_loss is None:
            self.musiq_loss = MUSIQLoss(self.device)
            logger.info("MUSIQ loss module initialized")

    def _init_clipiqa_loss(self):
        """初始化 CLIPIQA 损失"""
        if self.clipiqa_loss is None:
            self.clipiqa_loss = CLIPIQALoss(self.device)
            logger.info("CLIPIQA loss module initialized")

    def update_weights(self, pixel=None, perceptual=None, lpips=None, musiq=None, clipiqa=None):
        """
        更新损失权重（用于渐进训练）

        Args:
            pixel: 新的 L1 损失权重
            perceptual: 新的 VGG 感知损失权重
            lpips: 新的 LPIPS 损失权重
            musiq: 新的 MUSIQ 损失权重
            clipiqa: 新的 CLIPIQA 损失权重
        """
        if pixel is not None:
            self.pixel_weight = pixel
            logger.info(f"Updated pixel_weight to {pixel}")

        if perceptual is not None:
            self.perceptual_weight = perceptual
            if perceptual > 0:
                self._init_vgg_loss()
            logger.info(f"Updated perceptual_weight to {perceptual}")

        if lpips is not None:
            self.lpips_weight = lpips
            if lpips > 0:
                self._init_lpips_loss()
            logger.info(f"Updated lpips_weight to {lpips}")

        if musiq is not None:
            self.musiq_weight = musiq
            if musiq > 0:
                self._init_musiq_loss()
            logger.info(f"Updated musiq_weight to {musiq}")

        if clipiqa is not None:
            self.clipiqa_weight = clipiqa
            if clipiqa > 0:
                self._init_clipiqa_loss()
            logger.info(f"Updated clipiqa_weight to {clipiqa}")

    def forward(self, pred, target):
        """
        计算组合损失

        Args:
            pred: 预测图像 (B, C, H, W)，范围 [0, 1]
            target: 目标图像 (B, C, H, W)，范围 [0, 1]

        Returns:
            total_loss: 总损失值
            loss_dict: 各分量损失字典 {'pixel': x, 'perceptual': y, 'lpips': z, 'musiq': m, 'clipiqa': c, 'total': t}
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 1. L1 Loss (Pixel Loss)
        if self.pixel_weight > 0:
            pixel_loss = F.l1_loss(pred, target)
            loss_dict['pixel'] = pixel_loss.item()
            total_loss = total_loss + self.pixel_weight * pixel_loss
        else:
            loss_dict['pixel'] = 0.0

        # 2. VGG Perceptual Loss
        if self.perceptual_weight > 0 and self.vgg_loss is not None:
            perc_loss = self.vgg_loss(pred, target)
            loss_dict['perceptual'] = perc_loss.item()
            total_loss = total_loss + self.perceptual_weight * perc_loss
        else:
            loss_dict['perceptual'] = 0.0

        # 3. LPIPS Loss
        if self.lpips_weight > 0 and self.lpips_loss is not None:
            lpips_loss = self.lpips_loss(pred, target)
            loss_dict['lpips'] = lpips_loss.item()
            total_loss = total_loss + self.lpips_weight * lpips_loss
        else:
            loss_dict['lpips'] = 0.0

        # 4. MUSIQ Loss (无参考质量损失)
        if self.musiq_weight > 0 and self.musiq_loss is not None:
            musiq_loss = self.musiq_loss(pred)
            loss_dict['musiq'] = musiq_loss.item()
            total_loss = total_loss + self.musiq_weight * musiq_loss
        else:
            loss_dict['musiq'] = 0.0

        # 5. CLIPIQA Loss (无参考质量损失)
        if self.clipiqa_weight > 0 and self.clipiqa_loss is not None:
            clipiqa_loss = self.clipiqa_loss(pred)
            loss_dict['clipiqa'] = clipiqa_loss.item()
            total_loss = total_loss + self.clipiqa_weight * clipiqa_loss
        else:
            loss_dict['clipiqa'] = 0.0

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def get_current_weights(self):
        """获取当前损失权重配置"""
        return {
            'pixel': self.pixel_weight,
            'perceptual': self.perceptual_weight,
            'lpips': self.lpips_weight,
            'musiq': self.musiq_weight,
            'clipiqa': self.clipiqa_weight,
        }


if __name__ == "__main__":
    # 测试代码
    import sys

    # 检测设备
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        try:
            import torch_npu
            if torch.npu.is_available():
                device = 'npu:0'
            else:
                device = 'cpu'
        except ImportError:
            device = 'cpu'

    print(f"Testing on device: {device}")

    # 创建测试张量
    pred = torch.rand(2, 3, 128, 128).to(device)
    target = torch.rand(2, 3, 128, 128).to(device)

    # 测试 VGGPerceptualLoss
    print("\n--- Testing VGGPerceptualLoss ---")
    vgg_loss = VGGPerceptualLoss(device)
    loss = vgg_loss(pred, target)
    print(f"VGG Loss: {loss.item():.6f}")

    # 测试 LPIPSLoss
    print("\n--- Testing LPIPSLoss ---")
    lpips_loss = LPIPSLoss(device)
    loss = lpips_loss(pred, target)
    print(f"LPIPS Loss: {loss.item():.6f}")

    # 测试 CombinedLoss
    print("\n--- Testing CombinedLoss ---")
    combined_loss = CombinedLoss(
        device,
        pixel_weight=1.0,
        perceptual_weight=0.1,
        lpips_weight=0.5,
    )
    total_loss, loss_dict = combined_loss(pred, target)
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"Loss Dict: {loss_dict}")

    # 测试权重更新
    print("\n--- Testing weight update ---")
    combined_loss.update_weights(perceptual=0.2, lpips=0.3)
    total_loss, loss_dict = combined_loss(pred, target)
    print(f"After update - Total Loss: {total_loss.item():.6f}")
    print(f"Loss Dict: {loss_dict}")

    print("\n✓ All tests passed!")
