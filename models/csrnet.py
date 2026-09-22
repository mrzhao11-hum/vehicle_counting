"""CSRNet教师网络和轻量学生网络。

本文件保留CSRNet最核心的设计：VGG16前端提取图像特征，后端使用
扩张率为2的空洞卷积扩大感受野，并在1/8分辨率输出单通道密度图。

模型默认不在输出端添加ReLU。这与原始CSRNet/SKT基线保持一致，也避免
随机初始化阶段因为负输出被ReLU截断而产生零梯度。密度图可视化时可以只
显示非负部分，但训练和计数指标必须始终使用模型的原始输出。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


# 三次最大池化使输出步长为8。第四组VGG卷积被保留，但不再继续池化。
FRONTEND_CHANNELS: tuple[int | str, ...] = (
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
)
BACKEND_CHANNELS: tuple[int, ...] = (512, 512, 512, 256, 128, 64)

# SKT从教师/学生的六个对应位置提取特征。这里记录完整教师在这些位置的
# 通道数，后续B2可以据此建立1x1特征对齐层，而不把对齐层混入B1部署模型。
TEACHER_FEATURE_CHANNELS: tuple[int, ...] = (64, 64, 128, 256, 512, 256)


def make_layers(
    configuration: tuple[int | str, ...],
    *,
    input_channels: int,
    dilation: int = 1,
) -> nn.Sequential:
    """根据通道配置创建卷积网络。

    ``"M"``代表最大池化，其余整数代表卷积输出通道。卷积padding与
    dilation相同，因此卷积本身不会改变特征图宽高。
    """

    layers: list[nn.Module] = []
    current_channels = input_channels
    for value in configuration:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            continue

        output_channels = int(value)
        layers.extend(
            [
                nn.Conv2d(
                    current_channels,
                    output_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.ReLU(inplace=True),
            ]
        )
        current_channels = output_channels
    return nn.Sequential(*layers)


class CSRNetTeacher(nn.Module):
    """完整CSRNet教师模型。

    参数：
        pretrained_frontend: 是否加载ImageNet预训练VGG16的前十个卷积层。
            从断点恢复或只做评估时应传入False，因为checkpoint会覆盖参数。

    ``return_features=True``时额外返回六个中间特征。固定核教师训练暂时不
    使用这些特征，但保留此接口可供后续SKT蒸馏使用，避免通过易失效的hook
    从模型内部抓取结果。
    """

    output_stride = 8

    # 这些位置与原SKT教师代码中的特征hook位置一致。
    _frontend_feature_indices = frozenset({1, 4, 9, 16})
    _backend_feature_indices = frozenset({1, 7})

    def __init__(self, pretrained_frontend: bool = True) -> None:
        super().__init__()
        self.frontend = make_layers(
            FRONTEND_CHANNELS, input_channels=3, dilation=1
        )
        self.backend = make_layers(
            BACKEND_CHANNELS, input_channels=512, dilation=2
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()
        if pretrained_frontend:
            self._load_imagenet_frontend()

    def _initialize_weights(self) -> None:
        """使用CSRNet参考实现的初始化方式初始化所有卷积层。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _load_imagenet_frontend(self) -> None:
        """把ImageNet VGG16对应卷积参数复制到CSRNet前端。"""

        try:
            source = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception as error:  # pragma: no cover - 取决于服务器网络与缓存
            raise RuntimeError(
                "无法加载ImageNet预训练VGG16。请检查服务器网络/torch缓存，"
                "或在配置中将model.pretrained_frontend设为false。"
            ) from error

        source_convolutions = [m for m in source if isinstance(m, nn.Conv2d)]
        target_convolutions = [
            m for m in self.frontend if isinstance(m, nn.Conv2d)
        ]
        if len(target_convolutions) > len(source_convolutions):
            raise RuntimeError("VGG16卷积层数量不足，无法初始化CSRNet前端")

        with torch.no_grad():
            for target, pretrained in zip(
                target_convolutions, source_convolutions, strict=False
            ):
                target.weight.copy_(pretrained.weight)
                if target.bias is not None and pretrained.bias is not None:
                    target.bias.copy_(pretrained.bias)

    def forward(
        self, image: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """预测1/8分辨率密度图，可选返回蒸馏所需中间特征。"""

        features: list[torch.Tensor] = []
        feature = image
        for index, layer in enumerate(self.frontend):
            feature = layer(feature)
            if return_features and index in self._frontend_feature_indices:
                features.append(feature)

        for index, layer in enumerate(self.backend):
            feature = layer(feature)
            if return_features and index in self._backend_feature_indices:
                features.append(feature)

        density = self.output_layer(feature)
        if return_features:
            return {"density": density, "features": features}
        return density


class CSRNetStudent(nn.Module):
    """通道压缩后的轻量CSRNet学生网络。

    ``channel_ratio=4``对应SKT论文中的1/4-CSRNet：除最终单通道输出层外，
    每个卷积层的通道数均为完整CSRNet的四分之一。卷积参数量主要与输入、
    输出通道数的乘积成正比，因此核心参数量和计算量约为教师的1/16。

    B1只训练本类定义的核心计数网络，不包含知识蒸馏专用的1x1通道对齐层。
    当``return_features=True``时返回六个原始学生特征；B2会在损失模块中单独
    建立对齐层。这样B1与B2最终部署的是同一个学生主干，参数量和推理速度的
    比较不会被只在训练阶段需要的蒸馏分支干扰。
    """

    output_stride = 8
    _frontend_feature_indices = CSRNetTeacher._frontend_feature_indices
    _backend_feature_indices = CSRNetTeacher._backend_feature_indices

    def __init__(self, channel_ratio: int = 4) -> None:
        super().__init__()
        if channel_ratio <= 0:
            raise ValueError("channel_ratio必须为正整数")

        teacher_channels = (64, 128, 256, 512)
        if any(channel % channel_ratio != 0 for channel in teacher_channels):
            raise ValueError(
                "当前实现要求64、128、256、512均可被channel_ratio整除"
            )

        c1, c2, c3, c4 = (
            channel // channel_ratio for channel in teacher_channels
        )
        self.channel_ratio = channel_ratio
        self.frontend_channels: tuple[int | str, ...] = (
            c1,
            c1,
            "M",
            c2,
            c2,
            "M",
            c3,
            c3,
            c3,
            "M",
            c4,
            c4,
            c4,
        )
        self.backend_channels: tuple[int, ...] = (c4, c4, c4, c3, c2, c1)
        self.feature_channels: tuple[int, ...] = (c1, c1, c2, c3, c4, c3)

        self.frontend = make_layers(
            self.frontend_channels, input_channels=3, dilation=1
        )
        self.backend = make_layers(
            self.backend_channels, input_channels=c4, dilation=2
        )
        self.output_layer = nn.Conv2d(c1, 1, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """采用原SKT学生实现使用的Kaiming正态初始化。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self, image: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """预测密度图，并可选返回后续SKT使用的六个原始特征。"""

        features: list[torch.Tensor] = []
        feature = image
        for index, layer in enumerate(self.frontend):
            feature = layer(feature)
            if return_features and index in self._frontend_feature_indices:
                features.append(feature)

        for index, layer in enumerate(self.backend):
            feature = layer(feature)
            if return_features and index in self._backend_feature_indices:
                features.append(feature)

        density = self.output_layer(feature)
        if return_features:
            if len(features) != len(self.feature_channels):
                raise RuntimeError(
                    f"预期提取{len(self.feature_channels)}个特征，实际得到{len(features)}个"
                )
            return {"density": density, "features": features}
        return density


def count_trainable_parameters(model: nn.Module) -> int:
    """返回需要梯度的参数量，用于实验日志和论文效率统计。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
