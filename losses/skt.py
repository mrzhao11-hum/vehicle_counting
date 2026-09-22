"""SKT结构化知识迁移使用的特征适配与蒸馏损失。

本实现对应原SKT公开代码的三个知识来源：教师输出密度图、Dense-FSP跨层
结构关系和逐位置通道余弦关系。学生通道只有教师的四分之一，因此先使用
训练期1x1卷积把六层学生特征映射到教师通道数。适配层不会并入学生主干，
部署和复杂度统计仍然只使用``CSRNetStudent``。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _require_feature_lists(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
) -> None:
    if len(student_features) != len(teacher_features):
        raise ValueError(
            f"学生有{len(student_features)}层特征，教师有{len(teacher_features)}层"
        )
    if not student_features:
        raise ValueError("蒸馏特征列表不能为空")


class SKTFeatureAdapters(nn.Module):
    """使用1x1卷积和ReLU把学生特征通道映射到教师通道。

    原SKT把这些变换层写在学生类内部。这里把它们作为独立训练模块，既保持
    相同的梯度路径，也保证B1和B2最终部署的学生主干参数量完全一致。
    """

    def __init__(
        self,
        student_channels: Sequence[int],
        teacher_channels: Sequence[int],
    ) -> None:
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError("学生和教师特征通道列表长度必须一致")
        if not student_channels:
            raise ValueError("特征通道列表不能为空")

        self.student_channels = tuple(int(value) for value in student_channels)
        self.teacher_channels = tuple(int(value) for value in teacher_channels)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(student_channel, teacher_channel, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            for student_channel, teacher_channel in zip(
                self.student_channels,
                self.teacher_channels,
                strict=True,
            )
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # 与原SKT学生及其feature_transform使用相同的Kaiming初始化。
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.adapters):
            raise ValueError(
                f"预期{len(self.adapters)}层学生特征，实际得到{len(features)}层"
            )

        aligned: list[torch.Tensor] = []
        for index, (feature, adapter) in enumerate(
            zip(features, self.adapters, strict=True)
        ):
            if feature.ndim != 4:
                raise ValueError(f"第{index}层学生特征必须是NCHW四维Tensor")
            if feature.shape[1] != self.student_channels[index]:
                raise ValueError(
                    f"第{index}层学生特征通道为{feature.shape[1]}，"
                    f"预期{self.student_channels[index]}"
                )
            aligned.append(adapter(feature))
        return aligned


def scale_features_for_fsp(
    features: Sequence[torch.Tensor],
    scales: Sequence[int] = (3, 2, 1),
) -> list[torch.Tensor]:
    """把前三层高分辨率特征池化到1/8尺度，复现SKT的scale_process。

    ``scales=(3, 2, 1)``分别使用8、4、2倍最大池化；后三层已经处于
    CSRNet的1/8输出尺度，因此保持不变。
    """

    processed: list[torch.Tensor] = []
    for index, feature in enumerate(features):
        if index < len(scales):
            exponent = int(scales[index])
            if exponent < 0:
                raise ValueError("FSP池化尺度指数不能为负数")
            ratio = 2**exponent
            feature = F.max_pool2d(
                feature,
                kernel_size=ratio,
                stride=ratio,
                ceil_mode=True,
            )
        processed.append(feature)

    spatial_shapes = {tuple(feature.shape[-2:]) for feature in processed}
    if len(spatial_shapes) != 1:
        raise ValueError(f"FSP特征空间尺寸没有对齐：{sorted(spatial_shapes)}")
    return processed


def _fsp_matrix(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """计算一对特征的归一化跨层Gram矩阵，支持任意batch大小。"""

    if first.shape[0] != second.shape[0] or first.shape[-2:] != second.shape[-2:]:
        raise ValueError("计算FSP的两层特征必须具有相同batch和空间尺寸")

    first = F.instance_norm(first)
    second = F.instance_norm(second)
    batch, _, height, width = first.shape
    first_flat = first.flatten(start_dim=2)
    second_flat = second.flatten(start_dim=2)
    return torch.bmm(first_flat, second_flat.transpose(1, 2)) / (height * width)


def dense_fsp_loss(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
    *,
    scales: Sequence[int] = (3, 2, 1),
) -> torch.Tensor:
    """计算输入特征所有两两组合的Dense-FSP平方误差。

    正式B2会传入六个中间特征和最终密度输出，共产生21个关系矩阵。每个
    矩阵先按样本求元素平方误差和，再对batch取平均；在当前batch=1实验中
    与原SKT的sum-MSE完全一致。
    """

    _require_feature_lists(student_features, teacher_features)
    student_scaled = scale_features_for_fsp(student_features, scales)
    teacher_scaled = scale_features_for_fsp(teacher_features, scales)
    total = student_scaled[0].new_zeros(())

    for first_index in range(len(student_scaled)):
        for second_index in range(first_index + 1, len(student_scaled)):
            student_matrix = _fsp_matrix(
                student_scaled[first_index], student_scaled[second_index]
            )
            teacher_matrix = _fsp_matrix(
                teacher_scaled[first_index], teacher_scaled[second_index]
            )
            if student_matrix.shape != teacher_matrix.shape:
                raise ValueError(
                    "学生和教师FSP矩阵尺寸不一致："
                    f"{student_matrix.shape} vs {teacher_matrix.shape}"
                )
            per_sample = (student_matrix - teacher_matrix).square().flatten(1).sum(1)
            total = total + per_sample.mean()
    return total


def cosine_feature_loss(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
) -> torch.Tensor:
    """计算六层对应特征在每个空间位置上的通道余弦距离和。"""

    _require_feature_lists(student_features, teacher_features)
    total = student_features[0].new_zeros(())
    for index, (student, teacher) in enumerate(
        zip(student_features, teacher_features, strict=True)
    ):
        if student.shape != teacher.shape:
            raise ValueError(
                f"第{index}层余弦特征尺寸不一致：{student.shape} vs {teacher.shape}"
            )
        distance = 1.0 - F.cosine_similarity(student, teacher, dim=1)
        total = total + distance.flatten(1).sum(1).mean()
    return total


def batch_mean_sum_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """先对每张样本求平方误差和，再对batch取平均。"""

    if prediction.shape != target.shape:
        raise ValueError(
            f"蒸馏Tensor尺寸不一致：{prediction.shape} vs {target.shape}"
        )
    return (prediction - target).square().flatten(1).sum(1).mean()


__all__ = [
    "SKTFeatureAdapters",
    "batch_mean_sum_mse",
    "cosine_feature_loss",
    "dense_fsp_loss",
    "scale_features_for_fsp",
]
