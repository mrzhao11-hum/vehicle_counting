"""密度图监督损失。

固定核基线的weight和valid_mask都为1，但训练入口仍显式传入二者。这样后续
切换到小目标权重或UAVDT忽略区域时，不需要偷偷修改训练循环，只需更换标签
目录与实验配置。
"""

from __future__ import annotations

import torch
from torch import nn


class DensityMSELoss(nn.Module):
    """支持空间权重和有效区域掩码的密度图均方误差。

    ``batch_mean_sum``先对每张图片的所有有效像素求和，再对batch取平均。
    当batch=1、权重和掩码均为1时，它与原CSRNet/SKT代码中的
    ``MSELoss(reduction="sum")``一致。

    ``weighted_mean``按有效权重总和归一化。它更适合后续权重消融，但会改变
    损失与梯度量级，所以不能在不同对照实验之间随意切换。
    """

    SUPPORTED_REDUCTIONS = {"batch_mean_sum", "weighted_mean"}

    def __init__(self, reduction: str = "batch_mean_sum") -> None:
        super().__init__()
        if reduction not in self.SUPPORTED_REDUCTIONS:
            raise ValueError(
                f"不支持的density loss reduction：{reduction}；"
                f"可选值为{sorted(self.SUPPORTED_REDUCTIONS)}"
            )
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"预测尺寸{tuple(prediction.shape)}与GT尺寸{tuple(target.shape)}不一致"
            )
        if weight is not None and weight.shape != target.shape:
            raise ValueError("weight尺寸必须与密度图完全一致")
        if valid_mask is not None and valid_mask.shape != target.shape:
            raise ValueError("valid_mask尺寸必须与密度图完全一致")

        effective_weight = torch.ones_like(target)
        if weight is not None:
            effective_weight = effective_weight * weight
        if valid_mask is not None:
            effective_weight = effective_weight * valid_mask

        weighted_squared_error = (prediction - target).square() * effective_weight
        if self.reduction == "batch_mean_sum":
            return weighted_squared_error.sum() / max(prediction.shape[0], 1)

        denominator = effective_weight.sum().clamp_min(1e-12)
        return weighted_squared_error.sum() / denominator
