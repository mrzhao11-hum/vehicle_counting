"""密度图回归使用的图像级计数约束。

密度图的积分（离散实现中就是所有像素求和）应等于图像中的真实目标数。
逐像素MSE主要约束密度分布的形状，但不一定能消除整幅图的系统性少计或多计。
本模块额外约束密度图总质量，同时保留原有密度图监督作为主要定位约束。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RelativeCountSmoothL1Loss(nn.Module):
    """计算预测密度积分与真实计数之间的相对Smooth L1损失。

    对第i张图，先计算相对计数残差：

        r_i = (sum(M_i * D_i) - C_i) / (abs(C_i) + offset)

    其中D是学生密度图，M是有效区域掩码，C是真实车辆数。最终对r使用
    Smooth L1并在batch维取平均。分母按真实计数归一化，使车辆数量不同的
    图像具有可比较的优化尺度；offset同时保护C=0的空场景。

    ``beta``控制二次区间宽度。这里推荐0.1：相对误差小于10%时使用平滑
    二次惩罚，大于10%时近似线性惩罚，降低异常样本对训练的冲击。
    """

    def __init__(self, *, offset: float = 1.0, beta: float = 0.1) -> None:
        super().__init__()
        if offset <= 0.0:
            raise ValueError("count loss的offset必须大于0")
        if beta <= 0.0:
            raise ValueError("count loss的beta必须大于0")
        self.offset = float(offset)
        self.beta = float(beta)

    def forward(
        self,
        prediction: torch.Tensor,
        target_count: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.ndim < 2:
            raise ValueError(
                f"prediction至少需要batch和特征两个维度，实际为{tuple(prediction.shape)}"
            )

        # 始终用FP32完成全图求和。即使以后启用AMP，也避免大量像素累加时
        # FP16舍入误差直接污染计数监督。
        density = prediction.float()
        if valid_mask is not None:
            mask = valid_mask.float()
            try:
                density = density * mask
            except RuntimeError as error:
                raise ValueError(
                    "valid_mask必须能够广播到prediction，"
                    f"实际为{tuple(mask.shape)}与{tuple(prediction.shape)}"
                ) from error

        batch_size = int(prediction.shape[0])
        target = target_count.float().reshape(-1)
        if target.numel() != batch_size:
            raise ValueError(
                "target_count每张图必须恰好有一个计数，"
                f"batch={batch_size}，实际元素数={target.numel()}"
            )

        predicted_count = density.reshape(batch_size, -1).sum(dim=1)
        relative_error = (predicted_count - target) / (
            target.abs() + self.offset
        )
        return F.smooth_l1_loss(
            relative_error,
            torch.zeros_like(relative_error),
            reduction="mean",
            beta=self.beta,
        )


__all__ = ["RelativeCountSmoothL1Loss"]
