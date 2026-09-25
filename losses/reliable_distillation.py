"""B3实例可靠性引导的教师输出蒸馏。

B2把教师密度图的每个像素都视为同等可信。B3先借助车辆框在输出密度图上
建立软实例区域，再分别测量教师在每辆车附近的局部误差与翻转稳定性，最后
把实例可靠性还原成空间权重图。该权重只作用于教师输出蒸馏项，不修改GT
密度监督、Dense-FSP或余弦特征蒸馏，便于把实验增益归因到可靠性机制。

这里生成的椭圆高斯仅用于划分实例区域，不会替换固定sigma=8生成的GT密度图。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .object_reliability import (
    InstanceMaskConfig,
    build_soft_instance_regions,
    compute_instance_errors,
    rasterize_instance_values,
    reliability_from_errors,
)


SUPPORTED_RELIABILITY_MODES = frozenset(
    {"uniform", "supervised", "consistency", "combined"}
)


@dataclass(frozen=True)
class ReliabilityDistillationConfig:
    """控制B3可靠性估计和空间加权。

    ``mode``的含义：

    - ``uniform``：所有有效像素权重为1，用于验证B3训练器能退化为B2；
    - ``supervised``：使用教师相对GT的局部密度和局部计数误差；
    - ``consistency``：只使用教师原图/水平翻转图的一致性误差；
    - ``combined``：监督可靠性与一致性可靠性的几何平均，为正式B3。

    三个tau必须来自训练集或验证集的先验分析，不能根据测试集调参。
    ``background_value``让非车辆区域仍保留少量教师监督。归一化会把每张图
    有效区域内的权重均值恢复为1，使B2和B3的输出蒸馏损失量级可比。

    ``strength``是残差式可靠性强度lambda，最终权重为
    ``(1-lambda) * 1 + lambda * normalized_reliability``。lambda=0严格退化
    为B2均匀蒸馏，lambda=1对应B3-v1，介于两者之间时可避免困难车辆的
    教师监督被过度削弱。
    """

    mode: str = "combined"
    tau_map: float = 1.0
    tau_count: float = 1.0
    tau_consistency: float = 1.0
    minimum: float = 0.05
    background_value: float = 0.05
    normalize_mean: bool = True
    strength: float = 1.0
    epsilon: float = 1e-8
    instance_mask: InstanceMaskConfig = field(default_factory=InstanceMaskConfig)

    def validate(self) -> None:
        if self.mode not in SUPPORTED_RELIABILITY_MODES:
            raise ValueError(
                f"不支持的可靠性模式：{self.mode}；"
                f"可选值为{sorted(SUPPORTED_RELIABILITY_MODES)}"
            )
        if min(self.tau_map, self.tau_count, self.tau_consistency) <= 0:
            raise ValueError("tau_map、tau_count和tau_consistency必须大于0")
        if not 0.0 <= self.minimum < 1.0:
            raise ValueError("minimum必须位于[0,1)")
        if not 0.0 <= self.background_value <= 1.0:
            raise ValueError("background_value必须位于[0,1]")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength必须位于[0,1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon必须大于0")
        self.instance_mask.validate()

    @property
    def needs_consistency_view(self) -> bool:
        """当前模式是否需要额外执行一次翻转教师前向。"""

        return self.mode in {"consistency", "combined"}


def normalize_reliability_map(
    reliability_map: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """把单张空间可靠性图在有效区域内归一到均值1。

    B2输出蒸馏的有效像素权重全部为1。若B3直接使用0到1的可靠性，损失变小
    既可能来自更好的空间分配，也可能只是总梯度被整体缩小。按均值归一化后，
    B3只重新分配像素梯度，不改变每张图的总权重预算。
    """

    if reliability_map.ndim == 3 and reliability_map.shape[0] == 1:
        reliability_map = reliability_map[0]
    if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
        valid_mask = valid_mask[0]
    if reliability_map.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("reliability_map和valid_mask必须是[H,W]或[1,H,W]")
    if reliability_map.shape != valid_mask.shape:
        raise ValueError("reliability_map与valid_mask尺寸必须一致")
    if epsilon <= 0:
        raise ValueError("epsilon必须大于0")

    valid = (valid_mask > 0.5).to(dtype=reliability_map.dtype)
    valid_pixels = valid.sum()
    if float(valid_pixels.item()) <= 0:
        raise ValueError("valid_mask中没有有效像素")

    masked = reliability_map.clamp_min(0.0) * valid
    weight_sum = masked.sum()
    if float(weight_sum.item()) <= epsilon:
        # 极端情况下所有可靠性都为0。退化为均匀权重比产生无梯度更安全。
        return valid
    return masked * (valid_pixels / weight_sum.clamp_min(epsilon))


@torch.no_grad()
def build_batch_reliability_maps(
    *,
    teacher_density: torch.Tensor,
    target_density: torch.Tensor,
    boxes: Sequence[torch.Tensor],
    input_size: tuple[int, int],
    valid_mask: torch.Tensor,
    config: ReliabilityDistillationConfig,
    transformed_teacher_density: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """为一个batch构造``[B,1,H,W]``可靠性图并返回诊断统计。

    可靠性由冻结教师和标签计算，全程不建立反向图。CARPK的每张图车辆数
    不同，因此逐图建立实例区域；当前正式实验batch=1，该实现也兼容更大batch。
    """

    config.validate()
    if teacher_density.shape != target_density.shape:
        raise ValueError("teacher_density与target_density尺寸必须一致")
    if teacher_density.ndim != 4 or teacher_density.shape[1] != 1:
        raise ValueError("密度图必须是[B,1,H,W]")
    if valid_mask.shape != target_density.shape:
        raise ValueError("valid_mask尺寸必须与密度图一致")

    batch_size = teacher_density.shape[0]
    if len(boxes) != batch_size:
        raise ValueError("boxes列表长度必须等于batch大小")
    if config.needs_consistency_view:
        if transformed_teacher_density is None:
            raise ValueError(f"{config.mode}模式必须提供变换视图教师密度图")
        if transformed_teacher_density.shape != teacher_density.shape:
            raise ValueError("变换视图教师密度图尺寸必须与原教师输出一致")

    output_size = tuple(int(value) for value in target_density.shape[-2:])
    maps: list[torch.Tensor] = []
    raw_map_means: list[float] = []
    normalized_map_means: list[float] = []
    maximum_weights: list[float] = []
    object_reliability_sum = 0.0
    object_count = 0

    for index in range(batch_size):
        sample_valid = valid_mask[index, 0].float()
        sample_boxes = boxes[index].to(
            device=teacher_density.device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if config.mode == "uniform":
            # 该模式用于单元测试和消融：应严格退化为B2均匀输出蒸馏。
            selected = torch.ones(
                sample_boxes.shape[0], device=teacher_density.device
            )
            raw_map = torch.ones_like(sample_valid)
        else:
            regions = build_soft_instance_regions(
                sample_boxes,
                input_size=input_size,
                output_size=output_size,
                config=config.instance_mask,
                valid_mask=sample_valid,
            )
            transformed = None
            if transformed_teacher_density is not None:
                transformed = transformed_teacher_density[index, 0].float()
            errors = compute_instance_errors(
                teacher_density[index, 0].float(),
                target_density[index, 0].float(),
                regions,
                transformed_teacher_density=transformed,
                epsilon=config.epsilon,
            )
            reliabilities = reliability_from_errors(
                errors["local_map_mae"],
                errors["local_count_error"],
                tau_map=config.tau_map,
                tau_count=config.tau_count,
                view_consistency_error=(
                    errors["view_consistency_mae"]
                    if config.needs_consistency_view
                    else None
                ),
                tau_consistency=(
                    config.tau_consistency
                    if config.needs_consistency_view
                    else None
                ),
                minimum=config.minimum,
            )
            selected = reliabilities[config.mode]
            raw_map = rasterize_instance_values(
                regions,
                selected,
                background_value=config.background_value,
            )

        if selected.numel() > 0:
            object_reliability_sum += float(selected.sum().item())
            object_count += int(selected.numel())

        valid = (sample_valid > 0.5).to(raw_map.dtype)
        valid_pixels = valid.sum().clamp_min(1.0)
        raw_map_means.append(float((raw_map * valid).sum().item() / valid_pixels.item()))

        if config.normalize_mean:
            spatial_map = normalize_reliability_map(
                raw_map,
                sample_valid,
                epsilon=config.epsilon,
            )
        else:
            spatial_map = raw_map * valid

        # B3-v2不再用可靠性图完全替换B2的均匀权重，而是通过lambda做
        # 残差式混合。两部分在有效区域均值都为1，因此开启归一化时，
        # 混合后仍保持均值1，不会改变输出蒸馏的整体梯度预算。
        spatial_map = (
            (1.0 - config.strength) * valid
            + config.strength * spatial_map
        )
        normalized_map_means.append(
            float((spatial_map * valid).sum().item() / valid_pixels.item())
        )
        maximum_weights.append(float(spatial_map.max().item()))
        maps.append(spatial_map)

    reliability_maps = torch.stack(maps, dim=0).unsqueeze(1).float()
    statistics = {
        "object_count": float(object_count),
        "object_reliability_mean": (
            object_reliability_sum / object_count if object_count else 0.0
        ),
        "raw_map_mean": sum(raw_map_means) / max(len(raw_map_means), 1),
        "normalized_map_mean": (
            sum(normalized_map_means) / max(len(normalized_map_means), 1)
        ),
        "maximum_normalized_weight": max(maximum_weights, default=0.0),
    }
    return reliability_maps, statistics


__all__ = [
    "ReliabilityDistillationConfig",
    "SUPPORTED_RELIABILITY_MODES",
    "build_batch_reliability_maps",
    "normalize_reliability_map",
]
