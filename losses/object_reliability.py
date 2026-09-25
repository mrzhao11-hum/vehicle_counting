"""车辆实例级教师可靠性分析与后续蒸馏共享的基础函数。

本模块只负责可微分张量计算，不读写文件，也不依赖具体模型。当前分析脚本
使用它回答两个问题：

1. 教师在不同车辆实例附近的误差是否存在明显差异；
2. 教师在几何等价视图之间是否稳定，以及这种稳定性是否能够反映真实误差。

CARPK 的 bbox 位于原图坐标系，而 CSRNet 输出为原图的 1/8 分辨率。因此
所有实例区域都在这里显式缩放到密度图坐标系，避免在分析脚本和训练器中
重复实现坐标换算。后续可靠性蒸馏也可以直接复用本文件。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InstanceMaskConfig:
    """控制 bbox 到软实例区域的转换。

    参数：
        sigma_scale: 高斯标准差相对 bbox 宽、高的比例。例如 0.35 表示
            ``sigma_x = 0.35 * bbox_width``。它控制实例区域覆盖范围，但不会
            改变训练使用的 GT 密度标签。
        sigma_min: 输出密度图坐标系中的最小标准差，防止远处或极小车辆
            缩放后只覆盖一个采样点。
        sigma_max: 输出密度图坐标系中的最大标准差，防止大框覆盖过多邻车。
        truncate: 在多少个标准差处截断高斯。截断后背景不会被远距离高斯
            尾部错误地划入任意车辆实例。
        epsilon: 除法和归一化时使用的数值稳定常数。
    """

    sigma_scale: float = 0.35
    sigma_min: float = 0.75
    sigma_max: float = 4.0
    truncate: float = 2.5
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.sigma_scale <= 0:
            raise ValueError("sigma_scale必须大于0")
        if self.sigma_min <= 0:
            raise ValueError("sigma_min必须大于0")
        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max不能小于sigma_min")
        if self.truncate <= 0:
            raise ValueError("truncate必须大于0")
        if self.epsilon <= 0:
            raise ValueError("epsilon必须大于0")


@dataclass
class InstanceRegions:
    """一张图像在密度图坐标系中的实例区域结果。

    ``ownership``的形状为 ``[N, H, W]``。在多个车辆高斯区域重叠时，
    同一像素处所有实例 ownership 之和为 1；背景像素处为 0。这样能够
    避免密集车辆重叠区域的误差被重复统计。
    """

    ownership: torch.Tensor
    centers: torch.Tensor
    scaled_boxes: torch.Tensor
    foreground: torch.Tensor


def _validate_density(name: str, density: torch.Tensor) -> torch.Tensor:
    """将 ``[1,H,W]`` 或 ``[H,W]`` 密度图统一为 ``[H,W]``。"""

    if density.ndim == 3 and density.shape[0] == 1:
        density = density[0]
    if density.ndim != 2:
        raise ValueError(f"{name}必须是[H,W]或[1,H,W]，实际为{tuple(density.shape)}")
    return density


def build_soft_instance_regions(
    boxes: torch.Tensor,
    *,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    config: InstanceMaskConfig | None = None,
    valid_mask: torch.Tensor | None = None,
) -> InstanceRegions:
    """由原图 bbox 构造输出密度图上的软实例归属区域。

    参数：
        boxes: 原图坐标系中的 ``[N,4+]``。前四列必须依次为
            ``x1,y1,x2,y2``，其余列可保存类别等元数据。CARPK 的实际格式
            是 ``[x1,y1,x2,y2,class_id]``，第五列不会参与区域计算。
        input_size: 原图 ``(height, width)``。
        output_size: 密度图 ``(height, width)``。
        config: 实例高斯区域配置。
        valid_mask: 可选的密度图有效区域。UAVDT 等数据集存在忽略区域时，
            无效位置不会参与实例误差；CARPK 中通常全部为 1。

    注意：这里生成的是“分析/蒸馏区域”，不是新的 GT 密度图。每个实例
    高斯只用于划分教师误差归属，因此不会改变现有固定 sigma8 基线。
    """

    resolved = config or InstanceMaskConfig()
    resolved.validate()

    if boxes.ndim != 2 or boxes.shape[-1] < 4:
        raise ValueError(f"boxes必须是[N,4+]，实际为{tuple(boxes.shape)}")

    input_height, input_width = input_size
    output_height, output_width = output_size
    if min(input_height, input_width, output_height, output_width) <= 0:
        raise ValueError("输入和输出尺寸必须为正数")

    device = boxes.device
    dtype = boxes.dtype if boxes.is_floating_point() else torch.float32
    # CARPK 的 bbox 含第五列 class_id。可靠性分析只使用前四列坐标，
    # 显式切片也让该函数兼容未来附带置信度或遮挡标记的数据集。
    coordinate_boxes = boxes[:, :4].to(dtype=dtype)

    if boxes.shape[0] == 0:
        empty_regions = torch.empty(
            (0, output_height, output_width), device=device, dtype=dtype
        )
        return InstanceRegions(
            ownership=empty_regions,
            centers=torch.empty((0, 2), device=device, dtype=dtype),
            scaled_boxes=torch.empty((0, 4), device=device, dtype=dtype),
            foreground=torch.zeros(
                (output_height, output_width), device=device, dtype=dtype
            ),
        )

    scale_x = float(output_width) / float(input_width)
    scale_y = float(output_height) / float(input_height)
    scale = coordinate_boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])
    scaled_boxes = coordinate_boxes * scale

    x1, y1, x2, y2 = scaled_boxes.unbind(dim=1)
    widths = (x2 - x1).clamp_min(resolved.epsilon)
    heights = (y2 - y1).clamp_min(resolved.epsilon)
    centers = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dim=1)

    sigma_x = (widths * resolved.sigma_scale).clamp(
        min=resolved.sigma_min, max=resolved.sigma_max
    )
    sigma_y = (heights * resolved.sigma_scale).clamp(
        min=resolved.sigma_min, max=resolved.sigma_max
    )

    grid_y, grid_x = torch.meshgrid(
        torch.arange(output_height, device=device, dtype=dtype),
        torch.arange(output_width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = grid_x.unsqueeze(0) - centers[:, 0, None, None]
    dy = grid_y.unsqueeze(0) - centers[:, 1, None, None]

    normalized_distance = (
        dx.square() / (2.0 * sigma_x[:, None, None].square())
        + dy.square() / (2.0 * sigma_y[:, None, None].square())
    )
    masks = torch.exp(-normalized_distance)

    # 截断高斯尾部非常重要。若不截断，理论上每个高斯在整张图上都非零，
    # 最终所有背景像素都会被错误地分配给某辆车。
    inside_window = (
        (dx.abs() <= resolved.truncate * sigma_x[:, None, None])
        & (dy.abs() <= resolved.truncate * sigma_y[:, None, None])
    )
    masks = masks * inside_window.to(dtype)

    if valid_mask is not None:
        valid = _validate_density("valid_mask", valid_mask).to(device=device, dtype=dtype)
        if valid.shape != (output_height, output_width):
            raise ValueError(
                f"valid_mask尺寸{tuple(valid.shape)}与输出尺寸{output_size}不一致"
            )
        masks = masks * (valid > 0.5).to(dtype).unsqueeze(0)

    support = masks.sum(dim=0)
    foreground = (support > resolved.epsilon).to(dtype)
    ownership = torch.where(
        support.unsqueeze(0) > resolved.epsilon,
        masks / support.clamp_min(resolved.epsilon).unsqueeze(0),
        torch.zeros_like(masks),
    )
    return InstanceRegions(
        ownership=ownership,
        centers=centers,
        scaled_boxes=scaled_boxes,
        foreground=foreground,
    )


def compute_instance_errors(
    teacher_density: torch.Tensor,
    target_density: torch.Tensor,
    regions: InstanceRegions,
    *,
    transformed_teacher_density: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """计算每辆车对应的教师误差与视图一致性误差。

    ``transformed_teacher_density``应当已经变换回原视图。例如分析水平翻转
    稳定性时，应先把教师对翻转图像的输出再次水平翻转回来，再传给本函数。

    返回值中的局部质量不是假设每辆车严格等于 1，而是使用同一个软归属
    区域分别积分 GT 和教师输出。这样即使相邻车辆高斯发生重叠，比较口径
    仍然一致。
    """

    teacher = _validate_density("teacher_density", teacher_density).float()
    target = _validate_density("target_density", target_density).float()
    if teacher.shape != target.shape:
        raise ValueError(
            f"教师密度尺寸{tuple(teacher.shape)}与GT尺寸{tuple(target.shape)}不一致"
        )

    ownership = regions.ownership.float()
    if ownership.shape[-2:] != teacher.shape:
        raise ValueError(
            f"实例区域尺寸{tuple(ownership.shape[-2:])}与密度图{tuple(teacher.shape)}不一致"
        )

    object_count = ownership.shape[0]
    if object_count == 0:
        empty = teacher.new_empty((0,))
        return {
            "region_mass": empty,
            "teacher_local_mass": empty,
            "target_local_mass": empty,
            "local_count_error": empty,
            "local_map_mae": empty,
            "centroid_error": empty,
            "view_consistency_mae": empty,
        }

    region_mass = ownership.sum(dim=(1, 2)).clamp_min(epsilon)
    residual = (teacher - target).abs()
    local_map_mae = (ownership * residual.unsqueeze(0)).sum(dim=(1, 2)) / region_mass

    teacher_local_mass = (ownership * teacher.unsqueeze(0)).sum(dim=(1, 2))
    target_local_mass = (ownership * target.unsqueeze(0)).sum(dim=(1, 2))
    local_count_error = (teacher_local_mass - target_local_mass).abs()

    # 质心仅用于诊断教师响应是否发生空间偏移。预测密度可能含负值，所以
    # 质心计算只使用非负部分；当局部没有正质量时，将误差设为实例区域对角线，
    # 表示该实例在教师预测中基本缺失。
    height, width = teacher.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=teacher.device, dtype=teacher.dtype),
        torch.arange(width, device=teacher.device, dtype=teacher.dtype),
        indexing="ij",
    )
    teacher_weights = ownership * teacher.clamp_min(0).unsqueeze(0)
    target_weights = ownership * target.clamp_min(0).unsqueeze(0)
    teacher_positive_mass = teacher_weights.sum(dim=(1, 2))
    target_positive_mass = target_weights.sum(dim=(1, 2))

    teacher_centroid_x = (teacher_weights * grid_x).sum(dim=(1, 2)) / teacher_positive_mass.clamp_min(epsilon)
    teacher_centroid_y = (teacher_weights * grid_y).sum(dim=(1, 2)) / teacher_positive_mass.clamp_min(epsilon)
    target_centroid_x = (target_weights * grid_x).sum(dim=(1, 2)) / target_positive_mass.clamp_min(epsilon)
    target_centroid_y = (target_weights * grid_y).sum(dim=(1, 2)) / target_positive_mass.clamp_min(epsilon)
    centroid_error = torch.sqrt(
        (teacher_centroid_x - target_centroid_x).square()
        + (teacher_centroid_y - target_centroid_y).square()
    )
    box_width = (regions.scaled_boxes[:, 2] - regions.scaled_boxes[:, 0]).clamp_min(1.0)
    box_height = (regions.scaled_boxes[:, 3] - regions.scaled_boxes[:, 1]).clamp_min(1.0)
    missing_penalty = torch.sqrt(box_width.square() + box_height.square())
    centroid_error = torch.where(
        (teacher_positive_mass > epsilon) & (target_positive_mass > epsilon),
        centroid_error,
        missing_penalty,
    )

    if transformed_teacher_density is None:
        view_consistency_mae = torch.full_like(local_map_mae, float("nan"))
    else:
        transformed = _validate_density(
            "transformed_teacher_density", transformed_teacher_density
        ).float()
        if transformed.shape != teacher.shape:
            raise ValueError(
                "变换视图教师密度尺寸与原视图教师密度尺寸不一致"
            )
        view_residual = (teacher - transformed).abs()
        view_consistency_mae = (
            ownership * view_residual.unsqueeze(0)
        ).sum(dim=(1, 2)) / region_mass

    return {
        "region_mass": region_mass,
        "teacher_local_mass": teacher_local_mass,
        "target_local_mass": target_local_mass,
        "local_count_error": local_count_error,
        "local_map_mae": local_map_mae,
        "centroid_error": centroid_error,
        "view_consistency_mae": view_consistency_mae,
    }


def reliability_from_errors(
    local_map_error: torch.Tensor,
    local_count_error: torch.Tensor,
    *,
    tau_map: float,
    tau_count: float,
    view_consistency_error: torch.Tensor | None = None,
    tau_consistency: float | None = None,
    minimum: float = 0.0,
) -> dict[str, torch.Tensor]:
    """把不同量纲的实例误差转换为 ``[0,1]`` 可靠性。

    两项监督误差先分别归一化再取指数。前面的 ``0.5`` 使两项都等于各自
    tau 时，监督可靠性约为 ``exp(-1)``，避免权重过快塌缩到 0。

    视图一致性不依赖 GT，可作为独立可靠性信号。组合时使用几何平均，既
    要求教师相对 GT 正确，也要求教师对等价视图稳定，同时不会像直接相乘
    那样让权重整体过小。
    """

    if tau_map <= 0 or tau_count <= 0:
        raise ValueError("tau_map和tau_count必须大于0")
    if not 0.0 <= minimum < 1.0:
        raise ValueError("minimum必须位于[0,1)")

    supervised = torch.exp(
        -0.5
        * (
            local_map_error / float(tau_map)
            + local_count_error / float(tau_count)
        )
    )

    if view_consistency_error is None:
        consistency = torch.ones_like(supervised)
    else:
        if tau_consistency is None or tau_consistency <= 0:
            raise ValueError("提供视图一致性误差时tau_consistency必须大于0")
        consistency = torch.exp(
            -view_consistency_error / float(tau_consistency)
        )

    combined = torch.sqrt(supervised * consistency)
    if minimum > 0:
        supervised = minimum + (1.0 - minimum) * supervised
        consistency = minimum + (1.0 - minimum) * consistency
        combined = minimum + (1.0 - minimum) * combined

    return {
        "supervised": supervised.clamp(0.0, 1.0),
        "consistency": consistency.clamp(0.0, 1.0),
        "combined": combined.clamp(0.0, 1.0),
    }


def rasterize_instance_values(
    regions: InstanceRegions,
    values: torch.Tensor,
    *,
    background_value: float = 0.0,
) -> torch.Tensor:
    """将每个实例的标量可靠性还原为空间可靠性图。

    重叠区域使用 ownership 加权平均；非车辆区域使用 ``background_value``。
    当前分析建议背景设为 0，以便图像直观显示车辆可靠性。后续蒸馏时可设
    为 0.05--0.1，使学生仍能少量学习教师的背景抑制能力。
    """

    if values.ndim != 1 or values.shape[0] != regions.ownership.shape[0]:
        raise ValueError("values必须是一维Tensor，且长度与实例数一致")
    if not 0.0 <= background_value <= 1.0:
        raise ValueError("background_value必须位于[0,1]")

    if values.numel() == 0:
        return torch.full_like(regions.foreground, float(background_value))

    foreground_values = (
        regions.ownership * values[:, None, None].to(regions.ownership.dtype)
    ).sum(dim=0)
    result = foreground_values + (1.0 - regions.foreground) * float(background_value)
    return result.clamp(0.0, 1.0)


__all__ = [
    "InstanceMaskConfig",
    "InstanceRegions",
    "build_soft_instance_regions",
    "compute_instance_errors",
    "rasterize_instance_values",
    "reliability_from_errors",
]
