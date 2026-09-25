"""B3实例可靠性引导SKT蒸馏的单epoch训练循环。"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from itertools import chain

import torch
from torch import nn

from losses import (
    ReliabilityDistillationConfig,
    SKTFeatureAdapters,
    build_batch_reliability_maps,
    cosine_feature_loss,
    dense_fsp_loss,
)

from .common import AverageMeter, CountingAccumulator, extract_density_output


def _feature_output(
    output: object, model_name: str
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """校验教师/学生带中间特征的统一输出格式。"""

    if not isinstance(output, dict):
        raise TypeError(f"{model_name}启用return_features后必须返回字典")
    density = output.get("density")
    features = output.get("features")
    if not isinstance(density, torch.Tensor):
        raise TypeError(f"{model_name}输出缺少density Tensor")
    if not isinstance(features, list) or not all(
        isinstance(feature, torch.Tensor) for feature in features
    ):
        raise TypeError(f"{model_name}输出缺少Tensor特征列表")
    return density, features


def train_reliable_skt_one_epoch(
    *,
    teacher: nn.Module,
    student: nn.Module,
    adapters: SKTFeatureAdapters,
    loader: torch.utils.data.DataLoader,
    density_criterion: Callable[..., torch.Tensor],
    count_criterion: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    print_frequency: int,
    gt_density_weight: float,
    output_distillation_weight: float,
    fsp_weight: float,
    cosine_weight: float,
    count_loss_weight: float,
    fsp_scales: Sequence[int],
    reliability_config: ReliabilityDistillationConfig,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    """冻结教师，使用B3可靠性输出蒸馏与原SKT结构损失训练学生。

    与B2唯一的算法差异是``output_distillation_loss``使用空间可靠性图。
    GT、Dense-FSP、余弦损失及其权重保持不变。需要一致性信号时，冻结教师
    会对水平翻转图额外前向一次；翻转输出映射回原坐标后再计算局部稳定性。
    """

    reliability_config.validate()
    teacher.eval()
    student.train()
    adapters.train()

    total_meter = AverageMeter()
    gt_meter = AverageMeter()
    output_meter = AverageMeter()
    fsp_meter = AverageMeter()
    cosine_meter = AverageMeter()
    count_loss_meter = AverageMeter()
    raw_map_meter = AverageMeter()
    normalized_map_meter = AverageMeter()
    time_meter = AverageMeter()
    counts = CountingAccumulator()
    epoch_start = time.perf_counter()
    trainable_parameters = list(chain(student.parameters(), adapters.parameters()))
    object_reliability_sum = 0.0
    object_count = 0
    maximum_normalized_weight = 0.0

    for step, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        image = batch["image"].to(device, non_blocking=True)
        target = batch["density"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        ground_truth_count = batch["count"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 教师及可靠性图都只提供监督信号。先完成教师前向和可靠性估计，
        # 避免这些运算进入学生反向图并占用额外显存。
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                teacher_density, teacher_features = _feature_output(
                    teacher(image, return_features=True), "教师"
                )
                transformed_teacher_density = None
                if reliability_config.needs_consistency_view:
                    flipped_output = teacher(torch.flip(image, dims=(-1,)))
                    transformed_teacher_density = torch.flip(
                        extract_density_output(flipped_output), dims=(-1,)
                    )

            reliability_map, reliability_statistics = build_batch_reliability_maps(
                teacher_density=teacher_density.float(),
                target_density=target.float(),
                boxes=batch["boxes"],
                input_size=(int(image.shape[-2]), int(image.shape[-1])),
                valid_mask=valid_mask.float(),
                config=reliability_config,
                transformed_teacher_density=(
                    transformed_teacher_density.float()
                    if transformed_teacher_density is not None
                    else None
                ),
            )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            student_density, raw_student_features = _feature_output(
                student(image, return_features=True), "学生"
            )
            aligned_student_features = adapters(raw_student_features)

            gt_density_loss = density_criterion(
                student_density,
                target,
                weight=weight,
                valid_mask=valid_mask,
            )
            # 可靠性图已经按单图有效区域均值归一到1。这里沿用B2相同的
            # batch_mean_sum MSE，只把像素权重从均匀分配改成实例可靠性分配。
            output_distillation_loss = density_criterion(
                student_density,
                teacher_density,
                weight=reliability_map,
                valid_mask=valid_mask,
            )

            # Dense-FSP和余弦蒸馏与B2完全相同，确保消融只改变输出蒸馏项。
            student_fsp_features = [*aligned_student_features, student_density]
            teacher_fsp_features = [*teacher_features, teacher_density]
            fsp_loss = dense_fsp_loss(
                student_fsp_features,
                teacher_fsp_features,
                scales=fsp_scales,
            )
            cosine_loss = cosine_feature_loss(
                aligned_student_features,
                teacher_features,
            )
            # 可靠性图只重分配教师输出蒸馏的空间梯度，不能保证整图密度
            # 积分正确；显式计数项用于纠正实验中持续出现的负偏差。
            if count_loss_weight > 0.0:
                count_loss = count_criterion(
                    student_density,
                    ground_truth_count,
                    valid_mask=valid_mask,
                )
            else:
                # 权重为0时仅监控相对计数误差，保持旧B3的反向图不变。
                with torch.no_grad():
                    count_loss = count_criterion(
                        student_density.detach(),
                        ground_truth_count,
                        valid_mask=valid_mask,
                    )
            total_loss = (
                gt_density_weight * gt_density_loss
                + output_distillation_weight * output_distillation_loss
                + fsp_weight * fsp_loss
                + cosine_weight * cosine_loss
            )
            if count_loss_weight > 0.0:
                total_loss = total_loss + count_loss_weight * count_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"epoch={epoch} step={step}出现非有限B3损失："
                f"total={total_loss.detach().float().item()}"
            )

        scaler.scale(total_loss).backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(gradient_clip_norm)
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(image.shape[0])
        total_meter.update(total_loss.detach().float().item(), batch_size)
        gt_meter.update(gt_density_loss.detach().float().item(), batch_size)
        output_meter.update(
            output_distillation_loss.detach().float().item(), batch_size
        )
        fsp_meter.update(fsp_loss.detach().float().item(), batch_size)
        cosine_meter.update(cosine_loss.detach().float().item(), batch_size)
        count_loss_meter.update(count_loss.detach().float().item(), batch_size)
        raw_map_meter.update(reliability_statistics["raw_map_mean"], batch_size)
        normalized_map_meter.update(
            reliability_statistics["normalized_map_mean"], batch_size
        )
        current_object_count = int(reliability_statistics["object_count"])
        object_reliability_sum += (
            reliability_statistics["object_reliability_mean"]
            * current_object_count
        )
        object_count += current_object_count
        maximum_normalized_weight = max(
            maximum_normalized_weight,
            reliability_statistics["maximum_normalized_weight"],
        )

        predicted_count = student_density.detach().float().sum(dim=(1, 2, 3))
        counts.update(predicted_count, ground_truth_count)
        time_meter.update(time.perf_counter() - step_start)

        if print_frequency > 0 and (
            step == 1 or step % print_frequency == 0 or step == len(loader)
        ):
            current_metrics = counts.compute()
            current_object_mean = (
                object_reliability_sum / object_count if object_count else 0.0
            )
            print(
                f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"total={total_meter.average:.6f} "
                f"gt={gt_meter.average:.6f} "
                f"out_rel={output_meter.average:.6f} "
                f"fsp={fsp_meter.average:.6f} "
                f"cos={cosine_meter.average:.6f} "
                f"count_loss={count_loss_meter.average:.6f} "
                f"rel_obj={current_object_mean:.3f} "
                f"rel_norm={normalized_map_meter.average:.3f} "
                f"count_mae={current_metrics['mae']:.3f} "
                f"time={time_meter.average:.3f}s/batch"
            )

    metrics = counts.compute()
    metrics.update(
        {
            "loss": total_meter.average,
            "gt_density_loss": gt_meter.average,
            "output_distillation_loss": output_meter.average,
            "fsp_loss": fsp_meter.average,
            "cosine_loss": cosine_meter.average,
            "count_loss": count_loss_meter.average,
            "object_reliability_mean": (
                object_reliability_sum / object_count if object_count else 0.0
            ),
            "reliability_raw_map_mean": raw_map_meter.average,
            "reliability_normalized_map_mean": normalized_map_meter.average,
            "maximum_normalized_weight": maximum_normalized_weight,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
    )
    return metrics


__all__ = ["train_reliable_skt_one_epoch"]
