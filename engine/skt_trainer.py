"""B2原始SKT蒸馏的单epoch训练循环。"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from itertools import chain

import torch
from torch import nn

from losses import (
    SKTFeatureAdapters,
    cosine_feature_loss,
    dense_fsp_loss,
)

from .common import AverageMeter, CountingAccumulator


def _feature_output(output: object, model_name: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
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


def train_skt_one_epoch(
    *,
    teacher: nn.Module,
    student: nn.Module,
    adapters: SKTFeatureAdapters,
    loader: torch.utils.data.DataLoader,
    density_criterion: Callable[..., torch.Tensor],
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
    fsp_scales: Sequence[int],
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    """冻结教师，使用四项SKT损失训练学生主干和特征适配层。"""

    teacher.eval()
    student.train()
    adapters.train()

    total_meter = AverageMeter()
    gt_meter = AverageMeter()
    output_meter = AverageMeter()
    fsp_meter = AverageMeter()
    cosine_meter = AverageMeter()
    time_meter = AverageMeter()
    counts = CountingAccumulator()
    epoch_start = time.perf_counter()
    trainable_parameters = list(chain(student.parameters(), adapters.parameters()))

    for step, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        image = batch["image"].to(device, non_blocking=True)
        target = batch["density"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        ground_truth_count = batch["count"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            # 教师只提供监督信号，不保留计算图，也不会更新参数。
            with torch.no_grad():
                teacher_density, teacher_features = _feature_output(
                    teacher(image, return_features=True), "教师"
                )

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
            # 固定核CARPK的valid_mask全为1；显式使用它可兼容后续UAVDT忽略区。
            output_distillation_loss = density_criterion(
                student_density,
                teacher_density,
                valid_mask=valid_mask,
            )
            # 原SKT的Dense-FSP包含六个中间特征和最终密度输出，共7个
            # Tensor、21个两两关系；余弦损失只比较六个中间特征。
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
            total_loss = (
                gt_density_weight * gt_density_loss
                + output_distillation_weight * output_distillation_loss
                + fsp_weight * fsp_loss
                + cosine_weight * cosine_loss
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"epoch={epoch} step={step}出现非有限蒸馏损失："
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

        batch_size = image.shape[0]
        total_meter.update(total_loss.detach().float().item(), batch_size)
        gt_meter.update(gt_density_loss.detach().float().item(), batch_size)
        output_meter.update(
            output_distillation_loss.detach().float().item(), batch_size
        )
        fsp_meter.update(fsp_loss.detach().float().item(), batch_size)
        cosine_meter.update(cosine_loss.detach().float().item(), batch_size)
        predicted_count = student_density.detach().float().sum(dim=(1, 2, 3))
        counts.update(predicted_count, ground_truth_count)
        time_meter.update(time.perf_counter() - step_start)

        if print_frequency > 0 and (
            step == 1 or step % print_frequency == 0 or step == len(loader)
        ):
            current_metrics = counts.compute()
            print(
                f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"total={total_meter.average:.6f} "
                f"gt={gt_meter.average:.6f} "
                f"out={output_meter.average:.6f} "
                f"fsp={fsp_meter.average:.6f} "
                f"cos={cosine_meter.average:.6f} "
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
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
    )
    return metrics


__all__ = ["train_skt_one_epoch"]
