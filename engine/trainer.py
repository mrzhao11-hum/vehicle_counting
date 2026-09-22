"""教师和学生共享的单epoch密度图监督训练逻辑。"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn

from .common import AverageMeter, CountingAccumulator, extract_density_output


def train_one_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    print_frequency: int,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    """训练一个epoch并返回训练损失和计数MAE/RMSE。"""

    model.train()
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    counts = CountingAccumulator()
    epoch_start = time.perf_counter()

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
            prediction = extract_density_output(model(image))
            loss = criterion(
                prediction,
                target,
                weight=weight,
                valid_mask=valid_mask,
            )

        scaler.scale(loss).backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = image.shape[0]
        loss_meter.update(loss.detach().item(), batch_size)
        # AMP前向时prediction通常为FP16。先转FP32再对整张密度图积分，
        # 避免大量像素求和产生可见的量化误差。
        predicted_count = prediction.detach().float().sum(dim=(1, 2, 3))
        counts.update(predicted_count, ground_truth_count)
        time_meter.update(time.perf_counter() - step_start)

        if print_frequency > 0 and (
            step == 1 or step % print_frequency == 0 or step == len(loader)
        ):
            current_metrics = counts.compute()
            print(
                f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"loss={loss_meter.average:.6f} "
                f"count_mae={current_metrics['mae']:.3f} "
                f"time={time_meter.average:.3f}s/batch"
            )

    metrics = counts.compute()
    metrics.update(
        {
            "loss": loss_meter.average,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
    )
    return metrics
