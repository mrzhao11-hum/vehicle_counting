"""CARPK密度模型验证、测试、逐图结果和可视化。"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .common import AverageMeter, CountingAccumulator, extract_density_output


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _restore_image(image: torch.Tensor) -> np.ndarray:
    """把ImageNet标准化后的CHW Tensor恢复为可显示RGB图像。"""

    array = image.detach().float().cpu().numpy()
    array = array * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(array.transpose(1, 2, 0), 0.0, 1.0)


def _safe_name(sample_id: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in sample_id)


def _save_comparison(
    *,
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    ground_truth_count: float,
    raw_predicted_count: float,
    positive_predicted_count: float,
    negative_mass: float,
    path: Path,
) -> None:
    """保存原图、GT密度图和预测密度图的并排结果。"""

    rgb = _restore_image(image)
    target_array = target.detach().float().cpu().squeeze().numpy()
    prediction_array = prediction.detach().float().cpu().squeeze().numpy()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Image | GT count={ground_truth_count:.2f}")
    axes[1].imshow(target_array, cmap="jet", vmin=0)
    axes[1].set_title(f"GT density | sum={target_array.sum():.2f}")

    # 负值只在显示时截为0；正式MAE/RMSE仍由原始网络输出积分计算。
    axes[2].imshow(np.maximum(prediction_array, 0.0), cmap="jet", vmin=0)
    axes[2].set_title(
        f"Prediction | raw={raw_predicted_count:.2f} "
        f"positive={positive_predicted_count:.2f} negative={negative_mass:.2f}"
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def evaluate_model(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Callable[..., torch.Tensor] | None = None,
    amp_enabled: bool = False,
    output_dir: str | Path | None = None,
    save_visualizations: int = 0,
    save_worst_visualizations: int = 0,
    description: str = "evaluation",
) -> dict[str, float]:
    """评估模型并可选保存逐图CSV和密度图对比。

    MAE和RMSE均按“每张图片的总车辆数误差”计算。预测数使用原始浮点密度
    积分，不取整，也不把负密度截断，从而避免评估阶段改变模型结果。
    """

    model.eval()
    losses = AverageMeter()
    counts = CountingAccumulator()
    positive_counts = CountingAccumulator()
    rows: list[dict[str, str | float]] = []
    worst_examples: list[dict[str, object]] = []
    visualized = 0
    resolved_output = Path(output_dir) if output_dir is not None else None
    raw_prediction_sum = 0.0
    positive_prediction_sum = 0.0
    negative_mass_sum = 0.0
    ground_truth_sum = 0.0
    signed_error_sum = 0.0
    negative_pixels = 0
    density_pixels = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["density"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        ground_truth_count = batch["count"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            prediction = extract_density_output(model(image))
            if criterion is not None:
                loss = criterion(
                    prediction,
                    target,
                    weight=weight,
                    valid_mask=valid_mask,
                )
                losses.update(loss.item(), image.shape[0])

        # 网络前向可以使用AMP，但计数积分必须回到FP32。否则对14400个密度
        # 像素直接执行FP16求和，会让pred_count出现0.0625等量化步进。
        prediction_fp32 = prediction.float()
        predicted_count = prediction_fp32.sum(dim=(1, 2, 3))
        positive_count = prediction_fp32.clamp_min(0).sum(dim=(1, 2, 3))
        negative_mass = (-prediction_fp32.clamp_max(0)).sum(dim=(1, 2, 3))
        counts.update(predicted_count, ground_truth_count)
        positive_counts.update(positive_count, ground_truth_count)

        raw_prediction_sum += predicted_count.sum().item()
        positive_prediction_sum += positive_count.sum().item()
        negative_mass_sum += negative_mass.sum().item()
        ground_truth_sum += ground_truth_count.sum().item()
        signed_error_sum += (predicted_count - ground_truth_count).sum().item()
        negative_pixels += int((prediction_fp32 < 0).sum().item())
        density_pixels += prediction_fp32.numel()

        for index, meta in enumerate(batch["meta"]):
            prediction_value = predicted_count[index].float().item()
            positive_value = positive_count[index].float().item()
            negative_value = negative_mass[index].float().item()
            target_value = ground_truth_count[index].float().item()
            error = prediction_value - target_value
            row = {
                "id": meta["id"],
                "sequence": meta["sequence"],
                "split": meta["split"],
                "gt_count": target_value,
                # 保留pred_count兼容已经生成的分析表。
                "pred_count": prediction_value,
                "raw_pred_count": prediction_value,
                "positive_pred_count": positive_value,
                "negative_mass": negative_value,
                "error": error,
                "abs_error": abs(error),
            }
            rows.append(row)

            if resolved_output is not None and visualized < save_visualizations:
                _save_comparison(
                    image=image[index],
                    target=target[index],
                    prediction=prediction[index],
                    ground_truth_count=target_value,
                    raw_predicted_count=prediction_value,
                    positive_predicted_count=positive_value,
                    negative_mass=negative_value,
                    path=resolved_output
                    / "visualizations"
                    / f"{_safe_name(meta['id'])}.jpg",
                )
                visualized += 1

            should_keep_worst = (
                resolved_output is not None
                and save_worst_visualizations > 0
                and (
                    len(worst_examples) < save_worst_visualizations
                    or abs(error) > float(worst_examples[0]["abs_error"])
                )
            )
            if should_keep_worst:
                # 只在内存中保留当前误差最大的少量样本。图像转FP16后再放到
                # CPU，避免对完整测试集保存中间Tensor造成过高内存占用。
                worst_examples.append(
                    {
                        "abs_error": abs(error),
                        "id": meta["id"],
                        "image": image[index].detach().to("cpu", dtype=torch.float16),
                        "target": target[index].detach().to("cpu", dtype=torch.float32),
                        "prediction": prediction_fp32[index].detach().cpu(),
                        "ground_truth_count": target_value,
                        "raw_predicted_count": prediction_value,
                        "positive_predicted_count": positive_value,
                        "negative_mass": negative_value,
                    }
                )
                worst_examples.sort(key=lambda item: float(item["abs_error"]))
                if len(worst_examples) > save_worst_visualizations:
                    worst_examples.pop(0)

    metrics = counts.compute()
    positive_metrics = positive_counts.compute()
    if criterion is not None:
        metrics["loss"] = losses.average
    metrics["samples"] = float(counts.samples)
    sample_denominator = max(counts.samples, 1)
    metrics.update(
        {
            "bias": signed_error_sum / sample_denominator,
            "mean_ground_truth_count": ground_truth_sum / sample_denominator,
            "mean_raw_pred_count": raw_prediction_sum / sample_denominator,
            "mean_positive_pred_count": positive_prediction_sum / sample_denominator,
            "mean_negative_mass": negative_mass_sum / sample_denominator,
            "negative_pixel_fraction": negative_pixels / max(density_pixels, 1),
            "positive_clipped_mae": positive_metrics["mae"],
            "positive_clipped_rmse": positive_metrics["rmse"],
        }
    )

    if resolved_output is not None:
        resolved_output.mkdir(parents=True, exist_ok=True)
        csv_path = resolved_output / "per_image_results.csv"
        fieldnames = [
            "id",
            "sequence",
            "split",
            "gt_count",
            "pred_count",
            "raw_pred_count",
            "positive_pred_count",
            "negative_mass",
            "error",
            "abs_error",
        ]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (resolved_output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for rank, example in enumerate(
            sorted(
                worst_examples,
                key=lambda item: float(item["abs_error"]),
                reverse=True,
            ),
            start=1,
        ):
            _save_comparison(
                image=example["image"],
                target=example["target"],
                prediction=example["prediction"],
                ground_truth_count=float(example["ground_truth_count"]),
                raw_predicted_count=float(example["raw_predicted_count"]),
                positive_predicted_count=float(example["positive_predicted_count"]),
                negative_mass=float(example["negative_mass"]),
                path=resolved_output
                / "worst_visualizations"
                / (
                    f"{rank:02d}_abs_{float(example['abs_error']):.2f}_"
                    f"{_safe_name(str(example['id']))}.jpg"
                ),
            )

    print(
        f"{description}: samples={counts.samples} "
        f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}"
        + (f" loss={metrics['loss']:.6f}" if "loss" in metrics else "")
    )
    return metrics
