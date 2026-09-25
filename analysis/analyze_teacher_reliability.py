"""分析 CARPK 教师在车辆实例级别上的预测可靠性。

本脚本不训练网络，也不修改数据。它固定教师模型，在验证集上统计：

* 每辆车软实例区域内的密度图误差和局部计数误差；
* 教师对原图和水平翻转图的预测一致性；
* 误差与车辆大小、局部拥挤程度、整图车辆数量之间的关系；
* 由监督误差和视图一致性构成的实例可靠性分数。

推荐从项目根目录运行：

    python analysis/analyze_teacher_reliability.py \
      --config configs/carpk_teacher_fixed_fp32.yaml \
      --checkpoint outputs/carpk/b0_teacher_fixed_sigma8_fp32_fixed_lr/best_mae.pth \
      --split val \
      --output-dir outputs/analysis/teacher_reliability_val \
      --num-visualizations 12

方法设计说明：

监督可靠性直接利用训练/验证阶段可用的 GT 判断教师是否正确，后续可以用来
避免学生在教师错误区域接受强蒸馏。水平翻转一致性不使用 GT，它是一个独立
信号；若翻转不稳定性与真实局部误差正相关，就说明“预测稳定性”能够帮助识别
教师的不可靠知识。测试集不应参与 tau 或其他超参数选择。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import cm, colors, patches
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# 直接执行 analysis/ 下的脚本时，Python 默认只把 analysis 目录加入模块
# 搜索路径。这里显式加入项目根目录，从而可以复用 datasets/models/engine。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import CARPKDataset, carpk_collate_fn  # noqa: E402
from engine.common import (  # noqa: E402
    extract_density_output,
    load_checkpoint,
    seed_worker,
    set_random_seed,
)
from losses.object_reliability import (  # noqa: E402
    InstanceMaskConfig,
    build_soft_instance_regions,
    compute_instance_errors,
    rasterize_instance_values,
    reliability_from_errors,
)
from models import CSRNetTeacher  # noqa: E402


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析CARPK教师实例级可靠性")
    parser.add_argument("--config", type=Path, required=True, help="教师训练配置")
    parser.add_argument("--checkpoint", type=Path, required=True, help="教师最佳权重")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "train_full"),
        default="val",
        help="建议首先只使用val；不要用test调节可靠性参数",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None, help="覆盖配置中的DataLoader进程数")
    parser.add_argument("--max-samples", type=int, default=0, help="仅分析前N张，0表示全部")
    parser.add_argument("--num-visualizations", type=int, default=12)
    parser.add_argument("--amp", action="store_true", help="使用FP16前向；正式分析建议保持关闭")

    # 以下参数只决定“每辆车附近多大范围属于该实例”，不会修改固定sigma8 GT。
    parser.add_argument("--mask-sigma-scale", type=float, default=0.35)
    parser.add_argument("--mask-sigma-min", type=float, default=0.75)
    parser.add_argument("--mask-sigma-max", type=float, default=4.0)
    parser.add_argument("--mask-truncate", type=float, default=2.5)

    # tau<=0时由验证集误差中位数自动给出推荐值。论文实验确定参数后，应把
    # tau写入训练配置，并且不能再根据测试集重新选择。
    parser.add_argument("--tau-map", type=float, default=0.0)
    parser.add_argument("--tau-count", type=float, default=0.0)
    parser.add_argument("--tau-consistency", type=float, default=0.0)
    parser.add_argument(
        "--minimum-reliability",
        type=float,
        default=0.05,
        help="可靠性下限；后续蒸馏保留少量教师监督，避免梯度完全消失",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是字典")
    return payload


def make_loader(
    *,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    workers_override: int | None,
    max_samples: int,
) -> DataLoader:
    """创建无随机增强、batch=1的分析DataLoader。

    每张图车辆数不同，实例可靠性需要逐图构造可变长度区域，因此分析阶段
    固定 batch=1。它不影响训练代码原有的 batch 设置。
    """

    experiment = config["experiment"]
    data_config = config["data"]
    manifest_key = f"{split}_manifest"
    if manifest_key not in data_config:
        raise KeyError(f"配置缺少data.{manifest_key}")

    dataset: torch.utils.data.Dataset = CARPKDataset(
        data_root=data_config["data_root"],
        manifest=data_config[manifest_key],
        target_root=data_config["target_root"],
        output_stride=int(data_config.get("output_stride", 8)),
        horizontal_flip_probability=0.0,
    )
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    workers = (
        int(workers_override)
        if workers_override is not None
        else int(data_config.get("workers", 4))
    )
    generator = torch.Generator().manual_seed(int(experiment.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=carpk_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _restore_image(image: torch.Tensor) -> np.ndarray:
    """把ImageNet标准化的CHW Tensor还原为可显示RGB图像。"""

    array = image.detach().float().cpu().numpy()
    array = array * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(array.transpose(1, 2, 0), 0.0, 1.0)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _nearest_neighbor_statistics(boxes: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """返回中心点最近邻距离及按车辆面积归一化后的距离。

    归一化距离较小通常表示车辆排列更密集。只有一个目标时不存在最近邻，
    使用NaN，后续相关分析会自动忽略。
    """

    boxes = boxes.detach().float().cpu()
    count = boxes.shape[0]
    if count == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty
    if count == 1:
        single = np.asarray([np.nan], dtype=np.float32)
        return single, single.copy()

    centers = torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5),
        dim=1,
    )
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values
    area = (
        (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
        * (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    )
    normalized = nearest / area.sqrt()
    return nearest.numpy(), normalized.numpy()


def _auto_tau(values: Iterable[float], requested: float) -> float:
    """使用正误差中位数生成稳定的tau，或采用用户显式值。"""

    if requested > 0:
        return float(requested)
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return 1.0
    return max(float(np.median(array)), 1e-8)


def _finite_spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int | None]:
    """忽略NaN/Inf并计算Spearman相关，常量数组返回空结果。"""

    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]
    if x_array.size < 3 or np.unique(x_array).size < 2 or np.unique(y_array).size < 2:
        return {"rho": None, "p_value": None, "samples": int(x_array.size)}
    result = spearmanr(x_array, y_array)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "samples": int(x_array.size),
    }


def _quantile_edges(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 0.0
    lower, upper = np.quantile(array, [1.0 / 3.0, 2.0 / 3.0])
    return float(lower), float(upper)


def _three_way_label(value: float, lower: float, upper: float, labels: tuple[str, str, str]) -> str:
    if not math.isfinite(value):
        return "unknown"
    if value <= lower:
        return labels[0]
    if value <= upper:
        return labels[1]
    return labels[2]


def _group_summary(
    rows: list[dict[str, Any]], group_key: str, ordered_groups: tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for group in ordered_groups:
        members = [row for row in rows if row[group_key] == group]
        if not members:
            continue
        result[group] = {
            "objects": len(members),
            "mean_local_map_mae": float(np.mean([row["local_map_mae"] for row in members])),
            "mean_local_count_error": float(np.mean([row["local_count_error"] for row in members])),
            "mean_view_consistency_mae": float(
                np.mean([row["view_consistency_mae"] for row in members])
            ),
            "mean_combined_reliability": float(
                np.mean([row["combined_reliability"] for row in members])
            ),
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_analysis(
    object_rows: list[dict[str, Any]],
    size_summary: dict[str, dict[str, float | int]],
    density_summary: dict[str, dict[str, float | int]],
    output_dir: Path,
) -> None:
    """保存四张论文前期诊断图，图中文字使用英文避免服务器缺少中文字体。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    reliability = np.asarray(
        [row["combined_reliability"] for row in object_rows], dtype=np.float64
    )
    local_error = np.asarray([row["local_map_mae"] for row in object_rows], dtype=np.float64)
    consistency_error = np.asarray(
        [row["view_consistency_mae"] for row in object_rows], dtype=np.float64
    )
    areas = np.asarray([row["box_area"] for row in object_rows], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(reliability, bins=20, color="#3478bf", edgecolor="white")
    axis.set_xlabel("Combined instance reliability")
    axis.set_ylabel("Number of vehicles")
    axis.set_title("Teacher instance reliability distribution")
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_distribution.png", dpi=160)
    plt.close(figure)

    valid = np.isfinite(local_error) & np.isfinite(consistency_error) & np.isfinite(areas)
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        consistency_error[valid],
        local_error[valid],
        c=np.log1p(areas[valid]),
        cmap="viridis",
        s=12,
        alpha=0.55,
        linewidths=0,
    )
    axis.set_xlabel("Flip-view consistency error")
    axis.set_ylabel("Teacher local map error")
    axis.set_title("View instability versus true local error")
    figure.colorbar(scatter, ax=axis, label="log(1 + bbox area)")
    figure.tight_layout()
    figure.savefig(output_dir / "consistency_vs_error.png", dpi=160)
    plt.close(figure)

    def save_group_bar(
        summary: dict[str, dict[str, float | int]], names: tuple[str, ...], title: str, filename: str
    ) -> None:
        available = [name for name in names if name in summary]
        values = [float(summary[name]["mean_local_map_mae"]) for name in available]
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(available, values, color=("#4c78a8", "#f2a541", "#d45050")[: len(available)])
        axis.set_ylabel("Mean teacher local map error")
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)

    save_group_bar(
        size_summary,
        ("small", "medium", "large"),
        "Teacher error by vehicle size",
        "error_by_size.png",
    )
    save_group_bar(
        density_summary,
        ("sparse", "medium", "dense"),
        "Teacher error by image density",
        "error_by_density.png",
    )


def _save_instance_visualization(
    *,
    image: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    boxes: torch.Tensor,
    reliability_values: torch.Tensor,
    reliability_map: torch.Tensor,
    sample_id: str,
    ground_truth_count: float,
    path: Path,
) -> None:
    """保存原图实例可靠性、GT、教师预测和可靠性空间图。"""

    rgb = _restore_image(image)
    target_array = target.detach().float().cpu().squeeze().numpy()
    teacher_array = teacher.detach().float().cpu().squeeze().numpy()
    reliability_array = reliability_map.detach().float().cpu().numpy()
    boxes_array = boxes.detach().float().cpu().numpy()
    values_array = reliability_values.detach().float().cpu().numpy()

    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb)
    normalization = colors.Normalize(vmin=0.0, vmax=1.0)
    colormap = cm.get_cmap("turbo")
    for box, value in zip(boxes_array, values_array, strict=False):
        # CARPK 的第五列是 class_id；绘制矩形只读取前四列坐标。
        x1, y1, x2, y2 = box[:4].tolist()
        rectangle = patches.Rectangle(
            (x1, y1),
            max(x2 - x1, 1.0),
            max(y2 - y1, 1.0),
            linewidth=1.1,
            edgecolor=colormap(normalization(float(value))),
            facecolor="none",
        )
        axes[0].add_patch(rectangle)
    axes[0].set_title(
        f"Instances | mean reliability={float(values_array.mean()) if values_array.size else 0.0:.3f}"
    )
    axes[1].imshow(target_array, cmap="jet", vmin=0)
    axes[1].set_title(f"GT density | sum={target_array.sum():.2f}")
    axes[2].imshow(np.maximum(teacher_array, 0.0), cmap="jet", vmin=0)
    axes[2].set_title(f"Teacher | raw sum={teacher_array.sum():.2f}")
    axes[3].imshow(reliability_array, cmap="turbo", vmin=0, vmax=1)
    axes[3].set_title("Spatial reliability map")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{sample_id} | GT count={ground_truth_count:.0f}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def collect_raw_statistics(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_config: InstanceMaskConfig,
    amp_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """第一遍前向：收集逐实例原始误差和逐图计数误差。"""

    model.eval()
    object_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="分析教师可靠性", unit="image"):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["density"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        boxes = batch["boxes"][0].to(device, non_blocking=True)
        meta = batch["meta"][0]
        ground_truth_count = float(batch["count"][0].item())

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            teacher = extract_density_output(model(image))
            # 水平翻转是严格保持车辆数量的几何等价变换。将翻转图的预测再次
            # 翻转回来后，理想教师应与原图预测一致。
            flipped_teacher = extract_density_output(model(torch.flip(image, dims=(-1,))))
            flipped_teacher = torch.flip(flipped_teacher, dims=(-1,))

        teacher = teacher.float()
        flipped_teacher = flipped_teacher.float()
        input_height, input_width = image.shape[-2:]
        output_height, output_width = target.shape[-2:]
        regions = build_soft_instance_regions(
            boxes,
            input_size=(input_height, input_width),
            output_size=(output_height, output_width),
            config=mask_config,
            valid_mask=valid_mask[0],
        )
        errors = compute_instance_errors(
            teacher[0],
            target[0],
            regions,
            transformed_teacher_density=flipped_teacher[0],
            epsilon=mask_config.epsilon,
        )

        teacher_count = float(teacher[0].sum().item())
        image_rows.append(
            {
                "id": meta["id"],
                "sequence": meta["sequence"],
                "split": meta["split"],
                "gt_count": ground_truth_count,
                "teacher_count": teacher_count,
                "count_error": teacher_count - ground_truth_count,
                "abs_count_error": abs(teacher_count - ground_truth_count),
            }
        )

        boxes_cpu = boxes.detach().float().cpu()
        nearest, normalized_nearest = _nearest_neighbor_statistics(boxes_cpu)
        for object_index in range(boxes_cpu.shape[0]):
            # 数据集可在 bbox 坐标后附带类别等元数据，统计时只取 xyxy。
            x1, y1, x2, y2 = boxes_cpu[object_index, :4].tolist()
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            object_rows.append(
                {
                    "id": meta["id"],
                    "sequence": meta["sequence"],
                    "split": meta["split"],
                    "object_index": object_index,
                    "image_count": ground_truth_count,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_width": width,
                    "bbox_height": height,
                    "box_area": width * height,
                    "nearest_center_distance": float(nearest[object_index]),
                    "normalized_neighbor_distance": float(normalized_nearest[object_index]),
                    "region_mass": float(errors["region_mass"][object_index].item()),
                    "target_local_mass": float(errors["target_local_mass"][object_index].item()),
                    "teacher_local_mass": float(errors["teacher_local_mass"][object_index].item()),
                    "local_count_error": float(errors["local_count_error"][object_index].item()),
                    "local_map_mae": float(errors["local_map_mae"][object_index].item()),
                    "centroid_error": float(errors["centroid_error"][object_index].item()),
                    "view_consistency_mae": float(
                        errors["view_consistency_mae"][object_index].item()
                    ),
                }
            )

    return object_rows, image_rows


def finalize_statistics(
    *,
    object_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    tau_map_request: float,
    tau_count_request: float,
    tau_consistency_request: float,
    minimum_reliability: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    """确定tau、计算可靠性、分组统计并判断研究假设。"""

    if not object_rows:
        raise RuntimeError("数据集中没有车辆实例，无法分析可靠性")

    tau_map = _auto_tau(
        (row["local_map_mae"] for row in object_rows), tau_map_request
    )
    tau_count = _auto_tau(
        (row["local_count_error"] for row in object_rows), tau_count_request
    )
    tau_consistency = _auto_tau(
        (row["view_consistency_mae"] for row in object_rows),
        tau_consistency_request,
    )
    taus = {
        "map": tau_map,
        "count": tau_count,
        "consistency": tau_consistency,
    }

    reliability = reliability_from_errors(
        torch.tensor([row["local_map_mae"] for row in object_rows]),
        torch.tensor([row["local_count_error"] for row in object_rows]),
        tau_map=tau_map,
        tau_count=tau_count,
        view_consistency_error=torch.tensor(
            [row["view_consistency_mae"] for row in object_rows]
        ),
        tau_consistency=tau_consistency,
        minimum=minimum_reliability,
    )
    for index, row in enumerate(object_rows):
        row["supervised_reliability"] = float(reliability["supervised"][index].item())
        row["consistency_reliability"] = float(reliability["consistency"][index].item())
        row["combined_reliability"] = float(reliability["combined"][index].item())

    size_lower, size_upper = _quantile_edges(row["box_area"] for row in object_rows)
    density_lower, density_upper = _quantile_edges(row["image_count"] for row in object_rows)
    crowd_lower, crowd_upper = _quantile_edges(
        row["normalized_neighbor_distance"] for row in object_rows
    )
    for row in object_rows:
        row["size_group"] = _three_way_label(
            row["box_area"], size_lower, size_upper, ("small", "medium", "large")
        )
        row["density_group"] = _three_way_label(
            row["image_count"], density_lower, density_upper, ("sparse", "medium", "dense")
        )
        # 最近邻距离越小越拥挤，所以标签顺序与数值大小相反。
        row["crowding_group"] = _three_way_label(
            row["normalized_neighbor_distance"],
            crowd_lower,
            crowd_upper,
            ("crowded", "medium", "isolated"),
        )

    object_rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in object_rows:
        object_rows_by_image[str(row["id"])].append(row)
    for row in image_rows:
        members = object_rows_by_image[str(row["id"])]
        row["objects"] = len(members)
        row["mean_local_map_mae"] = float(np.mean([item["local_map_mae"] for item in members]))
        row["mean_local_count_error"] = float(
            np.mean([item["local_count_error"] for item in members])
        )
        row["mean_view_consistency_mae"] = float(
            np.mean([item["view_consistency_mae"] for item in members])
        )
        row["mean_combined_reliability"] = float(
            np.mean([item["combined_reliability"] for item in members])
        )
        row["minimum_combined_reliability"] = float(
            np.min([item["combined_reliability"] for item in members])
        )

    image_errors = np.asarray([row["count_error"] for row in image_rows], dtype=np.float64)
    local_map_errors = np.asarray(
        [row["local_map_mae"] for row in object_rows], dtype=np.float64
    )
    q25, q50, q75 = np.quantile(local_map_errors, [0.25, 0.5, 0.75])
    heterogeneity_ratio = float(q75 / max(q25, 1e-12))

    correlations = {
        # 这是最重要的独立检验：翻转一致性没有使用GT，但与真实误差比较。
        "view_consistency_error_vs_local_map_error": _finite_spearman(
            (row["view_consistency_mae"] for row in object_rows),
            (row["local_map_mae"] for row in object_rows),
        ),
        "view_consistency_error_vs_local_count_error": _finite_spearman(
            (row["view_consistency_mae"] for row in object_rows),
            (row["local_count_error"] for row in object_rows),
        ),
        "box_area_vs_local_map_error": _finite_spearman(
            (row["box_area"] for row in object_rows),
            (row["local_map_mae"] for row in object_rows),
        ),
        "neighbor_distance_vs_local_map_error": _finite_spearman(
            (row["normalized_neighbor_distance"] for row in object_rows),
            (row["local_map_mae"] for row in object_rows),
        ),
        # 监督可靠性由真实误差构成，因此该相关性只是实现正确性的检查，
        # 不能当作论文中证明方法有效的独立证据。
        "supervised_reliability_vs_local_map_error_sanity_check": _finite_spearman(
            (row["supervised_reliability"] for row in object_rows),
            (row["local_map_mae"] for row in object_rows),
        ),
    }

    consistency_correlation = correlations[
        "view_consistency_error_vs_local_map_error"
    ]
    rho = consistency_correlation["rho"]
    p_value = consistency_correlation["p_value"]
    error_varies = heterogeneity_ratio >= 1.5
    stability_predicts_error = (
        rho is not None
        and p_value is not None
        and float(rho) >= 0.2
        and float(p_value) < 0.05
    )
    if error_varies and stability_predicts_error:
        conclusion = "supported"
    elif error_varies:
        conclusion = "partially_supported"
    else:
        conclusion = "not_supported"

    summary: dict[str, Any] = {
        "images": len(image_rows),
        "objects": len(object_rows),
        "teacher_image_metrics": {
            "mae": float(np.mean(np.abs(image_errors))),
            "rmse": float(np.sqrt(np.mean(np.square(image_errors)))),
            "bias": float(np.mean(image_errors)),
        },
        "recommended_taus_from_this_split": taus,
        "minimum_reliability": minimum_reliability,
        "local_map_error_quantiles": {
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "q75_div_q25": heterogeneity_ratio,
        },
        "group_boundaries": {
            "bbox_area_tertiles": [size_lower, size_upper],
            "image_count_tertiles": [density_lower, density_upper],
            "normalized_neighbor_distance_tertiles": [crowd_lower, crowd_upper],
        },
        "correlations": correlations,
        "groups": {
            "size": _group_summary(object_rows, "size_group", ("small", "medium", "large")),
            "density": _group_summary(
                object_rows, "density_group", ("sparse", "medium", "dense")
            ),
            "crowding": _group_summary(
                object_rows, "crowding_group", ("crowded", "medium", "isolated")
            ),
        },
        "hypothesis_check": {
            "teacher_error_varies_across_instances": error_varies,
            "view_stability_predicts_true_error": stability_predicts_error,
            "conclusion": conclusion,
            "interpretation": (
                "supported表示实例误差差异明显，且独立的翻转稳定性与真实误差显著相关；"
                "partially_supported表示教师实例误差确实不同，但当前翻转稳定性不足以单独预测错误；"
                "not_supported表示当前实例区域或误差定义需要重新检查。"
            ),
        },
    }
    return summary, taus


@torch.inference_mode()
def save_selected_visualizations(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_config: InstanceMaskConfig,
    object_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    output_dir: Path,
    limit: int,
    amp_enabled: bool,
) -> None:
    """第二遍仅对平均可靠性最低的图像保存可视化。"""

    if limit <= 0:
        return
    selected = sorted(
        image_rows,
        key=lambda row: float(row["mean_combined_reliability"]),
    )[:limit]
    selected_ids = {str(row["id"]) for row in selected}
    reliability_lookup = {
        (str(row["id"]), int(row["object_index"])): float(row["combined_reliability"])
        for row in object_rows
    }

    model.eval()
    saved = 0
    for batch in tqdm(loader, desc="保存低可靠性可视化", unit="image"):
        meta = batch["meta"][0]
        sample_id = str(meta["id"])
        if sample_id not in selected_ids:
            continue

        image = batch["image"].to(device, non_blocking=True)
        target = batch["density"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        boxes = batch["boxes"][0].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            teacher = extract_density_output(model(image)).float()

        regions = build_soft_instance_regions(
            boxes,
            input_size=tuple(image.shape[-2:]),
            output_size=tuple(target.shape[-2:]),
            config=mask_config,
            valid_mask=valid_mask[0],
        )
        values = torch.tensor(
            [
                reliability_lookup[(sample_id, object_index)]
                for object_index in range(boxes.shape[0])
            ],
            device=device,
            dtype=torch.float32,
        )
        reliability_map = rasterize_instance_values(
            regions, values, background_value=0.0
        )
        _save_instance_visualization(
            image=image[0],
            target=target[0],
            teacher=teacher[0],
            boxes=boxes,
            reliability_values=values,
            reliability_map=reliability_map,
            sample_id=sample_id,
            ground_truth_count=float(batch["count"][0].item()),
            path=output_dir / "visualizations" / f"{_safe_name(sample_id)}.jpg",
        )
        saved += 1
        if saved >= limit:
            break


def _write_markdown_report(
    *, output_dir: Path, split: str, checkpoint: Path, summary: dict[str, Any]
) -> None:
    metrics = summary["teacher_image_metrics"]
    quantiles = summary["local_map_error_quantiles"]
    correlation = summary["correlations"][
        "view_consistency_error_vs_local_map_error"
    ]
    hypothesis = summary["hypothesis_check"]
    report = f"""# 教师实例可靠性分析报告

## 基本信息

- 数据划分：`{split}`
- 教师权重：`{checkpoint}`
- 图像数量：{summary['images']}
- 车辆实例数量：{summary['objects']}
- 教师图像级 MAE：{metrics['mae']:.4f}
- 教师图像级 RMSE：{metrics['rmse']:.4f}
- 教师计数偏差：{metrics['bias']:.4f}

## 实例误差差异

- 局部密度误差 Q25：{quantiles['q25']:.8f}
- 局部密度误差中位数：{quantiles['median']:.8f}
- 局部密度误差 Q75：{quantiles['q75']:.8f}
- Q75 / Q25：{quantiles['q75_div_q25']:.4f}

`Q75/Q25`越大，说明教师在不同车辆实例上的质量差异越明显。

## 独立稳定性检验

- 翻转一致性误差与真实局部误差 Spearman rho：{correlation['rho']}
- p-value：{correlation['p_value']}
- 有效实例数：{correlation['samples']}

该相关性使用不依赖GT的翻转一致性误差预测真实局部误差，比直接将GT误差
转换成可靠性后再做相关分析更有研究意义。

## 假设判断

- 教师实例误差存在明显差异：{hypothesis['teacher_error_varies_across_instances']}
- 视图稳定性能够预测真实误差：{hypothesis['view_stability_predicts_true_error']}
- 综合结论：`{hypothesis['conclusion']}`

`supported`可以进入可靠性输出蒸馏实验；`partially_supported`表示可以先使用
监督可靠性，但需要继续改进独立稳定性信号；`not_supported`时应先检查软实例
区域、尺度参数和教师输出，不建议直接增加蒸馏模块。

## 推荐可靠性尺度

```json
{json.dumps(summary['recommended_taus_from_this_split'], ensure_ascii=False, indent=2)}
```

这些数值只能由训练/验证数据确定，禁止根据测试集重新调整。
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    seed = int(experiment.get("seed", 42))
    set_random_seed(seed, deterministic=False)

    device = torch.device(args.device or experiment.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求使用CUDA，但当前PyTorch检测不到GPU")
    if args.split == "test":
        print("警告：当前正在分析test。请勿使用test结果选择tau或修改方法。")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(
        config=config,
        split=args.split,
        device=device,
        workers_override=args.workers,
        max_samples=args.max_samples,
    )

    model = CSRNetTeacher(pretrained_frontend=False).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("checkpoint中没有model_state或state_dict")
    model.load_state_dict(state)
    model.eval()

    mask_config = InstanceMaskConfig(
        sigma_scale=args.mask_sigma_scale,
        sigma_min=args.mask_sigma_min,
        sigma_max=args.mask_sigma_max,
        truncate=args.mask_truncate,
    )
    mask_config.validate()
    amp_enabled = bool(args.amp) and device.type == "cuda"

    print(f"设备：{device}")
    print(f"划分：{args.split}，图像数：{len(loader.dataset)}")
    print(f"教师：{args.checkpoint}（epoch={checkpoint.get('epoch')}）")
    print(
        "实例区域："
        f"sigma_scale={mask_config.sigma_scale}, "
        f"sigma=[{mask_config.sigma_min}, {mask_config.sigma_max}], "
        f"truncate={mask_config.truncate}"
    )
    print(f"AMP：{amp_enabled}，输出目录：{output_dir}")

    object_rows, image_rows = collect_raw_statistics(
        model=model,
        loader=loader,
        device=device,
        mask_config=mask_config,
        amp_enabled=amp_enabled,
    )
    summary, taus = finalize_statistics(
        object_rows=object_rows,
        image_rows=image_rows,
        tau_map_request=args.tau_map,
        tau_count_request=args.tau_count,
        tau_consistency_request=args.tau_consistency,
        minimum_reliability=args.minimum_reliability,
    )

    summary.update(
        {
            "split": args.split,
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "amp": amp_enabled,
            "mask_configuration": {
                "sigma_scale": mask_config.sigma_scale,
                "sigma_min": mask_config.sigma_min,
                "sigma_max": mask_config.sigma_max,
                "truncate": mask_config.truncate,
            },
        }
    )

    _write_csv(output_dir / "per_object_reliability.csv", object_rows)
    _write_csv(output_dir / "per_image_summary.csv", image_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _plot_analysis(
        object_rows,
        summary["groups"]["size"],
        summary["groups"]["density"],
        output_dir,
    )
    _write_markdown_report(
        output_dir=output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        summary=summary,
    )

    # 自动tau确定后，第二遍只处理平均可靠性最低的少量样本，避免把整个验证集
    # 的图像和密度Tensor长期保存在内存中。
    save_selected_visualizations(
        model=model,
        loader=loader,
        device=device,
        mask_config=mask_config,
        object_rows=object_rows,
        image_rows=image_rows,
        output_dir=output_dir,
        limit=args.num_visualizations,
        amp_enabled=amp_enabled,
    )

    hypothesis = summary["hypothesis_check"]
    correlation = summary["correlations"][
        "view_consistency_error_vs_local_map_error"
    ]
    metrics = summary["teacher_image_metrics"]
    print("\n教师实例可靠性分析完成")
    print(f"- 图像级MAE/RMSE：{metrics['mae']:.3f}/{metrics['rmse']:.3f}")
    print(
        "- 推荐tau："
        f"map={taus['map']:.8f}, count={taus['count']:.8f}, "
        f"consistency={taus['consistency']:.8f}"
    )
    print(
        "- 翻转不稳定性 vs 真实局部误差："
        f"rho={correlation['rho']}，p={correlation['p_value']}"
    )
    print(f"- 研究假设结论：{hypothesis['conclusion']}")
    print(f"- 详细结果：{output_dir}")


if __name__ == "__main__":
    main()
